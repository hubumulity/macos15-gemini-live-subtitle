#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macOS 15 (Sequoia) Gemini 3.5 Live Translate 实时系统音频中文字幕软件
===========================================================================
基于 Google Gemini Live API (`gemini-3.5-live-translate-preview`) 构建的桌面级实时字幕工具。

核心特性：
1. 原生支持 macOS 15 (Sequoia) 系统音频内录：
   - 模式 A（推荐，免装虚拟声卡）：自动编译并调用内嵌的 Swift `ScreenCaptureKit` 音频捕获器，
     直接抓取 macOS 系统桌面音频并输出 16kHz 16-bit 单声道 PCM 流。
   - 模式 B（CoreAudio 虚拟声卡/输入设备）：支持 BlackHole 2ch、Loopback 或任意麦克风/聚合设备，
     内置实时重采样与声道下混。
2. 三重语气词与喘息/呻吟声过滤引擎：
   - 第一层（声学频谱门限 & 呼吸噪声过滤）：实时计算 RMS 能量、过零率 (ZCR) 与频谱平坦度
     (Spectral Flatness) 及人声共振峰频带占比 (300Hz-3400Hz)，精准拦截宽频喘息声、叹气声、
     麦克风吹气声与短促非语音呻吟。
   - 第二层（Gemini 3.5 Live 系统级指令约束）：通过 `system_instruction` 与 `TranslationConfig`
     指定 `target_language_code="zh-Hans"`，在模型生成侧抑制无意义语气词与非语义发声。
   - 第三层（中英日多语种语气词正则清洗）：实时剔除字幕首尾及独立的“嗯、啊、哦、哎、呃、唔、哼、呀”
     及 `[喘息]`、`(叹气)` 等非实质内容。
3. 流式字幕防重复与去重算法 (Anti-Duplication Engine)：
   - 针对 Live API 流式 `output_transcription` 增量分片，结合 `SequenceMatcher` 相似度比对、
     首尾重叠合并 (Overlap Merging)、句内 N-gram 连续复读折叠（如“好的好的好的”->“好的”）以及
     滑动时间窗历史去重，彻底消除静音或长句切换时的重复刷屏。
4. 永不休眠与长间隙秒级唤醒 (Zero-Sleep & Instant Wakeup)：
   - 进程级防休眠：自动启动 macOS `caffeinate -disu -w <pid>` 防止 macOS 15 App Nap 挂起后台线程。
   - 预卷环形缓冲 (300ms Pre-roll Ring Buffer)：即使经过数小时静音，第一声人声出现前 300ms 的音频
     也会完整送入模型，绝不吞字。
   - 静音期零帧心跳保活 (Comfort-Silence Keepalive)：长间隙静音期停止推送环境底噪（防止模型幻觉输出），
     每 8 秒发送一次微弱零电平保活帧维持 WebSocket 活跃。
   - 无缝会话恢复 (`SessionResumptionConfig` + `ContextWindowCompressionConfig`)：突破 Live API
     单次连接 10 分钟/15 分钟上下文限制，遇到 `go_away` 或网络闪断时自动携带 `new_handle` 秒级重连。
5. 双界面引擎支持：
   - 若系统安装了 `tkinter`，默认启动 macOS 原生置顶透明悬浮窗 (`--ui tk`)；
   - 若未安装 `tkinter` 或指定 `--ui web`，自动启动零依赖毛玻璃悬浮字幕 Web HUD (`http://127.0.0.1:18765`)
     并调用 macOS 独立 App 窗口模式打开。
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable, Deque, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np

# 可选导入 Tkinter (兼容未安装 python-tk 的 Homebrew Python 环境)
try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TKINTER = True
except ImportError:
    tk = None
    ttk = None
    HAS_TKINTER = False

# 尝试导入 Google GenAI SDK (支持 Gemini 3.5 Live Translate)
try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
except ImportError:
    genai = None
    types = None
    HAS_GENAI_SDK = False

# 尝试导入 SoundDevice / PyAudio 作为 CoreAudio 设备捕获后备
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    pyaudio = None
    HAS_PYAUDIO = False


# ==============================================================================
# 日志与常量配置
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GeminiLiveSubtitle")

CONFIG_DIR = Path.home() / ".config" / "macos15_gemini_live_subtitle"
CONFIG_FILE = CONFIG_DIR / "settings.json"
SWIFT_HELPER_DIR = CONFIG_DIR / "bin"
SWIFT_HELPER_BIN = SWIFT_HELPER_DIR / "sck_audio_helper"

# Gemini Live API 音频规范：输入必须为 16kHz, 16-bit, 单声道小端 PCM
TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # int16 = 2 bytes
CHUNK_DURATION_MS = 100  # 官方推荐 100ms 分片
CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1600 samples
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH_BYTES  # 3200 bytes

DEFAULT_MODEL = "gemini-3.5-live-translate-preview"
FALLBACK_MODEL = "gemini-3.1-flash-live-preview"
TARGET_LANGUAGE_BCP47 = "zh-Hans"  # 简体中文官方 BCP-47 代码


# ==============================================================================
# 1. 内嵌 macOS 15 ScreenCaptureKit 原生系统音频捕获器 (Swift 源码)
# ==============================================================================
SWIFT_SCK_SOURCE = r"""
import Foundation
import ScreenCaptureKit
import CoreMedia
import AVFoundation

@available(macOS 13.0, *)
class SystemAudioCaptureEngine: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let audioQueue = DispatchQueue(label: "com.gemini.livesubtitle.audioqueue", qos: .userInteractive)

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            FileHandle.standardError.write("ERROR: No display found for ScreenCaptureKit.\n".data(using: .utf8)!)
            exit(1)
        }

        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 16000
        config.channelCount = 1
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let scStream = SCStream(filter: filter, configuration: config, delegate: self)
        try scStream.addStreamOutput(self, type: .audio, sampleHandlerQueue: audioQueue)
        try await scStream.startCapture()
        self.stream = scStream
        FileHandle.standardError.write("INFO: ScreenCaptureKit system audio stream started (16kHz Mono).\n".data(using: .utf8)!)
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }

        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        let status = CMBlockBufferGetDataPointer(
            blockBuffer,
            atOffset: 0,
            lengthAtOffsetOut: nil,
            totalLengthOut: &length,
            dataPointerOut: &dataPointer
        )
        guard status == kCMBlockBufferNoErr, let ptr = dataPointer, length > 0 else { return }

        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbdPtr = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc) else { return }
        let asbd = asbdPtr.pointee

        if (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0 && asbd.mBitsPerChannel == 32 {
            let floatCount = length / MemoryLayout<Float32>.size
            ptr.withMemoryRebound(to: Float32.self, capacity: floatCount) { floatPtr in
                var pcm16 = [Int16](repeating: 0, count: floatCount)
                for i in 0..<floatCount {
                    let clamped = max(-1.0, min(1.0, floatPtr[i]))
                    pcm16[i] = Int16(clamped * 32767.0)
                }
                pcm16.withUnsafeBufferPointer { bufPtr in
                    let outData = Data(buffer: bufPtr)
                    FileHandle.standardOutput.write(outData)
                }
            }
        } else if asbd.mBitsPerChannel == 16 {
            let outData = Data(bytes: ptr, count: length)
            FileHandle.standardOutput.write(outData)
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("ERROR: SCStream stopped: \(error.localizedDescription)\n".data(using: .utf8)!)
        exit(2)
    }
}

if #available(macOS 13.0, *) {
    let engine = SystemAudioCaptureEngine()
    Task {
        do {
            try await engine.start()
        } catch {
            FileHandle.standardError.write("ERROR: Failed to start ScreenCaptureKit: \(error.localizedDescription)\n".data(using: .utf8)!)
            exit(1)
        }
    }
    RunLoop.main.run()
} else {
    FileHandle.standardError.write("ERROR: macOS 13.0 or newer required.\n".data(using: .utf8)!)
    exit(1)
}
"""


# ==============================================================================
# 2. 用户配置管理
# ==============================================================================
@dataclass
class AppSettings:
    api_key: str = ""
    model_name: str = DEFAULT_MODEL
    capture_backend: str = "auto"  # "auto", "screencapturekit", "audio_device"
    audio_device_name: str = ""    # 指定 BlackHole 2ch 等设备名（为空则自动匹配）
    vad_rms_threshold: float = 0.008      # 语音激活 RMS 阈值
    breath_filter_strength: float = 0.65  # 喘息/呻吟频谱过滤强度 (0.0 ~ 1.0)
    dedup_similarity: float = 0.82        # 重复字幕判定阈值
    font_size: int = 24
    window_alpha: float = 0.88
    show_bilingual: bool = False          # 是否同时显示源语言识别文本
    echo_target_language: bool = False    # 当源音频本身是中文时是否复述

    @classmethod
    def load(cls) -> "AppSettings":
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        env_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                settings = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
                if not settings.api_key and env_key:
                    settings.api_key = env_key
                return settings
            except Exception as exc:
                logger.warning("读取配置文件失败，使用默认配置: %s", exc)
        return cls(api_key=env_key)

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(self.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except OSError:
            pass


# ==============================================================================
# 3. 声学前端：VAD + 喘息/呻吟频谱过滤器 + 300ms 预卷缓冲
# ==============================================================================
class AcousticSpeechFilter:
    """
    区分有效人类语音与喘息声(Panting/Breathing)、呻吟(Groaning)、叹气及底噪的声学过滤器。
    并在静音转语音的瞬间提供 300ms Pre-roll 预卷音频，确保长间隙后唤醒不丢字。
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        rms_threshold: float = 0.008,
        breath_filter_strength: float = 0.65,
    ) -> None:
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.breath_filter_strength = breath_filter_strength

        # 300ms 预卷缓冲 (3 个 100ms 分片)
        self.preroll_buffer: Deque[bytes] = deque(maxlen=3)
        # 语音尾部保持帧数 (Hangover: 1.2 秒 = 12 个 100ms 分片，防止句中停顿被切断)
        self.hangover_max_frames = 12
        self.hangover_counter = 0
        self.in_speech_segment = False
        self.consecutive_voiced_frames = 0

    def update_params(self, rms_threshold: float, breath_filter_strength: float) -> None:
        self.rms_threshold = max(0.001, float(rms_threshold))
        self.breath_filter_strength = min(1.0, max(0.0, float(breath_filter_strength)))

    def analyze_chunk(self, pcm16_bytes: bytes) -> Tuple[bool, List[bytes], dict]:
        """
        分析单个 100ms PCM16 音频块。
        返回: (should_send_audio, chunks_to_send, metrics_dict)
        """
        if len(pcm16_bytes) < 4:
            return False, [], {"rms": 0.0, "is_speech": False, "reason": "empty"}

        samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))

        # 1. 能量门限检测
        if rms < self.rms_threshold:
            self.consecutive_voiced_frames = 0
            if self.in_speech_segment and self.hangover_counter > 0:
                self.hangover_counter -= 1
                return True, [pcm16_bytes], {"rms": rms, "is_speech": True, "reason": "hangover"}
            self.in_speech_segment = False
            self.preroll_buffer.append(pcm16_bytes)
            return False, [], {"rms": rms, "is_speech": False, "reason": "silence"}

        # 2. 频谱特征计算：区分清晰语音 vs 喘息/吹气/无意义呻吟
        zero_crossings = float(np.mean(np.abs(np.diff(np.signbit(samples)))))

        windowed = samples * np.hanning(len(samples))
        mag_spectrum = np.abs(np.fft.rfft(windowed)) + 1e-10
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / self.sample_rate)

        # 频谱平坦度 (Spectral Flatness = 几何平均 / 算术平均)
        log_mean = float(np.mean(np.log(mag_spectrum)))
        geom_mean = math.exp(log_mean)
        arith_mean = float(np.mean(mag_spectrum))
        spectral_flatness = geom_mean / (arith_mean + 1e-10)

        # 核心人声共振峰频带能量比 (300Hz ~ 3400Hz 占总能量比例)
        total_energy = float(np.sum(mag_spectrum ** 2)) + 1e-10
        speech_band_mask = (freqs >= 300.0) & (freqs <= 3400.0)
        low_rumble_mask = freqs < 220.0
        speech_band_ratio = float(np.sum(mag_spectrum[speech_band_mask] ** 2)) / total_energy
        low_rumble_ratio = float(np.sum(mag_spectrum[low_rumble_mask] ** 2)) / total_energy

        flatness_limit = 0.55 - 0.25 * self.breath_filter_strength
        zcr_limit = 0.42 - 0.14 * self.breath_filter_strength
        min_speech_ratio = 0.22 + 0.18 * self.breath_filter_strength

        is_breath_or_panting = (
            (spectral_flatness > flatness_limit and zcr_limit < zero_crossings)
            or (speech_band_ratio < min_speech_ratio and spectral_flatness > flatness_limit * 0.85)
        )

        is_low_groan = (
            low_rumble_ratio > (0.82 - 0.12 * self.breath_filter_strength)
            and speech_band_ratio < 0.20
            and zero_crossings < 0.06
        )

        if is_breath_or_panting or is_low_groan:
            reason = "breath_filtered" if is_breath_or_panting else "groan_filtered"
            if self.in_speech_segment and self.hangover_counter > 0:
                self.hangover_counter -= 1
                silent_replacement = bytes(len(pcm16_bytes))
                return True, [silent_replacement], {"rms": rms, "is_speech": False, "reason": reason}
            return False, [], {"rms": rms, "is_speech": False, "reason": reason}

        # 3. 确认是有效人声
        self.consecutive_voiced_frames += 1
        self.hangover_counter = self.hangover_max_frames

        if not self.in_speech_segment:
            self.in_speech_segment = True
            chunks = list(self.preroll_buffer) + [pcm16_bytes]
            self.preroll_buffer.clear()
            return True, chunks, {"rms": rms, "is_speech": True, "reason": "speech_onset"}

        return True, [pcm16_bytes], {"rms": rms, "is_speech": True, "reason": "speech_active"}


# ==============================================================================
# 4. 文本后处理：语气词/呻吟词过滤 + 流式防重复去重引擎
# ==============================================================================
class SubtitleCleanerAndDeduplicator:
    """
    负责对 Gemini 3.5 Live Translate 返回的简体中文字幕进行：
    1. 语气词（嗯、啊、哦、哎、呃、唔、哼、哈等）及声学描述标签（[喘息]、(叹气)）剥离；
    2. 句内连续重复词/短语折叠（防止模型复读）；
    3. 流式分片拼接与跨句滑动窗口相似度去重。
    """

    NON_VERBAL_TAG_RE = re.compile(
        r"[\[\(（【]\s*(?:喘息|喘气|呻吟|叹气|呼吸|呼吸声|急促呼吸|娇喘|闷哼|哼声|咳嗽|清嗓|"
        r"笑声|哭声|杂音|噪音|音乐|背景音|静音|空白|sigh|groan|panting|breathing|moan|laugh|noise|music)\s*[\]\)）】]",
        re.IGNORECASE,
    )

    PURE_FILLER_SENTENCE_RE = re.compile(
        r"^[\s\.,!?;:，。！？；：、…~～\-—\'\""
        r"嗯啊哦哎呃唔哼哈噢呀哟诶欸喔呵呸嘘嘶"
        r"uh|um|ah|oh|er|hm|hmm|mm|mhm|huh|eh|ha|hah|sigh|groan|moan]+$",
        re.IGNORECASE,
    )

    LEADING_FILLER_RE = re.compile(
        r"^(?:[嗯啊哦哎呃唔哼噢呀哟诶欸喔]+[\s,，。、！!？?\.~～…]+)+",
    )

    TRAILING_FILLER_RE = re.compile(
        r"(?:[,，。、！!？?\s]+[嗯啊哦哎呃唔哼噢]+)+[\s\.,!?;:，。！？…~～]*$",
    )

    MID_ISOLATED_FILLER_RE = re.compile(
        r"([,，。！？\s])[嗯啊哦哎呃唔哼噢]{1,4}([,，。！？\s])"
    )

    REPEATED_PHRASE_RE = re.compile(r"(.{1,8}?)\1{2,}")

    def __init__(self, similarity_threshold: float = 0.82) -> None:
        self.similarity_threshold = similarity_threshold
        self.recent_finalized: Deque[Tuple[str, float]] = deque(maxlen=20)
        self.last_emitted_live_text: str = ""
        self.current_turn_buffer: str = ""

    def update_threshold(self, similarity_threshold: float) -> None:
        self.similarity_threshold = min(0.98, max(0.50, float(similarity_threshold)))

    @staticmethod
    def _normalize_for_compare(text: str) -> str:
        return re.sub(r"[\s\.,!?;:，。！？；：、…~～\-—\'\"“”‘’（）\(\)\[\]【】]", "", text).lower()

    def clean_text(self, raw_text: str) -> str:
        if not raw_text:
            return ""

        text = raw_text.strip()
        text = self.NON_VERBAL_TAG_RE.sub("", text).strip()
        if not text:
            return ""

        if self.PURE_FILLER_SENTENCE_RE.match(text):
            return ""

        text = self.LEADING_FILLER_RE.sub("", text)
        text = self.MID_ISOLATED_FILLER_RE.sub(r"\1", text)
        text = self.TRAILING_FILLER_RE.sub("", text)
        text = self.REPEATED_PHRASE_RE.sub(r"\1", text)

        text = text.strip(" ,，、")
        if not text or self.PURE_FILLER_SENTENCE_RE.match(text):
            return ""

        return text

    def append_streaming_fragment(self, fragment: str) -> Optional[str]:
        if not fragment:
            return None

        if not self.current_turn_buffer:
            self.current_turn_buffer = fragment
        elif fragment.startswith(self.current_turn_buffer):
            self.current_turn_buffer = fragment
        elif self.current_turn_buffer.endswith(fragment.strip()):
            pass
        else:
            self.current_turn_buffer = self._merge_overlap(self.current_turn_buffer, fragment)

        cleaned = self.clean_text(self.current_turn_buffer)
        if not cleaned:
            return None

        if self._is_duplicate_of_history(cleaned):
            return None

        if cleaned == self.last_emitted_live_text:
            return None

        self.last_emitted_live_text = cleaned
        return cleaned

    def finalize_turn(self) -> Optional[str]:
        cleaned = self.clean_text(self.current_turn_buffer)
        self.current_turn_buffer = ""
        self.last_emitted_live_text = ""

        if not cleaned:
            return None

        if self._is_duplicate_of_history(cleaned):
            return None

        now = time.monotonic()
        norm_new = self._normalize_for_compare(cleaned)
        if self.recent_finalized:
            last_text, last_ts = self.recent_finalized[-1]
            norm_last = self._normalize_for_compare(last_text)
            if norm_new and norm_last and norm_new.startswith(norm_last) and (now - last_ts) < 8.0:
                self.recent_finalized[-1] = (cleaned, now)
                return cleaned

        self.recent_finalized.append((cleaned, now))
        return cleaned

    def _is_duplicate_of_history(self, candidate: str) -> bool:
        norm_cand = self._normalize_for_compare(candidate)
        if not norm_cand:
            return True

        now = time.monotonic()
        for past_text, ts in reversed(self.recent_finalized):
            if now - ts > 45.0:
                continue
            norm_past = self._normalize_for_compare(past_text)
            if not norm_past:
                continue
            if norm_cand == norm_past:
                return True
            if len(norm_cand) >= 4 and norm_cand in norm_past and (now - ts) < 12.0:
                return True
            ratio = SequenceMatcher(None, norm_cand, norm_past).ratio()
            if ratio >= self.similarity_threshold:
                return True
        return False

    @staticmethod
    def _merge_overlap(left: str, right: str) -> str:
        max_k = min(len(left), len(right), 24)
        for k in range(max_k, 1, -1):
            if left.endswith(right[:k]):
                return left + right[k:]
        return left + right


# ==============================================================================
# 5. macOS 15 系统音频捕获统一管理器 (ScreenCaptureKit + CoreAudio 双引擎)
# ==============================================================================
class MacOSAudioCaptureManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.running = False
        self._sck_proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None

    @staticmethod
    def ensure_sck_helper_compiled() -> Tuple[bool, str]:
        if sys.platform != "darwin":
            return False, "当前非 macOS 系统，无法编译 ScreenCaptureKit 组件。"

        # 若处于 PyInstaller 打包后的 macOS .app 环境，优先使用内置预编译好的二进制
        bundled_candidates = []
        if getattr(sys, "frozen", False):
            meipass = Path(getattr(sys, "_MEIPASS", ""))
            bundled_candidates.extend([
                meipass / "bin" / "sck_audio_helper",
                Path(sys.executable).resolve().parent / "bin" / "sck_audio_helper",
                Path(sys.executable).resolve().parent.parent / "Resources" / "bin" / "sck_audio_helper",
            ])
        for candidate in bundled_candidates:
            if candidate.exists():
                return True, str(candidate)

        SWIFT_HELPER_DIR.mkdir(parents=True, exist_ok=True)
        swift_src_path = SWIFT_HELPER_DIR / "sck_audio_helper.swift"

        if SWIFT_HELPER_BIN.exists() and swift_src_path.exists():
            try:
                if swift_src_path.read_text(encoding="utf-8") == SWIFT_SCK_SOURCE:
                    return True, str(SWIFT_HELPER_BIN)
            except OSError:
                pass

        swiftc_path = shutil.which("swiftc")
        if not swiftc_path:
            return False, "未检测到 `swiftc` 编译器（可执行 `xcode-select --install` 安装或切换至 BlackHole 模式）。"

        try:
            swift_src_path.write_text(SWIFT_SCK_SOURCE, encoding="utf-8")
            cmd = [
                swiftc_path,
                "-O",
                str(swift_src_path),
                "-o",
                str(SWIFT_HELPER_BIN),
                "-framework", "ScreenCaptureKit",
                "-framework", "CoreMedia",
                "-framework", "AVFoundation",
                "-framework", "Foundation",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            if proc.returncode != 0:
                return False, f"Swift 编译失败: {proc.stderr.strip()}"
            os.chmod(SWIFT_HELPER_BIN, 0o755)
            return True, str(SWIFT_HELPER_BIN)
        except Exception as exc:
            return False, f"编译 ScreenCaptureKit 助手异常: {exc}"

    def start(self, on_pcm_chunk: Callable[[bytes], None], on_error: Callable[[str], None]) -> str:
        self.running = True
        backend = self.settings.capture_backend

        if backend in ("auto", "screencapturekit") and sys.platform == "darwin":
            ok, msg = self.ensure_sck_helper_compiled()
            if ok:
                self._sck_bin_path = msg
                self._reader_thread = threading.Thread(
                    target=self._run_sck_loop,
                    args=(on_pcm_chunk, on_error),
                    daemon=True,
                )
                self._reader_thread.start()
                return "ScreenCaptureKit (macOS 15 原生系统内录)"
            elif backend == "screencapturekit":
                on_error(msg)
                return f"启动失败 ({msg})"
            else:
                logger.info("ScreenCaptureKit 不可用 (%s)，自动回退至 CoreAudio 设备模式", msg)

        self._reader_thread = threading.Thread(
            target=self._run_device_loop,
            args=(on_pcm_chunk, on_error),
            daemon=True,
        )
        self._reader_thread.start()
        dev_label = self.settings.audio_device_name or "默认系统音频输入/BlackHole"
        return f"CoreAudio 设备 ({dev_label})"

    def stop(self) -> None:
        self.running = False
        if self._sck_proc is not None:
            try:
                self._sck_proc.terminate()
                self._sck_proc.wait(timeout=2)
            except Exception:
                try:
                    self._sck_proc.kill()
                except Exception:
                    pass
            self._sck_proc = None

    def _run_sck_loop(self, on_pcm_chunk: Callable[[bytes], None], on_error: Callable[[str], None]) -> None:
        while self.running:
            try:
                self._sck_proc = subprocess.Popen(
                    [self._sck_bin_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=CHUNK_BYTES * 4,
                )
                assert self._sck_proc.stdout is not None
                while self.running:
                    chunk = self._sck_proc.stdout.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    if len(chunk) < CHUNK_BYTES:
                        chunk = chunk + bytes(CHUNK_BYTES - len(chunk))
                    on_pcm_chunk(chunk)

                if self.running and self._sck_proc is not None:
                    err_out = ""
                    if self._sck_proc.stderr:
                        err_out = self._sck_proc.stderr.read().decode("utf-8", errors="replace").strip()
                    if "permission" in err_out.lower() or "denied" in err_out.lower():
                        on_error("请在「系统设置 -> 隐私与安全性 -> 录屏与系统录音」中授予终端/Python 权限。")
                        break
                    time.sleep(1.0)
            except Exception as exc:
                if self.running:
                    on_error(f"ScreenCaptureKit 采集异常: {exc}")
                    time.sleep(2.0)

    def _run_device_loop(self, on_pcm_chunk: Callable[[bytes], None], on_error: Callable[[str], None]) -> None:
        target_name = self.settings.audio_device_name.strip().lower()

        if HAS_SOUNDDEVICE:
            try:
                dev_idx = None
                dev_sr = TARGET_SAMPLE_RATE
                dev_ch = 1
                for idx, dev in enumerate(sd.query_devices()):
                    if dev.get("max_input_channels", 0) > 0:
                        name_lower = dev["name"].lower()
                        if (target_name and target_name in name_lower) or (
                            not target_name and ("blackhole" in name_lower or "loopback" in name_lower)
                        ):
                            dev_idx = idx
                            dev_sr = int(dev.get("default_samplerate", TARGET_SAMPLE_RATE))
                            dev_ch = min(2, int(dev.get("max_input_channels", 1)))
                            break

                blocksize = int(dev_sr * CHUNK_DURATION_MS / 1000)

                with sd.InputStream(
                    device=dev_idx,
                    samplerate=dev_sr,
                    channels=dev_ch,
                    dtype="float32",
                    blocksize=blocksize,
                ) as stream:
                    while self.running:
                        data, _ = stream.read(blocksize)
                        pcm_bytes = self._convert_float_to_16k_pcm16(data, dev_sr)
                        on_pcm_chunk(pcm_bytes)
                return
            except Exception as exc:
                on_error(f"SoundDevice 音频流异常: {exc}")
                return

        if HAS_PYAUDIO:
            try:
                pa = pyaudio.PyAudio()
                dev_idx = None
                dev_sr = TARGET_SAMPLE_RATE
                dev_ch = 1
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    if info.get("maxInputChannels", 0) > 0:
                        name_lower = str(info.get("name", "")).lower()
                        if (target_name and target_name in name_lower) or (
                            not target_name and ("blackhole" in name_lower or "loopback" in name_lower)
                        ):
                            dev_idx = i
                            dev_sr = int(info.get("defaultSampleRate", TARGET_SAMPLE_RATE))
                            dev_ch = min(2, int(info.get("maxInputChannels", 1)))
                            break

                frames_per_buffer = int(dev_sr * CHUNK_DURATION_MS / 1000)
                stream = pa.open(
                    format=pyaudio.paFloat32,
                    channels=dev_ch,
                    rate=dev_sr,
                    input=True,
                    input_device_index=dev_idx,
                    frames_per_buffer=frames_per_buffer,
                )
                while self.running:
                    raw = stream.read(frames_per_buffer, exception_on_overflow=False)
                    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, dev_ch)
                    pcm_bytes = self._convert_float_to_16k_pcm16(arr, dev_sr)
                    on_pcm_chunk(pcm_bytes)
                stream.stop_stream()
                stream.close()
                pa.terminate()
                return
            except Exception as exc:
                on_error(f"PyAudio 音频流异常: {exc}")
                return

        on_error("未检测到可用的音频捕获后端。请使用 macOS 15 ScreenCaptureKit 或安装 `pip install sounddevice`。")

    @staticmethod
    def _convert_float_to_16k_pcm16(audio_arr: np.ndarray, orig_sr: int) -> bytes:
        if audio_arr.ndim == 2 and audio_arr.shape[1] > 1:
            mono = np.mean(audio_arr, axis=1)
        else:
            mono = audio_arr.reshape(-1)

        if orig_sr != TARGET_SAMPLE_RATE and len(mono) > 1:
            target_len = CHUNK_SAMPLES
            x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            mono = np.interp(x_new, x_old, mono)

        clamped = np.clip(mono, -1.0, 1.0)
        pcm16 = (clamped * 32767.0).astype("<i2")
        raw_bytes = pcm16.tobytes()
        if len(raw_bytes) < CHUNK_BYTES:
            raw_bytes += bytes(CHUNK_BYTES - len(raw_bytes))
        elif len(raw_bytes) > CHUNK_BYTES:
            raw_bytes = raw_bytes[:CHUNK_BYTES]
        return raw_bytes


# ==============================================================================
# 6. Gemini 3.5 Live Translate 永不休眠流式会话引擎
# ==============================================================================
class GeminiLiveTranslateService:
    SYSTEM_INSTRUCTION = (
        "你是一个高精度的实时同声传译字幕引擎，负责将输入的系统音频实时翻译为自然、流畅、准确的简体中文（zh-Hans）字幕。\n"
        "严格遵守以下规则：\n"
        "1. 彻底忽略并过滤音频中的所有无意义语气词（如：嗯、啊、哦、哎、呃、唔、哼、哈、uh、um、ah、oh、er、hm、mm、ええと、あの等）。\n"
        "2. 彻底忽略并过滤任何喘息声、急促呼吸声、呻吟声、叹气声、咳嗽声、笑声或背景杂音，绝对不要将它们转写为文字或拟声词（禁止输出“啊…”、“嗯…”、“[喘息]”等）。\n"
        "3. 严禁重复输出已经翻译过的句子，严禁在静音或停顿期间编造、复读任何内容。\n"
        "4. 仅当听到具有明确语义的词汇或句子时才输出简练准确的简体中文译文。"
    )

    def __init__(
        self,
        settings: AppSettings,
        on_subtitle_update: Callable[[str, str, bool], None],
        on_status_change: Callable[[str, str], None],
        on_audio_level: Callable[[float, str], None],
    ) -> None:
        self.settings = settings
        self.on_subtitle_update = on_subtitle_update
        self.on_status_change = on_status_change
        self.on_audio_level = on_audio_level

        self.acoustic_filter = AcousticSpeechFilter(
            sample_rate=TARGET_SAMPLE_RATE,
            rms_threshold=settings.vad_rms_threshold,
            breath_filter_strength=settings.breath_filter_strength,
        )
        self.deduplicator = SubtitleCleanerAndDeduplicator(
            similarity_threshold=settings.dedup_similarity,
        )
        self.capture_manager = MacOSAudioCaptureManager(settings)

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pcm_queue: Optional[asyncio.Queue[bytes]] = None
        self._resumption_handle: Optional[str] = None
        self._caffeinate_proc: Optional[subprocess.Popen] = None
        self._current_src_transcript: str = ""

    def start(self) -> None:
        if self._running:
            return
        if not self.settings.api_key.strip():
            self.on_status_change("未配置 API Key，请点击右上角「设置」填入", "#FF6B6B")
            return
        if not HAS_GENAI_SDK:
            self.on_status_change("缺少 google-genai 依赖，请先运行安装脚本", "#FF6B6B")
            return

        self._running = True
        self._start_macos_anti_sleep()
        self._worker_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._running = False
        self.capture_manager.stop()
        self._stop_macos_anti_sleep()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)
        self.on_status_change("已停止翻译", "#9AA0A6")

    def reload_runtime_params(self) -> None:
        self.acoustic_filter.update_params(
            self.settings.vad_rms_threshold,
            self.settings.breath_filter_strength,
        )
        self.deduplicator.update_threshold(self.settings.dedup_similarity)

    def _start_macos_anti_sleep(self) -> None:
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            try:
                self._caffeinate_proc = subprocess.Popen(
                    ["caffeinate", "-disu", "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("已启用 macOS caffeinate 防休眠守护 (PID=%s)", os.getpid())
            except Exception as exc:
                logger.warning("启动 caffeinate 失败: %s", exc)

    def _stop_macos_anti_sleep(self) -> None:
        if self._caffeinate_proc is not None:
            try:
                self._caffeinate_proc.terminate()
            except Exception:
                pass
            self._caffeinate_proc = None

    def _build_live_config(self, model_name: str) -> "types.LiveConnectConfig":
        kwargs = {
            "response_modalities": ["AUDIO"],
            "input_audio_transcription": types.AudioTranscriptionConfig(),
            "output_audio_transcription": types.AudioTranscriptionConfig(),
            "system_instruction": types.Content(
                parts=[types.Part.from_text(text=self.SYSTEM_INSTRUCTION)]
            ),
        }

        if hasattr(types, "TranslationConfig"):
            kwargs["translation_config"] = types.TranslationConfig(
                target_language_code=TARGET_LANGUAGE_BCP47,
                echo_target_language=self.settings.echo_target_language,
            )

        if hasattr(types, "ContextWindowCompressionConfig") and hasattr(types, "SlidingWindow"):
            kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
                trigger_tokens=12000,
                sliding_window=types.SlidingWindow(target_tokens=8000),
            )

        if hasattr(types, "SessionResumptionConfig"):
            if self._resumption_handle:
                kwargs["session_resumption"] = types.SessionResumptionConfig(
                    handle=self._resumption_handle
                )
            else:
                kwargs["session_resumption"] = types.SessionResumptionConfig()

        return types.LiveConnectConfig(**kwargs)

    def _run_async_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._pcm_queue = asyncio.Queue(maxsize=300)

        def _on_pcm_chunk(pcm_bytes: bytes) -> None:
            if not self._running or self._loop is None or self._pcm_queue is None:
                return

            should_send, chunks_to_send, metrics = self.acoustic_filter.analyze_chunk(pcm_bytes)
            self.on_audio_level(metrics["rms"], metrics["reason"])

            if should_send:
                for c in chunks_to_send:
                    try:
                        self._loop.call_soon_threadsafe(self._enqueue_nowait, c)
                    except RuntimeError:
                        pass

        def _on_capture_error(err_msg: str) -> None:
            self.on_status_change(err_msg, "#FFB74D")

        capture_desc = self.capture_manager.start(_on_pcm_chunk, _on_capture_error)
        self.on_status_change(f"正在连接 Gemini 3.5 Live ({capture_desc})...", "#64B5F6")

        try:
            self._loop.run_until_complete(self._persistent_session_supervisor(capture_desc))
        except Exception as exc:
            logger.exception("异步事件循环异常退出: %s", exc)
        finally:
            self.capture_manager.stop()
            self._loop.close()

    def _enqueue_nowait(self, chunk: bytes) -> None:
        if self._pcm_queue is None:
            return
        if self._pcm_queue.full():
            try:
                self._pcm_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._pcm_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    async def _persistent_session_supervisor(self, capture_desc: str) -> None:
        backoff = 0.5
        active_model = self.settings.model_name.strip() or DEFAULT_MODEL

        while self._running:
            try:
                client = genai.Client(
                    api_key=self.settings.api_key.strip(),
                    http_options={"api_version": "v1beta"},
                )
                config = self._build_live_config(active_model)

                async with client.aio.live.connect(model=active_model, config=config) as session:
                    backoff = 0.5
                    self.on_status_change(
                        f"实时翻译运行中 · {active_model} · {capture_desc}",
                        "#4CAF50",
                    )

                    send_task = asyncio.create_task(self._send_audio_and_keepalive_loop(session))
                    recv_task = asyncio.create_task(self._receive_subtitles_loop(session))

                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    for task in done:
                        if task.exception():
                            raise task.exception()

            except Exception as exc:
                if not self._running:
                    break
                err_str = str(exc)
                logger.warning("Live API 会话重连触发 (%s)，%.1f 秒后恢复...", err_str, backoff)

                if "1007" in err_str or "invalid" in err_str.lower() or "resumption" in err_str.lower():
                    self._resumption_handle = None
                if "not found" in err_str.lower() and active_model == DEFAULT_MODEL:
                    logger.info("自动尝试备用实时模型 %s", FALLBACK_MODEL)
                    active_model = FALLBACK_MODEL

                self.on_status_change(
                    f"连接自动恢复中 ({backoff:.1f}s) · 音频缓冲保护中...",
                    "#FFB74D",
                )
                await asyncio.sleep(backoff)
                backoff = min(5.0, backoff * 1.5)

    async def _send_audio_and_keepalive_loop(self, session) -> None:
        assert self._pcm_queue is not None
        last_send_ts = time.monotonic()
        sent_tail_flush = True
        zero_keepalive_frame = bytes(CHUNK_BYTES)

        while self._running:
            try:
                chunk = await asyncio.wait_for(self._pcm_queue.get(), timeout=1.2)
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={TARGET_SAMPLE_RATE}")
                )
                last_send_ts = time.monotonic()
                sent_tail_flush = False
            except asyncio.TimeoutError:
                now = time.monotonic()
                idle_sec = now - last_send_ts

                if not sent_tail_flush:
                    for _ in range(2):
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=zero_keepalive_frame,
                                mime_type=f"audio/pcm;rate={TARGET_SAMPLE_RATE}",
                            )
                        )
                    sent_tail_flush = True
                    last_send_ts = now
                    final_zh = self.deduplicator.finalize_turn()
                    if final_zh:
                        self.on_subtitle_update(final_zh, self._current_src_transcript, True)
                        self._current_src_transcript = ""

                elif idle_sec >= 8.0:
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=zero_keepalive_frame,
                            mime_type=f"audio/pcm;rate={TARGET_SAMPLE_RATE}",
                        )
                    )
                    last_send_ts = now

    async def _receive_subtitles_loop(self, session) -> None:
        while self._running:
            async for response in session.receive():
                if not self._running:
                    return

                resumption = getattr(response, "session_resumption_update", None)
                if resumption and getattr(resumption, "resumable", False):
                    new_handle = getattr(resumption, "new_handle", None)
                    if new_handle:
                        self._resumption_handle = new_handle

                go_away = getattr(response, "go_away", None)
                if go_away is not None:
                    logger.info("收到服务端 GoAway 轮转通知，准备无缝切换新连接...")
                    return

                server_content = getattr(response, "server_content", None)
                if not server_content:
                    continue

                input_trans = getattr(server_content, "input_transcription", None)
                if input_trans and getattr(input_trans, "text", None):
                    src_piece = input_trans.text.strip()
                    if src_piece:
                        self._current_src_transcript = src_piece

                output_trans = getattr(server_content, "output_transcription", None)
                if output_trans and getattr(output_trans, "text", None):
                    live_zh = self.deduplicator.append_streaming_fragment(output_trans.text)
                    if live_zh:
                        self.on_subtitle_update(live_zh, self._current_src_transcript, False)

                turn_complete = getattr(server_content, "turn_complete", False)
                generation_complete = getattr(server_content, "generation_complete", False)
                if turn_complete or generation_complete:
                    final_zh = self.deduplicator.finalize_turn()
                    if final_zh:
                        self.on_subtitle_update(final_zh, self._current_src_transcript, True)
                    self._current_src_transcript = ""


# ==============================================================================
# 7A. 零依赖本地 Web HUD 悬浮字幕界面 (兼容未安装 tkinter 的 macOS Python 环境)
# ==============================================================================
WEB_HUD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<title>Gemini 3.5 Live 实时中文字幕 (macOS 15)</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", sans-serif;
    background: #0f1218;
    color: #f8fafc;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #181c24;
    padding: 8px 16px;
    border-bottom: 1px solid #262c3a;
    font-size: 13px;
  }
  .status-wrap { display: flex; align-items: center; gap: 8px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #9aa0a6; }
  .controls { display: flex; align-items: center; gap: 8px; }
  button {
    background: #262c3a; color: #f8fafc; border: none; border-radius: 6px;
    padding: 5px 12px; font-size: 12px; cursor: pointer; font-weight: 500;
  }
  button.primary { background: #2563eb; font-weight: 600; }
  button.danger { background: #dc2626; font-weight: 600; }
  .subtitle-stage {
    flex: 1; display: flex; flex-direction: column; justify-content: center;
    align-items: center; padding: 18px 28px; text-align: center;
  }
  .prev-line { color: #94a3b8; font-size: 17px; margin-bottom: 10px; min-height: 24px; }
  .curr-line {
    color: #ffffff; font-size: 26px; font-weight: 700; line-height: 1.4;
    text-shadow: 0 2px 10px rgba(0,0,0,0.65);
  }
  .modal {
    position: fixed; inset: 0; background: rgba(0,0,0,0.75);
    display: none; align-items: center; justify-content: center;
  }
  .modal.open { display: flex; }
  .modal-card {
    background: #1a1f29; border: 1px solid #2e3646; border-radius: 10px;
    width: 480px; max-width: 92vw; padding: 20px;
  }
  .modal-card h3 { margin-bottom: 12px; font-size: 16px; }
  .field { margin-bottom: 12px; }
  .field label { display: block; font-size: 12px; color: #cbd5e1; margin-bottom: 4px; }
  .field input, .field select {
    width: 100%; padding: 8px 10px; border-radius: 6px; border: 1px solid #334155;
    background: #0f172a; color: #f8fafc; font-size: 13px;
  }
</style>
</head>
<body>
  <div class="topbar">
    <div class="status-wrap">
      <div id="dot" class="dot"></div>
      <span id="statusText">就绪 · 点击「开始实时翻译」捕获 macOS 系统音频</span>
    </div>
    <div class="controls">
      <span id="vadBadge" style="color:#94a3b8;font-size:12px;margin-right:8px;">静音热备 (防休眠)</span>
      <button id="toggleBtn" class="primary" onclick="toggleRun()">▶ 开始实时翻译</button>
      <button onclick="openSettings()">⚙ API Key 与过滤设置</button>
    </div>
  </div>
  <div class="subtitle-stage">
    <div id="prevLine" class="prev-line"></div>
    <div id="currLine" class="curr-line">等待系统音频输入…（已开启语气词/喘息声过滤与防重复保护）</div>
  </div>

  <div id="settingsModal" class="modal">
    <div class="modal-card">
      <h3>自定义 API Key 与音频过滤设置</h3>
      <div class="field">
        <label>Google AI Studio API Key (支持免费层级密钥):</label>
        <input id="apiKeyInput" type="password" placeholder="粘贴 AIzaSy... 密钥" />
      </div>
      <div class="field">
        <label>实时翻译模型:</label>
        <select id="modelInput">
          <option value="gemini-3.5-live-translate-preview">gemini-3.5-live-translate-preview</option>
          <option value="gemini-3.1-flash-live-preview">gemini-3.1-flash-live-preview</option>
        </select>
      </div>
      <div class="field">
        <label>macOS 15 音频捕获引擎:</label>
        <select id="backendInput">
          <option value="auto">auto (优先 ScreenCaptureKit 原生内录)</option>
          <option value="screencapturekit">screencapturekit (macOS 15 免虚拟声卡)</option>
          <option value="audio_device">audio_device (BlackHole / CoreAudio)</option>
        </select>
      </div>
      <div class="field">
        <label>语气词与喘息/呻吟过滤强度 (0.1 ~ 0.95，推荐 0.65):</label>
        <input id="breathInput" type="number" step="0.05" min="0.1" max="0.95" value="0.65" />
      </div>
      <div class="field">
        <label>防重复字幕相似度阈值 (0.60 ~ 0.95，推荐 0.82):</label>
        <input id="dedupInput" type="number" step="0.02" min="0.60" max="0.95" value="0.82" />
      </div>
      <div style="display:flex;justify-content:flex-end;gap:8px;margin-top:16px;">
        <button onclick="closeSettings()">取消</button>
        <button class="primary" onclick="saveSettings()">💾 保存并生效</button>
      </div>
    </div>
  </div>

<script>
let isRunning = false;
async function pollState() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();
    document.getElementById('statusText').textContent = d.status_text;
    document.getElementById('dot').style.background = d.status_color;
    document.getElementById('vadBadge').textContent = d.vad_text;
    if (d.prev_subtitle) document.getElementById('prevLine').textContent = d.prev_subtitle;
    if (d.curr_subtitle) document.getElementById('currLine').textContent = d.curr_subtitle;
    isRunning = d.is_running;
    const btn = document.getElementById('toggleBtn');
    btn.textContent = isRunning ? '⏹ 停止翻译' : '▶ 开始实时翻译';
    btn.className = isRunning ? 'danger' : 'primary';
  } catch (e) {}
}
setInterval(pollState, 120);

async function toggleRun() {
  const r = await fetch('/api/toggle', { method: 'POST' });
  const d = await r.json();
  if (d.need_api_key) openSettings();
}
async function openSettings() {
  const r = await fetch('/api/settings');
  const s = await r.json();
  document.getElementById('apiKeyInput').value = s.api_key || '';
  document.getElementById('modelInput').value = s.model_name || 'gemini-3.5-live-translate-preview';
  document.getElementById('backendInput').value = s.capture_backend || 'auto';
  document.getElementById('breathInput').value = s.breath_filter_strength ?? 0.65;
  document.getElementById('dedupInput').value = s.dedup_similarity ?? 0.82;
  document.getElementById('settingsModal').classList.add('open');
}
function closeSettings() {
  document.getElementById('settingsModal').classList.remove('open');
}
async function saveSettings() {
  const body = {
    api_key: document.getElementById('apiKeyInput').value.trim(),
    model_name: document.getElementById('modelInput').value,
    capture_backend: document.getElementById('backendInput').value,
    breath_filter_strength: parseFloat(document.getElementById('breathInput').value || '0.65'),
    dedup_similarity: parseFloat(document.getElementById('dedupInput').value || '0.82'),
  };
  await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  closeSettings();
}
</script>
</body>
</html>
"""


class WebHudSubtitleApp:
    """当系统未安装 python-tk 时自动启用的零依赖 Web HUD 字幕悬浮窗服务。"""

    def __init__(self, port: int = 18765) -> None:
        self.port = port
        self.settings = AppSettings.load()
        self.state = {
            "is_running": False,
            "status_text": "就绪 · 点击「开始实时翻译」捕获 macOS 系统音频",
            "status_color": "#9AA0A6",
            "vad_text": "静音热备 (防休眠)",
            "prev_subtitle": "",
            "curr_subtitle": "等待系统音频输入…（已开启语气词/喘息声过滤与防重复保护）",
        }
        self.service = GeminiLiveTranslateService(
            settings=self.settings,
            on_subtitle_update=self._on_subtitle,
            on_status_change=self._on_status,
            on_audio_level=self._on_level,
        )

    def _on_subtitle(self, zh_text: str, src_text: str, is_final: bool) -> None:
        if is_final:
            self.state["prev_subtitle"] = zh_text
        self.state["curr_subtitle"] = zh_text

    def _on_status(self, msg: str, color: str) -> None:
        self.state["status_text"] = msg
        self.state["status_color"] = color

    def _on_level(self, rms: float, reason: str) -> None:
        bars = min(8, int(rms * 160))
        meter = "▮" * bars + "▯" * (8 - bars)
        reason_map = {
            "speech_active": "人声翻译中",
            "speech_onset": "唤醒捕捉 (含300ms预卷)",
            "hangover": "句尾延续",
            "breath_filtered": "已过滤喘息/呼吸声",
            "groan_filtered": "已过滤低频呻吟",
            "silence": "静音热备 (零帧保活)",
        }
        self.state["vad_text"] = f"{meter} {reason_map.get(reason, '监听中')}"

    def run(self) -> None:
        app_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path == "/api/state":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(app_ref.state, ensure_ascii=False).encode("utf-8"))
                elif self.path == "/api/settings":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(app_ref.settings.__dict__, ensure_ascii=False).encode("utf-8"))
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(WEB_HUD_HTML.encode("utf-8"))

            def do_POST(self):
                if self.path == "/api/toggle":
                    if not app_ref.state["is_running"]:
                        if not app_ref.settings.api_key.strip():
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            self.wfile.write(b'{"need_api_key": true}')
                            return
                        app_ref.state["is_running"] = True
                        app_ref.service.start()
                    else:
                        app_ref.state["is_running"] = False
                        app_ref.service.stop()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"need_api_key": false}')
                elif self.path == "/api/settings":
                    length = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    for k, v in body.items():
                        if hasattr(app_ref.settings, k):
                            setattr(app_ref.settings, k, v)
                    app_ref.settings.save()
                    app_ref.service.reload_runtime_params()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ok": true}')

        server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        url = f"http://127.0.0.1:{self.port}"
        logger.info("已启动 Web HUD 实时字幕界面: %s", url)
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.service.stop()


# ==============================================================================
# 7B. macOS 15 悬浮毛玻璃感字幕 HUD 界面 (Tkinter 原生置顶窗口)
# ==============================================================================
class MacOSSubtitleOverlayApp:
    def __init__(self) -> None:
        self.settings = AppSettings.load()
        self.ui_queue: "queue.Queue[Tuple[str, tuple]]" = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Gemini 3.5 Live 实时中文字幕 (macOS 15)")
        self.root.geometry("920x210+220+680")
        self.root.minsize(560, 140)
        self.root.configure(bg="#121417")

        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", self.settings.window_alpha)
        except tk.TclError:
            pass

        self._drag_offset_x = 0
        self._drag_offset_y = 0
        self._is_running = False
        self._history_lines: Deque[str] = deque(maxlen=200)

        self._build_ui()

        self.service = GeminiLiveTranslateService(
            settings=self.settings,
            on_subtitle_update=lambda zh, src, final: self.ui_queue.put(("subtitle", (zh, src, final))),
            on_status_change=lambda msg, color: self.ui_queue.put(("status", (msg, color))),
            on_audio_level=lambda rms, reason: self.ui_queue.put(("level", (rms, reason))),
        )

        self.root.after(40, self._poll_ui_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if not self.settings.api_key:
            self.root.after(300, self._open_settings_dialog)

    def _build_ui(self) -> None:
        top_bar = tk.Frame(self.root, bg="#1A1D24", height=36)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        top_bar.bind("<ButtonPress-1>", self._on_drag_start)
        top_bar.bind("<B1-Motion>", self._on_drag_motion)

        self.status_dot = tk.Label(
            top_bar, text="●", fg="#9AA0A6", bg="#1A1D24", font=("PingFang SC", 12, "bold")
        )
        self.status_dot.pack(side=tk.LEFT, padx=(12, 4), pady=4)

        self.status_label = tk.Label(
            top_bar,
            text="就绪 · 点击「开始实时翻译」捕获 macOS 系统音频",
            fg="#CBD5E1",
            bg="#1A1D24",
            font=("PingFang SC", 11),
        )
        self.status_label.pack(side=tk.LEFT, padx=4, pady=4)
        self.status_label.bind("<ButtonPress-1>", self._on_drag_start)
        self.status_label.bind("<B1-Motion>", self._on_drag_motion)

        btn_style = {
            "bg": "#262B36",
            "fg": "#F8FAFC",
            "activebackground": "#3B4252",
            "activeforeground": "#FFFFFF",
            "bd": 0,
            "padx": 10,
            "pady": 3,
            "font": ("PingFang SC", 11),
            "cursor": "hand2",
        }

        self.btn_settings = tk.Button(
            top_bar, text="⚙ API Key 与过滤设置", command=self._open_settings_dialog, **btn_style
        )
        self.btn_settings.pack(side=tk.RIGHT, padx=(4, 10), pady=4)

        self.btn_history = tk.Button(
            top_bar, text="📜 字幕记录", command=self._open_history_dialog, **btn_style
        )
        self.btn_history.pack(side=tk.RIGHT, padx=4, pady=4)

        self.btn_toggle = tk.Button(
            top_bar,
            text="▶ 开始实时翻译",
            command=self._toggle_translation,
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            bd=0,
            padx=12,
            pady=3,
            font=("PingFang SC", 11, "bold"),
            cursor="hand2",
        )
        self.btn_toggle.pack(side=tk.RIGHT, padx=4, pady=4)

        self.vad_badge = tk.Label(
            top_bar,
            text="静音待命 (防休眠热备)",
            fg="#94A3B8",
            bg="#1A1D24",
            font=("PingFang SC", 10),
        )
        self.vad_badge.pack(side=tk.RIGHT, padx=10, pady=4)

        body = tk.Frame(self.root, bg="#121417", padx=20, pady=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.bind("<ButtonPress-1>", self._on_drag_start)
        body.bind("<B1-Motion>", self._on_drag_motion)

        self.prev_subtitle_var = tk.StringVar(value="")
        self.prev_subtitle_label = tk.Label(
            body,
            textvariable=self.prev_subtitle_var,
            fg="#94A3B8",
            bg="#121417",
            font=("PingFang SC", max(14, self.settings.font_size - 6)),
            wraplength=860,
            justify=tk.CENTER,
        )
        self.prev_subtitle_label.pack(fill=tk.X, pady=(0, 6))
        self.prev_subtitle_label.bind("<ButtonPress-1>", self._on_drag_start)
        self.prev_subtitle_label.bind("<B1-Motion>", self._on_drag_motion)

        self.curr_subtitle_var = tk.StringVar(
            value="等待系统音频输入…（已开启语气词/喘息声过滤与防重复保护）"
        )
        self.curr_subtitle_label = tk.Label(
            body,
            textvariable=self.curr_subtitle_var,
            fg="#F8FAFC",
            bg="#121417",
            font=("PingFang SC", self.settings.font_size, "bold"),
            wraplength=860,
            justify=tk.CENTER,
        )
        self.curr_subtitle_label.pack(fill=tk.BOTH, expand=True)
        self.curr_subtitle_label.bind("<ButtonPress-1>", self._on_drag_start)
        self.curr_subtitle_label.bind("<B1-Motion>", self._on_drag_motion)

        self.root.bind("<Configure>", self._on_window_resize)

    def _on_drag_start(self, event) -> None:
        self._drag_offset_x = event.x_root - self.root.winfo_x()
        self._drag_offset_y = event.y_root - self.root.winfo_y()

    def _on_drag_motion(self, event) -> None:
        x = event.x_root - self._drag_offset_x
        y = event.y_root - self._drag_offset_y
        self.root.geometry(f"+{x}+{y}")

    def _on_window_resize(self, event) -> None:
        if event.widget == self.root:
            wrap_w = max(420, event.width - 56)
            self.curr_subtitle_label.configure(wraplength=wrap_w)
            self.prev_subtitle_label.configure(wraplength=wrap_w)

    def _toggle_translation(self) -> None:
        if not self._is_running:
            if not self.settings.api_key.strip():
                self._open_settings_dialog()
                return
            self._is_running = True
            self.btn_toggle.configure(text="⏹ 停止翻译", bg="#DC2626", activebackground="#B91C1C")
            self.curr_subtitle_var.set("正在监听 macOS 系统音频…")
            self.service.start()
        else:
            self._is_running = False
            self.btn_toggle.configure(text="▶ 开始实时翻译", bg="#2563EB", activebackground="#1D4ED8")
            self.service.stop()

    def _poll_ui_events(self) -> None:
        try:
            while True:
                evt_type, payload = self.ui_queue.get_nowait()
                if evt_type == "subtitle":
                    zh_text, src_text, is_final = payload
                    display_text = zh_text
                    if self.settings.show_bilingual and src_text:
                        display_text = f"{zh_text}\n[{src_text}]"

                    if is_final:
                        self.prev_subtitle_var.set(zh_text)
                        self.curr_subtitle_var.set(display_text)
                        ts_str = time.strftime("%H:%M:%S")
                        self._history_lines.append(f"[{ts_str}] {zh_text}")
                    else:
                        self.curr_subtitle_var.set(display_text)

                elif evt_type == "status":
                    msg, color = payload
                    self.status_label.configure(text=msg)
                    self.status_dot.configure(fg=color)

                elif evt_type == "level":
                    rms, reason = payload
                    db_bar = min(10, int(rms * 180))
                    meter = "▮" * db_bar + "▯" * (10 - db_bar)
                    reason_map = {
                        "speech_active": ("人声翻译中", "#4ADE80"),
                        "speech_onset": ("唤醒捕捉 (含300ms预卷)", "#38BDF8"),
                        "hangover": ("句尾延续", "#A3E635"),
                        "breath_filtered": ("已过滤喘息/呼吸声", "#FBBF24"),
                        "groan_filtered": ("已过滤低频呻吟/闷哼", "#FBBF24"),
                        "silence": ("静音热备 (零帧保活)", "#94A3B8"),
                    }
                    label_txt, color = reason_map.get(reason, ("监听中", "#94A3B8"))
                    self.vad_badge.configure(text=f"{meter} {label_txt}", fg=color)
        except queue.Empty:
            pass
        finally:
            self.root.after(40, self._poll_ui_events)

    def _open_settings_dialog(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("自定义 API Key 与音频过滤设置")
        win.geometry("560x520")
        win.configure(bg="#1E222B")
        win.attributes("-topmost", True)
        win.transient(self.root)

        pad = {"padx": 16, "pady": 6}

        tk.Label(
            win,
            text="Google AI Studio API Key (支持免费层级密钥):",
            fg="#E2E8F0",
            bg="#1E222B",
            font=("PingFang SC", 11, "bold"),
        ).pack(anchor=tk.W, padx=16, pady=(14, 2))

        api_key_var = tk.StringVar(value=self.settings.api_key)
        api_entry = tk.Entry(
            win,
            textvariable=api_key_var,
            show="•",
            bg="#0F172A",
            fg="#F8FAFC",
            insertbackground="#F8FAFC",
            font=("Menlo", 12),
            bd=1,
            relief=tk.FLAT,
        )
        api_entry.pack(fill=tk.X, **pad)

        show_key_var = tk.BooleanVar(value=False)

        def _toggle_show_key() -> None:
            api_entry.configure(show="" if show_key_var.get() else "•")

        tk.Checkbutton(
            win,
            text="显示明文 API Key",
            variable=show_key_var,
            command=_toggle_show_key,
            bg="#1E222B",
            fg="#94A3B8",
            selectcolor="#0F172A",
            activebackground="#1E222B",
        ).pack(anchor=tk.W, padx=16)

        tk.Label(
            win,
            text="实时翻译模型 (Gemini Live API):",
            fg="#E2E8F0",
            bg="#1E222B",
            font=("PingFang SC", 11, "bold"),
        ).pack(anchor=tk.W, padx=16, pady=(10, 2))

        model_var = tk.StringVar(value=self.settings.model_name)
        model_combo = ttk.Combobox(
            win,
            textvariable=model_var,
            values=[
                "gemini-3.5-live-translate-preview",
                "gemini-3.1-flash-live-preview",
            ],
        )
        model_combo.pack(fill=tk.X, **pad)

        tk.Label(
            win,
            text="macOS 15 系统音频捕获方式:",
            fg="#E2E8F0",
            bg="#1E222B",
            font=("PingFang SC", 11, "bold"),
        ).pack(anchor=tk.W, padx=16, pady=(10, 2))

        backend_var = tk.StringVar(value=self.settings.capture_backend)
        backend_combo = ttk.Combobox(
            win,
            textvariable=backend_var,
            values=[
                "auto",
                "screencapturekit",
                "audio_device",
            ],
        )
        backend_combo.pack(fill=tk.X, **pad)

        tk.Label(
            win,
            text="语气词与喘息/呻吟声过滤强度 (0.0 ~ 1.0，推荐 0.65):",
            fg="#E2E8F0",
            bg="#1E222B",
            font=("PingFang SC", 11, "bold"),
        ).pack(anchor=tk.W, padx=16, pady=(10, 2))

        breath_var = tk.DoubleVar(value=self.settings.breath_filter_strength)
        tk.Scale(
            win,
            variable=breath_var,
            from_=0.1,
            to=0.95,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            bg="#1E222B",
            fg="#F8FAFC",
            highlightthickness=0,
        ).pack(fill=tk.X, padx=16)

        tk.Label(
            win,
            text="防重复字幕相似度阈值 (越低去重越严格，推荐 0.82):",
            fg="#E2E8F0",
            bg="#1E222B",
            font=("PingFang SC", 11, "bold"),
        ).pack(anchor=tk.W, padx=16, pady=(8, 2))

        dedup_var = tk.DoubleVar(value=self.settings.dedup_similarity)
        tk.Scale(
            win,
            variable=dedup_var,
            from_=0.60,
            to=0.95,
            resolution=0.02,
            orient=tk.HORIZONTAL,
            bg="#1E222B",
            fg="#F8FAFC",
            highlightthickness=0,
        ).pack(fill=tk.X, padx=16)

        bilingual_var = tk.BooleanVar(value=self.settings.show_bilingual)
        tk.Checkbutton(
            win,
            text="同时显示源语言识别字幕（双语对照）",
            variable=bilingual_var,
            bg="#1E222B",
            fg="#CBD5E1",
            selectcolor="#0F172A",
            activebackground="#1E222B",
        ).pack(anchor=tk.W, padx=16, pady=6)

        def _save_and_apply() -> None:
            self.settings.api_key = api_key_var.get().strip()
            self.settings.model_name = model_var.get().strip() or DEFAULT_MODEL
            self.settings.capture_backend = backend_var.get().strip() or "auto"
            self.settings.breath_filter_strength = float(breath_var.get())
            self.settings.dedup_similarity = float(dedup_var.get())
            self.settings.show_bilingual = bool(bilingual_var.get())
            self.settings.save()
            self.service.reload_runtime_params()
            win.destroy()
            self.status_label.configure(text="设置已保存并立即生效")

        tk.Button(
            win,
            text="💾 保存设置并生效",
            command=_save_and_apply,
            bg="#10B981",
            fg="#FFFFFF",
            font=("PingFang SC", 12, "bold"),
            bd=0,
            padx=18,
            pady=6,
            cursor="hand2",
        ).pack(pady=14)

    def _open_history_dialog(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("实时翻译字幕历史记录")
        win.geometry("620x420")
        win.configure(bg="#1E222B")
        win.attributes("-topmost", True)

        txt = tk.Text(
            win,
            bg="#0F172A",
            fg="#F8FAFC",
            font=("PingFang SC", 12),
            padx=12,
            pady=12,
            bd=0,
        )
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        content = "\n".join(self._history_lines) if self._history_lines else "暂无已归档的字幕记录。"
        txt.insert(tk.END, content)
        txt.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        self.service.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ==============================================================================
# 8. 自检测试模式 (`--self-test`) 供自动化验证核心算法
# ==============================================================================
def run_self_test() -> int:
    print("=== Running macOS 15 Gemini 3.5 Live Subtitle Engine Self-Test ===")

    # 1. 验证语气词与呻吟/喘息文本清洗器
    dedup = SubtitleCleanerAndDeduplicator(similarity_threshold=0.82)

    assert dedup.clean_text("嗯…啊…哦！") == "", "纯语气词应当被完全过滤"
    assert dedup.clean_text("[喘息] (叹气) 呃，今天我们来讨论系统架构，嗯。") == "今天我们来讨论系统架构", "首尾语气词与喘息标签应被剥离"
    assert dedup.clean_text("好的好的好的好的，马上处理") == "好的，马上处理", "连续复读词组应被折叠"

    # 2. 验证流式增量拼接与历史去重
    s1 = dedup.append_streaming_fragment("嗯，这是一个")
    s2 = dedup.append_streaming_fragment("这是一个实时字幕测试。")
    assert s2 == "这是一个实时字幕测试。", f"流式拼接异常: {s2}"
    final1 = dedup.finalize_turn()
    assert final1 == "这是一个实时字幕测试。", f"归档异常: {final1}"

    dup_live = dedup.append_streaming_fragment("这是一个实时字幕测试！")
    assert dup_live is None, "重复字幕应当在流式阶段被拦截"
    dup_final = dedup.finalize_turn()
    assert dup_final is None, "重复字幕应当在归档阶段被拦截"

    # 3. 验证声学频谱过滤器（静音 vs 宽频喘息白噪 vs 模拟人声谐波）
    acoustic = AcousticSpeechFilter(sample_rate=TARGET_SAMPLE_RATE, rms_threshold=0.008, breath_filter_strength=0.65)

    silent_pcm = bytes(CHUNK_BYTES)
    send_flag, _, meta = acoustic.analyze_chunk(silent_pcm)
    assert not send_flag and meta["reason"] == "silence", f"静音块判定异常: {meta}"

    rng = np.random.default_rng(42)
    breath_noise = (rng.uniform(-0.15, 0.15, size=CHUNK_SAMPLES) * np.tile([1.0, -1.0], CHUNK_SAMPLES // 2)).astype(np.float32)
    breath_pcm = (breath_noise * 32767.0).astype("<i2").tobytes()
    send_flag, _, meta = acoustic.analyze_chunk(breath_pcm)
    assert not send_flag and meta["reason"] == "breath_filtered", f"喘息噪声应当被拦截: {meta}"

    t = np.linspace(0, CHUNK_DURATION_MS / 1000.0, CHUNK_SAMPLES, endpoint=False)
    voice_wave = 0.25 * np.sin(2 * np.pi * 180 * t) + 0.35 * np.sin(2 * np.pi * 720 * t) + 0.20 * np.sin(2 * np.pi * 1600 * t)
    voice_pcm = (np.clip(voice_wave, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    send_flag, chunks, meta = acoustic.analyze_chunk(voice_pcm)
    assert send_flag and meta["reason"] == "speech_onset" and len(chunks) >= 2, f"人声唤醒与300ms预卷异常: {meta}"

    print("=== All Self-Tests Passed Successfully! ===")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(run_self_test())
    use_web = ("--ui" in sys.argv and "web" in sys.argv) or (not HAS_TKINTER)
    if use_web:
        WebHudSubtitleApp().run()
    else:
        MacOSSubtitleOverlayApp().run()