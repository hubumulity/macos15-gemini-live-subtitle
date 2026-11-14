#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macOS 15 (Sequoia) Gemini 3.5 Live Translate 实时系统音频中文字幕软件
===========================================================================
基于 Google Gemini Live API (`gemini-3.5-live-translate-preview`) 构建的桌面级实时字幕工具。

核心特性：
1. 原生支持 macOS 15 (Sequoia) 系统音频内录：
   - 模式 A（推荐，免装虚拟声卡）：自动调用内置预编译的 Swift `ScreenCaptureKit` 音频捕获器，
     直接抓取 macOS 系统桌面音频并输出 16kHz 16-bit 单声道 PCM 流。
   - 模式 B（CoreAudio 虚拟声卡/输入设备）：支持 BlackHole 2ch、Loopback 或任意麦克风/聚合设备，
     内置实时重采样与声道下混。
2. 纯透明悬浮字幕显示 & 5秒自动消隐防挤压布局：
   - 原生支持 macOS 透明背景模式 (`-transparent` + `systemTransparent`)，默认无边框纯文字悬浮显示，
     鼠标移入时自动浮现顶部控制栏，移出后自动隐藏，仅留清晰字幕文字。
   - 历史字幕（上一行）在显示满 5.0 秒后自动消失并释放布局空间 (`pack_forget`)。
   - 动态高度自适应引擎 (`_sync_window_height`)：根据新字幕行数实时动态扩展悬浮窗高度，确保新字幕
     永远不会被历史字幕或窗口边界压缩、截断。
3. 三重语气词与喘息/呻吟声过滤引擎 + 流式防重复去重算法。
4. 永不休眠与长间隙秒级唤醒 (`caffeinate` + 300ms 预卷缓冲 + 8秒零帧心跳 + `SessionResumptionConfig`)。
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
import webbrowser

import numpy as np

# 可选导入 Tkinter
try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TKINTER = True
except ImportError:
    tk = None
    ttk = None
    HAS_TKINTER = False

# 导入 Google GenAI SDK (支持 Gemini 3.5 Live Translate)
try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
except ImportError:
    genai = None
    types = None
    HAS_GENAI_SDK = False

# 可选导入 SoundDevice / PyAudio 作为 CoreAudio 设备捕获后备
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
SAMPLE_WIDTH_BYTES = 2
CHUNK_DURATION_MS = 100
CHUNK_SAMPLES = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1600 samples
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH_BYTES  # 3200 bytes

DEFAULT_MODEL = "gemini-3.5-live-translate-preview"
FALLBACK_MODEL = "gemini-3.1-flash-live-preview"
TARGET_LANGUAGE_BCP47 = "zh-Hans"

# 历史字幕自动消失时间（秒）
HISTORY_EXPIRE_SECONDS = 5.0


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
    capture_backend: str = "auto"         # "auto", "screencapturekit", "audio_device"
    audio_device_name: str = ""           # 指定 BlackHole 2ch 等设备名
    vad_rms_threshold: float = 0.008      # 语音激活 RMS 阈值
    breath_filter_strength: float = 0.65  # 喘息/呻吟频谱过滤强度 (0.0 ~ 1.0)
    dedup_similarity: float = 0.82        # 重复字幕判定阈值
    font_size: int = 26
    transparent_bg: bool = True           # 默认开启背景纯透明（只显示文字）
    history_Addressing_sec: float = 5.0   # 历史字幕自动消失秒数（默认 5 秒）
    window_alpha: float = 0.92
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

        self.preroll_buffer: Deque[bytes] = deque(maxlen=3)
        self.hangover_max_frames = 12
        self.hangover_counter = 0
        self.in_speech_segment = False
        self.consecutive_voiced_frames = 0

    def update_params(self, rms_threshold: float, breath_filter_strength: float) -> None:
        self.rms_threshold = max(0.001, float(rms_threshold))
        self.breath_filter_strength = min(1.0, max(0.0, float(breath_filter_strength)))

    def analyze_chunk(self, pcm16_bytes: bytes) -> Tuple[bool, List[bytes], dict]:
        if len(pcm16_bytes) < 4:
            return False, [], {"rms": 0.0, "is_speech": False, "reason": "empty"}

        samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))

        if rms < self.rms_threshold:
            self.consecutive_voiced_frames = 0
            if self.in_speech_segment and self.hangover_counter > 0:
                self.hangover_counter -= 1
                return True, [pcm16_bytes], {"rms": rms, "is_speech": True, "reason": "hangover"}
            self.in_speech_segment = False
            self.preroll_buffer.append(pcm16_bytes)
            return False, [], {"rms": rms, "is_speech": False, "reason": "silence"}

        zero_crossings = float(np.mean(np.abs(np.diff(np.signbit(samples)))))

        windowed = samples * np.hanning(len(samples))
        mag_spectrum = np.abs(np.fft.rfft(windowed)) + 1e-10
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / self.sample_rate)

        log_mean = float(np.mean(np.log(mag_spectrum)))
        geom_mean = math.exp(log_mean)
        arith_mean = float(np.mean(mag_spectrum))
        spectral_flatness = geom_mean / (arith_mean + 1e-10)

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
# 4B. 字幕显示状态控制器：管理 5 秒历史消隐与新字幕防压缩截断
# ==============================================================================
class SubtitleDisplayStateController:
    """
    负责管理悬浮窗的双行字幕生命周期：
    - 历史字幕 (`prev_subtitle`) 单行摘要化（防止多行历史字幕挤压新字幕空间）；
    - 历史字幕满 5.0 秒自动清空消失；
    - 新字幕 (`curr_subtitle`) 永不截断，完整返回给自适应高度悬浮窗展示。
    """

    def __init__(self, expire_seconds: float = HISTORY_EXPIRE_SECONDS, max_history_chars: int = 44) -> None:
        self.expire_seconds = float(expire_seconds)
        self.max_history_chars = max_history_chars
        self.prev_subtitle: str = ""
        self.prev_timestamp: float = 0.0
        self.curr_subtitle: str = ""
        self.curr_timestamp: float = 0.0
        self.curr_is_final: bool = False

    def on_new_subtitle(self, zh_text: str, src_text: str = "", show_bilingual: bool = False, is_final: bool = False, now: Optional[float] = None) -> Tuple[str, str]:
        now_ts = time.monotonic() if now is None else now
        display_curr = f"{zh_text}\n[{src_text}]" if (show_bilingual and src_text) else zh_text

        if is_final:
            # 若此前已有归档完结的旧句，将其转为单行历史字幕并启动 5 秒倒计时
            if self.curr_subtitle and self.curr_is_final and self.curr_subtitle != display_curr:
                self._promote_to_history(self.curr_subtitle, now_ts)
            self.curr_subtitle = display_curr
            self.curr_timestamp = now_ts
            self.curr_is_final = True
        else:
            # 正在流式生成新句时，如果上一句已是 final，将上一句移入历史字幕（5秒后自动消失）
            if self.curr_subtitle and self.curr_is_final:
                self._promote_to_history(self.curr_subtitle, now_ts)
                self.curr_is_final = False
            self.curr_subtitle = display_curr
            self.curr_timestamp = now_ts

        self.check_expiration(now_ts)
        return self.prev_subtitle, self.curr_subtitle

    def _promote_to_history(self, text: str, now_ts: float) -> None:
        single_line = text.split("\n")[0].strip()
        if len(single_line) > self.max_history_chars:
            single_line = "…" + single_line[-(self.max_history_chars - 1):]
        self.prev_subtitle = single_line
        self.prev_timestamp = now_ts

    def check_expiration(self, now: Optional[float] = None) -> bool:
        """
        检查历史字幕是否已满 5 秒需要自动消失。
        返回 True 表示状态发生了变更（需要刷新 UI 并收缩历史字幕行）。
        """
        now_ts = time.monotonic() if now is None else now
        changed = False
        if self.prev_subtitle and (now_ts - self.prev_timestamp) >= self.expire_seconds:
            self.prev_subtitle = ""
            self.prev_timestamp = 0.0
            changed = True
        return changed


# ==============================================================================
# 5. macOS 15 系统音频捕获统一管理器 (ScreenCaptureKit + CoreAudio 双引擎)
# ==============================================================================
class MacOSAudioCaptureManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.running = False
        self._sck_proc: Optional[subprocess.Popen] = None
        self._sck_bin_path: str = str(SWIFT_HELPER_BIN)
        self._reader_thread: Optional[threading.Thread] = None

    @staticmethod
    def ensure_sck_helper_compiled() -> Tuple[bool, str]:
        if sys.platform != "darwin":
            return False, "当前非 macOS 系统，无法编译 ScreenCaptureKit 组件。"

        # 优先查找 PyInstaller (--onefile / --onedir) 打包内置的预编译二进制，无需用户机器存在 py/swift 源码
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
                        on_error("请在「系统设置 -> 隐私与安全性 -> 录屏与系统录音」中授予本应用权限。")
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
            self.on_status_change("未配置 API Key，请点击「设置」填入", "#FF6B6B")
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
                        f"实时翻译中 · {active_model}",
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
                    active_model = FALLBACK_MODEL

                self.on_status_change(
                    f"连接自动恢复中 ({backoff:.1f}s)...",
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
# 7A. 零依赖本地 Web HUD 悬浮字幕界面 (支持透明背景、5秒历史消隐与新字幕自适应高度)
# ==============================================================================
WEB_HUD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<title>Gemini 3.5 Live 实时中文字幕 (macOS 15)</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", sans-serif;
    background: transparent;
    color: #ffffff;
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    overflow-x: hidden;
  }
  body.solid-bg { background: rgba(15, 18, 24, 0.88); }
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(24, 28, 36, 0.92);
    padding: 6px 14px;
    border-radius: 8px;
    margin: 6px 12px 0 12px;
    font-size: 12px;
    opacity: 0;
    transition: opacity 0.25s ease;
  }
  body:hover .topbar, .topbar.pinned { opacity: 1; }
  .status-wrap { display: flex; align-items: center; gap: 8px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; background: #9aa0a6; }
  .controls { display: flex; align-items: center; gap: 6px; }
  button {
    background: #262c3a; color: #f8fafc; border: none; border-radius: 6px;
    padding: 4px 10px; font-size: 12px; cursor: pointer; font-weight: 500;
  }
  button.primary { background: #2563eb; font-weight: 600; }
  button.danger { background: #dc2626; font-weight: 600; }
  .subtitle-stage {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    align-items: center;
    padding: 10px 24px 18px 24px;
    text-align: center;
  }
  .prev-line {
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 92vw;
    text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000, 0 2px 8px rgba(0,0,0,0.95);
    transition: opacity 0.3s ease;
  }
  .prev-line:empty { display: none; margin: 0; }
  .curr-line {
    color: #ffffff;
    font-size: 28px;
    font-weight: 800;
    line-height: 1.42;
    word-break: break-word;
    max-width: 95vw;
    text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000, 0 3px 12px rgba(0,0,0,0.98);
  }
</style>
</head>
<body>
  <div id="topbar" class="topbar pinned">
    <div class="status-wrap">
      <div id="dot" class="dot"></div>
      <span id="statusText">就绪 · 鼠标移开自动隐藏控制栏，仅显示透明字幕文字</span>
    </div>
    <div class="controls">
      <span id="vadBadge" style="color:#94a3b8;font-size:12px;margin-right:6px;">静音热备</span>
      <button id="toggleBtn" class="primary" onclick="toggleRun()">▶ 开始实时翻译</button>
      <button onclick="document.body.classList.toggle('solid-bg')">🔲 切换透明/底板</button>
    </div>
  </div>
  <div class="subtitle-stage">
    <div id="prevLine" class="prev-line"></div>
    <div id="currLine" class="curr-line">等待系统音频输入…</div>
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
    document.getElementById('prevLine').textContent = d.prev_subtitle || '';
    document.getElementById('currLine').textContent = d.curr_subtitle || '';
    isRunning = d.is_running;
    const tb = document.getElementById('topbar');
    if (isRunning) tb.classList.remove('pinned');
    const btn = document.getElementById('toggleBtn');
    btn.textContent = isRunning ? '⏹ 停止' : '▶ 开始实时翻译';
    btn.className = isRunning ? 'danger' : 'primary';
  } catch (e) {}
}
setInterval(pollState, 100);
async function toggleRun() {
  await fetch('/api/toggle', { method: 'POST' });
}
</script>
</body>
</html>
"""


class WebHudSubtitleApp:
    def __init__(self, port: int = 18765) -> None:
        self.port = port
        self.settings = AppSettings.load()
        self.display_ctrl = SubtitleDisplayStateController(expire_seconds=HISTORY_EXPIRE_SECONDS)
        self.state = {
            "is_running": False,
            "status_text": "就绪 · 点击「开始实时翻译」捕获 macOS 系统音频",
            "status_color": "#9AA0A6",
            "vad_text": "静音热备 (防休眠)",
            "prev_subtitle": "",
            "curr_subtitle": "等待系统音频输入…",
        }
        self.service = GeminiLiveTranslateService(
            settings=self.settings,
            on_subtitle_update=self._on_subtitle,
            on_status_change=self._on_status,
            on_audio_level=self._on_level,
        )

    def _on_subtitle(self, zh_text: str, src_text: str, is_final: bool) -> None:
        prev_txt, curr_txt = self.display_ctrl.on_new_subtitle(
            zh_text, src_text, self.settings.show_bilingual, is_final
        )
        self.state["prev_subtitle"] = prev_txt
        self.state["curr_subtitle"] = curr_txt

    def _on_status(self, msg: str, color: str) -> None:
        self.state["status_text"] = msg
        self.state["status_color"] = color

    def _on_level(self, rms: float, reason: str) -> None:
        if self.display_ctrl.check_expiration():
            self.state["prev_subtitle"] = self.display_ctrl.prev_subtitle
        bars = min(8, int(rms * 160))
        meter = "▮" * bars + "▯" * (8 - bars)
        reason_map = {
            "speech_active": "人声翻译中",
            "speech_onset": "唤醒捕捉",
            "hangover": "句尾延续",
            "breath_filtered": "已过滤喘息声",
            "groan_filtered": "已过滤呻吟声",
            "silence": "静音热备",
        }
        self.state["vad_text"] = f"{meter} {reason_map.get(reason, '监听中')}"

    def run(self) -> None:
        app_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path == "/api/state":
                    if app_ref.display_ctrl.check_expiration():
                        app_ref.state["prev_subtitle"] = app_ref.display_ctrl.prev_subtitle
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(app_ref.state, ensure_ascii=False).encode("utf-8"))
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(WEB_HUD_HTML.encode("utf-8"))

            def do_POST(self):
                if self.path == "/api/toggle":
                    if not app_ref.state["is_running"]:
                        app_ref.state["is_running"] = True
                        app_ref.service.start()
                    else:
                        app_ref.state["is_running"] = False
                        app_ref.service.stop()
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
# 7B. macOS 15 原生置顶纯透明字幕悬浮窗 (只显示文字 + 历史5秒自动消失 + 新字幕自适应高度防压缩)
# ==============================================================================
class MacOSSubtitleOverlayApp:
    """
    针对 macOS 15 深度优化的原生悬浮字幕窗口：
    1. 支持背景纯透明 (`systemTransparent` + 无边框模式)，鼠标移走后自动隐藏顶部工具栏，只显示字幕文字；
    2. 历史字幕显示满 5 秒后自动消失并释放空间 (`pack_forget`)；
    3. 动态自适应窗口高度 (`_sync_window_height`)：新字幕无论多长都会自动撑开显示空间，绝不被压缩截断。
    """

    def __init__(self) -> None:
        self.settings = AppSettings.load()
        self.ui_queue: "queue.Queue[Tuple[str, tuple]]" = queue.Queue()
        self.display_ctrl = SubtitleDisplayStateController(expire_seconds=HISTORY_EXPIRE_SECONDS)

        self.root = tk.Tk()
        self.root.title("Gemini 3.5 Live 实时中文字幕")
        self._window_width = 960
        self.root.geometry(f"{self._window_width}x150+200+700")
        self.root.minsize(560, 80)

        # 置顶属性
        self.root.attributes("-topmost", True)

        self._drag_offset_x = 0
        self._drag_offset_y = 0
        self._is_running = False
        self._topbar_visible = True
        self._hover_hide_job: Optional[str] = None
        self._history_lines: Deque[str] = deque(maxlen=200)

        self._build_ui()
        self._apply_transparency_mode()

        self.service = GeminiLiveTranslateService(
            settings=self.settings,
            on_subtitle_update=lambda zh, src, final: self.ui_queue.put(("subtitle", (zh, src, final))),
            on_status_change=lambda msg, color: self.ui_queue.put(("status", (msg, color))),
            on_audio_level=lambda rms, reason: self.ui_queue.put(("level", (rms, reason))),
        )

        self.root.after(50, self._poll_ui_events)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if not self.settings.api_key:
            self.root.after(300, self._open_settings_dialog)

    def _get_bg_color(self) -> str:
        """在 macOS 下启用透明背景时使用 Aqua 原生 systemTransparent。"""
        if self.settings.transparent_bg and sys.platform == "darwin":
            return "systemTransparent"
        return "#121417"

    def _apply_transparency_mode(self) -> None:
        bg_color = self._get_bg_color()
        if sys.platform == "darwin":
            try:
                if self.settings.transparent_bg:
                    # 启用无边框 + 原生透明背景（只显示文字，不显示灰黑底框）
                    self.root.overrideredirect(True)
                    self.root.wm_attributes("-transparent", True)
                    self.root.attributes("-alpha", 1.0)
                else:
                    self.root.overrideredirect(False)
                    self.root.wm_attributes("-transparent", False)
                    self.root.attributes("-alpha", self.settings.window_alpha)
            except tk.TclError:
                pass
        else:
            try:
                self.root.attributes("-alpha", self.settings.window_alpha)
            except tk.TclError:
                pass

        self.root.configure(bg=bg_color)
        self.body.configure(bg=bg_color)
        self.prev_subtitle_label.configure(bg=bg_color)
        self.curr_subtitle_label.configure(bg=bg_color)
        self._sync_window_height()

    def _build_ui(self) -> None:
        bg_color = self._get_bg_color()

        # 顶部控制栏（鼠标移入显示，移出且翻译运行时自动隐藏，保证只显示文字）
        self.top_bar = tk.Frame(self.root, bg="#1A1D24", height=34)
        self.top_bar.pack(fill=tk.X, side=tk.TOP)

        self.status_dot = tk.Label(
            self.top_bar, text="●", fg="#9AA0A6", bg="#1A1D24", font=("PingFang SC", 12, "bold")
        )
        self.status_dot.pack(side=tk.LEFT, padx=(10, 4), pady=3)

        self.status_label = tk.Label(
            self.top_bar,
            text="就绪 · 可直接拖动文字移动位置，开始翻译后工具栏自动隐去",
            fg="#CBD5E1",
            bg="#1A1D24",
            font=("PingFang SC", 11),
        )
        self.status_label.pack(side=tk.LEFT, padx=4, pady=3)

        btn_style = {
            "bg": "#262B36",
            "fg": "#F8FAFC",
            "activebackground": "#3B4252",
            "activeforeground": "#FFFFFF",
            "bd": 0,
            "padx": 8,
            "pady": 2,
            "font": ("PingFang SC", 11),
            "cursor": "hand2",
        }

        self.btn_quit = tk.Button(
            self.top_bar, text="✕", command=self._on_close, **btn_style
        )
        self.btn_quit.pack(side=tk.RIGHT, padx=(2, 8), pady=3)

        self.btn_settings = tk.Button(
            self.top_bar, text="⚙ 设置", command=self._open_settings_dialog, **btn_style
        )
        self.btn_settings.pack(side=tk.RIGHT, padx=3, pady=3)

        self.btn_bg_mode = tk.Button(
            self.top_bar,
            text="🔲 纯透明文字" if self.settings.transparent_bg else "⬛ 半透底板",
            command=self._toggle_bg_mode,
            **btn_style,
        )
        self.btn_bg_mode.pack(side=tk.RIGHT, padx=3, pady=3)

        self.btn_history = tk.Button(
            self.top_bar, text="📜 记录", command=self._open_history_dialog, **btn_style
        )
        self.btn_history.pack(side=tk.RIGHT, padx=3, pady=3)

        self.btn_toggle = tk.Button(
            self.top_bar,
            text="▶ 开始翻译",
            command=self._toggle_translation,
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            bd=0,
            padx=10,
            pady=2,
            font=("PingFang SC", 11, "bold"),
            cursor="hand2",
        )
        self.btn_toggle.pack(side=tk.RIGHT, padx=3, pady=3)

        self.vad_badge = tk.Label(
            self.top_bar,
            text="静音待命",
            fg="#94A3B8",
            bg="#1A1D24",
            font=("PingFang SC", 10),
        )
        self.vad_badge.pack(side=tk.RIGHT, padx=8, pady=3)

        # 字幕主显示区
        self.body = tk.Frame(self.root, bg=bg_color, padx=16, pady=6)
        self.body.pack(fill=tk.BOTH, expand=True)

        # 历史字幕（默认不挂载 pack，仅有内容时挂载，满 5 秒后自动 pack_forget 彻底释放空间）
        self.prev_subtitle_var = tk.StringVar(value="")
        self.prev_subtitle_label = tk.Label(
            self.body,
            textvariable=self.prev_subtitle_var,
            fg="#CBD5E1",
            bg=bg_color,
            font=("PingFang SC", max(15, self.settings.font_size - 8), "bold"),
            wraplength=self._window_width - 40,
            justify=tk.CENTER,
        )

        # 当前实时新字幕（高对比度亮白大字，始终拥有最高布局优先级并自动撑开窗口高度）
        self.curr_subtitle_var = tk.StringVar(
            value="等待系统音频输入…"
        )
        self.curr_subtitle_label = tk.Label(
            self.body,
            textvariable=self.curr_subtitle_var,
            fg="#FFFFFF",
            bg=bg_color,
            font=("PingFang SC", self.settings.font_size, "bold"),
            wraplength=self._window_width - 40,
            justify=tk.CENTER,
        )
        self.curr_subtitle_label.pack(side=tk.BOTTOM, fill=tk.X, expand=True, pady=(2, 4))

        # 绑定任意位置（包括文字本身）左键拖拽移动悬浮窗
        for widget in (
            self.top_bar,
            self.status_label,
            self.body,
            self.prev_subtitle_label,
            self.curr_subtitle_label,
        ):
            widget.bind("<ButtonPress-1>", self._on_drag_start)
            widget.bind("<B1-Motion>", self._on_drag_motion)

        # 鼠标悬停唤醒工具栏，移开后自动隐藏（实现纯文字显示）
        self.root.bind("<Enter>", self._on_mouse_enter)
        self.root.bind("<Leave>", self._on_mouse_leave)

    def _toggle_bg_mode(self) -> None:
        self.settings.transparent_bg = not self.settings.transparent_bg
        self.settings.save()
        self.btn_bg_mode.configure(
            text="🔲 纯透明文字" if self.settings.transparent_bg else "⬛ 半透底板"
        )
        self._apply_transparency_mode()

    def _on_mouse_enter(self, _event=None) -> None:
        if self._hover_hide_job is not None:
            try:
                self.root.after_cancel(self._hover_hide_job)
            except Exception:
                pass
            self._hover_hide_job = None
        if not self._topbar_visible:
            self._topbar_visible = True
            self.top_bar.pack(fill=tk.X, side=tk.TOP, before=self.body)
            self._sync_window_height()

    def _on_mouse_leave(self, _event=None) -> None:
        if self._is_running and self.settings.transparent_bg:
            if self._hover_hide_job is not None:
                try:
                    self.root.after_cancel(self._hover_hide_job)
                except Exception:
                    pass
            self._hover_hide_job = self.root.after(1800, self._hide_topbar_for_pure_text)

    def _hide_topbar_for_pure_text(self) -> None:
        self._hover_hide_job = None
        if self._is_running and self.settings.transparent_bg and self._topbar_visible:
            self._topbar_visible = False
            self.top_bar.pack_forget()
            self._sync_window_height()

    def _sync_window_height(self) -> None:
        """
        动态计算新字幕与历史字幕所需像素高度并实时调整窗口高度，
        确保多行新字幕永远不会被窗口边界或历史字幕压缩、截断。
        """
        try:
            self.root.update_idletasks()
            req_h = self.curr_subtitle_label.winfo_reqheight() + 24
            if self.prev_subtitle_var.get():
                req_h += self.prev_subtitle_label.winfo_reqheight() + 8
            if self._topbar_visible:
                req_h += 40
            needed_h = max(72, min(460, req_h))
            cur_x = self.root.winfo_x()
            cur_y = self.root.winfo_y()
            self.root.geometry(f"{self._window_width}x{needed_h}+{cur_x}+{cur_y}")
        except Exception:
            pass

    def _update_subtitle_widgets(self, prev_txt: str, curr_txt: str) -> None:
        # 1. 控制历史字幕的挂载与卸载（为空或满 5 秒消隐后立即 pack_forget 释放全部空间）
        old_prev = self.prev_subtitle_var.get()
        if prev_txt:
            self.prev_subtitle_var.set(prev_txt)
            if not old_prev:
                self.prev_subtitle_label.pack(
                    side=tk.TOP, fill=tk.X, pady=(0, 2), before=self.curr_subtitle_label
                )
        else:
            if old_prev:
                self.prev_subtitle_var.set("")
                self.prev_subtitle_label.pack_forget()

        # 2. 更新当前新字幕并动态扩展窗口高度，防止任何压缩截断
        if curr_txt != self.curr_subtitle_var.get() or prev_txt != old_prev:
            self.curr_subtitle_var.set(curr_txt)
            self._sync_window_height()

    def _on_drag_start(self, event) -> None:
        self._drag_offset_x = event.x_root - self.root.winfo_x()
        self._drag_offset_y = event.y_root - self.root.winfo_y()

    def _on_drag_motion(self, event) -> None:
        x = event.x_root - self._drag_offset_x
        y = event.y_root - self._drag_offset_y
        self.root.geometry(f"+{x}+{y}")

    def _toggle_translation(self) -> None:
        if not self._is_running:
            if not self.settings.api_key.strip():
                self._open_settings_dialog()
                return
            self._is_running = True
            self.btn_toggle.configure(text="⏹ 停止", bg="#DC2626", activebackground="#B91C1C")
            self.curr_subtitle_var.set("正在监听 macOS 系统音频…")
            self.service.start()
            if self.settings.transparent_bg:
                self.root.after(2200, self._hide_topbar_for_pure_text)
        else:
            self._is_running = False
            self.btn_toggle.configure(text="▶ 开始翻译", bg="#2563EB", activebackground="#1D4ED8")
            self.service.stop()
            self._on_mouse_enter()

    def _poll_ui_events(self) -> None:
        try:
            # 1. 每帧检查历史字幕是否满 5 秒需要自动消失
            if self.display_ctrl.check_expiration():
                self._update_subtitle_widgets(
                    self.display_ctrl.prev_subtitle,
                    self.display_ctrl.curr_subtitle or self.curr_subtitle_var.get(),
                )

            # 2. 处理异步事件队列
            while True:
                evt_type, payload = self.ui_queue.get_nowait()
                if evt_type == "subtitle":
                    zh_text, src_text, is_final = payload
                    prev_txt, curr_txt = self.display_ctrl.on_new_subtitle(
                        zh_text=zh_text,
                        src_text=src_text,
                        show_bilingual=self.settings.show_bilingual,
                        is_final=is_final,
                    )
                    self._update_subtitle_widgets(prev_txt, curr_txt)

                    if is_final:
                        ts_str = time.strftime("%H:%M:%S")
                        self._history_lines.append(f"[{ts_str}] {zh_text}")

                elif evt_type == "status":
                    msg, color = payload
                    self.status_label.configure(text=msg)
                    self.status_dot.configure(fg=color)

                elif evt_type == "level":
                    rms, reason = payload
                    db_bar = min(8, int(rms * 160))
                    meter = "▮" * db_bar + "▯" * (8 - db_bar)
                    reason_map = {
                        "speech_active": ("人声翻译中", "#4ADE80"),
                        "speech_onset": ("唤醒捕捉", "#38BDF8"),
                        "hangover": ("句尾延续", "#A3E635"),
                        "breath_filtered": ("已滤喘息", "#FBBF24"),
                        "groan_filtered": ("已滤呻吟", "#FBBF24"),
                        "silence": ("静音热备", "#94A3B8"),
                    }
                    label_txt, color = reason_map.get(reason, ("监听中", "#94A3B8"))
                    self.vad_badge.configure(text=f"{meter} {label_txt}", fg=color)
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self._poll_ui_events)

    def _open_settings_dialog(self) -> None:
        self._on_mouse_enter()
        win = tk.Toplevel(self.root)
        win.title("自定义 API Key 与透明字幕设置")
        win.geometry("560x560")
        win.configure(bg="#1E222B")
        win.attributes("-topmost", True)
        win.transient(self.root)

        pad = {"padx": 16, "pady": 5}

        tk.Label(
            win,
            text="Google AI Studio API Key (支持免费层级密钥):",
            fg="#E2E8F0",
            bg="#1E222B",
            font=("PingFang SC", 11, "bold"),
        ).pack(anchor=tk.W, padx=16, pady=(12, 2))

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
        ).pack(anchor=tk.W, padx=16, pady=(8, 2))

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
        ).pack(anchor=tk.W, padx=16, pady=(8, 2))

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
            text="语气词与喘息/呻吟声过滤强度 (0.1 ~ 0.95，推荐 0.65):",
            fg="#E2E8F0",
            bg="#1E222B",
            font=("PingFang SC", 11, "bold"),
        ).pack(anchor=tk.W, padx=16, pady=(8, 2))

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
            text="字幕字号大小 (推荐 26):",
            fg="#E2E8F0",
            bg="#1E222B",
            font=("PingFang SC", 11, "bold"),
        ).pack(anchor=tk.W, padx=16, pady=(8, 2))

        font_var = tk.IntVar(value=self.settings.font_size)
        tk.Scale(
            win,
            variable=font_var,
            from_=18,
            to=42,
            resolution=2,
            orient=tk.HORIZONTAL,
            bg="#1E222B",
            fg="#F8FAFC",
            highlightthickness=0,
        ).pack(fill=tk.X, padx=16)

        transparent_var = tk.BooleanVar(value=self.settings.transparent_bg)
        tk.Checkbutton(
            win,
            text="背景完全透明（只显示文字，鼠标移走自动隐藏工具栏）",
            variable=transparent_var,
            bg="#1E222B",
            fg="#4ADE80",
            selectcolor="#0F172A",
            activebackground="#1E222B",
        ).pack(anchor=tk.W, padx=16, pady=4)

        bilingual_var = tk.BooleanVar(value=self.settings.show_bilingual)
        tk.Checkbutton(
            win,
            text="同时显示源语言识别字幕（双语对照）",
            variable=bilingual_var,
            bg="#1E222B",
            fg="#CBD5E1",
            selectcolor="#0F172A",
            activebackground="#1E222B",
        ).pack(anchor=tk.W, padx=16, pady=2)

        def _save_and_apply() -> None:
            self.settings.api_key = api_key_var.get().strip()
            self.settings.model_name = model_var.get().strip() or DEFAULT_MODEL
            self.settings.capture_backend = backend_var.get().strip() or "auto"
            self.settings.breath_filter_strength = float(breath_var.get())
            self.settings.font_size = int(font_var.get())
            self.settings.transparent_bg = bool(transparent_var.get())
            self.settings.show_bilingual = bool(bilingual_var.get())
            self.settings.save()

            self.curr_subtitle_label.configure(font=("PingFang SC", self.settings.font_size, "bold"))
            self.prev_subtitle_label.configure(font=("PingFang SC", max(15, self.settings.font_size - 8), "bold"))
            self.btn_bg_mode.configure(
                text="🔲 纯透明文字" if self.settings.transparent_bg else "⬛ 半透底板"
            )
            self._apply_transparency_mode()
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
        ).pack(pady=12)

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

    # 3. 验证历史字幕 5 秒自动消失与新字幕完整保留逻辑
    disp = SubtitleDisplayStateController(expire_seconds=5.0)
    prev_t, curr_t = disp.on_new_subtitle("第一句已经完成的字幕", is_final=True, now=100.0)
    assert prev_t == "" and curr_t == "第一句已经完成的字幕"

    # 101秒时新字幕流入，第一句进入历史字幕
    prev_t, curr_t = disp.on_new_subtitle("第二句正在实时生成的新字幕不会被任何历史字幕压缩截断", is_final=False, now=101.0)
    assert prev_t == "第一句已经完成的字幕" and "第二句正在实时生成的新字幕" in curr_t

    # 104秒时（未满5秒），历史字幕仍在
    assert not disp.check_expiration(now=104.0)
    assert disp.prev_subtitle == "第一句已经完成的字幕"

    # 106.1秒时（满5秒），历史字幕自动消失，且新字幕完好无损
    assert disp.check_expiration(now=106.1)
    assert disp.prev_subtitle == "", "历史字幕应当在5秒后自动消失"
    assert disp.curr_subtitle == "第二句正在实时生成的新字幕不会被任何历史字幕压缩截断", "新字幕绝不能被截断或丢失"

    # 4. 验证声学频谱过滤器
    acoustic = AcousticSpeechFilter(sample_rate=TARGET_SAMPLE_RATE, rms_threshold=0.008, breath_filter_strength=0.65)

    silent_pcm = bytes(CHUNK_BYTES)
    send_flag, _, meta = acoustic.analyze_chunk(silent_pcm)
    assert not send_flag and meta["reason"] == "silence"

    rng = np.random.default_rng(42)
    breath_noise = (rng.uniform(-0.15, 0.15, size=CHUNK_SAMPLES) * np.tile([1.0, -1.0], CHUNK_SAMPLES // 2)).astype(np.float32)
    breath_pcm = (breath_noise * 32767.0).astype("<i2").tobytes()
    send_flag, _, meta = acoustic.analyze_chunk(breath_pcm)
    assert not send_flag and meta["reason"] == "breath_filtered"

    t = np.linspace(0, CHUNK_DURATION_MS / 1000.0, CHUNK_SAMPLES, endpoint=False)
    voice_wave = 0.25 * np.sin(2 * np.pi * 180 * t) + 0.35 * np.sin(2 * np.pi * 720 * t) + 0.20 * np.sin(2 * np.pi * 1600 * t)
    voice_pcm = (np.clip(voice_wave, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    send_flag, chunks, meta = acoustic.analyze_chunk(voice_pcm)
    assert send_flag and meta["reason"] == "speech_onset" and len(chunks) >= 2

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
