#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macOS 15 (Sequoia) Gemini 3.5 Live Translate 实时系统音频中文字幕软件 (v1.2.0)
===========================================================================
基于 Google Gemini Live API (`gemini-3.5-live-translate-preview`) 构建的桌面级实时字幕工具。

核心特性：
1. 原生支持 macOS 15 (Sequoia) 系统音频内录：
   - 模式 A（推荐，免装虚拟声卡）：自动调用内置预编译的 Swift `ScreenCaptureKit` 音频捕获器，
     直接抓取 macOS 系统桌面音频并输出 16kHz 16-bit 单声道 PCM 流。
   - 模式 B（CoreAudio 虚拟声卡/输入设备）：支持 BlackHole 2ch、Loopback 或任意麦克风/聚合设备。
2. 逐句换行显示 + 每句严格 5 秒自动清空 (`SentenceSegmentedSubtitleController`)：
   - 自动将长串流式翻译按标点（。！？；，、）及语义长度（每行不超过 24 字）切分为独立短句并逐句换行显示。
   - 屏幕上最多同时展示最近 2 句短字幕，每句字幕独立计时，满 5.0 秒后自动清空消失；若满 5 秒无新语音，屏幕字幕完全清空。
3. 强力跨应用全局置顶 (`macOS Cocoa NSWindow Level + hidesOnDeactivate=False`)：
   - 通过 macOS 原生 Objective-C Runtime (`libobjc.A.dylib` / `AppKit`) 将字幕窗口提升至
     `NSPopUpMenuWindowLevel (101)`，强制设置 `setHidesOnDeactivate(False)` 与
     `CanJoinAllSpaces | FullScreenAuxiliary`，彻底解决切换到浏览器或其他全屏应用时字幕被隐藏的问题。
4. 纯透明无边框文字悬浮 + 语气词/喘息/呻吟三重过滤 + 永不休眠长间隙秒级唤醒。
"""

from __future__ import annotations

import asyncio
from collections import deque
import ctypes
import ctypes.util
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

# 可选导入 SoundDevice / PyAudio
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

# 每句字幕保留时间（秒）与单句最大字数
SENTENCE_TTL_SECONDS = 5.0
MAX_CHARS_PER_LINE = 24
MAX_VISIBLE_LINES = 2


# ==============================================================================
# 0. macOS 原生 Cocoa 窗口全局置顶与防隐藏增强器 (解决打开浏览器被遮挡/隐藏问题)
# ==============================================================================
def enforce_macos_always_on_top_and_no_hide() -> bool:
    """
    通过 macOS 原生 Objective-C Runtime (`libobjc.A.dylib`) 配置当前进程的所有 NSWindow：
    1. `setHidesOnDeactivate_(False)`：禁止在切换到浏览器/其他 App 时自动隐藏窗口；
    2. `setLevel_(101)` (`NSPopUpMenuWindowLevel` / `kCGStatusWindowLevel`)：强制悬浮在所有普通窗口和浏览器之上；
    3. `setCollectionBehavior_(273)` (`CanJoinAllSpaces | Stationary | FullScreenAuxiliary`)：支持全屏空间置顶。
    """
    if sys.platform != "darwin":
        return False
    try:
        objc_path = ctypes.util.find_library("objc") or "/usr/lib/libobjc.A.dylib"
        objc = ctypes.cdll.LoadLibrary(objc_path)
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")

        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        def _sel(name: bytes) -> ctypes.c_void_p:
            return objc.sel_registerName(name)

        # 无参数返回指针的 objc_msgSend
        msg_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        # 单整型参数 (NSInteger / NSUInteger) 的 objc_msgSend
        msg_int = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long)(
            ("objc_msgSend", objc)
        )
        # 单布尔参数 (BOOL) 的 objc_msgSend
        msg_bool = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool)(
            ("objc_msgSend", objc)
        )
        # 返回 NSUInteger 的 objc_msgSend
        msg_ret_uint = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        # 按索引取元素的 objectAtIndex:
        msg_at_idx = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(
            ("objc_msgSend", objc)
        )

        ns_app_cls = objc.objc_getClass(b"NSApplication")
        shared_app = msg_ptr(ns_app_cls, _sel(b"sharedApplication"))
        if not shared_app:
            return False

        windows_array = msg_ptr(shared_app, _sel(b"windows"))
        if not windows_array:
            return False

        count = msg_ret_uint(windows_array, _sel(b"count"))
        sel_hides = _sel(b"setHidesOnDeactivate:")
        sel_level = _sel(b"setLevel:")
        sel_behavior = _sel(b"setCollectionBehavior:")
        sel_order_front = _sel(b"orderFrontRegardless")

        # CanJoinAllSpaces (1 << 0) | Stationary (1 << 4) | FullScreenAuxiliary (1 << 8) = 273
        behavior_flags = (1 << 0) | (1 << 4) | (1 << 8)
        # NSPopUpMenuWindowLevel = 101 (高于普通窗口 0、浮动窗口 3、状态栏 25)
        floating_level = 101

        for i in range(count):
            win = msg_at_idx(windows_array, _sel(b"objectAtIndex:"), i)
            if win:
                msg_bool(win, sel_hides, False)
                msg_int(win, sel_level, floating_level)
                msg_int(win, sel_behavior, behavior_flags)
                msg_ptr(win, sel_order_front)
        return True
    except Exception as exc:
        logger.debug("Cocoa 窗口置顶增强跳过: %s", exc)
        return False


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
    sentence_ttl_sec: float = 5.0         # 每句字幕保留秒数（默认 5.0 秒自动清空）
    max_chars_per_line: int = 24          # 单句自动换行最大字数
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
# 4B. 逐句换行切分 & 每句独立 5 秒自动清空控制器
# ==============================================================================
class SubtitleDisplayStateController:
    """
    负责将实时流式长字幕切分为短句，按句换行展示，且**每一句只保留 5 秒**：
    1. 遇到句末标点（。！？；!?;）立即断句换行；
    2. 若单句超过 `max_chars_per_line`（默认 24 字），优先在逗号/顿号（，、,）处断句换行，否则按字数软切分；
    3. 每个已切分完成的短句拥有自己独立的创建时间戳 `born_ts`，满 5.0 秒立即从屏幕上移除；
    4. 正在说的最新一句同样在最后一次更新满 5.0 秒后自动清空，保证停止说话 5 秒后屏幕 100% 清空！
    """

    SENTENCE_SPLIT_RE = re.compile(r"([^。！？；!?;]+[。！？；!?;]?)")
    CLAUSE_SPLIT_RE = re.compile(r"([^，、,：:]+[，、,：:]?)")

    def __init__(
        self,
        expire_seconds: float = SENTENCE_TTL_SECONDS,
        max_chars_per_line: int = MAX_CHARS_PER_LINE,
        max_visible_lines: int = MAX_VISIBLE_LINES,
    ) -> None:
        self.expire_seconds = float(expire_seconds)
        self.max_chars_per_line = int(max_chars_per_line)
        self.max_visible_lines = int(max_visible_lines)

        # 保存每个短句首次出现的时间戳: {normalized_clause: first_seen_timestamp}
        self._clause_birth_ts: dict[str, float] = {}
        self._active_lines: List[Tuple[str, float]] = []
        self.prev_subtitle: str = ""
        self.curr_subtitle: str = ""

    def split_into_short_sentences(self, text: str) -> List[str]:
        """将长文本按标点和长度切分为易读的短句列表（每句一行）。"""
        if not text or not text.strip():
            return []

        raw_sentences = [s.strip() for s in self.SENTENCE_SPLIT_RE.findall(text.strip()) if s.strip()]
        short_lines: List[str] = []

        for sent in raw_sentences:
            if len(sent) <= self.max_chars_per_line:
                short_lines.append(sent)
                continue

            # 句子过长时，先按逗号/顿号切分子句
            clauses = [c.strip() for c in self.CLAUSE_SPLIT_RE.findall(sent) if c.strip()]
            buf = ""
            for clause in clauses:
                if not buf:
                    buf = clause
                elif len(buf) + len(clause) <= self.max_chars_per_line:
                    buf += clause
                else:
                    short_lines.extend(self._chunk_by_length(buf, self.max_chars_per_line))
                    buf = clause
            if buf:
                short_lines.extend(self._chunk_by_length(buf, self.max_chars_per_line))

        return [line.strip(" ，、,") for line in short_lines if line.strip(" ，、,")]

    @staticmethod
    def _chunk_by_length(text: str, max_len: int) -> List[str]:
        if len(text) <= max_len:
            return [text]
        return [text[i : i + max_len] for i in range(0, len(text), max_len)]

    def on_new_subtitle(
        self,
        zh_text: str,
        src_text: str = "",
        show_bilingual: bool = False,
        is_final: bool = False,
        now: Optional[float] = None,
    ) -> Tuple[str, str]:
        now_ts = time.monotonic() if now is None else now
        sentences = self.split_into_short_sentences(zh_text)
        if not sentences:
            self.check_expiration(now_ts)
            return self.prev_subtitle, self.curr_subtitle

        updated_lines: List[Tuple[str, float]] = []
        for idx, line in enumerate(sentences):
            norm_key = re.sub(r"[\s，。！？；、,.!?;]", "", line)
            if not norm_key:
                continue

            # 已完成的前序短句沿用其首次上屏时间戳（满 5 秒即消隐，绝不残留成大段长文）
            is_last_streaming_line = (idx == len(sentences) - 1) and (not is_final)
            if norm_key in self._clause_birth_ts and not is_last_streaming_line:
                born = self._clause_birth_ts[norm_key]
            else:
                born = self._clause_birth_ts.get(norm_key, now_ts)
                if is_last_streaming_line:
                    born = now_ts
                self._clause_birth_ts[norm_key] = born

            if (now_ts - born) < self.expire_seconds:
                updated_lines.append((line, born))

        # 合并上一轮尚未过期的短句（当新一轮 turn 开始时，保留上一轮未满 5 秒的最后一句作为上方行）
        for old_line, old_ts in self._active_lines:
            if (now_ts - old_ts) < self.expire_seconds:
                old_key = re.sub(r"[\s，。！？；、,.!?;]", "", old_line)
                if all(old_key not in re.sub(r"[\s，。！？；、,.!?;]", "", u[0]) and re.sub(r"[\s，。！？；、,.!?;]", "", u[0]) not in old_key for u in updated_lines):
                    updated_lines.insert(0, (old_line, old_ts))

        # 只保留最近 max_visible_lines 句（默认最多 2 行：上一句 + 当前句，逐句换行）
        self._active_lines = updated_lines[-self.max_visible_lines :]
        self._rebuild_visible_pair()
        return self.prev_subtitle, self.curr_subtitle

    def check_expiration(self, now: Optional[float] = None) -> bool:
        """
        每 50ms 检查一次所有在屏短句：任何满 5.0 秒的句子立即清空移除！
        """
        now_ts = time.monotonic() if now is None else now
        before = list(self._active_lines)
        self._active_lines = [
            (line, ts) for (line, ts) in self._active_lines if (now_ts - ts) < self.expire_seconds
        ]

        # 定期清理过期的字典缓存
        expired_keys = [k for k, ts in self._clause_birth_ts.items() if (now_ts - ts) >= self.expire_seconds * 2]
        for k in expired_keys:
            self._clause_birth_ts.pop(k, None)

        if self._active_lines != before:
            self._rebuild_visible_pair()
            return True
        return False

    def _rebuild_visible_pair(self) -> None:
        if not self._active_lines:
            self.prev_subtitle = ""
            self.curr_subtitle = ""
        elif len(self._active_lines) == 1:
            self.prev_subtitle = ""
            self.curr_subtitle = self._active_lines[0][0]
        else:
            self.prev_subtitle = self._active_lines[-2][0]
            self.curr_subtitle = self._active_lines[-1][0]


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
        "你是一个高精度的实时同声传译字幕引擎，负责将输入的系统音频实时翻译为自然、简练、准确的简体中文（zh-Hans）短句字幕。\n"
        "严格遵守以下规则：\n"
        "1. 每次翻译请保持句子简短精炼，按意群及时使用句号或逗号断句，不要输出冗长不间断的长段落。\n"
        "2. 彻底忽略并过滤音频中的所有无意义语气词（如：嗯、啊、哦、哎、呃、唔、哼、哈、uh、um、ah、oh、er、hm、mm、ええと、あの等）。\n"
        "3. 彻底忽略并过滤任何喘息声、急促呼吸声、呻吟声、叹气声、咳嗽声、笑声或背景杂音，绝对不要将它们转写为文字或拟声词。\n"
        "4. 严禁重复输出已经翻译过的句子，严禁在静音或停顿期间编造、复读任何内容。"
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
# 7A. 零依赖本地 Web HUD 悬浮字幕界面
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
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 6px;
    max-width: 92vw;
    text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000, 0 2px 8px rgba(0,0,0,0.95);
  }
  .prev-line:empty { display: none; margin: 0; }
  .curr-line {
    color: #ffffff;
    font-size: 28px;
    font-weight: 800;
    line-height: 1.42;
    max-width: 95vw;
    text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000, 0 3px 12px rgba(0,0,0,0.98);
  }
</style>
</head>
<body>
  <div id="topbar" class="topbar pinned">
    <div class="status-wrap">
      <div id="dot" class="dot"></div>
      <span id="statusText">就绪 · 每句字幕保留 5 秒后自动清空</span>
    </div>
    <div class="controls">
      <span id="vadBadge" style="color:#94a3b8;font-size:12px;margin-right:6px;">静音热备</span>
      <button id="toggleBtn" class="primary" onclick="toggleRun()">▶ 开始实时翻译</button>
      <button onclick="document.body.classList.toggle('solid-bg')">🔲 切换透明/底板</button>
    </div>
  </div>
  <div class="subtitle-stage">
    <div id="prevLine" class="prev-line"></div>
    <div id="currLine" class="curr-line"></div>
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
        self.display_ctrl = SubtitleDisplayStateController(
            expire_seconds=self.settings.sentence_ttl_sec,
            max_chars_per_line=self.settings.max_chars_per_line,
        )
        self.state = {
            "is_running": False,
            "status_text": "就绪 · 点击「开始实时翻译」捕获 macOS 系统音频",
            "status_color": "#9AA0A6",
            "vad_text": "静音热备 (防休眠)",
            "prev_subtitle": "",
            "curr_subtitle": "",
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
            self.state["curr_subtitle"] = self.display_ctrl.curr_subtitle
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
                        app_ref.state["curr_subtitle"] = app_ref.display_ctrl.curr_subtitle
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
# 7B. macOS 15 原生置顶纯透明字幕悬浮窗 (跨浏览器强制置顶 + 逐句换行 + 5秒清空)
# ==============================================================================
class MacOSSubtitleOverlayApp:
    """
    针对 macOS 15 深度优化的原生悬浮字幕窗口：
    1. 强力全局置顶 (`enforce_macos_always_on_top_and_no_hide`)：设置 Cocoa `hidesOnDeactivate=False`
       及 `NSPopUpMenuWindowLevel (101)`，打开或点击浏览器、全屏视频时永不隐藏、永不掉到后台；
    2. 逐句换行短句切分：单句不超过 24 字，最多同屏显示 2 行短句；
    3. 每句严格保留 5 秒：每句满 5.0 秒后自动清空消失，停止说话 5 秒后屏幕完全清空。
    """

    def __init__(self) -> None:
        self.settings = AppSettings.load()
        self.ui_queue: "queue.Queue[Tuple[str, tuple]]" = queue.Queue()
        self.display_ctrl = SubtitleDisplayStateController(
            expire_seconds=self.settings.sentence_ttl_sec,
            max_chars_per_line=self.settings.max_chars_per_line,
        )

        self.root = tk.Tk()
        self.root.title("Gemini 3.5 Live 实时中文字幕")
        self._window_width = 860
        self.root.geometry(f"{self._window_width}x130+240+720")
        self.root.minsize(480, 68)

        self._drag_offset_x = 0
        self._drag_offset_y = 0
        self._is_running = False
        self._topbar_visible = True
        self._hover_hide_job: Optional[str] = None
        self._topmost_tick = 0
        self._history_lines: Deque[str] = deque(maxlen=200)

        self._build_ui()
        self._apply_transparency_and_topmost()

        # 监听窗口失焦事件（当用户点击浏览器时立即重新拉起置顶）
        self.root.bind("<FocusOut>", lambda _e: self._assert_topmost())
        self.root.bind("<Map>", lambda _e: self._assert_topmost())

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
        if self.settings.transparent_bg and sys.platform == "darwin":
            return "systemTransparent"
        return "#121417"

    def _assert_topmost(self) -> None:
        """强制维持窗口在 macOS 浏览器与全屏应用之上，禁止失焦隐藏。"""
        try:
            self.root.attributes("-topmost", True)
            self.root.lift()
        except Exception:
            pass
        enforce_macos_always_on_top_and_no_hide()

    def _apply_transparency_and_topmost(self) -> None:
        bg_color = self._get_bg_color()
        if sys.platform == "darwin":
            try:
                if self.settings.transparent_bg:
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
        self.root.after(80, self._assert_topmost)

    def _build_ui(self) -> None:
        bg_color = self._get_bg_color()

        self.top_bar = tk.Frame(self.root, bg="#1A1D24", height=34)
        self.top_bar.pack(fill=tk.X, side=tk.TOP)

        self.status_dot = tk.Label(
            self.top_bar, text="●", fg="#9AA0A6", bg="#1A1D24", font=("PingFang SC", 12, "bold")
        )
        self.status_dot.pack(side=tk.LEFT, padx=(10, 4), pady=3)

        self.status_label = tk.Label(
            self.top_bar,
            text="就绪 · 全局置顶已开启（切换浏览器不隐藏，每句5秒自动清空）",
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

        # 字幕主显示区（逐句换行：第一行显示上一短句，第二行显示当前短句，每句保留 5 秒自动清空）
        self.body = tk.Frame(self.root, bg=bg_color, padx=16, pady=6)
        self.body.pack(fill=tk.BOTH, expand=True)

        self.prev_subtitle_var = tk.StringVar(value="")
        self.prev_subtitle_label = tk.Label(
            self.body,
            textvariable=self.prev_subtitle_var,
            fg="#CBD5E1",
            bg=bg_color,
            font=("PingFang SC", max(16, self.settings.font_size - 6), "bold"),
            wraplength=self._window_width - 40,
            justify=tk.CENTER,
        )

        self.curr_subtitle_var = tk.StringVar(value="等待系统音频输入…")
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

        for widget in (
            self.top_bar,
            self.status_label,
            self.body,
            self.prev_subtitle_label,
            self.curr_subtitle_label,
        ):
            widget.bind("<ButtonPress-1>", self._on_drag_start)
            widget.bind("<B1-Motion>", self._on_drag_motion)

        self.root.bind("<Enter>", self._on_mouse_enter)
        self.root.bind("<Leave>", self._on_mouse_leave)

    def _toggle_bg_mode(self) -> None:
        self.settings.transparent_bg = not self.settings.transparent_bg
        self.settings.save()
        self.btn_bg_mode.configure(
            text="🔲 纯透明文字" if self.settings.transparent_bg else "⬛ 半透底板"
        )
        self._apply_transparency_and_topmost()

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
            self._assert_topmost()

    def _sync_window_height(self) -> None:
        try:
            self.root.update_idletasks()
            req_h = 20
            if self.curr_subtitle_var.get():
                req_h += self.curr_subtitle_label.winfo_reqheight() + 10
            if self.prev_subtitle_var.get():
                req_h += self.prev_subtitle_label.winfo_reqheight() + 8
            if self._topbar_visible:
                req_h += 40
            needed_h = max(68, min(360, req_h))
            cur_x = self.root.winfo_x()
            cur_y = self.root.winfo_y()
            self.root.geometry(f"{self._window_width}x{needed_h}+{cur_x}+{cur_y}")
        except Exception:
            pass

    def _update_subtitle_widgets(self, prev_txt: str, curr_txt: str) -> None:
        old_prev = self.prev_subtitle_var.get()
        old_curr = self.curr_subtitle_var.get()

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

        if curr_txt != old_curr or prev_txt != old_prev:
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
            # 立即激活一次 5 秒倒计时提示，5 秒后无语音则自动清空屏幕
            prev_t, curr_t = self.display_ctrl.on_new_subtitle("正在监听系统音频…", is_final=True)
            self._update_subtitle_widgets(prev_t, curr_t)
            self.service.start()
            self._assert_topmost()
            if self.settings.transparent_bg:
                self.root.after(2000, self._hide_topbar_for_pure_text)
        else:
            self._is_running = False
            self.btn_toggle.configure(text="▶ 开始翻译", bg="#2563EB", activebackground="#1D4ED8")
            self.service.stop()
            self._on_mouse_enter()

    def _poll_ui_events(self) -> None:
        try:
            # 1. 每 50ms 检查每句字幕是否满 5 秒，满 5 秒自动清空（包括当前字幕）
            if self.display_ctrl.check_expiration():
                self._update_subtitle_widgets(
                    self.display_ctrl.prev_subtitle,
                    self.display_ctrl.curr_subtitle,
                )

            # 2. 每隔约 0.8 秒执行一次 Cocoa 窗口置顶防隐藏加固（确保切换浏览器绝不消失）
            self._topmost_tick += 1
            if self._topmost_tick >= 16:
                self._topmost_tick = 0
                self._assert_topmost()

            # 3. 处理异步事件队列
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
            text="单行每句最大字数 (超出自动断句换行，默认 24 字):",
            fg="#E2E8F0",
            bg="#1E222B",
            font=("PingFang SC", 11, "bold"),
        ).pack(anchor=tk.W, padx=16, pady=(8, 2))

        chars_var = tk.IntVar(value=self.settings.max_chars_per_line)
        tk.Scale(
            win,
            variable=chars_var,
            from_=14,
            to=36,
            resolution=2,
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

        def _save_and_apply() -> None:
            self.settings.api_key = api_key_var.get().strip()
            self.settings.model_name = model_var.get().strip() or DEFAULT_MODEL
            self.settings.max_chars_per_line = int(chars_var.get())
            self.settings.font_size = int(font_var.get())
            self.settings.transparent_bg = bool(transparent_var.get())
            self.settings.save()

            self.display_ctrl.max_chars_per_line = self.settings.max_chars_per_line
            self.curr_subtitle_label.configure(font=("PingFang SC", self.settings.font_size, "bold"))
            self.prev_subtitle_label.configure(font=("PingFang SC", max(16, self.settings.font_size - 6), "bold"))
            self.btn_bg_mode.configure(
                text="🔲 纯透明文字" if self.settings.transparent_bg else "⬛ 半透底板"
            )
            self._apply_transparency_and_topmost()
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
    print("=== Running macOS 15 Gemini 3.5 Live Subtitle Engine Self-Test (v1.2.0) ===")

    # 1. 验证语气词与呻吟/喘息文本清洗器
    dedup = SubtitleCleanerAndDeduplicator(similarity_threshold=0.82)

    assert dedup.clean_text("嗯…啊…哦！") == "", "纯语气词应当被完全过滤"
    assert dedup.clean_text("[喘息] (叹气) 呃，今天我们来讨论系统架构，嗯。") == "今天我们来讨论系统架构", "首尾语气词与喘息标签应被剥离"
    assert dedup.clean_text("好的好的好的好的，马上处理") == "好的，马上处理", "连续复读词组应被折叠"

    # 2. 验证逐句换行切分 & 每句严格 5 秒自动清空逻辑
    disp = SubtitleDisplayStateController(expire_seconds=5.0, max_chars_per_line=20, max_visible_lines=2)

    # 输入一段包含多句的长字幕：应当自动按句切分，且最多保留最近2句换行显示
    long_stream = "第一句简短的话。第二句是比较核心的内容！第三句正在实时流式输出"
    prev_t, curr_t = disp.on_new_subtitle(long_stream, is_final=False, now=100.0)
    assert prev_t == "第二句是比较核心的内容！" and curr_t == "第三句正在实时流式输出", f"逐句切分换行异常: prev={prev_t}, curr={curr_t}"

    # 105.1 秒时（满 5 秒无新字幕），屏幕上所有字幕必须完全自动清空！
    changed = disp.check_expiration(now=105.1)
    assert changed and disp.prev_subtitle == "" and disp.curr_subtitle == "", (
        f"5秒后所有字幕应当完全清空: prev={disp.prev_subtitle}, curr={disp.curr_subtitle}"
    )

    # 3. 验证声学频谱过滤器
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
