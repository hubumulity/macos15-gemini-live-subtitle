#!/usr/bin/env bash
# ==============================================================================
# macOS 15 (Sequoia) Gemini 3.5 Live 实时系统音频字幕软件 - 一键部署与启动脚本
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_SCRIPT="${SCRIPT_DIR}/macos15_gemini_live_subtitle.py"
VENV_DIR="${SCRIPT_DIR}/.venv_gemini_subtitle"

echo "=================================================================="
echo "  Gemini 3.5 Live Translate · macOS 15 实时系统音频中文字幕安装与启动"
echo "=================================================================="

# 1. 检查 macOS 版本与 Python3
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[提示] 当前系统非 macOS，ScreenCaptureKit 原生内录仅在 macOS 13~15+ 上生效。"
fi

if ! command -v python3 &>/dev/null; then
    echo "[错误] 未找到 python3，请先通过 Homebrew 安装: brew install python-tk@3.12"
    exit 1
fi

# 2. 创建并激活独立虚拟环境
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[1/3] 正在创建 Python 虚拟环境: ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# 3. 安装 Google GenAI SDK 与音频计算依赖
echo "[2/3] 正在检查并安装依赖 (google-genai, numpy, sounddevice) ..."
pip install --upgrade pip --quiet
pip install --upgrade google-genai numpy sounddevice --quiet

# 4. 预编译 macOS 15 ScreenCaptureKit 原生系统音频捕获器 (免装 BlackHole)
if command -v swiftc &>/dev/null; then
    echo "[3/3] 检测到 Swift 编译器，将在启动时自动启用 macOS 15 ScreenCaptureKit 原生系统内录。"
else
    echo "[提示] 未检测到 Xcode Command Line Tools (swiftc)。"
    echo "       若需免驱动直接内录 macOS 15 系统音频，可运行: xcode-select --install"
    echo "       或者通过 brew install blackhole-2ch 使用虚拟声卡模式。"
fi

echo "------------------------------------------------------------------"
echo "正在启动悬浮字幕窗 (首次启动请在「录屏与系统录音」权限弹窗中点击允许)..."
echo "------------------------------------------------------------------"
exec python3 "${APP_SCRIPT}" "$@"