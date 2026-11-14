#!/usr/bin/env bash
# ==============================================================================
# 一键发布到 https://github.com/hubumulity/macos15-gemini-live-subtitle
# 并触发 GitHub Actions 在线编译 macOS 15 (.app & .dmg) 安装包
# ==============================================================================
set -e

GITHUB_USER="hubumulity"
REPO_NAME="${1:-macos15-gemini-live-subtitle}"
REPO_FULL="${GITHUB_USER}/${REPO_NAME}"
REPO_GIT_URL="https://github.com/${REPO_FULL}.git"
VISIBILITY="--public"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=================================================================="
echo "  一键发布至 https://github.com/${REPO_FULL} 并启动 macOS 15 在线编译"
echo "=================================================================="

# 1. 自动整合并创建标准的 GitHub Actions 目录结构
mkdir -p .github/workflows
if [[ -f "build-macos15-dmg.yml" ]]; then
    cp -f "build-macos15-dmg.yml" ".github/workflows/build-macos15-dmg.yml"
fi

cat << 'EOF' > requirements.txt
google-genai>=1.0.0
numpy>=1.26.0
sounddevice>=0.4.6
EOF

cat << 'EOF' > .gitignore
.venv_gemini_subtitle/
__pycache__/
*.pyc
build/
dist/
dmg_staging/
build_bin/
*.dmg
*.sha256
.DS_Store
EOF

# 2. 初始化 Git 仓库并提交所有构建源码
if [[ ! -d ".git" ]]; then
    git init
    git branch -M main
fi

git add macos15_gemini_live_subtitle.py setup_and_run_macos15.sh publish_to_github.sh requirements.txt .gitignore .github/workflows/build-macos15-dmg.yml
git commit -m "feat: macOS 15 Gemini 3.5 Live Translate real-time subtitle app + GitHub Actions DMG builder" || true

# 3. 推送至 GitHub 仓库 (https://github.com/hubumulity/macos15-gemini-live-subtitle)
if command -v gh &>/dev/null; then
    if ! gh auth status &>/dev/null; then
        echo "[步骤 1] 请先登录 GitHub CLI (将在浏览器中完成授权)..."
        gh auth login
    fi

    if ! git remote get-url origin &>/dev/null; then
        echo "[步骤 2] 正在创建公开仓库并推送源码: ${REPO_FULL} (${VISIBILITY})..."
        gh repo create "${REPO_FULL}" "${VISIBILITY}" --source=. --remote=origin --push
    else
        echo "[步骤 2] 正在推送最新代码到 https://github.com/${REPO_FULL} ..."
        git push -u origin main
    fi

    echo "[步骤 3] 正在推送 v1.0.0 标签并触发 GitHub Actions 在线编译 macOS 15 DMG 安装包..."
    git tag -f v1.0.0
    git push -f origin v1.0.0
    gh workflow run build-macos15-dmg.yml -f create_release=true || true

    echo "=================================================================="
    echo "✅ 发布成功！GitHub 正在云端 macos-15 实例上编译 .app 与 .dmg 安装包："
    echo "🔗 仓库主页: https://github.com/${REPO_FULL}"
    echo "⚙️ 在线编译进度: https://github.com/${REPO_FULL}/actions"
    echo "📦 安装包下载页 (编译完成后自动生成): https://github.com/${REPO_FULL}/releases"
    echo "=================================================================="
else
    echo "[步骤 1] 请确保已在 https://github.com/new 创建公开仓库: ${REPO_NAME}"
    if git remote get-url origin &>/dev/null; then
        git remote set-url origin "${REPO_GIT_URL}"
    else
        git remote add origin "${REPO_GIT_URL}"
    fi

    echo "[步骤 2] 正在推送源码与 GitHub Actions 工作流至 ${REPO_GIT_URL} ..."
    git push -u origin main

    echo "[步骤 3] 正在推送 v1.0.0 标签以自动触发 GitHub Release DMG 安装包编译..."
    git tag -f v1.0.0
    git push -f origin v1.0.0

    echo "=================================================================="
    echo "✅ 推送完成！GitHub Actions 已自动开始在线编译 macOS 15 (.dmg) 安装包："
    echo "🔗 仓库主页: https://github.com/${REPO_FULL}"
    echo "⚙️ 查看在线编译进度: https://github.com/${REPO_FULL}/actions"
    echo "📦 下载编译好的 .dmg 安装包: https://github.com/${REPO_FULL}/releases"
    echo "=================================================================="
fi