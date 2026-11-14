#!/usr/bin/env bash
# ==============================================================================
# 一键修复 GitHub 鉴权报错、自动创建公开仓库并触发 macOS 15 (.dmg) 在线编译
# 目标仓库: https://github.com/hubumulity/macos15-gemini-live-subtitle
# ==============================================================================
set -e

GITHUB_USER="hubumulity"
REPO_NAME="${1:-macos15-gemini-live-subtitle}"
REPO_FULL="${GITHUB_USER}/${REPO_NAME}"
VISIBILITY="--public"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=================================================================="
echo "  一键修复 GitHub 鉴权并发布至 https://github.com/${REPO_FULL}"
echo "=================================================================="

# 1. 清除 macOS 钥匙串 (osxkeychain) 中缓存的旧密码/失效 Token，避免 Git 反复发送旧凭据报错
if command -v git-credential-osxkeychain &>/dev/null || [[ "$(uname -s)" == "Darwin" ]]; then
    echo "[1/4] 正在清理 macOS 钥匙串中残留的旧 GitHub 密码缓存..."
    printf "protocol=https\nhost=github.com\n\n" | git credential-osxkeychain erase 2>/dev/null || true
    printf "protocol=https\nhost=github.com\nusername=%s\n\n" "${GITHUB_USER}" | git credential-osxkeychain erase 2>/dev/null || true
fi

# 2. 自动整合并创建标准的 GitHub Actions 目录结构
echo "[2/4] 正在准备项目文件与 GitHub Actions 工作流..."
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

if [[ ! -d ".git" ]]; then
    git init
    git branch -M main
fi

git add macos15_gemini_live_subtitle.py setup_and_run_macos15.sh publish_to_github.sh requirements.txt .gitignore .github/workflows/build-macos15-dmg.yml
git commit -m "feat: macOS 15 Gemini 3.5 Live Translate real-time subtitle app + GitHub Actions DMG builder" || true

# 3. 优先使用 GitHub CLI (gh) 浏览器免密授权；若未安装则使用 Personal Access Token 自动创建仓库并推送
if command -v gh &>/dev/null; then
    echo "[3/4] 检测到 GitHub CLI (gh)，正在配置 repo 与 workflow 权限..."
    if ! gh auth status &>/dev/null; then
        gh auth login --hostname github.com --git-protocol https --scopes repo,workflow --web
    else
        # 确保当前 Token 具备推送 .github/workflows 的 workflow 权限
        gh auth refresh --hostname github.com --scopes repo,workflow || true
    fi
    # 关键步骤：将 gh 注册为 Git 凭据助手，替代失效的密码验证
    gh auth setup-git --hostname github.com

    if ! gh repo view "${REPO_FULL}" &>/dev/null; then
        echo "[4/4] 正在创建公开仓库 https://github.com/${REPO_FULL} 并推送代码..."
        git remote remove origin 2>/dev/null || true
        gh repo create "${REPO_FULL}" "${VISIBILITY}" --source=. --remote=origin --push
    else
        git remote remove origin 2>/dev/null || true
        git remote add origin "https://github.com/${REPO_FULL}.git"
        git push -u origin main --force
    fi

    git tag -f v1.0.0
    git push -f origin v1.0.0
    gh workflow run build-macos15-dmg.yml --repo "${REPO_FULL}" -f create_release=true || true
else
    echo "------------------------------------------------------------------"
    echo "GitHub 已停用账号密码直接进行 Git 推送，需要使用 Personal Access Token (PAT)。"
    TOKEN_URL="https://github.com/settings/tokens/new?scopes=repo,workflow&description=macos15-gemini-live-subtitle"
    echo "正在为你打开 GitHub Token 生成页面（已自动勾选 repo 和 workflow 权限）："
    echo "👉 ${TOKEN_URL}"
    if [[ "$(uname -s)" == "Darwin" ]]; then
        open "${TOKEN_URL}" 2>/dev/null || true
    fi
    echo "------------------------------------------------------------------"
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        read -r -s -p "请在网页点击底部绿色按钮 Generate token，复制以 ghp_ 开头的 Token 粘贴到此处 (输入不回显，粘贴后按回车): " GITHUB_TOKEN
        echo ""
    fi

    if [[ -z "${GITHUB_TOKEN}" ]]; then
        echo "[错误] Token 不能为空。请重新运行 bash publish_to_github.sh 并粘贴 Token。"
        exit 1
    fi

    echo "[3/4] 正在通过 GitHub API 自动检查/创建公开仓库 ${REPO_FULL} ..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: token ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/${REPO_FULL}")

    if [[ "${HTTP_CODE}" == "404" ]]; then
        curl -s -X POST \
            -H "Authorization: token ${GITHUB_TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            https://api.github.com/user/repos \
            -d "{\"name\":\"${REPO_NAME}\",\"description\":\"macOS 15 Gemini 3.5 Live Translate real-time subtitle app\",\"private\":false}" >/dev/null
        echo "      已自动创建公开仓库: https://github.com/${REPO_FULL}"
    fi

    echo "[4/4] 正在使用 Token 安全推送源码与 GitHub Actions 工作流..."
    AUTH_REMOTE="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${REPO_FULL}.git"
    git push "${AUTH_REMOTE}" main:main --force
    git tag -f v1.0.0
    git push "${AUTH_REMOTE}" refs/tags/v1.0.0 --force

    # 恢复干净的远程地址（不在本地 .git/config 中明文残留 Token）
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://github.com/${REPO_FULL}.git"
fi

echo "=================================================================="
echo "✅ 发布成功！GitHub Actions 已开始在线编译 macOS 15 (.app & .dmg) 安装包："
echo "🔗 仓库主页: https://github.com/${REPO_FULL}"
echo "⚙️ 在线编译进度: https://github.com/${REPO_FULL}/actions"
echo "📦 安装包下载页 (编译完成后自动生成): https://github.com/${REPO_FULL}/releases"
echo "=================================================================="