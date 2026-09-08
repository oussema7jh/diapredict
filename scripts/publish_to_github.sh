#!/usr/bin/env bash
#
# DiaPredict AI — publish to GitHub (macOS / Linux / Git Bash).
#
# Creates the repository on GitHub and pushes the main branch.
#
# Prerequisites (choose one):
#   * GitHub CLI:  https://cli.github.com  ->  `gh auth login`
#   * or a Personal Access Token set as the GITHUB_TOKEN environment variable.
#
# Usage:
#   ./scripts/publish_to_github.sh [repo-name] [visibility]
#
# Examples:
#   ./scripts/publish_to_github.sh                       # diapredict / public
#   ./scripts/publish_to_github.sh diapredict private    # custom
#
set -euo pipefail

REPO_NAME="${1:-diapredict}"
VISIBILITY="${2:-public}"
BRANCH="main"

cd "$(dirname "$0")/.."

echo ">> Checking working directory: $(pwd)"
if [ ! -d ".git" ]; then
  echo "!! Not a git repository. Run: git init -b main"
  exit 1
fi

if [ -n "${GITHUB_TOKEN:-}" ] && ! command -v gh >/dev/null 2>&1; then
  echo ">> GITHUB_TOKEN detected (gh CLI not installed). Using token-based push."
  REMOTE_URL="https://${GITHUB_TOKEN}@github.com/$(git config --get user.name || echo 'you')/${REPO_NAME}.git"
  # Note: creating the repo requires the gh CLI or the GitHub REST API. If the
  # repo already exists, this pushes to it; otherwise create it on github.com
  # first (or install gh).
  REMOTE_URL="https://github.com/$(git config --get user.name || echo 'you')/${REPO_NAME}.git"
  echo ">> Setting remote origin -> ${REMOTE_URL} (repo must already exist)"
  git remote set-url origin "${REMOTE_URL}" 2>/dev/null || git remote add origin "${REMOTE_URL}"
  git push -u origin "${BRANCH}"
  exit 0
fi

if command -v gh >/dev/null 2>&1; then
  echo ">> gh CLI found."
  gh auth status >/dev/null 2>&1 || { echo "!! Not authenticated. Run: gh auth login"; exit 1; }
  if gh repo view "${REPO_NAME}" >/dev/null 2>&1; then
    echo ">> Repo '${REPO_NAME}' already exists — skipping creation."
  else
    echo ">> Creating GitHub repo '${REPO_NAME}' (${VISIBILITY})..."
    gh repo create "${REPO_NAME}" --"${VISIBILITY}" --source=. --remote=origin --push
    echo ">> Created and pushed."
  fi
  # Ensure remote + branch are set even if repo already existed.
  git remote set-url origin "https://github.com/$(gh api user -q .login)/${REPO_NAME}.git" 2>/dev/null || true
  git push -u origin "${BRANCH}"
  echo ">> Done. View at: https://github.com/$(gh api user -q .login)/${REPO_NAME}"
  exit 0
fi

echo "!! No GitHub CLI and no GITHUB_TOKEN found."
echo "   Install:   brew install gh   (macOS)  /  see https://cli.github.com"
echo "   or export GITHUB_TOKEN=<your-personal-access-token>"
echo "   then re-run this script."
exit 1
