# DiaPredict AI - publish to GitHub (Windows PowerShell).
#
# Creates the repository on GitHub and pushes the main branch.
#
# Prerequisites (choose one):
#   * GitHub CLI: https://cli.github.com  ->  `gh auth login`
#   * or a Personal Access Token set as the GITHUB_TOKEN environment variable.
#
# Usage:
#   .\scripts\publish_to_github.ps1                 # diapredict / public
#   .\scripts\publish_to_github.ps1 diapredict private
#
param(
    [string]$RepoName = "diapredict",
    [string]$Visibility = "public"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host ">> Working directory: $(Get-Location)" -ForegroundColor Cyan
if (-not (Test-Path ".git")) {
    Write-Host "!! Not a git repository. Run: git init -b main"
    exit 1
}

# If a token is provided and gh is unavailable, push with the token.
if ($env:GITHUB_TOKEN -and -not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host ">> GITHUB_TOKEN detected. Setting remote -> token push." -ForegroundColor Cyan
    $owner = git config --get user.name
    if (-not $owner) { $owner = "you" }
    # NOTE: the token path cannot create a repo by itself; create it on
    # github.com or install gh. It pushes to an existing repo.
    $url = "https://github.com/$owner/$RepoName.git"
    git remote remove origin 2>$null
    git remote add origin $url
    git push -u origin main
    Write-Host ">> Pushed. View at: $url"
    exit 0
}

$gh = Get-Command gh -ErrorAction SilentlyContinue
if ($gh) {
    Write-Host ">> gh CLI found." -ForegroundColor Cyan
    gh auth status *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!! Not authenticated. Run: gh auth login"
        exit 1
    }
    $exists = (gh repo view $RepoName *> $null; $LASTEXITCODE -eq 0)
    if ($exists) {
        Write-Host ">> Repo '$RepoName' already exists." -ForegroundColor Yellow
    } else {
        Write-Host ">> Creating repo '$RepoName' ($Visibility)..." -ForegroundColor Cyan
        gh repo create $RepoName --$Visibility --source . --remote origin --push
    }
    $login = (gh api user -q .login).Trim()
    git push -u origin main
    Write-Host ">> Done. View at: https://github.com/$login/$RepoName" -ForegroundColor Green
    exit 0
}

Write-Host "!! No GitHub CLI and no GITHUB_TOKEN found." -ForegroundColor Red
Write-Host "   Install: winget install GitHub.cli   (then 'gh auth login')"
Write-Host "   or set \$env:GITHUB_TOKEN = '<your-personal-access-token>'"
Write-Host "   then re-run this script."
exit 1
