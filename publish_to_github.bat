@echo off
setlocal

REM Usage:
REM   publish_to_github.bat https://github.com/<username>/<repo>.git

if "%~1"=="" (
  echo Please pass your GitHub repository URL.
  echo Example:
  echo   publish_to_github.bat https://github.com/PonnarasuP/impact.git
  exit /b 1
)

set REPO_URL=%~1

echo Initializing git repository (if needed)...
if not exist .git (
  git init
)

echo Staging files...
git add .

echo Creating commit...
git commit -m "Initial commit: stock predictor app"

echo Setting main branch...
git branch -M main

echo Adding or updating remote origin...
git remote remove origin >nul 2>nul
git remote add origin %REPO_URL%

echo Pushing to GitHub...
git push -u origin main

if errorlevel 1 (
  echo.
  echo Push failed. Make sure:
  echo 1) You are logged in with GitHub credentials
  echo 2) The repository URL is correct
  echo 3) The repository exists on GitHub
  exit /b 1
)

echo.
echo Done. Your project is now on GitHub.
exit /b 0
