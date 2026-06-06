@echo off
setlocal

:: ==============================================================================
:: PLEASE UPDATE THE REPO NAME BELOW BEFORE RUNNING THE SCRIPT
:: Replace "your-repo-name" with your actual GitHub repository name
:: ==============================================================================
set REPO_URL=https://github.com/hafeezug/Balochi-Tokenizer

echo ----------------------------------------
echo Starting GitHub Push Process
echo ----------------------------------------

:: Check if git is initialized; if not, initialize it
if not exist ".git\" (
    echo Initializing Git repository...
    git init
) else (
    echo Git repository already initialized.
)

echo Adding files...
git add .

echo Committing changes...
:: If there are no changes, this might throw an error, but the script will continue
git commit -m "Update repository"

echo Renaming branch to main...
git branch -M main

:: Check if origin exists, if not add it, if yes, set URL just in case
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo Adding remote origin...
    git remote add origin %REPO_URL%
) else (
    echo Updating remote origin...
    git remote set-url origin %REPO_URL%
)

echo Pushing to GitHub...
git push -u origin main

echo ----------------------------------------
echo Done!
echo ----------------------------------------
pause
