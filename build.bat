@echo off
REM This script cleans up old build files and creates a new one-file executable.

echo.
echo Closing any running instances of the application...
REM Forcefully closes the gui.exe process if it's running. The >nul 2>&1 part hides any "process not found" errors.
taskkill /F /IM gui.exe >nul 2>&1
echo Cleaning up old build files...

REM Remove the build and dist folders if they exist
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo Starting a new build...

REM Build using the existing spec file (which includes MediaPipe data files)
python -m PyInstaller --noconfirm --clean gui.spec

echo.
echo Packaging release archive...
python3 package_release.py

echo.
echo Build complete! Release archive created.
echo.
pause
