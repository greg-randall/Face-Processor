#!/bin/bash
# This script cleans up old build files and creates a new one-file executable.

echo "Cleaning up old build files..."

# Remove the build and dist folders if they exist
rm -rf build/
rm -rf dist/

echo "Starting a new build..."

# Build using the existing spec file (which includes MediaPipe data files)
python3 -m PyInstaller --noconfirm --clean gui.spec


echo "Build complete! The new executable is in the 'dist' folder."
