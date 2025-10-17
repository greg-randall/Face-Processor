# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

# Find MediaPipe installation path
import mediapipe as mp
mp_path = Path(mp.__file__).parent

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include MediaPipe data files
        (str(mp_path / 'modules'), 'mediapipe/modules'),
        # Include median_landmarks.json from the project root
        ('median_landmarks.json', '.'),
    ],
    hiddenimports=['mediapipe'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='gui',
)
