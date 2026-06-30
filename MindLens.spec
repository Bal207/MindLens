# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None

# SPECPATH is the directory that contains this .spec file (the repo root),
# which is exactly where main.py, website/ and CameraDetection/ live.
project_dir = os.path.abspath(SPECPATH)

added_files = [
    (os.path.join(project_dir, 'website'), 'website'),
    (os.path.join(project_dir, 'CameraDetection', 'yolo26n.pt'), 'CameraDetection'),
]

a = Analysis(
    ['main.py'],
    pathex=[project_dir],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'flask',
        'webview',
        'mss',
        'pyautogui',
        'cv2',
        'easyocr',
        'torch',
        'transformers',
        'ultralytics',
        'mediapipe',
        'numpy',
        'pillow'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MindLens',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # windowed app — no terminal window pops up
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MindLens',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='MindLens.app',
        icon=None,
        bundle_identifier='com.mindlens.app',
        info_plist={
            'NSHighResolutionCapable': True,
            'LSApplicationCategoryType': 'public.app-category.productivity',
            # macOS requires a usage description or it kills the app the moment
            # it touches the camera / screen recording.
            'NSCameraUsageDescription':
                'MindLens uses your camera locally to detect phone use and posture.',
            'NSMicrophoneUsageDescription':
                'MindLens does not record audio.',
        },
    )
