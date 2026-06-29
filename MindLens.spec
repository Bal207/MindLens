# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None

project_dir = os.path.abspath(os.path.dirname(SPECPATH))

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
    console=True,
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
    )
