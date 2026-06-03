# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

openvino_binaries = collect_dynamic_libs('openvino')
openvino_datas = collect_data_files('openvino')
certifi_datas = collect_data_files('certifi')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=openvino_binaries,
    datas=openvino_datas + certifi_datas,
    hiddenimports=[
        'huggingface_hub',
        'numpy',
        'PIL',
        'csv',
        'argparse',
        'unicodedata',
        '_hashlib',
        '_ssl',
        '_blake2',
        '_sha3'
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

pyz = PYZ(
    a.pure, 
    a.zipped_data, 
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='autotagger',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
