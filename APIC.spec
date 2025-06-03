# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

# Obtener el directorio base del proyecto
base_dir = os.path.dirname(os.path.abspath(SPEC))

# Definir directorios de datos
data_dir = os.path.join(base_dir, 'data')
logo_dir = os.path.join(data_dir, 'logo')
processed_dir = os.path.join(data_dir, 'processed')
raw_dir = os.path.join(data_dir, 'raw')

# Archivos de datos para incluir
datas = []

# Incluir toda la carpeta data
if os.path.exists(data_dir):
    datas.append((data_dir, 'data'))

# Incluir específicamente el logo de CNTA
cnta_logo_path = os.path.join(logo_dir, 'cnta.jpg')
if os.path.exists(cnta_logo_path):
    print(f"Incluyendo logo CNTA: {cnta_logo_path}")
    datas.append((cnta_logo_path, 'data/logo'))
else:
    print(f"¡ADVERTENCIA! No se encontró el logo CNTA en: {cnta_logo_path}")

# Incluir específicamente el icono de la aplicación
icon_path = os.path.join(logo_dir, 'icon.ico')
if os.path.exists(icon_path):
    print(f"Incluyendo icono: {icon_path}")
    datas.append((icon_path, 'data/logo'))
else:
    print(f"¡ADVERTENCIA! No se encontró el icono en: {icon_path}")

# Incluir archivos de descripción de las tabs
descriptions_dir = os.path.join(base_dir, 'app', 'tabs', 'descriptions')
if os.path.exists(descriptions_dir):
    datas.append((descriptions_dir, 'app/tabs/descriptions'))

# Incluir archivos de configuración
config_files = ['config.py', 'requirements.txt', 'README.md']
for config_file in config_files:
    config_path = os.path.join(base_dir, config_file)
    if os.path.exists(config_path):
        datas.append((config_path, '.'))

# Incluir archivos de datos de matplotlib (fuentes, configuraciones)
import matplotlib
mpl_data_dir = matplotlib.get_data_path()
datas.append((mpl_data_dir, 'matplotlib/mpl-data'))

# Incluir archivos de configuración de matplotlib
try:
    import matplotlib.font_manager
    font_paths = matplotlib.font_manager.findSystemFonts()
    for font_path in font_paths[:10]:  # Limitar a 10 fuentes para no hacer el exe muy grande
        if os.path.exists(font_path):
            datas.append((font_path, 'matplotlib/fonts'))
except:
    pass

# Lista completa de importaciones ocultas basada en requirements.txt y el código
hiddenimports = [
    # Bibliotecas de análisis y procesamiento numérico
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.lib.format',
    'numpy.random',
    'numpy.linalg',
    'scipy',
    'scipy.spatial.distance',
    'scipy.ndimage',
    'scipy.signal',
    'scipy.optimize',
    'scipy.stats',
    'scipy.sparse',
    'scipy.interpolate',
    
    # Procesamiento de imágenes
    'cv2',
    'skimage',
    'skimage.feature',
    'skimage.filters',
    'skimage.measure',
    'skimage.morphology',
    'skimage.segmentation',
    'skimage.transform',
    'skimage.util',
    'skimage.exposure',
    'skimage.restoration',
    'skimage.color',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL._tkinter_finder',
    
    # Visualización de resultados - CRÍTICO para matplotlib
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
    'matplotlib.backends._backend_agg',
    'matplotlib.figure',
    'matplotlib.patches',
    'matplotlib.collections',
    'matplotlib.colors',
    'matplotlib.cm',
    'matplotlib.ticker',
    'matplotlib.font_manager',
    'matplotlib.ft2font',
    'matplotlib._path',
    'matplotlib.mathtext',
    'matplotlib.dviread',
    'matplotlib.tri',
    'matplotlib.axes',
    'matplotlib.axis',
    'matplotlib.cbook',
    'matplotlib.dates',
    'matplotlib.gridspec',
    'matplotlib.image',
    'matplotlib.legend',
    'matplotlib.lines',
    'matplotlib.markers',
    'matplotlib.mlab',
    'matplotlib.path',
    'matplotlib.patches',
    'matplotlib.pylab',
    'matplotlib.scale',
    'matplotlib.spines',
    'matplotlib.style',
    'matplotlib.text',
    'matplotlib.transforms',
    'matplotlib.widgets',
    'matplotlib.rcsetup',
    'matplotlib._version',
    
    # Análisis de datos
    'pandas',
    'pandas.plotting',
    'pandas.io.formats.excel',
    'openpyxl',
    'openpyxl.workbook',
    'openpyxl.worksheet',
    'openpyxl.styles',
    
    # Interfaz gráfica
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.scrolledtext',
    'ttkbootstrap',
    'ttkbootstrap.constants',
    'ttkbootstrap.themes',
    'ttkbootstrap.style',
    'ttkbootstrap.widgets',
    
    # Módulos estándar de Python que a veces faltan
    'json',
    'argparse',
    'os',
    'sys',
    'pathlib',
    'threading',
    'queue',
    'time',
    'datetime',
    'collections',
    'itertools',
    'functools',
    'warnings',
    'logging',
    
    # Módulos específicos del proyecto
    'src',
    'src.core',
    'src.analysis',
    'src.preprocessing',
    'src.segmentation',
    'src.detection_v2_0',
    'src.features',
    'src.visualization',
    'utils',
    'utils.file_operations',
    'utils.visualization',
    'app',
    'app.components',
    'app.components.progress_tracker',
    'app.components.results_viewer',
    'app.tabs',
    'app.tabs.analysis_tab',
    'app.tabs.methodology_tab',
    'app.tabs.info_tab',
    'config',
]

a = Analysis(
    ['app\\gui.py'],
    pathex=[base_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        # Configuración específica para matplotlib
        "matplotlib": {
            "backends": ["TkAgg", "Agg"]
        }
    },
    runtime_hooks=[],
    excludes=[
        # Excluir módulos que no necesitas para reducir tamaño
        'IPython',
        'jupyter',
        'notebook',
        'qtconsole',
        'spyder',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'wx',
        # Excluir backends de matplotlib que no necesitamos
        'matplotlib.backends.backend_qt4agg',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_gtk3agg',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# Configurar el ejecutable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='APIC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # False para aplicación GUI (sin ventana de consola)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Configurar el icono - IMPORTANTE: usar ruta absoluta
    icon=icon_path if os.path.exists(icon_path) else None,
)

# Crear la colección final
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='APIC',
)