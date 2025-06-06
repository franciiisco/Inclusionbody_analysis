\
# filepath: c:\\Users\\fmarquez\\Desktop\\POLIP_Analizer\\config.py

# --- Modos de Ejecución y Visualización ---
DEVELOPMENT_MODE = False  # True para modo desarrollo (con visualizaciones), False para modo estándar

# Opciones de visualización detallada (solo aplican si DEVELOPMENT_MODE es True)
VISUALIZATION_SETTINGS = {
    'show_preprocessing_steps': False,
    'show_segmentation_results': False,
    'show_inclusion_detection': False,
    'show_summary_plots': False,
    'save_intermediate_images': False

}

# Configuración para el modo estándar (actualmente solo define el comportamiento por defecto)
# La elección entre 'single' o 'batch' se manejará en main.py
STANDARD_MODE_CONFIG = {
    'run_type': 'single'  # Puede ser 'single' o 'batch'
}

# --- Configuración de Preprocesamiento, Segmentación y Detección ---

PREPROCESS_CONFIG = {
    'normalize': {'method': 'clahe', 'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
    'denoise': {'method': 'bilateral', 'params': {'d': 3, 'sigma_color': 50, 'sigma_space': 50}},
    'correct_illumination': {'method': 'morphological', 'params': {'kernel_size': 51}},
    'invert': True
}

SEGMENT_CONFIG = {
    'use_enhanced': True,
    'min_cell_size': 200,
    'min_distance': 25,  # Reducido de 25 para mayor sensibilidad a objetos cercanos
    'gaussian_sigma': 1.5,
    'find_markers_method': 'distance',
    'threshold': {
        'method': 'adaptive',
        'params': {'block_size': 51, 'C': 5}
    },
    'morphological': [
        ('open', {'kernel_size': 3, 'iterations': 2}),
        ('close', {'kernel_size': 3, 'iterations': 3})
    ],
    'filter': {
        'min_area': 200,
        'max_area': 2000,
        'min_circularity': 0.1,
        'max_aspect_ratio': 10.0
    }
}

# ------------------------------------------------------
# |   Configuración de Detección de Inclusiones (v2)   |
# ------------------------------------------------------

'''
Esta configuración es para la versión 2 de detección de inclusiones
Se recomienda usar esta versión para un mejor rendimiento y resultados más precisos.
'''

DETECTION_V2_CONFIG = {
    'preprocessing': {
        'cell_normalization': True,
        'contrast_enhancement': 'clahe',  # 'clahe', 'histogram_equalization', None
        'edge_enhancement': True
    },
    
    'thresholding': {
        'method': 'multi_level',  # 'multi_level', 'adaptive', 'otsu'
        'sensitivity': 0.45,       # Factor de ajuste (0.5-1.5)
        'adaptive_block_size': 5  # Para umbralización adaptativa
    },
    
    'separation': {
        'method': 'watershed',    # 'watershed', 'distance', 'contour_analysis'
        'min_distance': 1,        # Distancia mínima entre inclusiones
        'intensity_weight': 0.1   # Peso de la intensidad vs. distancia (0-1)
    },
    
    'filtering': {
        'min_size': 50,           # Tamaño mínimo en píxeles
        'max_size': 500,        # Tamaño máximo en píxeles
        'min_circularity': 0.15,  # Circularidad mínima (0-1)
        'min_contrast': 0.03,    # Contraste mínimo con el entorno
        'texture_analysis': False  # DESACTIVADO: para evitar filtrar inclusiones válidas
    },
    
    'debug': {
        'save_intermediate_steps': False,  # Guardar pasos intermedios para depuración
        'save_intermediate_images': False,  # Nueva opción para imágenes de cada paso
        'specific_cell_ids': []  # Para depuración de células específicas
    }
}
