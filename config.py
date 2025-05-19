\
# filepath: c:\\Users\\fmarquez\\Desktop\\POLIP_Analizer\\config.py


# --- Modos de Ejecución y Visualización ---
DEVELOPMENT_MODE = True  # True para modo desarrollo (con visualizaciones), False para modo estándar

# Opciones de visualización detallada (solo aplican si DEVELOPMENT_MODE es True)
VISUALIZATION_SETTINGS = {
    'show_preprocessing_steps': False,
    'show_segmentation_results': False,
    'show_inclusion_detection': True,
    'show_summary_plots': False
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

DETECTION_CONFIG = {
    'min_size': 2,  
    'max_size': 600,
    'threshold_offset': 0.05,    # Reducido de 0.1 para mayor sensibilidad
    'min_contrast': 0.03,       # Reducido de 0.12 para mayor sensibilidad
    'contrast_window': 3,       # Aumentado de 3 para mejorar el cálculo de contraste en áreas con inclusiones cercanas
    'remove_border': False,
    'min_circularity': 0.7      # Reducido de 0.7 para permitir objetos menos circulares (posiblemente fusionados)
}

# --- Configuración de la nueva detección de inclusiones v2.0 ---
DETECTION_V2_CONFIG = {
    'preprocessing': {
        'cell_normalization': True,
        'contrast_enhancement': 'clahe',  # 'clahe', 'histogram_equalization', None
        'edge_enhancement': True
    },
    
    'thresholding': {
        'method': 'multi_level',  # 'multi_level', 'adaptive', 'otsu'
        'sensitivity': 0.6,       # Factor de ajuste (0.5-1.5)
        'adaptive_block_size': 15  # Para umbralización adaptativa
    },
    
    'separation': {
        'method': 'watershed',    # 'watershed', 'distance', 'contour_analysis'
        'min_distance': 5,        # Distancia mínima entre inclusiones
        'intensity_weight': 0.8   # Peso de la intensidad vs. distancia (0-1)
    },
    
    'filtering': {
        'min_size': 3,           # Tamaño mínimo en píxeles
        'max_size': 1500,        # Tamaño máximo en píxeles
        'min_circularity': 0.3,  # Circularidad mínima (0-1)
        'min_contrast': 0.05,    # Contraste mínimo con el entorno
        'texture_analysis': True  # Análisis de textura para validación
    },
    
    'debug': {
        'save_intermediate_steps': False,  # Guardar pasos intermedios para depuración
        'specific_cell_ids': []  # Para depuración de células específicas
    }
}
