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
