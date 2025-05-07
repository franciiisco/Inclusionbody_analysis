"""
Módulo para la segmentación de células bacterianas en imágenes de microscopía.

Este módulo proporciona funciones para identificar y delimitar células
bacterianas en imágenes preprocesadas, permitiendo su posterior análisis.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
from skimage import measure, filters, segmentation, morphology
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


def threshold_image(
    image: np.ndarray, 
    method: str = 'adaptive', 
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Aplica umbralización a la imagen para separar células del fondo.
    
    Args:
        image: Imagen preprocesada
        method: Método de umbralización ('adaptive', 'otsu', 'binary')
        params: Parámetros específicos para el método seleccionado
    
    Returns:
        Imagen binaria (máscara) donde las células son blanco (255) y el fondo negro (0)
    """
    if params is None:
        params = {}
    
    # Valores predeterminados para cada método
    defaults = {
        'adaptive': {'block_size': 51, 'C': 5},
        'otsu': {'max_value': 255},
        'binary': {'threshold': 127, 'max_value': 255}
    }
    
    # Actualizar valores predeterminados con los proporcionados
    if method in defaults:
        for key, default_value in defaults[method].items():
            if key not in params:
                params[key] = default_value
    
    # Aplicar el método seleccionado
    if method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            params['block_size'], 
            params['C']
        )
    
    elif method == 'otsu':
        _, binary = cv2.threshold(
            image, 
            0, 
            params['max_value'], 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    
    elif method == 'binary':
        _, binary = cv2.threshold(
            image, 
            params['threshold'], 
            params['max_value'], 
            cv2.THRESH_BINARY
        )
    
    else:
        raise ValueError(f"Método de umbralización no reconocido: {method}")
    
    return binary


def apply_morphological_operations(
    binary_image: np.ndarray, 
    operations: List[Tuple[str, Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Aplica una serie de operaciones morfológicas para mejorar la segmentación.
    
    Args:
        binary_image: Imagen binaria obtenida tras la umbralización
        operations: Lista de tuplas (operación, parámetros)
            Operaciones disponibles: 'erode', 'dilate', 'open', 'close'
    
    Returns:
        Imagen binaria después de aplicar las operaciones morfológicas
    """
    if operations is None:
        operations = [
            ('open', {'kernel_size': 3, 'iterations': 2}),
            ('close', {'kernel_size': 3, 'iterations': 1})
        ]
    
    result = binary_image.copy()
    
    for operation, params in operations:
        # Crear kernel
        kernel_size = params.get('kernel_size', 3)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        iterations = params.get('iterations', 1)
        
        # Aplicar operación
        if operation == 'erode':
            result = cv2.erode(result, kernel, iterations=iterations)
        elif operation == 'dilate':
            result = cv2.dilate(result, kernel, iterations=iterations)
        elif operation == 'open':
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'close':
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            raise ValueError(f"Operación morfológica no reconocida: {operation}")
    
    return result


def segment_cells_enhanced(
    image: np.ndarray,
    min_cell_size: int = 60,
    min_distance: int = 20,
    gaussian_sigma: float = 1.0,
    find_markers_method: str = 'distance'
) -> np.ndarray:
    """
    Segmenta células utilizando transformada de distancia y watershed, 
    inspirado en la función count_bacteria.
    
    Args:
        image: Imagen preprocesada
        min_cell_size: Tamaño mínimo de célula para filtrar objetos pequeños
        min_distance: Distancia mínima entre máximos locales
        gaussian_sigma: Sigma para el suavizado gaussiano
        find_markers_method: Método para encontrar marcadores ('distance' o 'threshold')
    
    Returns:
        Imagen etiquetada donde cada célula tiene etiqueta única
    """
    # 1. Suavizado para reducir ruido
    img_blurred = cv2.GaussianBlur(image, (0, 0), gaussian_sigma)
    
    # 2. Umbralización mediante Otsu
    _, img_thresh = cv2.threshold(
        img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # 3. Calcular transformada de distancia
    distance = ndi.distance_transform_edt(img_thresh)
    
    # 4. Encontrar marcadores para watershed
    if find_markers_method == 'distance':
        # Método 1: Usando detección de máximos locales (como en count_bacteria)
        local_maxi_coords = peak_local_max(
            distance, 
            min_distance=min_distance,
            footprint=np.ones((3, 3)), 
            labels=img_thresh
        )
        
        # Crear array de marcadores
        markers = np.zeros_like(img_thresh, dtype=np.int32)
        for i, coord in enumerate(local_maxi_coords, start=1):
            markers[tuple(coord)] = i
    
    else:  # find_markers_method == 'threshold'
        # Método 2: Usando umbralización de la transformada de distancia
        # (Similar a nuestro enfoque original)
        distance_normalized = cv2.normalize(distance, None, 0, 1.0, cv2.NORM_MINMAX)
        _, markers = cv2.threshold(distance_normalized, 0.3, 1, cv2.THRESH_BINARY)
        markers = np.uint8(markers)
        _, markers = cv2.connectedComponents(markers)
    
    # 5. Aplicar watershed
    labels_ws = segmentation.watershed(-distance, markers, mask=img_thresh)
    
    # 6. Eliminar objetos pequeños
    labels_ws = morphology.remove_small_objects(labels_ws, min_size=min_cell_size)
    
    return labels_ws


def filter_regions(
    labeled_image: np.ndarray, 
    min_area: int = 50, 
    max_area: int = 2000, 
    min_circularity: float = 0.2,
    max_aspect_ratio: float = 3.0
) -> np.ndarray:
    """
    Filtra regiones segmentadas basadas en propiedades geométricas.
    
    Args:
        labeled_image: Imagen con etiquetas únicas para cada célula
        min_area: Área mínima de una célula válida
        max_area: Área máxima de una célula válida
        min_circularity: Circularidad mínima (4π*área/perímetro²)
        max_aspect_ratio: Relación máxima entre ejes mayor y menor
    
    Returns:
        Imagen etiquetada con regiones filtradas
    """
    # Crear imagen de salida
    filtered = np.zeros_like(labeled_image, dtype=np.int32)
    
    # Obtener propiedades de regiones
    regions = measure.regionprops(labeled_image)
    
    new_label = 1  # Comenzar con etiqueta 1
    
    for region in regions:
        # Omitir el fondo (etiqueta 0 o 1 en watershed)
        if region.label <= 0:
            continue
        
        area = region.area
        perimeter = region.perimeter
        
        # Calcular circularidad (4π*área/perímetro²)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calcular relación de aspecto
        aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else float('inf')
        
        # Aplicar filtros
        if (min_area <= area <= max_area and 
            circularity >= min_circularity and 
            aspect_ratio <= max_aspect_ratio):
            # La región pasa los filtros, copiar a la imagen de salida con nueva etiqueta
            coords = region.coords
            filtered[coords[:, 0], coords[:, 1]] = new_label
            new_label += 1
    
    return filtered


def segment_cells(
    image: np.ndarray, 
    config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Pipeline completo de segmentación de células.
    
    Args:
        image: Imagen preprocesada
        config: Configuración para los distintos pasos de segmentación
            {
                'use_enhanced': True/False,  # Si usar el método mejorado
                'threshold': {'method': 'adaptive', 'params': {...}},
                'morphological': [('open', {...}), ('close', {...})],
                'min_cell_size': 60,
                'min_distance': 20,
                'gaussian_sigma': 1.0,
                'find_markers_method': 'distance',
                'filter': {'min_area': 50, 'max_area': 2000, ...}
            }
    
    Returns:
        Imagen etiquetada donde cada célula tiene una etiqueta única
    """
    if config is None:
        config = {
            'use_enhanced': True,  # Por defecto usamos el método mejorado
            'threshold': {
                'method': 'adaptive', 
                'params': {'block_size': 51, 'C': 5}
            },
            'morphological': [
                ('open', {'kernel_size': 3, 'iterations': 2}),
                ('close', {'kernel_size': 3, 'iterations': 1})
            ],
            'min_cell_size': 60,
            'min_distance': 20,
            'gaussian_sigma': 1.0,
            'find_markers_method': 'distance',
            'filter': {
                'min_area': 50, 
                'max_area': 2000,
                'min_circularity': 0.2,
                'max_aspect_ratio': 3.0
            }
        }
    
    # Usar el método mejorado inspirado en count_bacteria o el método original
    if config.get('use_enhanced', True):
        # Método mejorado
        segmented = segment_cells_enhanced(
            image,
            min_cell_size=config.get('min_cell_size', 60),
            min_distance=config.get('min_distance', 20),
            gaussian_sigma=config.get('gaussian_sigma', 1.0),
            find_markers_method=config.get('find_markers_method', 'distance')
        )
    else:
        # Método original
        # 1. Umbralización
        cfg = config.get('threshold', {})
        binary = threshold_image(
            image, 
            method=cfg.get('method', 'adaptive'),
            params=cfg.get('params', {})
        )
        
        # 2. Operaciones morfológicas
        morpho_ops = config.get('morphological', [])
        binary = apply_morphological_operations(binary, morpho_ops)
        
        # 3. Separación de células (watershed)
        cfg = config.get('watershed', {})
        segmented = watershed_segmentation(
            image, 
            binary,
            min_distance=cfg.get('min_distance', 10)
        )
    
    # 4. Filtrado de regiones
    cfg = config.get('filter', {})
    filtered = filter_regions(
        segmented,
        min_area=cfg.get('min_area', 50),
        max_area=cfg.get('max_area', 2000),
        min_circularity=cfg.get('min_circularity', 0.2),
        max_aspect_ratio=cfg.get('max_aspect_ratio', 3.0)
    )
    
    return filtered


def visualize_segmentation(
    original_image: np.ndarray,
    segmented_image: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
    draw_contours: bool = True
) -> None:
    """
    Visualiza los resultados de la segmentación.
    
    Args:
        original_image: Imagen original preprocesada
        segmented_image: Imagen segmentada con etiquetas
        binary_mask: Máscara binaria opcional para mostrar
        draw_contours: Si es True, dibuja contornos de células; si es False, muestra cells como formas coloreadas
    """
    fig, axes = plt.subplots(1, 3 if binary_mask is not None else 2, figsize=(15, 5))
    
    # Imagen original
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    # Máscara binaria (si está disponible)
    if binary_mask is not None:
        axes[1].imshow(binary_mask, cmap='gray')
        axes[1].set_title('Máscara Binaria')
        axes[1].axis('off')
        idx = 2
    else:
        idx = 1
    
    # Imagen segmentada
    if draw_contours:
        # Crear una copia RGB de la imagen original para dibujar contornos
        img_segmented = np.dstack([original_image] * 3) if len(original_image.shape) == 2 else original_image.copy()
        
        # Encontrar contornos de las células segmentadas
        for region in measure.regionprops(segmented_image):
            # Get contours using scikit-image
            contour = measure.find_contours(region.image, 0.5)[0]
            # Adjust contour coordinates to image coordinates
            contour_coords = contour + np.array([region.bbox[0], region.bbox[1]])
            # Flip for OpenCV
            contour_coords_cv = np.flip(contour_coords, axis=1).astype(np.int32)
            # Draw contours
            cv2.drawContours(img_segmented, [contour_coords_cv], -1, (0, 255, 0), 1)
        
        axes[idx].imshow(img_segmented)
        axes[idx].set_title(f'Contornos ({np.max(segmented_image)} células)')
    else:
        # Imagen segmentada con formas coloreadas (overlay)
        from skimage.color import label2rgb
        # Usar alpha=0.5 para mayor opacidad de las células segmentadas
        overlay = label2rgb(segmented_image, original_image, alpha=0.5, bg_label=0, colors=None)
        axes[idx].imshow(overlay)
        axes[idx].set_title(f'Células Segmentadas ({np.max(segmented_image)} células)')
    
    axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def watershed_segmentation(
    image: np.ndarray, 
    binary_mask: np.ndarray, 
    min_distance: int = 10
) -> np.ndarray:
    """
    Aplica el algoritmo watershed para separar células adyacentes.
    
    Args:
        image: Imagen original preprocesada
        binary_mask: Máscara binaria de las células
        min_distance: Distancia mínima entre marcadores
    
    Returns:
        Imagen etiquetada donde cada célula tiene un valor único
    """
    # Calcular transformada de distancia
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Normalizar para visualización
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Umbralizar para obtener marcadores seguros
    _, sure_fg = cv2.threshold(dist_transform, 0.3, 1, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    
    # Obtener región desconocida (bordes de células)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(binary_mask, kernel, iterations=3)
    unknown = cv2.subtract(dilated, sure_fg)
    
    # Etiquetado para watershed
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Añadir 1 a todos los marcadores para que el fondo sea 1 en lugar de 0
    markers = markers + 1
    
    # Marcar la región desconocida con 0
    markers[unknown == 255] = 0
    
    # Convertir la imagen a BGR si está en escala de grises
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
    
    # Aplicar watershed
    cv2.watershed(image_colored, markers)
    
    return markers