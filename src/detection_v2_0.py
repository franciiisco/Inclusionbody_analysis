"""
Módulo para la detección de inclusiones de polifosfatos dentro de células bacterianas (Versión 2.0).

Este módulo proporciona funciones mejoradas para identificar y caracterizar las 
inclusiones de polifosfatos dentro de células previamente segmentadas, con mejor
manejo de inclusiones cercanas y variaciones de contraste.
"""
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import cv2
from skimage import measure, filters, feature, segmentation, morphology, transform


def create_cell_masks(segmented_image: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Crea máscaras individuales para cada célula segmentada.
    
    Args:
        segmented_image: Imagen segmentada donde cada célula tiene una etiqueta única
    
    Returns:
        Diccionario que mapea IDs de células a sus máscaras binarias
    """
    unique_labels = np.unique(segmented_image)
    
    # Excluir el fondo (etiqueta 0)
    unique_labels = unique_labels[unique_labels > 0]
    
    cell_masks = {}
    for label in unique_labels:
        # Crear máscara binaria para esta célula
        mask = (segmented_image == label).astype(np.uint8) * 255
        cell_masks[int(label)] = mask
    
    return cell_masks


def enhance_cell_contrast(image: np.ndarray, cell_mask: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Mejora el contraste dentro de una célula individual.
    
    Args:
        image: Imagen original
        cell_mask: Máscara binaria de la célula
        method: Método de mejora ('clahe', 'histogram_equalization', None)
    
    Returns:
        Imagen con contraste mejorado dentro de la célula
    """
    # Crear una copia de la región de la imagen
    cell_region = image.copy()
    
    # Si no se especifica un método, devolver la imagen original
    if method is None:
        return cell_region
    
    # Extraer solo la región de la célula
    x, y, w, h = cv2.boundingRect(cell_mask.astype(np.uint8))
    roi = image[y:y+h, x:x+w]
    mask_roi = cell_mask[y:y+h, x:x+w]
    
    # Aplicar mejora de contraste según el método especificado
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Crear una imagen de trabajo con solo los píxeles de la célula
        cell_only = roi.copy()
        cell_only[mask_roi == 0] = 0
        
        # Aplicar CLAHE
        enhanced_roi = clahe.apply(cell_only)
        
        # Combinar el resultado con la imagen original
        result_roi = roi.copy()
        result_roi[mask_roi > 0] = enhanced_roi[mask_roi > 0]
        
        # Copiar el resultado de vuelta a la imagen completa
        result = image.copy()
        result[y:y+h, x:x+w] = result_roi
        return result
        
    elif method == 'histogram_equalization':
        # Similar al anterior pero con ecualización de histograma normal
        cell_only = roi.copy()
        cell_only[mask_roi == 0] = 0
        
        # Aplicar ecualización de histograma
        enhanced_roi = cv2.equalizeHist(cell_only)
        
        # Combinar el resultado con la imagen original
        result_roi = roi.copy()
        result_roi[mask_roi > 0] = enhanced_roi[mask_roi > 0]
        
        # Copiar el resultado de vuelta a la imagen completa
        result = image.copy()
        result[y:y+h, x:x+w] = result_roi
        return result
    
    return cell_region


def apply_edge_enhancement(image: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
    """
    Mejora los bordes dentro de una célula para ayudar a separar inclusiones cercanas.
    
    Args:
        image: Imagen original
        cell_mask: Máscara binaria de la célula
    
    Returns:
        Imagen con bordes mejorados dentro de la célula
    """
    # Aplicar filtro de Sobel para detección de bordes
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combinar los bordes en todas las direcciones
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalizar a 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Mejorar la imagen original usando los bordes
    enhanced = cv2.addWeighted(image, 0.7, magnitude, 0.3, 0)
    
    # Aplicar solo dentro de la célula
    result = image.copy()
    result[cell_mask > 0] = enhanced[cell_mask > 0]
    
    return result


def multilevel_threshold(image: np.ndarray, cell_mask: np.ndarray, 
                        sensitivity: float = 0.8, n_levels: int = 3) -> np.ndarray:
    """
    Aplica umbrales múltiples para mejorar la detección de inclusiones con diferentes intensidades.
    
    Args:
        image: Imagen original (inclusiones brillantes)
        cell_mask: Máscara binaria de la célula
        sensitivity: Factor de sensibilidad (0.5-1.5)
        n_levels: Número de niveles de umbral a aplicar
    
    Returns:
        Máscara binaria de inclusiones candidatas
    """
    # Extraer valores de píxeles dentro de la célula
    cell_pixels = image[cell_mask > 0]
    
    if len(cell_pixels) == 0:
        return np.zeros_like(cell_mask)
    
    # Calcular estadísticas básicas
    mean_val = np.mean(cell_pixels)
    std_val = np.std(cell_pixels)
    
    # Generar múltiples umbrales
    thresholds = [mean_val + sensitivity * std_val * (i/n_levels) for i in range(1, n_levels+1)]
    
    # Inicializar máscara de salida
    inclusion_mask = np.zeros_like(cell_mask)
    
    # Aplicar cada umbral y combinar resultados
    for threshold in thresholds:
        temp_mask = np.zeros_like(cell_mask)
        temp_mask[(image > threshold) & (cell_mask > 0)] = 255
        
        # Combinar con la máscara actual
        inclusion_mask = cv2.bitwise_or(inclusion_mask, temp_mask)
    
    return inclusion_mask


def adaptive_local_threshold(image: np.ndarray, cell_mask: np.ndarray, 
                          block_size: int = 15, sensitivity: float = 0.8) -> np.ndarray:
    """
    Aplica umbralización adaptativa local para detectar inclusiones.
    
    Args:
        image: Imagen original (inclusiones brillantes)
        cell_mask: Máscara binaria de la célula
        block_size: Tamaño del bloque para adaptación local
        sensitivity: Factor de ajuste para sensibilidad
    
    Returns:
        Máscara binaria de inclusiones candidatas
    """
    # Asegurarse que el tamaño del bloque es impar
    if block_size % 2 == 0:
        block_size += 1
    
    # Calcular C (constante de ajuste)
    cell_pixels = image[cell_mask > 0]
    if len(cell_pixels) == 0:
        return np.zeros_like(cell_mask)
    
    mean_intensity = np.mean(cell_pixels)
    std_intensity = np.std(cell_pixels)
    C = -(sensitivity * std_intensity)
    
    # Aplicar umbralización adaptativa solo en la región de la célula
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[cell_mask > 0] = image[cell_mask > 0]
    
    # Umbralización adaptativa
    binary = cv2.adaptiveThreshold(
        mask, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        C
    )
    
    # Asegurar que solo consideramos la región de la célula
    binary[cell_mask == 0] = 0
    
    return binary


def separate_inclusions_watershed(binary_mask: np.ndarray, original_image: np.ndarray, 
                               min_distance: int = 5, intensity_weight: float = 0.7) -> np.ndarray:
    """
    Separa inclusiones cercanas utilizando el algoritmo watershed con mejoras para
    la detección de inclusiones conectadas por líneas delgadas.
    
    Args:
        binary_mask: Máscara binaria inicial de inclusiones
        original_image: Imagen original para usar en la transformada watershed
        min_distance: Distancia mínima entre marcadores
        intensity_weight: Peso para la intensidad vs distancia
    
    Returns:
        Máscara binaria mejorada con inclusiones separadas
    """
    # Si no hay objetos en la máscara, devolver la máscara original
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    # MEJORA 1: Aplicar un filtro top-hat para resaltar pequeñas estructuras
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tophat = cv2.morphologyEx(binary_mask, cv2.MORPH_TOPHAT, kernel)
    enhanced_mask = cv2.add(binary_mask, tophat)
    
    # MEJORA 2: Reducir conexiones delgadas utilizando apertura
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Transformada de distancia
    dist = cv2.distanceTransform(opened_mask, cv2.DIST_L2, 5)
    
    # MEJORA 3: Aplicar un filtro gaussiano para suavizar la transformada y resaltar máximos
    dist_smooth = cv2.GaussianBlur(dist, (3, 3), 0)
    
    # Normalizar para visualización
    dist_normalized = cv2.normalize(dist_smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Encontrar máximos locales (obtener coordenadas)
    # MEJORA 4: Reducir threshold para detectar más picos
    coords = feature.peak_local_max(
        dist_smooth, 
        min_distance=min_distance,
        footprint=np.ones((3, 3)),
        exclude_border=False,
        threshold_rel=0.3  # Umbral relativo más bajo para detectar más máximos
    )
    
    # Crear una máscara con los picos
    dist_peaks = np.zeros_like(dist, dtype=bool)
    dist_peaks[tuple(coords.T)] = True
    
    # Etiquetar marcadores
    markers = measure.label(dist_peaks)
    
    # Crear una imagen combinada que considere intensidad y distancia
    # Las inclusiones son brillantes, por lo que mayor intensidad debe corresponder a mayor probabilidad de ser inclusión
    combined_image = (intensity_weight * original_image + 
                     (1-intensity_weight) * dist_normalized).astype(np.uint8)
    
    # MEJORA 5: Invertir la imagen combinada para watershed
    # Esto mejora la detección de fronteras entre regiones conectadas
    combined_image_inv = 255 - combined_image
    
    # Aplicar watershed
    watershed_result = segmentation.watershed(combined_image_inv, markers, mask=binary_mask)
    
    # Convertir resultado a máscara binaria
    result_mask = (watershed_result > 0).astype(np.uint8) * 255
    
    # MEJORA 6: Análisis final para separar regiones alargadas
    # Si hay pocas regiones (menos de lo esperado), intentar dividir las más alargadas
    labels = measure.label(result_mask)
    props = measure.regionprops(labels)
    
    # Si hay regiones con alta excentricidad (alargadas), podrían ser inclusiones conectadas
    for prop in props:
        if prop.eccentricity > 0.8 and prop.area > 20:  # Regiones alargadas y no muy pequeñas
            # Crear una máscara para esta región
            region_mask = (labels == prop.label).astype(np.uint8) * 255
            
            # Intentar dividir utilizando esqueletización y detección de puntos de unión
            skeleton = morphology.skeletonize(region_mask > 0)
            skeleton_labeled = measure.label(skeleton)
            
            if np.max(skeleton_labeled) > 1:  # Si hay múltiples partes en el esqueleto
                # Dividir la región original usando watershed con nuevos marcadores
                new_markers = measure.label(np.logical_and(skeleton, dist > 0.3 * np.max(dist)))
                if np.max(new_markers) > 1:
                    # Aplicar watershed específicamente a esta región
                    region_watershed = segmentation.watershed(combined_image_inv, new_markers, mask=region_mask)
                    
                    # Actualizar la máscara de resultado
                    result_mask[labels == prop.label] = 0
                    result_mask[region_watershed > 0] = 255
    
    return result_mask


def filter_inclusions(labeled_mask: np.ndarray, intensity_image: np.ndarray, 
                   min_size: int = 5, max_size: int = 1500, 
                   min_circularity: float = 0.4, min_contrast: float = 0.08,
                   texture_analysis: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Filtra y caracteriza inclusiones basadas en tamaño, forma, y contraste.
    
    Args:
        labeled_mask: Máscara etiquetada de inclusiones candidatas
        intensity_image: Imagen original para medir intensidad y contraste
        min_size: Tamaño mínimo en píxeles
        max_size: Tamaño máximo en píxeles
        min_circularity: Circularidad mínima (0-1)
        min_contrast: Contraste mínimo con el entorno
        texture_analysis: Si se debe aplicar análisis de textura
    
    Returns:
        Tupla con (máscara filtrada, lista de propiedades de inclusiones)
    """
    # Obtener propiedades de regiones
    props = measure.regionprops(labeled_mask, intensity_image=intensity_image)
    
    valid_regions = []
    inclusion_props = []
    
    for prop in props:
        # Filtrar por tamaño
        if prop.area < min_size or prop.area > max_size:
            continue
        
        # Calcular circularidad
        perimeter = prop.perimeter
        if perimeter == 0:  # Evitar división por cero
            circularity = 0
        else:
            circularity = 4 * np.pi * prop.area / (perimeter * perimeter)
        
        # Filtrar por circularidad
        if circularity < min_circularity:
            continue
        
        # Obtener coordenadas y crear una máscara dilatada para medir el contraste
        coords = prop.coords
        inclusion_mask = np.zeros_like(intensity_image, dtype=np.uint8)
        inclusion_mask[coords[:, 0], coords[:, 1]] = 1
        
        # Dilatar para obtener el entorno
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(inclusion_mask, kernel, iterations=1)
        
        # Máscara del entorno (excluyendo la inclusión)
        surround_mask = dilated & ~inclusion_mask
        
        # Calcular contraste
        if np.sum(surround_mask) > 0:
            inclusion_mean = prop.mean_intensity
            surround_mean = np.mean(intensity_image[surround_mask == 1])
            contrast = abs(inclusion_mean - surround_mean) / 255.0
        else:
            contrast = 0
        
        # Filtrar por contraste
        if contrast < min_contrast:
            continue
        
        # Análisis de textura (homogeneidad interna)
        texture_valid = True
        if texture_analysis:
            # Calcular desviación estándar dentro de la inclusión como medida simple de homogeneidad
            inclusion_std = np.std(intensity_image[inclusion_mask == 1])
            # Una inclusión real debería ser relativamente homogénea
            if inclusion_std > 0.2 * 255:  # Umbral arbitrario, ajustar según necesidad
                texture_valid = False
        
        if not texture_valid:
            continue
        
        # Si pasa todos los filtros, agregar a las regiones válidas
        valid_regions.append(prop.label)
        
        # Guardar propiedades
        centroid = prop.centroid
        bbox = prop.bbox
        
        # Crear una máscara para esta inclusión y obtener su contorno
        inclusion_mask = np.zeros_like(intensity_image, dtype=np.uint8)
        inclusion_mask[labeled_mask == prop.label] = 255
        contours, _ = cv2.findContours(inclusion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_points = []
        if contours:  # Asegurarse de que se encontraron contornos
            contour_points = contours[0].reshape(-1, 2).tolist()  # Convertir a lista de puntos
        
        inclusion_props.append({
            'label': prop.label,
            'area': float(prop.area),
            'centroid_x': float(centroid[1]),
            'centroid_y': float(centroid[0]),
            'bbox': [int(x) for x in bbox],
            'circularity': float(circularity),
            'contrast': float(contrast),
            'mean_intensity': float(prop.mean_intensity),
            'contour': contour_points  # Añadir el contorno
        })
    
    # Crear máscara filtrada
    filtered_mask = np.zeros_like(labeled_mask)
    for label in valid_regions:
        filtered_mask[labeled_mask == label] = label
    
    return filtered_mask, inclusion_props


def detect_inclusions_in_cell_v2(original_image: np.ndarray, cell_mask: np.ndarray, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detecta inclusiones dentro de una célula utilizando el nuevo algoritmo.
    
    Args:
        original_image: Imagen original (inclusiones brillantes)
        cell_mask: Máscara binaria de la célula
        config: Configuración de la detección
    
    Returns:
        Lista de diccionarios con propiedades de cada inclusión detectada
    """
    # --- PASO 1: Preprocesamiento específico por célula ---
    
    # Configuración
    preprocessing_config = config.get('preprocessing', {})
    cell_normalization = preprocessing_config.get('cell_normalization', True)
    contrast_method = preprocessing_config.get('contrast_enhancement', 'clahe')
    edge_enhancement = preprocessing_config.get('edge_enhancement', True)
    
    # Preprocesar imagen
    processed_image = original_image.copy()
    
    if cell_normalization:
        processed_image = enhance_cell_contrast(processed_image, cell_mask, method=contrast_method)
    
    if edge_enhancement:
        processed_image = apply_edge_enhancement(processed_image, cell_mask)
    
    # --- PASO 2: Detección de inclusiones por umbralización local ---
    
    # Configuración
    thresholding_config = config.get('thresholding', {})
    method = thresholding_config.get('method', 'multi_level')
    sensitivity = thresholding_config.get('sensitivity', 0.8)
    adaptive_block_size = thresholding_config.get('adaptive_block_size', 15)
    
    # Aplicar umbralización
    if method == 'multi_level':
        # Umbralización multinivel
        binary_mask = multilevel_threshold(processed_image, cell_mask, sensitivity=sensitivity)
    elif method == 'adaptive':
        # Umbralización adaptativa
        binary_mask = adaptive_local_threshold(
            processed_image, cell_mask, 
            block_size=adaptive_block_size, 
            sensitivity=sensitivity
        )
    else:  # 'otsu'
        # Umbralización Otsu solo dentro de la célula
        masked_cell = np.zeros_like(processed_image)
        masked_cell[cell_mask > 0] = processed_image[cell_mask > 0]
        
        # Aplicar Otsu
        _, binary_mask = cv2.threshold(
            masked_cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    
    # Aplicar operaciones morfológicas para limpiar el resultado
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # --- PASO 3: Separación de inclusiones cercanas ---
    
    # Configuración
    separation_config = config.get('separation', {})
    separation_method = separation_config.get('method', 'watershed')
    min_distance = separation_config.get('min_distance', 5)
    intensity_weight = separation_config.get('intensity_weight', 0.7)
    
    # Verificar si hay suficientes píxeles para procesar
    if np.sum(binary_mask) > 0:
        if separation_method == 'watershed':
            # Aplicar watershed para separar inclusiones cercanas
            binary_mask = separate_inclusions_watershed(
                binary_mask, processed_image,
                min_distance=min_distance,
                intensity_weight=intensity_weight
            )
    
    # --- PASO 4: Filtrado y validación ---
    
    # Configuración
    filtering_config = config.get('filtering', {})
    min_size = filtering_config.get('min_size', 5)
    max_size = filtering_config.get('max_size', 1500)
    min_circularity = filtering_config.get('min_circularity', 0.4)
    min_contrast = filtering_config.get('min_contrast', 0.08)
    texture_analysis = filtering_config.get('texture_analysis', True)
    
    # Etiquetar las regiones conectadas
    labeled_mask = measure.label(binary_mask)
    
    # Filtrar y caracterizar las inclusiones
    filtered_mask, inclusion_props = filter_inclusions(
        labeled_mask, processed_image,
        min_size=min_size,
        max_size=max_size,
        min_circularity=min_circularity,
        min_contrast=min_contrast,
        texture_analysis=texture_analysis
    )
    
    # --- PASO 5: Visualización de debug (opcional) ---
    
    debug_config = config.get('debug', {})
    save_intermediate_steps = debug_config.get('save_intermediate_steps', False)
    specific_cell_ids = debug_config.get('specific_cell_ids', [])
    
    if save_intermediate_steps:
        # Esta es una función opcional que podría implementarse para guardar
        # imágenes intermedias para depuración
        pass
    
    return inclusion_props


def detect_all_inclusions_v2(image: np.ndarray, segmented_image: np.ndarray, 
                          config: Optional[Dict[str, Any]] = None) -> Dict[int, List[Dict[str, Any]]]:
    """
    Detecta inclusiones en todas las células segmentadas.
    
    Args:
        image: Imagen original (inclusiones brillantes)
        segmented_image: Imagen segmentada donde cada célula tiene una etiqueta única
        config: Configuración para la detección
    
    Returns:
        Diccionario que mapea IDs de células a listas de inclusiones detectadas
    """
    from config import DETECTION_V2_CONFIG
    
    if config is None:
        config = DETECTION_V2_CONFIG
    
    # Crear máscaras individuales para cada célula
    cell_masks = create_cell_masks(segmented_image)
    
    # Detectar inclusiones en cada célula
    all_inclusions = {}
    
    for cell_id, mask in cell_masks.items():
        inclusions = detect_inclusions_in_cell_v2(image, mask, config)
        all_inclusions[cell_id] = inclusions
    
    return all_inclusions


def visualize_inclusions_v2(original_image: np.ndarray, segmented_image: np.ndarray,
                        all_inclusions: Dict[int, List[Dict[str, Any]]],
                        show_visualization: bool = True) -> np.ndarray:
    """
    Visualiza las inclusiones detectadas sobre la imagen original.
    
    Args:
        original_image: Imagen original
        segmented_image: Imagen segmentada
        all_inclusions: Diccionario de inclusiones por célula
        show_visualization: Si es True, muestra la visualización
    
    Returns:
        Imagen con células segmentadas e inclusiones marcadas
    """
    from skimage.color import label2rgb
    import matplotlib.pyplot as plt
    
    # Convertir a BGR si está en escala de grises
    if len(original_image.shape) == 2:
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        original_image_color = original_image.copy()
    
    # Colorear células segmentadas
    segmentation_overlay = label2rgb(segmented_image, image=original_image, bg_label=0, alpha=0.3)
    segmentation_overlay = (segmentation_overlay * 255).astype(np.uint8)
    
    # Convertir a BGR para visualización con OpenCV
    if segmentation_overlay.shape[2] == 3 and segmentation_overlay.dtype == np.uint8:
        segmentation_overlay_bgr = cv2.cvtColor(segmentation_overlay, cv2.COLOR_RGB2BGR)
    else:
        segmentation_overlay_bgr = segmentation_overlay
    
    # Superponer visualización de segmentación
    result = segmentation_overlay_bgr.copy()
    
    # Marcar inclusiones con sus contornos reales (o con círculos si no hay contorno disponible)
    for cell_id, inclusions in all_inclusions.items():
        for inc in inclusions:
            # Color basado en el tamaño relativo a la célula
            # Verde para inclusiones pequeñas, amarillo para medianas, rojo para grandes
            cell_mask = (segmented_image == cell_id)
            cell_area = np.sum(cell_mask)
            ratio = inc['area'] / cell_area if cell_area > 0 else 0
            
            if ratio < 0.1:
                color = (0, 255, 0)  # Verde: inclusión pequeña
            elif ratio < 0.3:
                color = (0, 255, 255)  # Amarillo: inclusión mediana
            else:
                color = (0, 0, 255)  # Rojo: inclusión grande
            
            # Dibujar contorno real si está disponible
            if 'contour' in inc and inc['contour']:
                # Convertir la lista de puntos a formato numpy para OpenCV
                contour = np.array(inc['contour']).reshape((-1, 1, 2)).astype(np.int32)
                cv2.drawContours(result, [contour], 0, color, 1)
            else:
                # Caer en método anterior (círculo) si no hay contorno disponible
                cx, cy = int(inc['centroid_x']), int(inc['centroid_y'])
                radius = int(np.sqrt(inc['area'] / np.pi))
                cv2.circle(result, (cx, cy), max(1, radius), color, 1)
    
    # Mostrar la visualización si está habilitado
    if show_visualization:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Detección de Inclusiones V2")
        plt.axis('off')
        plt.show()
    
    return result


def summarize_inclusions_v2(all_inclusions: Dict[int, List[Dict[str, Any]]],
                        segmented_image: np.ndarray) -> Dict[str, Any]:
    """
    Genera un resumen estadístico de las inclusiones detectadas.
    
    Args:
        all_inclusions: Diccionario de inclusiones por célula
        segmented_image: Imagen segmentada para calcular áreas celulares
    
    Returns:
        Diccionario con estadísticas resumidas
    """
    # Inicializar estadísticas
    total_cells = len(all_inclusions)
    cells_with_inclusions = sum(1 for cell_id, incs in all_inclusions.items() if len(incs) > 0)
    total_inclusions = sum(len(incs) for incs in all_inclusions.values())
    
    # Células sin inclusiones
    cells_without_inclusions = total_cells - cells_with_inclusions
    
    # Porcentaje de células con inclusiones
    percent_cells_with_inclusions = (cells_with_inclusions / total_cells * 100) if total_cells > 0 else 0
    
    # Número de inclusiones por célula
    inclusions_per_cell = [len(incs) for incs in all_inclusions.values()]
    avg_inclusions_per_cell = np.mean(inclusions_per_cell) if inclusions_per_cell else 0
    std_inclusions_per_cell = np.std(inclusions_per_cell) if inclusions_per_cell else 0
    
    # Tamaño de inclusiones
    all_inclusion_areas = [inc['area'] for incs in all_inclusions.values() for inc in incs]
    avg_inclusion_area = np.mean(all_inclusion_areas) if all_inclusion_areas else 0
    std_inclusion_area = np.std(all_inclusion_areas) if all_inclusion_areas else 0
    
    # Ratio de área de inclusiones respecto a células
    inclusion_ratios = []
    for cell_id, incs in all_inclusions.items():
        # Calcular área de la célula
        cell_mask = (segmented_image == cell_id)
        cell_area = np.sum(cell_mask)
        
        if cell_area > 0 and incs:
            # Suma de áreas de todas las inclusiones en esta célula
            total_inclusion_area = sum(inc['area'] for inc in incs)
            ratio = total_inclusion_area / cell_area
            inclusion_ratios.append(ratio)
    
    avg_inclusion_ratio = np.mean(inclusion_ratios) if inclusion_ratios else 0
    std_inclusion_ratio = np.std(inclusion_ratios) if inclusion_ratios else 0
    
    return {
        'total_cells': total_cells,
        'cells_with_inclusions': cells_with_inclusions,
        'cells_without_inclusions': cells_without_inclusions,
        'percent_cells_with_inclusions': percent_cells_with_inclusions,
        'total_inclusions': total_inclusions,
        'avg_inclusions_per_cell': avg_inclusions_per_cell,
        'std_inclusions_per_cell': std_inclusions_per_cell,
        'avg_inclusion_area': avg_inclusion_area,
        'std_inclusion_area': std_inclusion_area,
        'avg_inclusion_ratio': avg_inclusion_ratio,
        'std_inclusion_ratio': std_inclusion_ratio
    }


def plot_inclusion_statistics_v2(summary: Dict[str, Any],
                              all_inclusions: Dict[int, List[Dict[str, Any]]]) -> None:
    """
    Genera visualizaciones estadísticas de las inclusiones detectadas.
    
    Args:
        summary: Resumen estadístico generado por summarize_inclusions_v2
        all_inclusions: Diccionario de inclusiones por célula
    """
    import matplotlib.pyplot as plt
    
    # Crear figura con 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distribución del número de inclusiones por célula
    inclusions_per_cell = [len(incs) for incs in all_inclusions.values()]
    axes[0, 0].hist(inclusions_per_cell, bins=max(10, max(inclusions_per_cell) if inclusions_per_cell else 1), 
                   color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribución de Inclusiones por Célula')
    axes[0, 0].set_xlabel('Número de Inclusiones')
    axes[0, 0].set_ylabel('Frecuencia (Células)')
    
    # 2. Distribución de tamaños de inclusiones
    all_inclusion_areas = [inc['area'] for incs in all_inclusions.values() for inc in incs]
    if all_inclusion_areas:
        axes[0, 1].hist(all_inclusion_areas, bins=20, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribución de Tamaños de Inclusiones')
        axes[0, 1].set_xlabel('Área (píxeles)')
        axes[0, 1].set_ylabel('Frecuencia')
    
    # 3. Pie chart: Porcentaje de células con/sin inclusiones
    with_without = [summary['cells_with_inclusions'], summary['cells_without_inclusions']]
    axes[1, 0].pie(with_without, labels=['Con inclusiones', 'Sin inclusiones'], 
                  autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue'])
    axes[1, 0].set_title('Células con/sin Inclusiones')
    
    # 4. Distribución de ratio inclusión/célula
    inclusion_ratios = []
    for cell_id, incs in all_inclusions.items():
        if incs:  # Solo células con inclusiones
            total_inclusion_area = sum(inc['area'] for inc in incs)
            # Asumimos que la información de área de célula está en los datos
            # Si no hay datos directos, tendríamos que calcularla en este punto
            cell_area = 1000  # Valor arbitrario si no tenemos el dato real
            inclusion_ratios.append(total_inclusion_area / cell_area * 100)  # En porcentaje
    
    if inclusion_ratios:
        axes[1, 1].hist(inclusion_ratios, bins=20, color='plum', edgecolor='black')
        axes[1, 1].set_title('Ratio Área Inclusión/Célula')
        axes[1, 1].set_xlabel('Porcentaje de Célula (%)')
        axes[1, 1].set_ylabel('Frecuencia (Células)')
    
    # Ajustar diseño y mostrar
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Código de prueba
    import matplotlib.pyplot as plt
    
    # Cargar una imagen de ejemplo
    image_path = "data/raw/sample_image.png"
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"No se pudo cargar la imagen: {image_path}")
        else:
            print(f"Imagen cargada correctamente: {image_path}")
            # Aquí se podría agregar código para probar las funciones
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")