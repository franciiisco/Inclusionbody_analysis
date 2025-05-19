"""
Módulo para la detección de inclusiones de polifosfatos dentro de células bacterianas.

Este módulo proporciona funciones para identificar y caracterizar las inclusiones
de polifosfatos dentro de células previamente segmentadas.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
from skimage import measure, filters, feature


def create_cell_masks(
    segmented_image: np.ndarray
) -> Dict[int, np.ndarray]:
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
        mask = (segmented_image == label).astype(np.uint8) * 255
        cell_masks[int(label)] = mask
    
    return cell_masks


def detect_inclusions_in_cell(
    original_image_inclusions_bright: np.ndarray, # Renombrado para claridad
    cell_mask: np.ndarray,
    min_size: int = 3,
    max_size: int = 30,
    threshold_offset: float = 0.2,  # Ahora positivo: inclusiones más brillantes que la media
    min_contrast: float = 0.1, 
    contrast_window: int = 3,
    remove_border: bool = True,
    min_circularity: float = 0.6  # Nuevo parámetro para filtrar por forma
) -> List[Dict[str, Any]]:
    """
    Detecta inclusiones (esperadas como brillantes) dentro de una célula específica.
    
    Args:
        original_image_inclusions_bright: Imagen donde las inclusiones son brillantes y el fondo/citoplasma oscuro.
        cell_mask: Máscara binaria de la célula
        min_size: Tamaño mínimo de inclusión en píxeles
        max_size: Tamaño máximo de inclusión en píxeles
        threshold_offset: Ajuste para el umbral de detección (positivo para detectar
                         áreas más brillantes que la célula)
        min_contrast: Contraste mínimo entre inclusión y su entorno
        contrast_window: Tamaño de ventana para evaluación de contraste
        remove_border: Si es True, elimina detecciones en el borde de la célula
        min_circularity: Circularidad mínima para que una detección sea considerada válida
    
    Returns:
        Lista de diccionarios con propiedades de cada inclusión detectada
    """
    # Aplicar la máscara a la imagen (inclusiones brillantes)
    masked_cell = cv2.bitwise_and(original_image_inclusions_bright, original_image_inclusions_bright, mask=cell_mask.astype(np.uint8))
    
    # Calcular estadísticas de intensidad dentro de la célula
    cell_pixels = original_image_inclusions_bright[cell_mask > 0]
    if len(cell_pixels) == 0:
        return []
    
    mean_intensity = np.mean(cell_pixels) # Media del citoplasma (ahora más oscuro)
    std_intensity = np.std(cell_pixels)
    
    # Crear una versión dilatada de la célula para detectar bordes
    if remove_border:
        kernel_border = np.ones((3, 3), np.uint8)
        inner_mask = cv2.erode(cell_mask, kernel_border, iterations=1)
        border_mask = cell_mask - inner_mask
    else:
        border_mask = np.zeros_like(cell_mask)
    
    # Umbral global: inclusiones son más brillantes que la media + offset
    threshold_global = mean_intensity + threshold_offset * std_intensity 
    
    # Crear una máscara inicial de candidatos a inclusiones (brillantes)
    candidate_mask = np.zeros_like(cell_mask)
    # CAMBIO: Condición para píxeles más brillantes
    candidate_mask[(masked_cell > threshold_global) & (cell_mask > 0)] = 255 
    
    # Aplicar umbral adaptativo para refinar la detección
    # Solo donde la máscara de célula es positiva
    cell_region = original_image_inclusions_bright.copy()
    cell_region[cell_mask == 0] = 0 # Fondo de la región celular es negro
    
    # Si hay suficientes píxeles, aplicar umbral adaptativo
    if np.sum(cell_mask > 0) > 100:  # Verificar que la célula tenga tamaño suficiente
        try:
            # Usar umbral adaptativo con un tamaño de bloque proporcional al tamaño celular
            cell_size = np.sqrt(np.sum(cell_mask > 0))
            block_size = max(int(cell_size / 4) * 2 + 1, 11)  # Asegurar que sea impar
            block_size = min(block_size, 51)  # Limitar el tamaño máximo
            
            adaptive_thresh = cv2.adaptiveThreshold(
                cell_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, 2 # CAMBIO: THRESH_BINARY para objetos brillantes
            )
            
            # Combinar umbral global y adaptativo
            inclusion_mask = cv2.bitwise_and(candidate_mask, adaptive_thresh)
        except Exception:
            # Si falla el adaptivo, usar solo el umbral global
            inclusion_mask = candidate_mask
    else:
        inclusion_mask = candidate_mask
    
    # Eliminar ruido pequeño mediante apertura morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)) 
    inclusion_mask = cv2.morphologyEx(inclusion_mask, cv2.MORPH_OPEN, kernel)
    
    # Remover detecciones en el borde de la célula si está habilitado
    if remove_border:
        inclusion_mask[border_mask > 0] = 0
    
    # Etiquetar componentes conectados en la máscara de inclusiones
    # Usar original_image_inclusions_bright para regionprops
    labeled_inclusions = measure.label(inclusion_mask)
    inclusion_props = measure.regionprops(labeled_inclusions, intensity_image=original_image_inclusions_bright)
    
    inclusions = []
    
    # Filtrar y caracterizar inclusiones
    for prop in inclusion_props:
        # Filtrar por tamaño
        if not (min_size <= prop.area <= max_size):
            continue

        # Calcular y filtrar por circularidad
        perimeter = prop.perimeter
        if perimeter == 0: # Evitar división por cero
            circularity = 0
        else:
            circularity = 4 * np.pi * prop.area / (perimeter * perimeter)
        
        if circularity < min_circularity:
            continue
            
        # Calcular centroide
        y, x = prop.centroid
        
        # Crear una máscara para esta inclusión específica
        inclusion_specific_mask = (labeled_inclusions == prop.label).astype(np.uint8)
        
        # Calcular contraste local: diferencia entre la inclusión y su entorno inmediato
        kernel_env = np.ones((contrast_window, contrast_window), np.uint8)
        environment_mask = cv2.dilate(inclusion_specific_mask, kernel_env, iterations=1)
        environment_mask = environment_mask - inclusion_specific_mask
        environment_mask = environment_mask & cell_mask # Asegurar que el entorno esté dentro de la célula
        
        inclusion_intensity = prop.mean_intensity # Intensidad de la inclusión (brillante)
        
        if np.sum(environment_mask) > 0:
            # Intensidad del entorno (citoplasma, ahora más oscuro)
            environment_intensity = np.mean(original_image_inclusions_bright[environment_mask > 0])
            # Contraste: inclusión brillante vs entorno oscuro
            if inclusion_intensity > environment_intensity and inclusion_intensity > 0:
                 local_contrast = (inclusion_intensity - environment_intensity) / inclusion_intensity
            elif environment_intensity > 0: # Fallback si la inclusión no es más brillante o entorno es 0
                 local_contrast = abs(inclusion_intensity - environment_intensity) / environment_intensity
            else: # Fallback si ambas intensidades son muy bajas o cero
                 local_contrast = abs(inclusion_intensity - environment_intensity) / 255.0 if 255.0 > 0 else 0

        else:
            local_contrast = 0 # No hay entorno para comparar
            environment_intensity = None
        
        # Filtrar por contraste mínimo
        if local_contrast >= min_contrast:
            # Extraer propiedades relevantes
            inclusion_details = { # Renombrado para evitar conflicto con la lista 'inclusions'
                'centroid': (x, y),
                'area': prop.area,
                'perimeter': perimeter, # Usar la variable perimeter calculada
                'mean_intensity': prop.mean_intensity, # Esta es la intensidad en la imagen invertida
                'min_intensity': prop.min_intensity,   # Esta es la intensidad en la imagen invertida
                'contrast': local_contrast,
                'environment_intensity': environment_intensity,
                'circularity': circularity, # Usar la variable circularity calculada
                'bbox': prop.bbox  # (min_row, min_col, max_row, max_col)
            }
            
            inclusions.append(inclusion_details) # Añadir el diccionario a la lista
    
    return inclusions


def detect_all_inclusions(
    image_for_inclusion_detection: np.ndarray, # Nombre genérico
    segmented_image: np.ndarray,
    detection_params: Optional[Dict[str, Any]] = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Detecta inclusiones en todas las células segmentadas.
    
    Args:
        image_for_inclusion_detection: Imagen a usar para la detección.
                                       Si se esperan inclusiones brillantes, esta imagen debe estar invertida.
        segmented_image: Imagen segmentada donde cada célula tiene una etiqueta única
        detection_params: Parámetros para la detección de inclusiones
    
    Returns:
        Diccionario que mapea IDs de células a listas de inclusiones detectadas
    """
    if detection_params is None:
        # Valores por defecto actualizados para inclusiones brillantes y filtro de circularidad
        detection_params = {
            'min_size': 3,
            'max_size': 50, 
            'threshold_offset': 0.2,  # Positivo para inclusiones brillantes
            'min_contrast': 0.1,
            'contrast_window': 3,
            'remove_border': True,
            'min_circularity': 0.6  # Nuevo parámetro por defecto
        }
    
    # Crear máscaras individuales para cada célula
    cell_masks = create_cell_masks(segmented_image)
    
    # Detectar inclusiones en cada célula
    all_inclusions_map = {} 
    
    for cell_id, mask in cell_masks.items():
        inclusions_list = detect_inclusions_in_cell(
            original_image_inclusions_bright=image_for_inclusion_detection, 
            cell_mask=mask,
            min_size=detection_params.get('min_size', 3),
            max_size=detection_params.get('max_size', 50),
            threshold_offset=detection_params.get('threshold_offset', 0.2),
            min_contrast=detection_params.get('min_contrast', 0.1),
            contrast_window=detection_params.get('contrast_window', 3),
            remove_border=detection_params.get('remove_border', True),
            min_circularity=detection_params.get('min_circularity', 0.6)
        )
        
        all_inclusions_map[cell_id] = inclusions_list
    
    return all_inclusions_map


def visualize_inclusions(
    original_image: np.ndarray,
    segmented_image: np.ndarray,
    all_inclusions: Dict[int, List[Dict[str, Any]]],
    show_visualization: bool = True
) -> np.ndarray:
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
    # Crear imagen colorizada para visualización
    from skimage.color import label2rgb
    
    # Convertir a BGR si está en escala de grises
    if len(original_image.shape) == 2:
        display_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        display_img = original_image.copy()
    
    # Colorear células segmentadas
    segmentation_overlay = label2rgb(segmented_image, image=original_image, bg_label=0, alpha=0.3)
    segmentation_overlay = (segmentation_overlay * 255).astype(np.uint8)
    
    # Convertir a BGR si es necesario
    if segmentation_overlay.shape[2] == 3 and segmentation_overlay.dtype == np.uint8:
        segmentation_overlay_bgr = cv2.cvtColor(segmentation_overlay, cv2.COLOR_RGB2BGR)
    else:
        segmentation_overlay_bgr = segmentation_overlay
    
    # Superponer visualización de segmentación
    result = segmentation_overlay_bgr.copy()
    
    # Marcar inclusiones
    for cell_id, inclusions in all_inclusions.items():
        for inclusion in inclusions:
            # Obtener coordenadas del centroide
            x, y = int(inclusion['centroid'][0]), int(inclusion['centroid'][1])
            
            # Dibujar círculo en la posición de la inclusión
            radius = int(np.sqrt(inclusion['area'] / np.pi))
            radius = max(radius, 2)  # Asegurar un radio mínimo visible
            
            # Color basado en la intensidad (más oscuro = más polifosfatos)
            intensity_normalized = (inclusion['mean_intensity'] / 255)
            color = (0, 0, 255)  # Rojo en BGR
            
            cv2.circle(result, (x, y), radius, color, 1)
    
    # Mostrar la visualización si está habilitado
    if show_visualization:
        # Convertir de BGR (OpenCV) a RGB (matplotlib)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(result_rgb)
        plt.title(f"Células segmentadas con inclusiones de polifosfatos ({sum(len(incs) for incs in all_inclusions.values())} inclusiones)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return result


def summarize_inclusions(
    all_inclusions: Dict[int, List[Dict[str, Any]]],
    segmented_image: np.ndarray
) -> Dict[str, Any]:
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
    
    # Inclusiones por célula
    inclusions_per_cell = [len(incs) for incs in all_inclusions.values()]
    avg_inclusions_per_cell = np.mean(inclusions_per_cell) if inclusions_per_cell else 0
    std_inclusions_per_cell = np.std(inclusions_per_cell) if inclusions_per_cell else 0
    
    # Tamaños de inclusiones
    all_areas = [inc['area'] for cell_incs in all_inclusions.values() for inc in cell_incs]
    avg_inclusion_area = np.mean(all_areas) if all_areas else 0
    std_inclusion_area = np.std(all_areas) if all_areas else 0
    
    # Áreas de células
    cell_props = measure.regionprops(segmented_image)
    cell_areas = [prop.area for prop in cell_props]
    avg_cell_area = np.mean(cell_areas) if cell_areas else 0
    
    # Ratio de áreas (inclusiones/célula)
    inclusion_to_cell_ratios = []
    for cell_id, inclusions in all_inclusions.items():
        if not inclusions:
            continue
        
        # Buscar el área de la célula
        cell_area = next((prop.area for prop in cell_props if prop.label == cell_id), None)
        if cell_area:
            total_inclusion_area = sum(inc['area'] for inc in inclusions)
            ratio = total_inclusion_area / cell_area
            inclusion_to_cell_ratios.append(ratio)
    
    avg_inclusion_ratio = np.mean(inclusion_to_cell_ratios) if inclusion_to_cell_ratios else 0
    std_inclusion_ratio = np.std(inclusion_to_cell_ratios) if inclusion_to_cell_ratios else 0
    
    # Crear resumen
    summary = {
        'total_cells': total_cells,
        'cells_with_inclusions': cells_with_inclusions,
        'cells_without_inclusions': cells_without_inclusions,
        'percent_cells_with_inclusions': percent_cells_with_inclusions,
        'total_inclusions': total_inclusions,
        'avg_inclusions_per_cell': avg_inclusions_per_cell,
        'std_inclusions_per_cell': std_inclusions_per_cell,
        'avg_inclusion_area': avg_inclusion_area,
        'std_inclusion_area': std_inclusion_area,
        'avg_cell_area': avg_cell_area,
        'avg_inclusion_ratio': avg_inclusion_ratio,
        'std_inclusion_ratio': std_inclusion_ratio
    }
    
    return summary


def plot_inclusion_statistics(
    summary: Dict[str, Any],
    all_inclusions: Dict[int, List[Dict[str, Any]]]
) -> None:
    """
    Genera gráficos estadísticos sobre las inclusiones detectadas.
    
    Args:
        summary: Resumen estadístico de inclusiones
        all_inclusions: Diccionario de inclusiones por célula
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Distribución de inclusiones por célula
    plt.subplot(2, 2, 1)
    inclusions_per_cell = [len(incs) for incs in all_inclusions.values()]
    
    # Verificar si hay datos significativos para mostrar
    if max(inclusions_per_cell) == 0:
        plt.text(0.5, 0.5, 'Todas las células tienen 0 inclusiones', 
                ha='center', va='center', transform=plt.gca().transAxes)
    else:
        # Determinar el número de bins apropiado
        num_unique_values = len(np.unique(inclusions_per_cell))
        num_bins = min(10, max(num_unique_values, 2))
        
        # Crear el histograma con conteo explícito
        values, counts = np.unique(inclusions_per_cell, return_counts=True)
        plt.bar(values, counts, alpha=0.7, width=0.8)
        
        plt.axvline(summary['avg_inclusions_per_cell'], color='r', linestyle='dashed', 
                   linewidth=1, label=f'Media: {summary["avg_inclusions_per_cell"]:.2f}')
        plt.xlabel('Número de inclusiones')
        plt.ylabel('Número de células')
        plt.title('Distribución de inclusiones por célula')
        plt.legend()
        
        # Ajustar el eje X para mostrar solo valores enteros
        plt.xticks(np.arange(0, max(inclusions_per_cell) + 1, 1))

    
    # 2. Tamaño de las inclusiones
    plt.subplot(2, 2, 2)
    all_areas = [inc['area'] for cell_incs in all_inclusions.values() for inc in cell_incs]
    if all_areas:
        plt.hist(all_areas, bins=20, alpha=0.7)
        plt.axvline(summary['avg_inclusion_area'], color='r', linestyle='dashed', 
                   linewidth=1, label=f'Media: {summary["avg_inclusion_area"]:.2f}')
        plt.xlabel('Área de inclusión (píxeles)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de tamaños de inclusiones')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No hay datos de inclusiones', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    # 3. Gráfico circular de células con/sin inclusiones
    plt.subplot(2, 2, 3)
    plt.pie([summary['cells_with_inclusions'], summary['cells_without_inclusions']], 
           labels=['Con inclusiones', 'Sin inclusiones'],
           autopct='%1.1f%%', startangle=90, explode=[0.1, 0])
    plt.axis('equal')
    plt.title('Proporción de células con inclusiones')
    
    # 4. Relación área de inclusiones / área de célula
    plt.subplot(2, 2, 4)
    
    # Calcular la relación para cada célula
    ratios = []
    for cell_id, inclusions in all_inclusions.items():
        if not inclusions:
            continue
        total_inc_area = sum(inc['area'] for inc in inclusions)
        ratios.append((len(inclusions), total_inc_area))
    
    if ratios:
        x = [r[0] for r in ratios]  # Número de inclusiones
        y = [r[1] for r in ratios]  # Área total de inclusiones
        
        plt.scatter(x, y, alpha=0.7)
        plt.xlabel('Número de inclusiones')
        plt.ylabel('Área total de inclusiones (píxeles)')
        plt.title('Relación entre cantidad y área de inclusiones')
        
        # Línea de tendencia
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.7)
    else:
        plt.text(0.5, 0.5, 'No hay datos suficientes', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    import os
    
    # Añadir el directorio raíz del proyecto al path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    from src.preprocessing import preprocess_pipeline
    from src.segmentation import segment_cells
    
    # Cargar imagen
    image_path = "../data/raw/sample_image.jpg"
    try:
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if original_image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
        
        # Preprocesar imagen - guardamos original y preprocesada
        original_for_detection = original_image.copy()  # Guardar original para detección
        
        preprocess_config = {
            'normalize': {'method': 'clahe', 'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
            'denoise': {'method': 'gaussian', 'params': {'ksize': (5, 5), 'sigma': 0}},
            'correct_illumination': {'method': 'subtract_background', 'params': {'kernel_size': 51}},
            'invert': True  # Invertir para segmentación
        }
        
        preprocessed = preprocess_pipeline(original_image, preprocess_config)
        
        # Segmentar células
        segment_config = {
            'threshold': {'method': 'otsu', 'params': {}},
            'morphological': [
                ('open', {'kernel_size': 3, 'iterations': 2}),
                ('close', {'kernel_size': 5, 'iterations': 1})
            ],
            'watershed': {'min_distance': 10},
            'filter': {
                'min_area': 30,
                'max_area': 1000,
                'min_circularity': 0.3,
                'max_aspect_ratio': 5.0
            }
        }
        
        segmented = segment_cells(preprocessed, segment_config)
        
        # Para detección de inclusiones, usamos la imagen original sin inversión
        # Las inclusiones son más oscuras que las células
        detection_params = {
            'min_size': 3,
            'max_size': 30,
            'threshold_offset': -0.2,  # Negativo para detectar áreas más oscuras
            'min_contrast': 0.05,
            'contrast_window': 3,
            'remove_border': True
        }
        
        # Detectar inclusiones
        all_inclusions = detect_all_inclusions(
            original_for_detection,  # Usamos la original no invertida
            segmented,
            detection_params
        )
        
        # Visualizar resultados
        result_image = visualize_inclusions(original_image, segmented, all_inclusions)
        
        # Generar y mostrar estadísticas
        summary = summarize_inclusions(all_inclusions, segmented)
        
        print("\nResumen de detección de inclusiones:")
        print(f"Total de células: {summary['total_cells']}")
        print(f"Células con inclusiones: {summary['cells_with_inclusions']} ({summary['percent_cells_with_inclusions']:.1f}%)")
        print(f"Total de inclusiones: {summary['total_inclusions']}")
        print(f"Promedio de inclusiones por célula: {summary['avg_inclusions_per_cell']:.2f} ± {summary['std_inclusions_per_cell']:.2f}")
        print(f"Tamaño promedio de inclusiones: {summary['avg_inclusion_area']:.2f} ± {summary['std_inclusion_area']:.2f} píxeles")
        print(f"Ratio promedio inclusiones/célula: {summary['avg_inclusion_ratio']*100:.2f}% ± {summary['std_inclusion_ratio']*100:.2f}%")
        
        # Mostrar gráficos estadísticos
        plot_inclusion_statistics(summary, all_inclusions)
        
        # Guardar imagen resultado
        cv2.imwrite("../data/processed/inclusions_detected.png", result_image)
        
    except Exception as e:
        print(f"Error durante la detección de inclusiones: {e}")
        import traceback
        traceback.print_exc()