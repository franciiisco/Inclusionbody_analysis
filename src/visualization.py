"""
Módulo para la visualización de resultados de detección y análisis
de inclusiones de polifosfatos en células bacterianas.
"""
from typing import Dict, Any, List
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import label2rgb


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


def visualize_inclusions_v2(
    original_image: np.ndarray, 
    segmented_image: np.ndarray,
    all_inclusions: Dict[int, List[Dict[str, Any]]],
    show_visualization: bool = True
) -> np.ndarray:
    """
    Visualiza las inclusiones detectadas sobre la imagen original con la versión 2.0.
    
    Args:
        original_image: Imagen original
        segmented_image: Imagen segmentada
        all_inclusions: Diccionario de inclusiones por célula
        show_visualization: Si es True, muestra la visualización
    
    Returns:
        Imagen con células segmentadas e inclusiones marcadas
    """
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


def plot_inclusion_statistics(
    summary: Dict[str, Any],
    all_inclusions: Dict[int, List[Dict[str, Any]]]
) -> None:
    """
    Genera visualizaciones estadísticas de las inclusiones detectadas.
    
    Args:
        summary: Resumen estadístico generado por summarize_inclusions
        all_inclusions: Diccionario de inclusiones por célula
    """
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