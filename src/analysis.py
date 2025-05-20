"""
Módulo para el análisis estadístico y la visualización de los resultados
de la detección de inclusiones de polifosfatos.
"""
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


def summarize_inclusions(all_inclusions: Dict[int, List[Dict[str, Any]]],
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


def plot_inclusion_statistics(summary: Dict[str, Any],
                             all_inclusions: Dict[int, List[Dict[str, Any]]]) -> None:
    """
    Genera visualizaciones estadísticas de las inclusiones detectadas.
    
    Args:
        summary: Resumen estadístico generado por summarize_inclusions
        all_inclusions: Diccionario de inclusiones por célula
    """
    from visualization import plot_inclusion_statistics as plot_stats
    plot_stats(summary, all_inclusions)
