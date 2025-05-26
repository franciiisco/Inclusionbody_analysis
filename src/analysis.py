"""
Módulo para el análisis estadístico y la visualización de los resultados
de la detección de inclusiones de polifosfatos.
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from datetime import datetime
from collections import defaultdict


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
    for cell_id, incs in all_inclusions.items():        # Calcular área de la célula
        cell_mask = (segmented_image == cell_id)
        cell_area = np.sum(cell_mask)
        
        if cell_area > 0 and incs:
            # Suma de áreas de todas las inclusiones en esta célula
            total_inclusion_area = sum(inc['area'] for inc in incs)
            ratio = total_inclusion_area / cell_area
            inclusion_ratios.append(ratio)
    
    avg_inclusion_ratio = np.mean(inclusion_ratios) if inclusion_ratios else 0
    std_inclusion_ratio = np.std(inclusion_ratios) if inclusion_ratios else 0
    
    # Calcular áreas totales
    total_cell_area = 0
    total_inclusion_area = 0
    
    for cell_id, incs in all_inclusions.items():
        # Calcular área de la célula
        cell_mask = (segmented_image == cell_id)
        cell_area = np.sum(cell_mask)
        total_cell_area += cell_area
        
        # Calcular área total de inclusiones para esta célula
        cell_inclusion_area = sum(inc['area'] for inc in incs)
        total_inclusion_area += cell_inclusion_area
    
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
        'std_inclusion_ratio': std_inclusion_ratio,
        'total_cell_area': total_cell_area,
        'total_inclusion_area': total_inclusion_area
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


def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extrae los metadatos de un nombre de archivo con formato:
    CONDICION_BOTE_REPLICA_TIEMPO_NºIMAGEN.TIFF o .PNG
    
    Ejemplo: MEI_B1_R3_t4_026_BF1.png
    
    Args:
        filename: Nombre del archivo a analizar
        
    Returns:
        Diccionario con los metadatos extraídos o None si el formato no es válido
    """
    # Eliminar extensión y cualquier ruta
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Patrón para extraer los componentes
    pattern = r"^([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)(?:_.*)?$"
    match = re.match(pattern, name_without_ext)
    
    if match:
        return {
            'condicion': match.group(1),
            'bote': match.group(2),
            'replica': match.group(3),
            'tiempo': match.group(4),
            'numero_imagen': match.group(5)
        }
    else:
        return None


def validate_filename_format(filename: str) -> bool:
    """
    Valida si un nombre de archivo cumple con el formato requerido:
    CONDICION_BOTE_REPLICA_TIEMPO_NºIMAGEN.TIFF o .PNG
    
    Args:
        filename: Nombre del archivo a validar
        
    Returns:
        True si el formato es válido, False en caso contrario
    """
    metadata = extract_metadata_from_filename(filename)
    return metadata is not None


def aggregate_inclusion_data(image_results: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Agrega los resultados de análisis por CONDICION/TIEMPO/REPLICA y CONDICION/TIEMPO.
    
    Args:
        image_results: Lista de tuplas (nombre_archivo, resultados_análisis)
                      donde resultados_análisis es el diccionario devuelto por summarize_inclusions
        
    Returns:
        Diccionario con resultados agregados por jerarquía
    """
    # Inicializar estructuras para agrupar datos
    condition_time_replicate = defaultdict(list)
    condition_time = defaultdict(list)
    
    # Agrupar resultados por las jerarquías requeridas
    for filename, result in image_results:
        metadata = extract_metadata_from_filename(filename)
        if not metadata:
            print(f"Advertencia: El archivo {filename} no tiene un formato válido. Se omite en la agregación.")
            continue
            
        # Clave para CONDICION/TIEMPO/REPLICA
        ctr_key = f"{metadata['condicion']}/{metadata['tiempo']}/{metadata['replica']}"
        
        # Clave para CONDICION/TIEMPO
        ct_key = f"{metadata['condicion']}/{metadata['tiempo']}"
        
        # Agregar resultados a los grupos correspondientes
        condition_time_replicate[ctr_key].append(result)
        condition_time[ct_key].append(result)
    
    # Resultados agregados
    aggregated_results = {
        'condition_time_replicate': {},
        'condition_time': {}
    }
    
    # Calcular estadísticas para cada grupo CONDICION/TIEMPO/REPLICA
    for key, results in condition_time_replicate.items():
        aggregated_results['condition_time_replicate'][key] = _calculate_aggregate_statistics(results)
    
    # Calcular estadísticas para cada grupo CONDICION/TIEMPO
    for key, results in condition_time.items():
        aggregated_results['condition_time'][key] = _calculate_aggregate_statistics(results)
    
    return aggregated_results


def _calculate_aggregate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula estadísticas agregadas a partir de múltiples resultados de análisis.
    
    Args:
        results: Lista de diccionarios de resultados de análisis
        
    Returns:
        Diccionario con estadísticas agregadas
    """
    if not results:
        return {}
    
    # Métricas a agregar
    metrics = [
        'avg_inclusions_per_cell',
        'avg_inclusion_area',
        'avg_inclusion_ratio',
        'percent_cells_with_inclusions'
    ]
    
    aggregated = {}
    
    # Calcular media y desviación estándar para cada métrica
    for metric in metrics:
        values = [r[metric] for r in results if metric in r]
        if values:
            aggregated[f'mean_{metric}'] = np.mean(values)
            aggregated[f'std_{metric}'] = np.std(values)
    
    # Agregar conteo total
    aggregated['total_images'] = len(results)
    
    return aggregated


def export_results_to_excel(image_results: List[Tuple[str, Dict[str, Any]]], output_dir: str, progress_reporter=None) -> str:
    """
    Exporta los resultados agregados a un archivo Excel con múltiples hojas.

    Args:
        image_results: Lista de tuplas (nombre_archivo, resultados_análisis)
        output_dir: Directorio donde guardar el archivo Excel
        progress_reporter: Función opcional para reportar el progreso

    Returns:
        Ruta al archivo Excel generado
    """
    def update_progress(value=None, detail=None):
        if progress_reporter:
            progress_reporter(value=value, detail=detail)

    try:
        update_progress(value=0.1, detail="Procesando datos para Excel...")
        
        # Transformar los datos para el formato del Excel
        transformed_data = []
        
        for filename, result in image_results:
            metadata = extract_metadata_from_filename(filename)
            if not metadata:
                print(f"Advertencia: El archivo {filename} no tiene un formato válido.")
                continue
                
            # Convertir metadata a datos para DataFrame
            row = {
                'Medio': metadata['condicion'],
                'Replica': metadata['replica'],
                'Tiempo (h)': metadata['tiempo'].replace('t', '')
            }
              # Añadir métricas calculadas
            row.update({
                'Recuento_Celulas': result.get('total_cells', 0),
                'Recuento_Inclusiones': result.get('total_inclusions', 0),
                'Area_Celulas_px': result.get('total_cell_area', 0),
                'Area_Inclusiones_px': result.get('total_inclusion_area', 0),
                'Inclusiones/Celula': result.get('avg_inclusions_per_cell', 0),
                'Area_Inclusiones/Celula_perc': result.get('avg_inclusion_ratio', 0) * 100,  # Convertir a porcentaje
                # Campos adicionales para completar manualmente
                'UFC/mL': np.nan,
                'Log (UFC/mL)': np.nan,
                'DO600': np.nan
            })
            
            transformed_data.append(row)
        
        # Convertir a DataFrame
        df = pd.DataFrame(transformed_data)
        
        # Convertir Tiempo a numérico
        df['Tiempo (h)'] = pd.to_numeric(df['Tiempo (h)'], errors='coerce')
        
        update_progress(value=0.2, detail="Calculando estadísticas...")
        
        # Calcular estadísticas por Medio/Replica/Tiempo
        tabla_formateada = df.copy()
        
        # Ordenar tabla
        tabla_formateada.sort_values(by=['Medio', 'Replica', 'Tiempo (h)'], inplace=True)
        
        # Calcular promedios por Medio y Tiempo
        update_progress(value=0.3, detail="Calculando promedios generales...")
        promedios_generales = tabla_formateada.groupby(['Medio', 'Tiempo (h)']).agg(
            UFC_mL_Promedio=('UFC/mL', lambda x: x.mean(skipna=True)),
            Log_UFC_mL_Promedio=('Log (UFC/mL)', lambda x: x.mean(skipna=True)),
            DO600_Promedio=('DO600', lambda x: x.mean(skipna=True)),
            Promedio_Recuento_Celulas=('Recuento_Celulas', lambda x: x.mean(skipna=True)),
            Promedio_Recuento_Inclusiones=('Recuento_Inclusiones', lambda x: x.mean(skipna=True)),
            Promedio_Area_Celulas_px=('Area_Celulas_px', lambda x: x.mean(skipna=True)),
            Promedio_Area_Inclusiones_px=('Area_Inclusiones_px', lambda x: x.mean(skipna=True)),
            Promedio_Inclusiones_Celula=('Inclusiones/Celula', lambda x: x.mean(skipna=True)),
            SD_Inclusiones_Celula=('Inclusiones/Celula', lambda x: x.std(skipna=True)),
            Promedio_Area_Inclusiones_Celula_perc=('Area_Inclusiones/Celula_perc', lambda x: x.mean(skipna=True)),
            SD_Area_Inclusiones_Celula_perc=('Area_Inclusiones/Celula_perc', lambda x: x.std(skipna=True)),
            Numero_Replicas=('Replica', 'count')
        ).reset_index()
        
        # Ordenar promedios
        promedios_generales.sort_values(by=['Medio', 'Tiempo (h)'], inplace=True)
        
        update_progress(value=0.4, detail="Generando archivo Excel...")
        
        # Exportar a Excel
        output_filename = "analisis_polifosfatos_resumen.xlsx"
        ruta_excel = os.path.join(output_dir, output_filename)
        
        try:
            with pd.ExcelWriter(ruta_excel, engine='openpyxl') as writer:
                # Escribir hojas por réplica
                for name, group in tabla_formateada.groupby(['Medio', 'Replica']):
                    sheet_name = f"{name[0]}-{name[1]}"
                    # Seleccionar columnas excluyendo Medio y Replica para la hoja
                    group_to_write = group.drop(columns=['Medio', 'Replica'])
                    group_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Escribir hoja de promedios
                promedios_generales.to_excel(writer, sheet_name="Promedios_Generales", index=False)
            
            update_progress(value=0.1, detail="¡Completado!")
            return f"Archivo Excel guardado en: {ruta_excel}"
            
        except Exception as e_save:
            print(f"Advertencia: Error al guardar {output_filename} - puede que esté abierto. Intentando nombre alternativo.")
            print(f"Error: {e_save}")
            
            try:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                ruta_excel_alt = os.path.join(output_dir, f"analisis_polifosfatos_resumen_{timestamp}.xlsx")
                
                with pd.ExcelWriter(ruta_excel_alt, engine='openpyxl') as writer:
                    # Reintentar escritura
                    for name, group in tabla_formateada.groupby(['Medio', 'Replica']):
                        sheet_name = f"{name[0]}-{name[1]}"
                        group_to_write = group.drop(columns=['Medio', 'Replica'])
                        group_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    promedios_generales.to_excel(writer, sheet_name="Promedios_Generales", index=False)
                
                update_progress(value=0.1, detail="¡Completado!")
                return f"Archivo Excel guardado con nombre alternativo: {ruta_excel_alt}"
                
            except Exception as e_alt:
                raise IOError(f"No se pudo guardar el archivo Excel. Error: {e_alt}")
    
    except Exception as e:
        raise RuntimeError(f"Error en export_results_to_excel: {e}")
