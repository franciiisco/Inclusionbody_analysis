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
    std_inclusion_area = np.std(all_inclusion_areas) if all_inclusion_areas else 0    # Ratio de área de inclusiones respecto a células
    inclusion_ratios = []
    for cell_id, incs in all_inclusions.items():
        # Calcular área de la célula
        cell_mask = (segmented_image == cell_id)
        cell_area = np.sum(cell_mask)
        
        if cell_area > 0:  # Considera TODAS las células con área válida
            # Suma de áreas de todas las inclusiones en esta célula (0 si no hay inclusiones)
            total_inclusion_area = sum(inc['area'] for inc in incs) if incs else 0
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
    
    # Calcular ratio global para verificación
    global_inclusion_ratio = (total_inclusion_area / total_cell_area) if total_cell_area > 0 else 0
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
        'global_inclusion_ratio': global_inclusion_ratio,  # Añadido: Ratio global
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


def extract_metadata_from_filename(filename: str, flexible: bool = True) -> Dict[str, str]:
    """
    Extrae los metadatos de un nombre de archivo con formato flexible.
    Formato ideal: CONDICION_BOTE_REPLICA_TIEMPO_NºIMAGEN.EXT
    
    Args:
        filename: Nombre del archivo a analizar
        flexible: Si True, usa patrones flexibles; si False, usa el patrón estricto original
        
    Returns:
        Diccionario con los metadatos extraídos o None si no se puede extraer información mínima
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    if not flexible:
        # Patrón original estricto
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
    
    # Modo flexible: múltiples patrones con fallbacks
    metadata = {
        'condicion': 'UNKNOWN',
        'bote': 'B1',  # Valor por defecto
        'replica': 'R1',
        'tiempo': 't0',
        'numero_imagen': '001'  # Valor por defecto
    }
    
    # Dividir por guiones bajos
    parts = name_without_ext.split('_')
    
    # Estrategia 1: Patrón completo (5 partes)
    if len(parts) >= 5:
        pattern1 = r"^([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)(?:_.*)?$"
        match1 = re.match(pattern1, name_without_ext)
        if match1:
            return {
                'condicion': match1.group(1),
                'bote': match1.group(2),
                'replica': match1.group(3),
                'tiempo': match1.group(4),
                'numero_imagen': match1.group(5)
            }
    
    # Estrategia 2: Sin BOTE (4 partes: CONDICION_REPLICA_TIEMPO_NUMERO)
    if len(parts) >= 4:
        pattern2 = r"^([^_]+)_([^_]+)_([^_]+)_([^_]+)(?:_.*)?$"
        match2 = re.match(pattern2, name_without_ext)
        if match2:
            # Verificar si la segunda parte es réplica (empieza con R)
            if match2.group(2).upper().startswith('R'):
                return {
                    'condicion': match2.group(1),
                    'bote': 'B1',  # Valor por defecto
                    'replica': match2.group(2),
                    'tiempo': match2.group(3),
                    'numero_imagen': match2.group(4)
                }
            # O si la segunda parte es bote y no hay número de imagen
            elif match2.group(2).upper().startswith('B'):
                # Buscar número de imagen en cualquier parte
                num_match = re.search(r'(\d{2,3})', name_without_ext)
                numero_imagen = num_match.group(1) if num_match else '001'
                
                return {
                    'condicion': match2.group(1),
                    'bote': match2.group(2),
                    'replica': match2.group(3),
                    'tiempo': match2.group(4),
                    'numero_imagen': numero_imagen
                }
    
    # Estrategia 3: Sin NUMERO_IMAGEN (4 partes: CONDICION_BOTE_REPLICA_TIEMPO)
    if len(parts) >= 4:
        pattern3 = r"^([^_]+)_([^_]+)_([^_]+)_([^_]+)(?:_.*)?$"
        match3 = re.match(pattern3, name_without_ext)
        if match3 and match3.group(2).upper().startswith('B') and match3.group(3).upper().startswith('R'):
            # Buscar número de imagen en cualquier parte del nombre
            num_match = re.search(r'(\d{2,3})', name_without_ext)
            numero_imagen = num_match.group(1) if num_match else '001'
            
            return {
                'condicion': match3.group(1),
                'bote': match3.group(2),
                'replica': match3.group(3),
                'tiempo': match3.group(4),
                'numero_imagen': numero_imagen
            }
    
    # Estrategia 4: Solo mínimo requerido (3 partes: CONDICION_REPLICA_TIEMPO)
    if len(parts) >= 3:
        pattern4 = r"^([^_]+)_([^_]+)_([^_]+)(?:_.*)?$"
        match4 = re.match(pattern4, name_without_ext)
        if match4:
            # Verificar si hay patrón de réplica y tiempo
            replica_candidate = match4.group(2)
            tiempo_candidate = match4.group(3)
            
            # Si el segundo elemento parece réplica
            if replica_candidate.upper().startswith('R'):
                # Buscar número de imagen
                num_match = re.search(r'(\d{2,3})', name_without_ext)
                numero_imagen = num_match.group(1) if num_match else '001'
                
                return {
                    'condicion': match4.group(1),
                    'bote': 'B1',  # Valor por defecto
                    'replica': replica_candidate,
                    'tiempo': tiempo_candidate,
                    'numero_imagen': numero_imagen
                }
    
    # Estrategia 5: Búsqueda de patrones individuales (último recurso)
    found_patterns = 0
    
    # Buscar condición (primera parte o patrón alfabético)
    if parts:
        cond_match = re.search(r'^([A-Za-z]+)', parts[0])
        if cond_match:
            metadata['condicion'] = cond_match.group(1)
            found_patterns += 1
    
    # Buscar bote (B seguido de número)
    bote_match = re.search(r'(B\d+)', name_without_ext, re.IGNORECASE)
    if bote_match:
        metadata['bote'] = bote_match.group(1).upper()
        found_patterns += 1
    
    # Buscar réplica (R seguido de número)
    replica_match = re.search(r'(R\d+)', name_without_ext, re.IGNORECASE)
    if replica_match:
        metadata['replica'] = replica_match.group(1).upper()
        found_patterns += 1
    
    # Buscar tiempo (t seguido de número)
    tiempo_match = re.search(r'(t\d+)', name_without_ext, re.IGNORECASE)
    if tiempo_match:
        metadata['tiempo'] = tiempo_match.group(1).lower()
        found_patterns += 1
    
    # Buscar número de imagen (2-3 dígitos consecutivos)
    num_match = re.search(r'(\d{2,3})', name_without_ext)
    if num_match:
        metadata['numero_imagen'] = num_match.group(1)
        found_patterns += 1
    
    # Retornar metadatos si encontramos al menos 3 patrones (mínimo viable)
    if found_patterns >= 3:
        return metadata
    else:
        # Si no encontramos suficientes patrones, retornar None
        return None


def validate_filename_format(filename: str, flexible: bool = True, min_required_fields: int = 3) -> bool:
    """
    Valida si un nombre de archivo puede ser procesado.
    
    Args:
        filename: Nombre del archivo a validar
        flexible: Si usar validación flexible o estricta
        min_required_fields: Número mínimo de campos requeridos en modo flexible
        
    Returns:
        True si el formato es procesable, False en caso contrario
    """
    if not flexible:
        # Validación estricta original
        metadata = extract_metadata_from_filename(filename, flexible=False)
        return metadata is not None
    
    # Validación flexible
    metadata = extract_metadata_from_filename(filename, flexible=True)
    
    if metadata is None:
        return False
    
    # Verificar que tenemos al menos los campos mínimos requeridos
    required_fields = ['condicion', 'replica', 'tiempo']
    found_required = sum(1 for field in required_fields if metadata.get(field, 'UNKNOWN') != 'UNKNOWN')
    
    return found_required >= min_required_fields


def aggregate_inclusion_data(image_results: List[Tuple[str, Dict[str, Any]]], flexible_parsing: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Agrega los resultados de análisis por CONDICION/TIEMPO/REPLICA y CONDICION/TIEMPO.
    
    Args:
        image_results: Lista de tuplas (nombre_archivo, resultados_análisis)
                      donde resultados_análisis es el diccionario devuelto por summarize_inclusions
        flexible_parsing: Si usar parsing flexible de nombres de archivo
        
    Returns:
        Diccionario con resultados agregados por jerarquía
    """
    # Inicializar estructuras para agrupar datos
    condition_time_replicate = defaultdict(list)
    condition_time = defaultdict(list)
    
    # Contador de archivos procesados vs omitidos
    processed_files = 0
    skipped_files = 0
    files_with_defaults = 0
    
    # Agrupar resultados por las jerarquías requeridas
    for filename, result in image_results:
        metadata = extract_metadata_from_filename(filename, flexible=flexible_parsing)
        
        if not metadata:
            print(f"Advertencia: El archivo {filename} no pudo ser procesado. Se omite en la agregación.")
            skipped_files += 1
            continue
        
        # Verificar si se usaron valores por defecto
        has_defaults = (metadata['bote'] == 'B1' and 'B1' not in filename) or \
                      (metadata['numero_imagen'] == '001' and '001' not in filename)
        
        if has_defaults:
            files_with_defaults += 1
            print(f"Info: {filename} procesado con valores por defecto - Bote: {metadata['bote']}, Num: {metadata['numero_imagen']}")
        
        processed_files += 1
            
        # Clave para CONDICION/TIEMPO/REPLICA
        ctr_key = f"{metadata['condicion']}/{metadata['tiempo']}/{metadata['replica']}"
        
        # Clave para CONDICION/TIEMPO
        ct_key = f"{metadata['condicion']}/{metadata['tiempo']}"
        
        # Agregar resultados a los grupos correspondientes
        condition_time_replicate[ctr_key].append(result)
        condition_time[ct_key].append(result)
    
    print(f"Resumen: {processed_files} archivos procesados, {skipped_files} omitidos, {files_with_defaults} con valores por defecto")
    
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
        'global_inclusion_ratio',
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
            metadata = extract_metadata_from_filename(filename, flexible=True)
            if not metadata:
                print(f"Advertencia: El archivo {filename} no tiene un formato válido.")
                continue
                
            # Convertir metadata a datos para DataFrame
            row = {
                'Medio': metadata['condicion'],
                'Replica': metadata['replica'],
                'Tiempo (h)': metadata['tiempo'].replace('t', '')
            }
            # Añadir métricas calculadas (sin OD600, UFC/mL, Log UFC/mL)
            row.update({
                'Recuento_Celulas': result.get('total_cells', 0),
                'Recuento_Inclusiones': result.get('total_inclusions', 0),
                'Area_Celulas_px': result.get('total_cell_area', 0),
                'Area_Inclusiones_px': result.get('total_inclusion_area', 0),
                'Inclusiones/Celula': result.get('avg_inclusions_per_cell', 0),
                'Area_Inclusiones/Celula_perc': result.get('global_inclusion_ratio', 0) * 100,
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
        
        # Calcular sumatorios por Medio y Tiempo
        update_progress(value=0.3, detail="Calculando sumatorios generales...")
        promedios_generales = tabla_formateada.groupby(['Medio', 'Tiempo (h)']).agg(
            SUMATORIO_RECUENTO_CELULAS=('Recuento_Celulas', lambda x: x.sum(skipna=True)),
            SUMATORIO_RECUENTO_INCLUSIONES=('Recuento_Inclusiones', lambda x: x.sum(skipna=True)),
            SUMATORIO_AREA_CELULAS=('Area_Celulas_px', lambda x: x.sum(skipna=True)),
            SUMATORIO_AREA_INCLUSIONES=('Area_Inclusiones_px', lambda x: x.sum(skipna=True)),
            NUMERO_IMAGENES=('Replica', 'count')
        ).reset_index()

        # Calcular AREA_INCLUSIONES_CELULA_PERC usando los sumatorios
        # Evitar división por cero
        promedios_generales['AREA_INCLUSIONES_CELULA_PERC'] = np.where(
            promedios_generales['SUMATORIO_AREA_CELULAS'] > 0,
            (promedios_generales['SUMATORIO_AREA_INCLUSIONES'] / promedios_generales['SUMATORIO_AREA_CELULAS']) * 100,
            0  # O np.nan si prefieres NaN en lugar de 0 cuando el área de células es 0
        )
        
        # Ordenar promedios
        promedios_generales.sort_values(by=['Medio', 'Tiempo (h)'], inplace=True)
        
        update_progress(value=0.4, detail="Generando archivo Excel...")
        
        # Exportar a Excel
        output_filename = "resultados_analisis_polifosfatos.xlsx"
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