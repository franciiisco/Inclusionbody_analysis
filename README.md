# POLIP_Analizer

## Descripción del Proyecto

POLIP_Analizer es una herramienta avanzada diseñada para el análisis automatizado de imágenes de microscopía de células bacterianas con inclusiones de polifosfatos. El sistema proporciona un pipeline completo que incluye preprocesamiento de imágenes, segmentación celular, detección avanzada de inclusiones y análisis estadístico detallado. Cuenta con una interfaz gráfica moderna y capacidades de procesamiento por lotes.

## Características Principales

- **Interfaz gráfica moderna** con ttkbootstrap para una experiencia de usuario mejorada
- **Algoritmos de detección avanzados** (versión 2.0) con mejor separación de inclusiones cercanas
- **Procesamiento por lotes** con validación de convenciones de nomenclatura
- **Múltiples métodos de umbralización** (multinivel, adaptativa, Otsu)
- **Análisis estadístico completo** con exportación a Excel
- **Visualizaciones interactivas** y generación de reportes
- **Modo desarrollo** con visualizaciones paso a paso

## Estructura del Proyecto

### Módulos Principales (`src/`)
- **`core.py`**: Pipeline principal de procesamiento y funciones de batch
- **`preprocessing.py`**: Normalización, reducción de ruido, corrección de iluminación
- **`segmentation.py`**: Segmentación celular con métodos watershed y operaciones morfológicas
- **`detection_v2_0.py`**: Detección avanzada de inclusiones (versión 2.0) con:
  - Umbralización multinivel y adaptativa
  - Separación por watershed de inclusiones conectadas
  - Filtrado por tamaño, circularidad y contraste
  - Análisis de textura opcional
- **`visualization.py`**: Visualización de resultados con código de colores por tamaño
- **`analysis.py`**: Análisis estadístico, agregación de datos y exportación a Excel
- **`features.py`**: Extracción de características morfológicas

### Interfaz Gráfica (`app/`)
- **`gui.py`**: Aplicación principal con interfaz moderna
- **`components/`**: Componentes reutilizables (barra de progreso, visor de resultados)
- **`tabs/`**: Pestañas organizadas (análisis, metodología, información)
  - **`descriptions/`**: Documentación técnica detallada de cada módulo

### Utilidades (`utils/`)
- **`file_operations.py`**: Operaciones de archivos y detección automática de patrones
- **`visualization.py`**: Funciones auxiliares de visualización

### Datos y Resultados
- **`data/`**: Datos de entrada y procesados
  - `raw/`: Imágenes originales
  - `processed/`: Resultados con imágenes segmentadas y estadísticas
  - `logo/`: Recursos gráficos de la aplicación
- **`results/`**: Almacena figuras y estadísticas generadas
- **`notebooks/`**: Jupyter notebooks para análisis interactivo y ajuste de parámetros

### Archivos de Configuración
- **`config.py`**: Configuración completa para todos los módulos
- **`main.py`**: Script principal con argumentos de línea de comandos
- **`run.py`**: Launcher para la interfaz gráfica
- **`requirements.txt`**: Dependencias del proyecto

## Instalación

1. Clona este repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd POLIP_Analizer
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Formas de Uso

### 1. Interfaz Gráfica (Recomendado)

Ejecuta la interfaz gráfica moderna:
```bash
python run.py
```

La interfaz incluye:
- **Pestaña de Análisis**: Selección de archivos, configuración de parámetros y procesamiento
- **Pestaña de Metodología**: Explicación detallada de cada algoritmo
- **Pestaña de Información**: Detalles del proyecto y referencias

### 2. Línea de Comandos

#### Procesamiento de una imagen individual:
```bash
python main.py
```

#### Procesamiento por lotes:
```bash
python main.py --batch --input "ruta/a/imagenes" --output "ruta/salida" --pattern "*.tif"
```

#### Opciones avanzadas:
```bash
# Sin validación de nombres de archivo
python main.py --batch --input "data/raw" --no-enforce-naming

# Con patrón específico de archivos
python main.py --batch --input "data/raw" --pattern "*.{tif,png,jpg}"
```

### 3. Notebooks Interactivos

Explora los notebooks en `notebooks/` para:
- **`parameter_tuning.ipynb`**: Ajuste fino de parámetros de detección
- **`result_analysis.ipynb`**: Análisis estadístico avanzado de resultados

## Funcionalidades Avanzadas

### 1. Preprocesamiento Inteligente
- **Normalización adaptativa**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Reducción de ruido**: Filtro bilateral preservando bordes
- **Corrección de iluminación**: Métodos morfológicos para iluminación no uniforme
- **Inversión automática**: Para adaptarse a diferentes tipos de contraste

### 2. Segmentación Celular Robusta
- **Métodos múltiples**: Umbralización Otsu, watershed, operaciones morfológicas
- **Filtrado inteligente**: Por área, circularidad y relación de aspecto
- **Validación automática**: Eliminación de artefactos y objetos no celulares

### 3. Detección de Inclusiones (Versión 2.0)

#### Algoritmos de Umbralización:
- **Multinivel**: Detección de inclusiones con diferentes intensidades
- **Adaptativa local**: Ajuste automático a variaciones de iluminación
- **Otsu específico**: Aplicado solo dentro de cada célula

#### Separación de Inclusiones Conectadas:
- **Algoritmo watershed mejorado**: Con análisis de intensidad y distancia
- **Detección de líneas delgadas**: Separación de inclusiones unidas por conexiones débiles
- **Análisis de excentricidad**: Identificación y división de regiones alargadas

#### Filtrado y Validación:
- **Múltiples criterios**: Tamaño, circularidad, contraste y textura
- **Análisis de homogeneidad**: Validación de textura interna
- **Contraste con entorno**: Medición de diferencia con píxeles circundantes

### 4. Análisis Estadístico Completo
- **Estadísticas por célula**: Número, área y distribución de inclusiones
- **Agregación por condiciones**: Según convención de nomenclatura experimental
- **Exportación a Excel**: Reportes detallados con múltiples hojas de cálculo
- **Metadatos automáticos**: Extracción de información experimental de nombres de archivo

### 5. Visualizaciones Avanzadas
- **Código de colores**: Verde (pequeñas), amarillo (medianas), rojo (grandes)
- **Contornos reales**: Visualización de formas exactas de inclusiones
- **Gráficos estadísticos**: Histogramas, distribuciones y comparaciones
- **Modo desarrollo**: Visualización paso a paso del pipeline

### 6. Configuración Flexible
- **Modos de operación**: Desarrollo (con visualizaciones) y estándar (automático)
- **Parámetros ajustables**: Para cada etapa del pipeline
- **Configuraciones predefinidas**: Optimizadas para diferentes tipos de muestras

## Convención de Nomenclatura

Para un análisis óptimo, se recomienda seguir la convención de nomenclatura:
```
CONDICION_BOTE_REPLICA_TIEMPO_NUMERO.extension
```

**Ejemplo**: `Control_B1_R2_24h_001.tif`

Donde:
- **CONDICION**: Condición experimental (ej: Control, Tratamiento1)
- **BOTE**: Identificador del recipiente/muestra (ej: B1, B2)
- **REPLICA**: Número de réplica (ej: R1, R2, R3)
- **TIEMPO**: Punto temporal (ej: 0h, 24h, 48h)
- **NUMERO**: Número de imagen (ej: 001, 002)

Esta convención permite:
- Agregación automática de datos por condiciones experimentales
- Análisis estadístico agrupado
- Generación de reportes organizados

## Formatos de Archivo Soportados

- **Imágenes de entrada**: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`
- **Salida de datos**: `.json`, `.xlsx` (Excel)
- **Visualizaciones**: `.png`, `.jpg`

## Modos de Operación

### Modo Desarrollo
- Visualizaciones interactivas paso a paso
- Ideal para ajuste de parámetros y depuración
- Activar en `config.py`: `DEVELOPMENT_MODE = True`

### Modo Estándar
- Procesamiento automático sin interrupciones
- Optimizado para análisis por lotes
- Genera solo los resultados finales

## Configuración Avanzada

El archivo `config.py` permite personalizar:

### Preprocesamiento
- Métodos de normalización (CLAHE, ecualización de histograma)
- Parámetros de reducción de ruido
- Configuración de corrección de iluminación

### Segmentación
- Umbrales y métodos de segmentación
- Filtros morfológicos
- Criterios de validación celular

### Detección de Inclusiones V2
- Algoritmos de umbralización (multinivel, adaptativa, Otsu)
- Parámetros de separación watershed
- Criterios de filtrado (tamaño, forma, contraste)
- Análisis de textura

## Resultados Generados

### Archivos de Salida

Para cada imagen procesada se generan:

1. **Estadísticas en JSON**: Datos cuantitativos detallados
2. **Imágenes procesadas**: 
   - Imagen preprocesada
   - Células segmentadas
   - Inclusiones detectadas
   - Visualización con código de colores
3. **Gráficos estadísticos**: Histogramas y distribuciones
4. **Reporte Excel**: Análisis agregado por condiciones experimentales

### Métricas Calculadas

#### Por Célula:
- Número de inclusiones
- Área total de inclusiones
- Ratio de inclusión (área inclusiones/área célula)
- Distribución de tamaños

#### Por Imagen:
- Total de células detectadas
- Células con inclusiones
- Estadísticas agregadas (media, desviación estándar)
- Distribuciones de propiedades

#### Por Condición Experimental:
- Comparación entre grupos
- Análisis estadístico descriptivo
- Tablas y gráficos comparativos

## Limitaciones Conocidas

- **Calidad de imagen**: Requiere imágenes con contraste suficiente entre células e inclusiones
- **Resolución**: Optimizado para microscopía de fluorescencia con resolución típica
- **Solapamiento**: Células muy superpuestas pueden afectar la segmentación
- **Artefactos**: Objetos brillantes no celulares pueden ser detectados como inclusiones

## Requisitos del Sistema

- **Python**: 3.8 o superior
- **RAM**: Mínimo 4GB, recomendado 8GB para procesamiento por lotes
- **Almacenamiento**: Variable según el tamaño de las imágenes
- **Sistema operativo**: Windows, macOS, Linux

## Troubleshooting

### Problemas Comunes

1. **Error de importación de módulos**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Memoria insuficiente para imágenes grandes**:
   - Procesar imágenes individualmente
   - Reducir resolución si es posible
   - Cerrar otras aplicaciones

3. **Detección incorrecta**:
   - Ajustar parámetros en `config.py`
   - Usar modo desarrollo para visualizar pasos
   - Verificar calidad de la imagen de entrada

4. **Problemas con la interfaz gráfica**:
   ```bash
   pip install ttkbootstrap --upgrade
   ```

## Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Realiza tus cambios y añade tests si es necesario
4. Commit tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
5. Push a la rama (`git push origin feature/nueva-funcionalidad`)
6. Abre un Pull Request

### Áreas de Mejora

- Algoritmos de detección alternativos
- Soporte para nuevos formatos de imagen
- Análisis estadístico avanzado
- Optimización de rendimiento
- Interfaz web

## Licencia

Este proyecto está licenciado bajo [especificar licencia]. Ver el archivo LICENSE para más detalles.

## Citas y Referencias

Si utilizas POLIP_Analizer en tu investigación, por favor cita:

```
[Pendiente: añadir información de citación]
```

## Contacto

- **Desarrollador**: [Francisco Márquez Urbano]
- **Email**: [fmarquez@cnta.es]

---

**Nota**: Este README se actualiza regularmente. Para la documentación más reciente, consulta la versión en línea del repositorio.
