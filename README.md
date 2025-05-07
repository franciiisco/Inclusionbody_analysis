# POLIP_Analizer

## Descripción del Proyecto

POLIP_Analizer es una herramienta diseñada para analizar imágenes de microscopía de células bacterianas con inclusiones de polifosfatos. Este proyecto permite preprocesar imágenes, segmentar células, detectar inclusiones de polifosfatos y generar estadísticas detalladas sobre las inclusiones detectadas. Además, incluye visualizaciones para facilitar la interpretación de los resultados.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

- **`src/`**: Contiene los módulos principales del proyecto.
  - `preprocessing.py`: Funciones para preprocesar imágenes (normalización, reducción de ruido, corrección de iluminación, etc.).
  - `segmentation.py`: Funciones para segmentar células bacterianas en las imágenes.
  - `detection.py`: Funciones para detectar inclusiones de polifosfatos dentro de las células segmentadas.
  - `visualization.py`: Funciones para visualizar los resultados de segmentación y detección.
  - `features.py`: (Pendiente) Funciones para la extracción de características.
  - `analysis.py`: (Pendiente) Funciones para el análisis estadístico.

- **`data/`**: Contiene los datos de entrada y salida.
  - `raw/`: Imágenes originales sin procesar.
  - `processed/`: Resultados procesados, incluyendo imágenes segmentadas, estadísticas y visualizaciones.

- **`results/`**: Almacena figuras y estadísticas generadas.
  - `figures/`: Figuras generadas durante el análisis.
  - `stats/`: Datos estadísticos generados.

- **`notebooks/`**: Notebooks de Jupyter para análisis interactivo y ajuste de parámetros.

- **`main.py`**: Script principal para ejecutar el pipeline completo de análisis.

- **`config.py`**: Archivo de configuración (actualmente vacío).

- **`requirements.txt`**: Lista de dependencias necesarias para ejecutar el proyecto.

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

## Uso

### Procesamiento de una Imagen

Ejecuta el script principal para procesar una imagen:
```bash
python main.py
```
Por defecto, el script procesará la imagen `data/raw/sample_image.png` y generará los resultados en `data/processed/`.

### Procesamiento por Lotes

Para procesar múltiples imágenes en un directorio, utiliza la función `batch_process` en `main.py`.

### Notebooks

Explora los notebooks en la carpeta `notebooks/` para realizar análisis interactivos y ajustar parámetros.

## Funcionalidades

1. **Preprocesamiento**:
   - Normalización del contraste y brillo.
   - Reducción de ruido.
   - Corrección de iluminación no uniforme.

2. **Segmentación**:
   - Identificación y delimitación de células bacterianas.
   - Métodos avanzados como watershed y transformada de distancia.

3. **Detección de Inclusiones**:
   - Identificación de inclusiones de polifosfatos dentro de las células segmentadas.
   - Caracterización de inclusiones (área, circularidad, intensidad, etc.).

4. **Visualización**:
   - Visualización de células segmentadas y sus inclusiones.
   - Gráficos estadísticos sobre las inclusiones detectadas.

5. **Análisis Estadístico**:
   - Resumen de estadísticas clave (número de células, inclusiones, tamaños, etc.).
   - Gráficos para interpretar los resultados.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request con tus sugerencias o mejoras.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.