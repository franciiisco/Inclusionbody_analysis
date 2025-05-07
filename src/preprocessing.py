"""
Módulo para el preprocesamiento de imágenes de microscopía con células bacterianas
e inclusiones de polifosfatos.

Este módulo proporciona funciones para normalizar el contraste, reducir el ruido y
corregir problemas de iluminación no uniforme en imágenes de microscopía.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, Union


def normalize_image(
    image: np.ndarray, 
    method: str = 'minmax', 
    clip_limit: float = 2.0, 
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Normaliza el contraste y brillo de una imagen de microscopía.
    
    Args:
        image: Imagen de entrada en formato numpy array
        method: Método de normalización ('minmax', 'clahe', o 'histogram')
        clip_limit: Límite de contraste para CLAHE
        tile_grid_size: Tamaño de la cuadrícula para CLAHE
        
    Returns:
        Imagen normalizada
    """
    # Verificar que la imagen es válida
    if not isinstance(image, np.ndarray):
        raise TypeError("La imagen debe ser un numpy array")
    
    # Convertir a escala de grises si la imagen tiene 3 canales
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Asegurar que la imagen esté en uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Aplicar el método de normalización seleccionado
    if method == 'minmax':
        # Normalización min-max simple
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    elif method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        normalized = clahe.apply(image)
    
    elif method == 'histogram':
        # Ecualización de histograma tradicional
        normalized = cv2.equalizeHist(image)
    
    else:
        raise ValueError(f"Método de normalización no reconocido: {method}")
    
    return normalized


def denoise_image(
    image: np.ndarray, 
    method: str = 'gaussian', 
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Aplica filtros para reducir el ruido en la imagen preservando detalles importantes.
    
    Args:
        image: Imagen de entrada
        method: Método de filtrado ('gaussian', 'median', 'bilateral')
        params: Diccionario de parámetros específicos para el método seleccionado
            - Para 'gaussian': {'ksize': tamaño del kernel, 'sigma': valor sigma}
            - Para 'median': {'ksize': tamaño del kernel}
            - Para 'bilateral': {'d': diámetro, 'sigma_color': sigma color, 'sigma_space': sigma espacio}
    
    Returns:
        Imagen filtrada
    """
    if params is None:
        params = {}
    
    # Valores predeterminados para cada método
    defaults = {
        'gaussian': {'ksize': (5, 5), 'sigma': 0},
        'median': {'ksize': 5},
        'bilateral': {'d': 7, 'sigma_color': 50, 'sigma_space': 50}
    }
    
    # Actualizar valores predeterminados con los proporcionados
    if method in defaults:
        for key, default_value in defaults[method].items():
            if key not in params:
                params[key] = default_value
    
    # Aplicar el filtro seleccionado
    if method == 'gaussian':
        denoised = cv2.GaussianBlur(image, params['ksize'], params['sigma'])
    
    elif method == 'median':
        denoised = cv2.medianBlur(image, params['ksize'])
    
    elif method == 'bilateral':
        denoised = cv2.bilateralFilter(
            image, params['d'], params['sigma_color'], params['sigma_space']
        )
    
    else:
        raise ValueError(f"Método de filtrado no reconocido: {method}")
    
    return denoised


def correct_illumination(
    image: np.ndarray, 
    method: str = 'subtract_background', 
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Corrige problemas de iluminación no uniforme en la imagen.
    
    Args:
        image: Imagen de entrada
        method: Método de corrección ('subtract_background', 'morphological', 'homomorphic')
        params: Parámetros específicos para cada método
    
    Returns:
        Imagen con iluminación corregida
    """
    if params is None:
        params = {}
    
    # Valores predeterminados para cada método
    defaults = {
        'subtract_background': {'kernel_size': 51},
        'morphological': {'kernel_size': 15},
        'homomorphic': {'gauss_size': 15, 'gamma1': 1.5, 'gamma2': 0.5}
    }
    
    # Actualizar valores predeterminados con los proporcionados
    if method in defaults:
        for key, default_value in defaults[method].items():
            if key not in params:
                params[key] = default_value
    
    # Aplicar el método seleccionado
    if method == 'subtract_background':
        # Estimación de fondo usando filtro de mediana
        kernel_size = params['kernel_size']
        background = cv2.medianBlur(image, kernel_size)
        # Restar el fondo estimado
        corrected = cv2.subtract(image, background)
        # Normalizar resultado
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    
    elif method == 'morphological':
        # Corrección basada en operaciones morfológicas
        kernel_size = params['kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        corrected = cv2.divide(image, background, scale=255)
    
    elif method == 'homomorphic':
        # Filtro homomorfo
        # 1. Pasar al dominio logarítmico
        log_image = np.log1p(np.array(image, dtype=np.float32))
        
        # 2. Transformada de Fourier
        img_fft = np.fft.fft2(log_image)
        img_fft_shift = np.fft.fftshift(img_fft)
        
        # 3. Filtro Butterworth/Gaussiano
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        mask = np.ones((rows, cols), np.uint8)
        gauss_size = params['gamma1']
        mask = np.exp(-((np.arange(rows) - crow) ** 2)[:, np.newaxis] / (2 * gauss_size ** 2) -
                      ((np.arange(cols) - ccol) ** 2) / (2 * gauss_size ** 2))
        
        # 4. Aplicar filtro en dominio de frecuencia
        img_fft_shift_filtered = img_fft_shift * mask
        
        # 5. Transformada inversa
        img_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft_shift_filtered)))
        
        # 6. Volver del dominio logarítmico
        corrected = np.expm1(img_filtered)
        
        # 7. Normalizar resultado
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
        corrected = np.uint8(corrected)
    
    else:
        raise ValueError(f"Método de corrección de iluminación no reconocido: {method}")
    
    return corrected

def invert_image(image: np.ndarray) -> np.ndarray:
    """
    Invierte los valores de intensidad de una imagen.
    
    Args:
        image: Imagen de entrada
        
    Returns:
        Imagen invertida donde los píxeles oscuros se vuelven brillantes y viceversa
    """
    return cv2.bitwise_not(image)

def preprocess_pipeline(
    image: np.ndarray, 
    config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Aplica toda la secuencia de preprocesamiento a una imagen.
    
    Args:
        image: Imagen de entrada
        config: Configuración para los distintos pasos del preprocesamiento
            {
                'normalize': {'method': 'minmax', ...},
                'denoise': {'method': 'gaussian', 'params': {...}},
                'correct_illumination': {'method': 'subtract_background', 'params': {...}},
                'invert': True/False
            }
    
    Returns:
        Imagen preprocesada
    """
    if config is None:
        config = {
            'normalize': {'method': 'clahe', 'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
            'denoise': {'method': 'gaussian', 'params': {'ksize': (5, 5), 'sigma': 0}},
            'correct_illumination': {'method': 'subtract_background', 'params': {'kernel_size': 51}},
            'invert': True  # Por defecto invertimos la imagen para facilitar la detección posterior
        }
    
    # 1. Corrección de iluminación
    if 'correct_illumination' in config:
        cfg = config['correct_illumination']
        image = correct_illumination(
            image, 
            method=cfg.get('method', 'subtract_background'),
            params=cfg.get('params', {})
        )
    
    # 2. Normalización
    if 'normalize' in config:
        cfg = config['normalize']
        normalize_params = {k: v for k, v in cfg.items() if k != 'method'}
        image = normalize_image(image, method=cfg.get('method', 'minmax'), **normalize_params)
    
    # 3. Filtrado de ruido
    if 'denoise' in config:
        cfg = config['denoise']
        image = denoise_image(
            image,
            method=cfg.get('method', 'gaussian'),
            params=cfg.get('params', {})
        )
    
    # 4. Inversión de la imagen (si está configurado)
    if config.get('invert', False):
        image = invert_image(image)
    
    return image

def visualize_preprocessing_steps(
    original_image: np.ndarray, 
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Visualiza cada paso del preprocesamiento para facilitar el ajuste de parámetros.
    
    Args:
        original_image: Imagen original
        config: Configuración para el preprocesamiento
    """
    if config is None:
        config = {
            'normalize': {'method': 'clahe', 'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
            'denoise': {'method': 'gaussian', 'params': {'ksize': (5, 5), 'sigma': 0}},
            'correct_illumination': {'method': 'subtract_background', 'params': {'kernel_size': 51}},
            'invert': True
        }
    
    # Crear figura para mostrar resultados
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Inicializar las imágenes de cada paso
    processed_images = {
        'original': original_image,
        'illumination_corrected': None,
        'normalized': None,
        'denoised': None,
        'inverted': None
    }
    
    # Imagen original
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Imagen Original')
    
    # Corrección de iluminación
    if 'correct_illumination' in config:
        cfg = config['correct_illumination']
        processed_images['illumination_corrected'] = correct_illumination(
            original_image,
            method=cfg.get('method', 'subtract_background'),
            params=cfg.get('params', {})
        )
        axes[0, 1].imshow(processed_images['illumination_corrected'], cmap='gray')
        axes[0, 1].set_title(f'Corrección de Iluminación: {cfg.get("method")}')
    
    # Normalización
    if 'normalize' in config:
        cfg = config['normalize']
        normalize_params = {k: v for k, v in cfg.items() if k != 'method'}
        base_image = processed_images['illumination_corrected'] if processed_images['illumination_corrected'] is not None else original_image
        processed_images['normalized'] = normalize_image(
            base_image,
            method=cfg.get('method', 'minmax'),
            **normalize_params
        )
        axes[0, 2].imshow(processed_images['normalized'], cmap='gray')
        axes[0, 2].set_title(f'Normalización: {cfg.get("method")}')
    
    # Filtrado de ruido
    if 'denoise' in config:
        cfg = config['denoise']
        base_image = (processed_images['normalized'] if processed_images['normalized'] is not None 
                     else (processed_images['illumination_corrected'] if processed_images['illumination_corrected'] is not None
                          else original_image))
        processed_images['denoised'] = denoise_image(
            base_image,
            method=cfg.get('method', 'gaussian'),
            params=cfg.get('params', {})
        )
        axes[1, 0].imshow(processed_images['denoised'], cmap='gray')
        axes[1, 0].set_title(f'Reducción de Ruido: {cfg.get("method")}')
    
    # Inversión
    if config.get('invert', False):
        base_image = (processed_images['denoised'] if processed_images['denoised'] is not None
                     else (processed_images['normalized'] if processed_images['normalized'] is not None
                          else (processed_images['illumination_corrected'] if processed_images['illumination_corrected'] is not None
                               else original_image)))
        processed_images['inverted'] = invert_image(base_image)
        axes[1, 1].imshow(processed_images['inverted'], cmap='gray')
        axes[1, 1].set_title('Imagen Invertida')
    
    # Resultado final
    final_image = next(processed_images[key] for key in ['inverted', 'denoised', 'normalized', 'illumination_corrected', 'original'] 
                      if processed_images[key] is not None)
    axes[1, 2].imshow(final_image, cmap='gray')
    axes[1, 2].set_title('Resultado Final')
    
    # Ajustar diseño y mostrar
    plt.tight_layout()
    plt.show()
    
    return final_image


if __name__ == "__main__":
    # Ejemplo de uso
    image_path = "C:/Users/fmarquez/Desktop/POLIP_Analizer/data/raw/sample_image.png"
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
        
        # Configuración personalizada
        config = {
            'normalize': {'method': 'clahe', 'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
            'denoise': {'method': 'bilateral', 'params': {'d': 3, 'sigma_color': 50, 'sigma_space': 50}},
            'correct_illumination': {'method': 'morphological', 'params': {'kernel_size': 31}},
            'invert': True
        }
        
        # Visualizar los pasos de preprocesamiento
        visualize_preprocessing_steps(image, config)
        
        # Aplicar todo el pipeline
        processed_image = preprocess_pipeline(image, config)
        
        # Guardar la imagen procesada
        cv2.imwrite("C:/Users/fmarquez/Desktop/POLIP_Analizer/data/processed/preprocessed_sample.png", processed_image)
        print("Preprocesamiento completado con éxito.")
        
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")