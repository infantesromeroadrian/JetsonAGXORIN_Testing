#!/usr/bin/env python3
"""
Utilidades para procesamiento y manejo de imágenes.

Este módulo proporciona funciones para codificar imágenes a base64
y gestionar archivos de imagen para su uso con modelos de visión.
"""

import base64
import sys
from pathlib import Path
from typing import Optional, List, Union


class ImageProcessor:
    """Procesador de imágenes para modelos de visión."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    def __init__(self):
        """Inicializa el procesador de imágenes."""
        pass
    
    @staticmethod
    def encode_to_base64(image_path: Union[str, Path]) -> str:
        """
        Codifica una imagen a base64.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            str: Imagen codificada en base64
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato no es soportado
            IOError: Si hay error leyendo el archivo
        """
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        if path.suffix.lower() not in ImageProcessor.SUPPORTED_FORMATS:
            raise ValueError(f"Formato no soportado: {path.suffix}")
        
        try:
            with open(path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded
        except IOError as e:
            raise IOError(f"Error leyendo imagen {image_path}: {e}")
    
    @staticmethod
    def validate_image_path(image_path: Union[str, Path]) -> bool:
        """
        Valida si una ruta de imagen es válida.
        
        Args:
            image_path: Ruta a validar
            
        Returns:
            bool: True si la imagen es válida
        """
        try:
            path = Path(image_path)
            return (path.exists() and 
                   path.is_file() and 
                   path.suffix.lower() in ImageProcessor.SUPPORTED_FORMATS)
        except Exception:
            return False
    
    @staticmethod
    def find_default_image(search_paths: Optional[List[Union[str, Path]]] = None) -> Optional[Path]:
        """
        Busca una imagen por defecto en las rutas especificadas.
        
        Args:
            search_paths: Lista de rutas donde buscar
            
        Returns:
            Optional[Path]: Ruta de la imagen encontrada o None
        """
        if search_paths is None:
            # Rutas por defecto relativas al proyecto
            base_dir = Path(__file__).parent.parent.parent
            search_paths = [
                base_dir / "assets",
                base_dir / "data",
                Path.cwd() / "assets"
            ]
        
        # Nombres de archivo por defecto a buscar
        default_names = [
            "test_image.jpg",
            "puerto-new-york-1068x570.webp",
            "example.png",
            "sample.jpg"
        ]
        
        for search_path in search_paths:
            try:
                path = Path(search_path)
                if not path.exists():
                    continue
                
                for image_name in default_names:
                    image_path = path / image_name
                    if ImageProcessor.validate_image_path(image_path):
                        return image_path
            except Exception:
                continue
        
        return None
    
    def get_image_info(self, image_path: Union[str, Path]) -> dict:
        """
        Obtiene información básica de una imagen.
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            dict: Información de la imagen
        """
        path = Path(image_path)
        
        if not self.validate_image_path(path):
            return {"error": "Imagen no válida"}
        
        try:
            return {
                "path": str(path.absolute()),
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "format": path.suffix.lower(),
                "exists": True,
                "is_supported": True
            }
        except Exception as e:
            return {"error": f"Error obteniendo info: {e}"}
    
    def batch_encode(self, image_paths: List[Union[str, Path]]) -> List[str]:
        """
        Codifica múltiples imágenes a base64.
        
        Args:
            image_paths: Lista de rutas de imágenes
            
        Returns:
            List[str]: Lista de imágenes codificadas (vacías si hay error)
        """
        encoded_images = []
        
        for image_path in image_paths:
            try:
                encoded = self.encode_to_base64(image_path)
                encoded_images.append(encoded)
            except Exception as e:
                print(f"Warning: Error codificando {image_path}: {e}", file=sys.stderr)
                encoded_images.append("")  # Placeholder para mantener el orden
        
        return encoded_images


def safe_encode_image(image_path: Union[str, Path]) -> str:
    """
    Función de conveniencia para codificar imágenes de forma segura.
    
    Args:
        image_path: Ruta de la imagen
        
    Returns:
        str: Imagen codificada en base64 o cadena vacía si hay error
    """
    try:
        return ImageProcessor.encode_to_base64(image_path)
    except Exception as e:
        print(f"Error codificando imagen {image_path}: {e}", file=sys.stderr)
        return ""
