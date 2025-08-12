#!/usr/bin/env python3
"""
Cliente modular para comunicación con servidor Ollama.

Este módulo encapsula toda la lógica de comunicación con el servidor Ollama,
incluyendo verificación de conectividad y generación de respuestas.
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any

import requests


class OllamaClient:
    """Cliente para interactuar con el servidor Ollama."""
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 600):
        """
        Inicializa el cliente Ollama.
        
        Args:
            host: URL del servidor Ollama
            timeout: Timeout por defecto en segundos
        """
        self.host = host.rstrip('/')
        self.timeout = timeout
        self._session = requests.Session()
    
    def is_server_available(self, tries: int = 10, delay: float = 1.0) -> bool:
        """
        Verifica si el servidor Ollama está disponible.
        
        Args:
            tries: Número de intentos
            delay: Retraso entre intentos en segundos
            
        Returns:
            bool: True si el servidor responde
        """
        url = f"{self.host}/api/version"
        
        for _ in range(tries):
            try:
                response = self._session.get(url, timeout=2)
                if response.ok:
                    return True
            except Exception:
                time.sleep(delay)
        
        return False
    
    def generate_response(
        self,
        model: str,
        prompt: str,
        images: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        Genera una respuesta del modelo.
        
        Args:
            model: Nombre del modelo
            prompt: Texto de entrada
            images: Lista de imágenes codificadas en base64
            options: Opciones de generación
            stream: Si hacer streaming de la respuesta
            
        Returns:
            Tuple de (respuesta, estadísticas, tiempo_transcurrido)
            
        Raises:
            requests.RequestException: Si hay error en la petición
        """
        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if options:
            payload["options"] = options
        
        if images:
            payload["images"] = images

        start_time = time.perf_counter()
        
        if stream:
            return self._handle_streaming_response(url, payload)
        else:
            return self._handle_single_response(url, payload, start_time)
    
    def _handle_streaming_response(
        self, 
        url: str, 
        payload: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        Maneja respuesta de streaming.
        
        Args:
            url: URL de la API
            payload: Datos de la petición
            
        Returns:
            Tuple de (respuesta, estadísticas, tiempo_transcurrido)
        """
        start_time = time.perf_counter()
        text_chunks = []
        stats = {}
        
        with self._session.post(url, json=payload, stream=True, 
                               timeout=self.timeout) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                data = json.loads(line)
                
                if "response" in data:
                    chunk = data["response"]
                    print(chunk, end="", flush=True)
                    text_chunks.append(chunk)
                
                if data.get("done"):
                    stats = data
                    break
        
        print()  # Nueva línea al final del stream
        elapsed = time.perf_counter() - start_time
        
        return "".join(text_chunks), stats, elapsed
    
    def _handle_single_response(
        self, 
        url: str, 
        payload: Dict[str, Any], 
        start_time: float
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        Maneja respuesta única (no streaming).
        
        Args:
            url: URL de la API
            payload: Datos de la petición
            start_time: Tiempo de inicio
            
        Returns:
            Tuple de (respuesta, estadísticas, tiempo_transcurrido)
        """
        response = self._session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        elapsed = time.perf_counter() - start_time
        
        return data.get("response", ""), data, elapsed
    
    def warmup(self, model: str, prompt: str = "Hello.") -> bool:
        """
        Realiza warmup del modelo.
        
        Args:
            model: Nombre del modelo
            prompt: Prompt simple para warmup
            
        Returns:
            bool: True si el warmup fue exitoso
        """
        try:
            self.generate_response(model, prompt, stream=False)
            return True
        except Exception:
            return False
    
    def __enter__(self):
        """Soporte para context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cierra la sesión al salir del context manager."""
        self._session.close()
