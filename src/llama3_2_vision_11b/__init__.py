"""
Módulo de testing para Llama 3.2 Vision 11B en Jetson AGX Orin.

Este paquete proporciona herramientas modularizadas para evaluar el rendimiento
del modelo multimodal llama3.2-vision:11b.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"

# Imports principales para facilitar el uso
from .ollama_client import OllamaClient
from .image_utils import ImageProcessor
from .metrics import MetricsAnalyzer
from .test_runner import VisionTestRunner
from .sweep_runner import SweepRunner, ParameterCombinationGenerator, PromptManager

__all__ = [
    "OllamaClient",
    "ImageProcessor", 
    "MetricsAnalyzer",
    "VisionTestRunner",
    "SweepRunner",
    "ParameterCombinationGenerator", 
    "PromptManager"
]
