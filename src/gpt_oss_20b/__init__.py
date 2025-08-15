"""
Módulo de benchmarking para el modelo GPT-OSS 20B
GPT-OSS 20B - Modelo de lenguaje de código abierto de 20 mil millones de parámetros
"""

__version__ = "1.0.0"
__model__ = "gpt-oss:20b"
__description__ = "Benchmarking suite for GPT-OSS 20B model on Jetson AGX Orin"

# Configuración por defecto del modelo
DEFAULT_MODEL_CONFIG = {
    "model_name": "gpt-oss:20b",
    "default_context": 8192,
    "default_temperature": 0.7,
    "default_num_predict": 256,
    "model_type": "text",  # no multimodal
    "recommended_memory_gb": 16,
    "supports_streaming": True
}
