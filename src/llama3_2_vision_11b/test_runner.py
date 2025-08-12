#!/usr/bin/env python3
"""
Ejecutor de tests para modelos de visión.

Este módulo proporciona la lógica principal para ejecutar tests
de rendimiento en modelos multimodales de forma estructurada.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .ollama_client import OllamaClient
from .image_utils import ImageProcessor, safe_encode_image
from .metrics import MetricsAnalyzer


class VisionTestRunner:
    """Ejecutor de tests para modelos de visión."""
    
    def __init__(self, host: str = "http://localhost:11434", 
                 timeout: int = 600):
        """
        Inicializa el ejecutor de tests.
        
        Args:
            host: URL del servidor Ollama
            timeout: Timeout por defecto
        """
        self.client = OllamaClient(host, timeout)
        self.image_processor = ImageProcessor()
        self.metrics_analyzer = MetricsAnalyzer()
        self.host = host
    
    def verify_setup(self) -> bool:
        """
        Verifica que el setup esté listo para ejecutar tests.
        
        Returns:
            bool: True si todo está listo
        """
        if not self.client.is_server_available():
            print(f"❌ Error: No se puede conectar con {self.host}", file=sys.stderr)
            print("   Ejecuta: sudo systemctl start ollama", file=sys.stderr)
            return False
        
        print(f"✅ Servidor Ollama disponible en {self.host}")
        return True
    
    def run_warmup(self, model: str) -> bool:
        """
        Ejecuta warmup del modelo.
        
        Args:
            model: Nombre del modelo
            
        Returns:
            bool: True si el warmup fue exitoso
        """
        print("🔥 Ejecutando warmup...")
        
        if self.client.warmup(model):
            print("✅ Warmup completado\n")
            return True
        else:
            print("⚠️  Warning: Warmup falló (continuamos)\n", file=sys.stderr)
            return False
    
    def run_text_only_test(self, 
                          model: str,
                          prompt: str,
                          options: Dict[str, Any],
                          runs: int = 3,
                          stream: bool = False) -> List[Dict[str, Any]]:
        """
        Ejecuta test solo con texto.
        
        Args:
            model: Nombre del modelo
            prompt: Texto de entrada
            options: Opciones de generación
            runs: Número de repeticiones
            stream: Si hacer streaming
            
        Returns:
            Lista de métricas por ejecución
        """
        print("\n📝 === TEST MODO TEXTO ===")
        print(f"Prompt: {prompt[:80]}...")
        
        metrics_list = []
        
        for run in range(1, runs + 1):
            print(f"\n🔄 Run {run}/{runs} (texto)")
            
            try:
                response, stats, elapsed = self.client.generate_response(
                    model, prompt, None, options, stream
                )
                
                if not stream and response:
                    print(f"💬 Respuesta: {response[:150]}...")
                
                # Procesar métricas
                metrics = self.metrics_analyzer.process_ollama_stats(stats, elapsed)
                metrics.update({"mode": "text", "run": run})
                metrics_list.append(metrics)
                
                # Mostrar estadísticas
                self._print_run_stats(metrics)
                self.metrics_analyzer.add_metrics(metrics)
                
            except Exception as e:
                print(f"❌ Error en run {run}: {e}", file=sys.stderr)
                continue
        
        return metrics_list
    
    def run_vision_test(self,
                       model: str, 
                       prompt: str,
                       image_path: Union[str, Path],
                       options: Dict[str, Any],
                       runs: int = 3,
                       stream: bool = False) -> List[Dict[str, Any]]:
        """
        Ejecuta test con imagen y texto.
        
        Args:
            model: Nombre del modelo
            prompt: Texto de entrada
            image_path: Ruta a la imagen
            options: Opciones de generación
            runs: Número de repeticiones
            stream: Si hacer streaming
            
        Returns:
            Lista de métricas por ejecución
        """
        print("\n🖼️  === TEST MODO VISIÓN ===")
        print(f"Imagen: {image_path}")
        print(f"Prompt: {prompt[:80]}...")
        
        # Validar y codificar imagen
        if not self.image_processor.validate_image_path(image_path):
            print(f"❌ Error: Imagen no válida: {image_path}", file=sys.stderr)
            return []
        
        image_base64 = safe_encode_image(image_path)
        if not image_base64:
            print("❌ Error: No se pudo codificar la imagen", file=sys.stderr)
            return []
        
        # Mostrar info de imagen
        image_info = self.image_processor.get_image_info(image_path)
        if "error" not in image_info:
            print(f"📊 Imagen: {image_info['name']} ({image_info['size_mb']} MB)")
        
        metrics_list = []
        
        for run in range(1, runs + 1):
            print(f"\n🔄 Run {run}/{runs} (visión)")
            
            try:
                response, stats, elapsed = self.client.generate_response(
                    model, prompt, [image_base64], options, stream
                )
                
                if not stream and response:
                    print(f"💬 Respuesta: {response[:150]}...")
                
                # Procesar métricas
                metrics = self.metrics_analyzer.process_ollama_stats(stats, elapsed)
                metrics.update({
                    "mode": "vision", 
                    "run": run,
                    "image_path": str(image_path)
                })
                metrics_list.append(metrics)
                
                # Mostrar estadísticas
                self._print_run_stats(metrics)
                print("   ⓘ  Procesamiento de imagen incluido en prefill")
                self.metrics_analyzer.add_metrics(metrics)
                
            except Exception as e:
                print(f"❌ Error en run {run}: {e}", file=sys.stderr)
                continue
        
        return metrics_list
    
    def run_comparison_test(self,
                           model: str,
                           text_prompt: str,
                           vision_prompt: str,
                           image_path: Union[str, Path],
                           options: Dict[str, Any],
                           runs: int = 3,
                           stream: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Ejecuta test comparativo entre modo texto y visión.
        
        Args:
            model: Nombre del modelo
            text_prompt: Prompt para modo texto
            vision_prompt: Prompt para modo visión
            image_path: Ruta a la imagen
            options: Opciones de generación
            runs: Número de repeticiones por modo
            stream: Si hacer streaming
            
        Returns:
            Dict con métricas de ambos modos
        """
        results = {
            "text_metrics": [],
            "vision_metrics": []
        }
        
        # Test modo texto
        results["text_metrics"] = self.run_text_only_test(
            model, text_prompt, options, runs, stream
        )
        
        # Test modo visión
        results["vision_metrics"] = self.run_vision_test(
            model, vision_prompt, image_path, options, runs, stream
        )
        
        # Mostrar comparación
        if results["text_metrics"] and results["vision_metrics"]:
            self._print_comparison(results["text_metrics"], results["vision_metrics"])
        
        return results
    
    def _print_run_stats(self, metrics: Dict[str, Any]) -> None:
        """
        Imprime estadísticas de un run.
        
        Args:
            metrics: Métricas del run
        """
        prefill_tps = metrics.get('prefill_tps')
        decode_tps = metrics.get('decode_tps')
        
        prefill_str = f"{prefill_tps:.1f}" if prefill_tps else "n/a"
        decode_str = f"{decode_tps:.1f}" if decode_tps else "n/a"
        
        print(f"   📊 wall={metrics['wall_time_s']:.2f}s | "
              f"prefill={metrics.get('prefill_tokens', 0)} tok @ {prefill_str} t/s | "
              f"decode={metrics.get('decode_tokens', 0)} tok @ {decode_str} t/s")
    
    def _print_comparison(self, text_metrics: List[Dict[str, Any]], 
                         vision_metrics: List[Dict[str, Any]]) -> None:
        """
        Imprime comparación entre modos.
        
        Args:
            text_metrics: Métricas del modo texto
            vision_metrics: Métricas del modo visión
        """
        print("\n" + "="*70)
        print("📊 COMPARACIÓN TEXTO vs VISIÓN")
        print("="*70)
        
        # Obtener estadísticas resumidas
        text_summary = self.metrics_analyzer.get_summary_stats("text")
        vision_summary = self.metrics_analyzer.get_summary_stats("vision")
        
        if "error" not in text_summary:
            print(f"\n📝 MODO TEXTO:")
            print(f"   Velocidad promedio: {text_summary['decode_tps']['mean']:.1f} t/s")
            print(f"   Tiempo promedio: {text_summary['wall_time_s']['mean']:.2f} s")
        
        if "error" not in vision_summary:
            print(f"\n🖼️  MODO VISIÓN:")
            print(f"   Velocidad promedio: {vision_summary['decode_tps']['mean']:.1f} t/s")
            print(f"   Tiempo promedio: {vision_summary['wall_time_s']['mean']:.2f} s")
            
            # Calcular overhead
            if "error" not in text_summary:
                overhead = (vision_summary['wall_time_s']['mean'] - 
                           text_summary['wall_time_s']['mean'])
                ratio = (text_summary['decode_tps']['mean'] / 
                        vision_summary['decode_tps']['mean'])
                
                print(f"   Overhead por imagen: ~{overhead:.1f} s")
                print(f"\n⚡ Factor velocidad (texto/visión): {ratio:.2f}x")
    
    def save_results(self, filepath: str, 
                    additional_fields: Optional[Dict[str, Any]] = None) -> bool:
        """
        Guarda resultados en archivo JSONL.
        
        Args:
            filepath: Ruta del archivo
            additional_fields: Campos adicionales
            
        Returns:
            bool: True si se guardó correctamente
        """
        return self.metrics_analyzer.save_to_jsonl(filepath, additional_fields)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de todas las métricas.
        
        Returns:
            Dict con resumen completo
        """
        return {
            "overall": self.metrics_analyzer.get_summary_stats(),
            "text_mode": self.metrics_analyzer.get_summary_stats("text"),
            "vision_mode": self.metrics_analyzer.get_summary_stats("vision")
        }
