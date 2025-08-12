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
from .system_monitor import SystemMonitor, get_instant_metrics


class VisionTestRunner:
    """Ejecutor de tests para modelos de visión."""
    
    def __init__(self, host: str = "http://localhost:11434", 
                 timeout: int = 600,
                 enable_system_monitoring: bool = True):
        """
        Inicializa el ejecutor de tests.
        
        Args:
            host: URL del servidor Ollama
            timeout: Timeout por defecto
            enable_system_monitoring: Si activar monitoreo del sistema
        """
        self.client = OllamaClient(host, timeout)
        self.image_processor = ImageProcessor()
        self.metrics_analyzer = MetricsAnalyzer()
        self.host = host
        
        # Sistema de monitoreo
        self.system_monitor = SystemMonitor() if enable_system_monitoring else None
        self.enable_system_monitoring = enable_system_monitoring
    
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
            
            # Capturar métricas pre-ejecución
            if self.system_monitor:
                pre_metrics = get_instant_metrics()
                cpu_str = f"{pre_metrics.cpu_percent:.1f}" if isinstance(pre_metrics.cpu_percent, (int, float)) else "n/a"
                ram_gb_str = f"{pre_metrics.ram_used_gb:.1f}" if isinstance(pre_metrics.ram_used_gb, (int, float)) else "n/a"
                ram_pct_str = f"{pre_metrics.ram_percent:.1f}" if isinstance(pre_metrics.ram_percent, (int, float)) else "n/a"
                print(f"[pre] CPU: {cpu_str}% | RAM: {ram_gb_str}GB ({ram_pct_str}%)")
                if pre_metrics.cpu_temp and isinstance(pre_metrics.cpu_temp, (int, float)):
                    cpu_temp_str = f"{pre_metrics.cpu_temp:.1f}"
                    print(f"[pre] CPU Temp: {cpu_temp_str}°C")
            
            try:
                # Iniciar monitoreo del sistema
                system_metrics_summary = None
                if self.system_monitor:
                    self.system_monitor.start_monitoring()
                
                response, stats, elapsed = self.client.generate_response(
                    model, prompt, None, options, stream
                )
                
                # Detener monitoreo y obtener resumen
                if self.system_monitor:
                    monitor_history = self.system_monitor.stop_monitoring()
                    system_metrics_summary = self.system_monitor.get_metrics_summary()
                
                if not stream and response:
                    print(f"💬 Respuesta: {response[:150]}...")
                
                # Procesar métricas combinadas
                metrics = self.metrics_analyzer.process_ollama_stats(
                    stats, elapsed, system_metrics_summary
                )
                metrics.update({"mode": "text", "run": run})
                metrics_list.append(metrics)
                
                # Mostrar estadísticas
                self._print_run_stats(metrics, system_metrics_summary)
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
            
            # Capturar métricas pre-ejecución
            if self.system_monitor:
                pre_metrics = get_instant_metrics()
                cpu_str = f"{pre_metrics.cpu_percent:.1f}" if isinstance(pre_metrics.cpu_percent, (int, float)) else "n/a"
                ram_gb_str = f"{pre_metrics.ram_used_gb:.1f}" if isinstance(pre_metrics.ram_used_gb, (int, float)) else "n/a"
                ram_pct_str = f"{pre_metrics.ram_percent:.1f}" if isinstance(pre_metrics.ram_percent, (int, float)) else "n/a"
                print(f"[pre] CPU: {cpu_str}% | RAM: {ram_gb_str}GB ({ram_pct_str}%)")
                if pre_metrics.cpu_temp and isinstance(pre_metrics.cpu_temp, (int, float)):
                    cpu_temp_str = f"{pre_metrics.cpu_temp:.1f}"
                    print(f"[pre] CPU Temp: {cpu_temp_str}°C")
                if pre_metrics.gpu_usage_percent and isinstance(pre_metrics.gpu_usage_percent, (int, float)):
                    gpu_str = f"{pre_metrics.gpu_usage_percent:.1f}"
                    print(f"[pre] GPU: {gpu_str}%")
            
            try:
                # Iniciar monitoreo del sistema
                system_metrics_summary = None
                if self.system_monitor:
                    self.system_monitor.start_monitoring()
                
                response, stats, elapsed = self.client.generate_response(
                    model, prompt, [image_base64], options, stream
                )
                
                # Detener monitoreo y obtener resumen
                if self.system_monitor:
                    monitor_history = self.system_monitor.stop_monitoring()
                    system_metrics_summary = self.system_monitor.get_metrics_summary()
                
                if not stream and response:
                    print(f"💬 Respuesta: {response[:150]}...")
                
                # Procesar métricas combinadas
                metrics = self.metrics_analyzer.process_ollama_stats(
                    stats, elapsed, system_metrics_summary
                )
                metrics.update({
                    "mode": "vision", 
                    "run": run,
                    "image_path": str(image_path)
                })
                metrics_list.append(metrics)
                
                # Mostrar estadísticas
                self._print_run_stats(metrics, system_metrics_summary)
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
    
    def _print_run_stats(self, metrics: Dict[str, Any], 
                        system_metrics_summary: Optional[Dict[str, Any]] = None) -> None:
        """
        Imprime estadísticas de un run incluyendo métricas del sistema.
        
        Args:
            metrics: Métricas del run
            system_metrics_summary: Resumen de métricas del sistema (opcional)
        """
        prefill_tps = metrics.get('prefill_tps')
        decode_tps = metrics.get('decode_tps')
        
        prefill_str = f"{prefill_tps:.1f}" if prefill_tps and isinstance(prefill_tps, (int, float)) else "n/a"
        decode_str = f"{decode_tps:.1f}" if decode_tps and isinstance(decode_tps, (int, float)) else "n/a"
        
        # Métricas básicas de Ollama
        wall_time = metrics.get('wall_time_s', 0)
        wall_str = f"{wall_time:.2f}" if isinstance(wall_time, (int, float)) else str(wall_time)
        print(f"   📊 [ollama] wall={wall_str}s | "
              f"prefill={metrics.get('prefill_tokens', 0)} tok @ {prefill_str} t/s | "
              f"decode={metrics.get('decode_tokens', 0)} tok @ {decode_str} t/s")
        
        # Mostrar métricas del sistema si están disponibles
        if system_metrics_summary and self.enable_system_monitoring:
            cpu_mean = metrics.get('cpu_usage_mean_percent')
            cpu_max = metrics.get('cpu_usage_max_percent')
            ram_mean = metrics.get('ram_usage_mean_percent')
            ram_max = metrics.get('ram_usage_max_percent')
            
            if cpu_mean is not None and isinstance(cpu_mean, (int, float)):
                cpu_mean_str = f"{cpu_mean:.1f}"
                cpu_max_str = f"{cpu_max:.1f}" if isinstance(cpu_max, (int, float)) else str(cpu_max)
                print(f"   📊 [system] CPU: {cpu_mean_str}% avg, {cpu_max_str}% max")
            if ram_mean is not None and isinstance(ram_mean, (int, float)):
                ram_mean_str = f"{ram_mean:.1f}"
                ram_max_str = f"{ram_max:.1f}" if isinstance(ram_max, (int, float)) else str(ram_max)
                print(f"   📊 [system] RAM: {ram_mean_str}% avg, {ram_max_str}% max")
                
            # Métricas de temperatura si están disponibles
            cpu_temp_mean = metrics.get('cpu_temp_mean_c')
            if cpu_temp_mean is not None and isinstance(cpu_temp_mean, (int, float)):
                cpu_temp_mean_str = f"{cpu_temp_mean:.1f}"
                cpu_temp_max = metrics.get('cpu_temp_max_c')
                cpu_temp_max_str = f"{cpu_temp_max:.1f}" if isinstance(cpu_temp_max, (int, float)) else str(cpu_temp_max)
                print(f"   📊 [system] CPU Temp: {cpu_temp_mean_str}°C avg, {cpu_temp_max_str}°C max")
                
            # Métricas de GPU si están disponibles  
            gpu_mean = metrics.get('gpu_usage_mean_percent')
            if gpu_mean is not None and isinstance(gpu_mean, (int, float)):
                gpu_mean_str = f"{gpu_mean:.1f}"
                gpu_max = metrics.get('gpu_usage_max_percent')
                gpu_max_str = f"{gpu_max:.1f}" if isinstance(gpu_max, (int, float)) else str(gpu_max)
                print(f"   📊 [system] GPU: {gpu_mean_str}% avg, {gpu_max_str}% max")
                
            # Métricas de potencia si están disponibles
            power_mean = metrics.get('power_mean_watts')
            if power_mean is not None and isinstance(power_mean, (int, float)):
                power_mean_str = f"{power_mean:.1f}"
                power_max = metrics.get('power_max_watts')
                power_max_str = f"{power_max:.1f}" if isinstance(power_max, (int, float)) else str(power_max)
                print(f"   📊 [system] Power: {power_mean_str}W avg, {power_max_str}W max")
            
            monitoring_duration = metrics.get('monitoring_duration_s')
            if monitoring_duration is not None and isinstance(monitoring_duration, (int, float)):
                monitoring_duration_str = f"{monitoring_duration:.1f}"
                print(f"   📊 [system] Monitoring: {monitoring_duration_str}s, {metrics.get('monitoring_samples')} samples")
    
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
            text_decode_mean = text_summary.get('decode_tps', {}).get('mean')
            text_wall_mean = text_summary.get('wall_time_s', {}).get('mean')
            text_decode_str = f"{text_decode_mean:.1f}" if isinstance(text_decode_mean, (int, float)) else "n/a"
            text_wall_str = f"{text_wall_mean:.2f}" if isinstance(text_wall_mean, (int, float)) else "n/a"
            print(f"   Velocidad promedio: {text_decode_str} t/s")
            print(f"   Tiempo promedio: {text_wall_str} s")
        
        if "error" not in vision_summary:
            print(f"\n🖼️  MODO VISIÓN:")
            vision_decode_mean = vision_summary.get('decode_tps', {}).get('mean')
            vision_wall_mean = vision_summary.get('wall_time_s', {}).get('mean')
            vision_decode_str = f"{vision_decode_mean:.1f}" if isinstance(vision_decode_mean, (int, float)) else "n/a"
            vision_wall_str = f"{vision_wall_mean:.2f}" if isinstance(vision_wall_mean, (int, float)) else "n/a"
            print(f"   Velocidad promedio: {vision_decode_str} t/s")
            print(f"   Tiempo promedio: {vision_wall_str} s")
            
            # Calcular overhead si ambos valores están disponibles
            if ("error" not in text_summary and 
                isinstance(text_wall_mean, (int, float)) and isinstance(vision_wall_mean, (int, float)) and
                isinstance(text_decode_mean, (int, float)) and isinstance(vision_decode_mean, (int, float))):
                overhead = vision_wall_mean - text_wall_mean
                ratio = text_decode_mean / vision_decode_mean
                
                overhead_str = f"{overhead:.1f}" if isinstance(overhead, (int, float)) else "n/a"
                ratio_str = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "n/a"
                
                print(f"   Overhead por imagen: ~{overhead_str} s")
                print(f"\n⚡ Factor velocidad (texto/visión): {ratio_str}x")
    
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
