#!/usr/bin/env python3
"""
Ejecutor de barridos paramétricos para modelos de visión.

Este módulo proporciona la funcionalidad para ejecutar tests sistemáticos
variando múltiples parámetros y analizando el rendimiento resultante.
"""

import csv
import hashlib
import itertools
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

from .ollama_client import OllamaClient
from .image_utils import ImageProcessor
from .metrics import MetricsAnalyzer
from .system_monitor import SystemMonitor, get_instant_metrics


class ParameterCombinationGenerator:
    """Generador de combinaciones de parámetros para barridos."""
    
    @staticmethod
    def parse_int_list(value: str) -> List[int]:
        """
        Parsea lista de enteros separados por comas.
        
        Args:
            value: String con enteros separados por comas
            
        Returns:
            Lista de enteros
        """
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    
    @staticmethod
    def parse_float_list(value: str) -> List[float]:
        """
        Parsea lista de floats separados por comas.
        
        Args:
            value: String con floats separados por comas
            
        Returns:
            Lista de floats
        """
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    
    @staticmethod
    def parse_seed_list(value: str) -> List[Optional[int]]:
        """
        Parsea lista de semillas.
        
        Args:
            value: String con semillas separadas por comas
            
        Returns:
            Lista de semillas (puede incluir None)
        """
        if not value.strip():
            return [None]
        
        seeds = []
        for seed_str in value.split(","):
            seed_str = seed_str.strip()
            if seed_str:
                seeds.append(int(seed_str))
            else:
                seeds.append(None)
        
        return seeds or [None]


class PromptManager:
    """Gestor de prompts para barridos."""
    
    @staticmethod
    def load_prompts(prompt: Optional[str] = None, 
                    prompt_file: Optional[str] = None) -> List[str]:
        """
        Carga prompts desde archivo y/o argumento.
        
        Args:
            prompt: Prompt único
            prompt_file: Archivo con prompts
            
        Returns:
            Lista de prompts
        """
        prompts = []
        
        # Cargar desde archivo
        if prompt_file and Path(prompt_file).exists():
            with open(prompt_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        prompts.append(line)
        
        # Añadir prompt individual
        if prompt:
            prompts.append(prompt)
        
        # Prompts por defecto si no hay ninguno
        if not prompts:
            prompts = [
                "Describe esta imagen en detalle.",
                "¿Qué objetos puedes identificar en esta imagen?", 
                "Analiza los colores y la composición de esta imagen.",
                "Resume en 5 líneas las capacidades del modelo Llama 3.2 Vision."
            ]
        
        return prompts
    
    @staticmethod
    def create_prompt_hash(prompt: str, length: int = 8) -> str:
        """
        Crea hash corto de un prompt para identificación.
        
        Args:
            prompt: Prompt a hashear
            length: Longitud del hash
            
        Returns:
            Hash hexadecimal corto
        """
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:length]


class SweepRunner:
    """Ejecutor principal de barridos paramétricos."""
    
    def __init__(self, host: str = "http://localhost:11434", 
                 timeout: int = 600,
                 enable_system_monitoring: bool = True):
        """
        Inicializa el ejecutor de barridos.
        
        Args:
            host: URL del servidor Ollama
            timeout: Timeout por defecto
            enable_system_monitoring: Si activar monitoreo del sistema
        """
        self.client = OllamaClient(host, timeout)
        self.image_processor = ImageProcessor()
        self.metrics_analyzer = MetricsAnalyzer()
        self.param_generator = ParameterCombinationGenerator()
        self.prompt_manager = PromptManager()
        
        # Sistema de monitoreo
        self.system_monitor = SystemMonitor() if enable_system_monitoring else None
        self.enable_system_monitoring = enable_system_monitoring
        
        # Estadísticas del barrido
        self.results_by_mode = {"text": [], "vision": []}
        self.total_jobs = 0
        self.completed_jobs = 0
        self.start_time = None
        
    def verify_setup(self) -> bool:
        """
        Verifica que el setup esté listo.
        
        Returns:
            bool: True si todo está configurado correctamente
        """
        return self.client.is_server_available()
    
    def load_images(self, image_path: Optional[str] = None,
                   image_dir: Optional[str] = None) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Carga imágenes para el barrido.
        
        Args:
            image_path: Ruta a imagen única
            image_dir: Directorio con imágenes
            
        Returns:
            Lista de tuplas (ruta, base64). Incluye (None, None) si no hay imágenes
        """
        images = []
        
        # Imagen única
        if image_path and self.image_processor.validate_image_path(image_path):
            try:
                encoded = self.image_processor.encode_to_base64(image_path)
                images.append((image_path, encoded))
            except Exception as e:
                print(f"Warning: Error cargando imagen {image_path}: {e}")
        
        # Directorio de imágenes  
        if image_dir:
            image_dir_path = Path(image_dir)
            if image_dir_path.is_dir():
                for img_file in image_dir_path.glob("*"):
                    if self.image_processor.validate_image_path(img_file):
                        try:
                            encoded = self.image_processor.encode_to_base64(img_file)
                            images.append((str(img_file), encoded))
                        except Exception as e:
                            print(f"Warning: Error cargando {img_file}: {e}")
        
        # Buscar imagen por defecto si no se encontró ninguna
        if not images:
            default_image = self.image_processor.find_default_image()
            if default_image:
                try:
                    encoded = self.image_processor.encode_to_base64(default_image)
                    images.append((str(default_image), encoded))
                except Exception:
                    pass
        
        # Siempre incluir modo texto (sin imagen)
        if not any(img[0] is None for img in images):
            images.append((None, None))
        
        return images
    
    def generate_combinations(self, 
                            prompts: List[str],
                            images: List[Tuple[Optional[str], Optional[str]]],
                            contexts: List[int],
                            num_predicts: List[int], 
                            temperatures: List[float],
                            seeds: List[Optional[int]],
                            test_mode: str) -> List[Tuple]:
        """
        Genera todas las combinaciones de parámetros según el modo de test.
        
        Args:
            prompts: Lista de prompts
            images: Lista de imágenes
            contexts: Tamaños de contexto
            num_predicts: Números de tokens a predecir
            temperatures: Temperaturas
            seeds: Semillas
            test_mode: "text", "vision", o "both"
            
        Returns:
            Lista de combinaciones como tuplas
        """
        if test_mode == "text":
            # Solo modo texto
            image_subset = [(None, None)]
            
        elif test_mode == "vision":
            # Solo modo visión, requiere al menos una imagen
            image_subset = [(path, b64) for path, b64 in images if path is not None]
            if not image_subset:
                raise ValueError("Modo visión requiere al menos una imagen")
                
        else:  # "both"
            # Todos los modos
            image_subset = images
        
        combinations = list(itertools.product(
            prompts, image_subset, contexts, num_predicts, temperatures, seeds
        ))
        
        return combinations
    
    def setup_csv_output(self, csv_path: str) -> None:
        """
        Configura archivo CSV de salida con headers incluyendo métricas del sistema.
        
        Args:
            csv_path: Ruta del archivo CSV
        """
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            base_headers = [
                "timestamp", "host", "model", "mode", "prompt_hash", "prompt_length",
                "image_path", "context", "num_predict", "temperature", "seed",
                "cycle", "run", "wall_time_s", "total_duration_s", "load_duration_s", 
                "prefill_tokens", "decode_tokens", "prefill_tps", "decode_tps"
            ]
            
            # Añadir cabeceras de métricas del sistema si está habilitado
            if self.enable_system_monitoring:
                system_headers = [
                    "cpu_usage_mean_percent", "cpu_usage_max_percent", "cpu_usage_min_percent",
                    "ram_usage_mean_percent", "ram_usage_max_percent",
                    "ram_used_mean_gb", "ram_used_max_gb",
                    "gpu_usage_mean_percent", "gpu_usage_max_percent", "gpu_usage_min_percent",
                    "cpu_temp_mean_c", "cpu_temp_max_c",
                    "gpu_temp_mean_c", "gpu_temp_max_c",
                    "power_mean_watts", "power_max_watts",
                    "monitoring_duration_s", "monitoring_samples"
                ]
                base_headers.extend(system_headers)
            
            writer.writerow(base_headers)
    
    def execute_single_run(self,
                          model: str,
                          prompt: str, 
                          image_base64: Optional[str],
                          options: Dict[str, Any],
                          seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecuta una sola ejecución del modelo con monitoreo del sistema.
        
        Args:
            model: Nombre del modelo
            prompt: Prompt de entrada
            image_base64: Imagen codificada (opcional)
            options: Opciones del modelo
            seed: Semilla (opcional)
            
        Returns:
            Dict con métricas procesadas combinadas
        """
        images = [image_base64] if image_base64 else None
        
        # Iniciar monitoreo del sistema si está habilitado
        system_metrics_summary = None
        if self.system_monitor:
            self.system_monitor.start_monitoring()
        
        response, stats, wall_time = self.client.generate_response(
            model, prompt, images, options, stream=False
        )
        
        # Detener monitoreo y obtener resumen
        if self.system_monitor:
            monitor_history = self.system_monitor.stop_monitoring()
            system_metrics_summary = self.system_monitor.get_metrics_summary()
        
        return self.metrics_analyzer.process_ollama_stats(
            stats, wall_time, system_metrics_summary
        )
    
    def run_sweep(self,
                 model: str,
                 prompts: List[str],
                 images: List[Tuple[Optional[str], Optional[str]]],
                 contexts: List[int],
                 num_predicts: List[int],
                 temperatures: List[float], 
                 seeds: List[Optional[int]],
                 test_mode: str = "both",
                 runs_per_combo: int = 3,
                 cycles: int = 1,
                 warmup: bool = False,
                 sleep_time: float = 1.0,
                 jsonl_output: Optional[str] = None,
                 csv_output: Optional[str] = None) -> Dict[str, Any]:
        """
        Ejecuta el barrido paramétrico completo.
        
        Args:
            model: Nombre del modelo
            prompts: Lista de prompts
            images: Lista de imágenes
            contexts: Tamaños de contexto
            num_predicts: Tokens a predecir
            temperatures: Temperaturas
            seeds: Semillas
            test_mode: Modo de test
            runs_per_combo: Runs por combinación
            cycles: Ciclos del barrido completo
            warmup: Si hacer warmup
            sleep_time: Pausa entre runs
            jsonl_output: Archivo JSONL de salida
            csv_output: Archivo CSV de salida
            
        Returns:
            Dict con resultados del barrido
        """
        # Generar combinaciones
        combinations = self.generate_combinations(
            prompts, images, contexts, num_predicts, temperatures, seeds, test_mode
        )
        
        self.total_jobs = len(combinations) * runs_per_combo * cycles
        self.completed_jobs = 0
        self.start_time = time.time()
        
        print(f"🚀 Iniciando barrido: {len(combinations)} combinaciones × "
              f"{runs_per_combo} runs × {cycles} ciclos = {self.total_jobs} ejecuciones\n")
        
        # Configurar salidas
        if csv_output:
            self.setup_csv_output(csv_output)
        
        # Diccionario para agregados por combinación
        combination_results = {}
        
        for cycle in range(1, cycles + 1):
            for prompt, (img_path, img_b64), ctx, num_predict, temp, seed in combinations:
                
                # Crear clave única para esta combinación
                prompt_hash = self.prompt_manager.create_prompt_hash(prompt)
                combo_key = (prompt_hash, img_path, ctx, num_predict, temp, seed)
                
                if combo_key not in combination_results:
                    combination_results[combo_key] = []
                
                # Configurar opciones del modelo
                model_options = {
                    "num_ctx": ctx,
                    "temperature": temp,
                    "num_predict": num_predict
                }
                
                mode = "vision" if img_path else "text"
                
                # Warmup si está habilitado
                if warmup:
                    try:
                        self.client.generate_response(
                            model, "Warmup.", 
                            [img_b64] if img_b64 else None, 
                            model_options, stream=False
                        )
                    except Exception as e:
                        print(f"Warning: Warmup error: {e}")
                
                # Ejecutar runs para esta combinación
                for run in range(1, runs_per_combo + 1):
                    self.completed_jobs += 1
                    
                    try:
                        # Ejecutar modelo
                        metrics = self.execute_single_run(
                            model, prompt, img_b64, model_options, seed
                        )
                        
                        # Añadir metadata
                        metrics.update({
                            "model": model,
                            "mode": mode,
                            "prompt_hash": prompt_hash,
                            "prompt_length": len(prompt),
                            "image_path": img_path,
                            "context": ctx,
                            "num_predict": num_predict,
                            "temperature": temp,
                            "seed": seed,
                            "cycle": cycle,
                            "run": run
                        })
                        
                        # Almacenar para análisis
                        combination_results[combo_key].append(metrics["decode_tps"])
                        self.results_by_mode[mode].append(metrics["decode_tps"])
                        
                        # Mostrar progreso
                        self._print_progress(metrics, self.completed_jobs, self.total_jobs)
                        
                        # Guardar resultados
                        self._save_results(metrics, jsonl_output, csv_output)
                        
                        # Pausa entre runs
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                            
                    except Exception as e:
                        print(f"❌ Error en job {self.completed_jobs}: {e}")
                        import traceback
                        print("\n🔍 TRACEBACK COMPLETO:")
                        traceback.print_exc()
                        print("-" * 60)
                        continue
        
        # Generar resumen
        return self._generate_summary(combination_results)
    
    def _print_progress(self, metrics: Dict[str, Any], 
                       current_job: int, total_jobs: int) -> None:
        """Imprime progreso de una ejecución con métricas del sistema."""
        pf_tps = metrics.get('prefill_tps', 0)
        dc_tps = metrics.get('decode_tps', 0)
        
        # Formatear datos con verificación de tipo
        cycle = metrics.get('cycle', '?')
        run = metrics.get('run', '?')
        mode = metrics.get('mode', '?')
        context = metrics.get('context', '?')
        num_predict = metrics.get('num_predict', '?')
        temperature = metrics.get('temperature', '?')
        seed = metrics.get('seed', 'none')
        
        print(f"[{current_job}/{total_jobs}] cycle={cycle} run={run} mode={mode}")
        print(f"  ctx={context} np={num_predict} temp={temperature} seed={seed}")
        
        image_path = metrics.get('image_path')
        if image_path:
            try:
                img_name = Path(image_path).name
                print(f"  img={img_name}")
            except Exception:
                print(f"  img={image_path}")
        else:
            print(f"  modo texto puro")
            
        # Formatear valores de manera segura
        pf_tps_str = f"{pf_tps:.1f}" if (pf_tps and isinstance(pf_tps, (int, float))) else "0"
        dc_tps_str = f"{dc_tps:.1f}" if (dc_tps and isinstance(dc_tps, (int, float))) else "0"
        
        print(f"  [ollama] prefill={metrics.get('prefill_tokens', 0)} @ "
              f"{pf_tps_str} t/s | "
              f"decode={metrics.get('decode_tokens', 0)} @ "
              f"{dc_tps_str} t/s")
        
        wall_time = metrics.get('wall_time_s', 0)
        if isinstance(wall_time, (int, float)):
            print(f"  [ollama] wall={wall_time:.2f}s")
        else:
            print(f"  [ollama] wall={wall_time}s")
        
        # Mostrar métricas del sistema si están disponibles
        if self.enable_system_monitoring:
            cpu_mean = metrics.get('cpu_usage_mean_percent')
            ram_mean = metrics.get('ram_usage_mean_percent')
            gpu_mean = metrics.get('gpu_usage_mean_percent')
            
            system_info = []
            if cpu_mean is not None and isinstance(cpu_mean, (int, float)):
                cpu_mean_str = f"{cpu_mean:.1f}"
                system_info.append(f"CPU:{cpu_mean_str}%")
            if ram_mean is not None and isinstance(ram_mean, (int, float)):
                ram_mean_str = f"{ram_mean:.1f}"
                system_info.append(f"RAM:{ram_mean_str}%")
            if gpu_mean is not None and isinstance(gpu_mean, (int, float)):
                gpu_mean_str = f"{gpu_mean:.1f}"
                system_info.append(f"GPU:{gpu_mean_str}%")
                
            # Añadir temperatura si está disponible
            cpu_temp = metrics.get('cpu_temp_mean_c')
            if cpu_temp is not None and isinstance(cpu_temp, (int, float)):
                cpu_temp_str = f"{cpu_temp:.1f}"
                system_info.append(f"T:{cpu_temp_str}°C")
                
            # Añadir potencia si está disponible
            power = metrics.get('power_mean_watts')
            if power is not None and isinstance(power, (int, float)):
                power_str = f"{power:.1f}"
                system_info.append(f"P:{power_str}W")
            
            if system_info:
                print(f"  [system] {' '.join(system_info)}")
        
        print()  # línea en blanco
    
    def _save_results(self, metrics: Dict[str, Any], 
                     jsonl_path: Optional[str], 
                     csv_path: Optional[str]) -> None:
        """Guarda resultados en archivos."""
        timestamp = time.time()
        
        if jsonl_path:
            record = {"timestamp": timestamp, **metrics}
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        
        if csv_path:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Función auxiliar para formateo seguro
                def safe_round(value, decimals=3):
                    if value is None or not isinstance(value, (int, float)):
                        return 0
                    return round(value, decimals)
                
                base_row = [
                    timestamp, self.client.host, 
                    metrics.get("model", ""), metrics.get("mode", ""),
                    metrics.get("prompt_hash", ""), metrics.get("prompt_length", 0),
                    metrics.get("image_path", ""), metrics.get("context", 0),
                    metrics.get("num_predict", 0), metrics.get("temperature", 0), 
                    metrics.get("seed", ""), metrics.get("cycle", 0), metrics.get("run", 0),
                    safe_round(metrics.get("wall_time_s"), 3),
                    safe_round(metrics.get("total_duration_s"), 3),
                    safe_round(metrics.get("load_duration_s"), 3),
                    metrics.get("prefill_tokens", 0),
                    metrics.get("decode_tokens", 0),
                    safe_round(metrics.get("prefill_tps"), 2),
                    safe_round(metrics.get("decode_tps"), 2)
                ]
                
                # Añadir métricas del sistema si está habilitado
                if self.enable_system_monitoring:
                    system_row = [
                        safe_round(metrics.get("cpu_usage_mean_percent"), 2),
                        safe_round(metrics.get("cpu_usage_max_percent"), 2),
                        safe_round(metrics.get("cpu_usage_min_percent"), 2),
                        safe_round(metrics.get("ram_usage_mean_percent"), 2),
                        safe_round(metrics.get("ram_usage_max_percent"), 2),
                        safe_round(metrics.get("ram_used_mean_gb"), 2),
                        safe_round(metrics.get("ram_used_max_gb"), 2),
                        safe_round(metrics.get("gpu_usage_mean_percent"), 2),
                        safe_round(metrics.get("gpu_usage_max_percent"), 2),
                        safe_round(metrics.get("gpu_usage_min_percent"), 2),
                        safe_round(metrics.get("cpu_temp_mean_c"), 1),
                        safe_round(metrics.get("cpu_temp_max_c"), 1),
                        safe_round(metrics.get("gpu_temp_mean_c"), 1),
                        safe_round(metrics.get("gpu_temp_max_c"), 1),
                        safe_round(metrics.get("power_mean_watts"), 2),
                        safe_round(metrics.get("power_max_watts"), 2),
                        safe_round(metrics.get("monitoring_duration_s"), 1),
                        metrics.get("monitoring_samples", 0)
                    ]
                    base_row.extend(system_row)
                
                writer.writerow(base_row)
    
    def _generate_summary(self, combination_results: Dict) -> Dict[str, Any]:
        """Genera resumen final del barrido."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Estadísticas por modo
        text_stats = self._calculate_mode_stats("text")
        vision_stats = self._calculate_mode_stats("vision") 
        
        # Resumen por combinación
        combo_summary = []
        for combo_key, speeds in combination_results.items():
            if speeds:
                valid_speeds = [s for s in speeds if s and s > 0]
                if valid_speeds:
                    combo_summary.append({
                        "combination": combo_key,
                        "mean_speed": statistics.mean(valid_speeds),
                        "median_speed": statistics.median(valid_speeds),
                        "samples": len(valid_speeds)
                    })
        
        return {
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "elapsed_time_minutes": elapsed_time / 60,
            "text_mode": text_stats,
            "vision_mode": vision_stats, 
            "combinations": combo_summary,
            "speed_comparison": self._compare_modes(text_stats, vision_stats)
        }
    
    def _calculate_mode_stats(self, mode: str) -> Dict[str, Any]:
        """Calcula estadísticas para un modo específico."""
        speeds = self.results_by_mode.get(mode, [])
        valid_speeds = [s for s in speeds if s and s > 0]
        
        if not valid_speeds:
            return {"samples": 0, "error": "No hay datos válidos"}
        
        # Función auxiliar para estadísticas seguras
        def safe_stats_round(value, decimals=2):
            try:
                if value is None or not isinstance(value, (int, float)):
                    return 0.0
                return round(value, decimals)
            except (TypeError, ValueError):
                return 0.0
        
        try:
            return {
                "samples": len(valid_speeds),
                "mean": safe_stats_round(statistics.mean(valid_speeds), 2),
                "median": safe_stats_round(statistics.median(valid_speeds), 2),
                "min": safe_stats_round(min(valid_speeds), 2),
                "max": safe_stats_round(max(valid_speeds), 2),
                "stdev": safe_stats_round(statistics.stdev(valid_speeds), 2) if len(valid_speeds) > 1 else 0
            }
        except Exception:
            return {"samples": 0, "error": "Error calculando estadísticas"}
    
    def _compare_modes(self, text_stats: Dict, vision_stats: Dict) -> Dict[str, Any]:
        """Compara estadísticas entre modos texto y visión."""
        if "error" in text_stats or "error" in vision_stats:
            return {"error": "No hay suficientes datos para comparar"}
        
        if text_stats["mean"] > 0 and vision_stats["mean"] > 0:
            ratio = text_stats["mean"] / vision_stats["mean"]
            # Usar la función safe_stats_round definida anteriormente
            def safe_stats_round_local(value, decimals=2):
                try:
                    if value is None or not isinstance(value, (int, float)):
                        return 0.0
                    return round(value, decimals)
                except (TypeError, ValueError):
                    return 0.0
                    
            return {
                "text_vs_vision_ratio": safe_stats_round_local(ratio, 2),
                "text_faster_by_percent": safe_stats_round_local((ratio - 1) * 100, 1),
                "faster_mode": "text" if ratio > 1 else "vision"
            }
        
        return {"error": "No se pueden calcular ratios"}
