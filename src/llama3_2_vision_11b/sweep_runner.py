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
                 timeout: int = 600):
        """
        Inicializa el ejecutor de barridos.
        
        Args:
            host: URL del servidor Ollama
            timeout: Timeout por defecto
        """
        self.client = OllamaClient(host, timeout)
        self.image_processor = ImageProcessor()
        self.metrics_analyzer = MetricsAnalyzer()
        self.param_generator = ParameterCombinationGenerator()
        self.prompt_manager = PromptManager()
        
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
        Configura archivo CSV de salida con headers.
        
        Args:
            csv_path: Ruta del archivo CSV
        """
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "host", "model", "mode", "prompt_hash", "prompt_length",
                "image_path", "context", "num_predict", "temperature", "seed",
                "cycle", "run", "wall_time_s", "total_duration_s", "load_duration_s", 
                "prefill_tokens", "decode_tokens", "prefill_tps", "decode_tps"
            ])
    
    def execute_single_run(self,
                          model: str,
                          prompt: str, 
                          image_base64: Optional[str],
                          options: Dict[str, Any],
                          seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecuta una sola ejecución del modelo.
        
        Args:
            model: Nombre del modelo
            prompt: Prompt de entrada
            image_base64: Imagen codificada (opcional)
            options: Opciones del modelo
            seed: Semilla (opcional)
            
        Returns:
            Dict con métricas procesadas
        """
        images = [image_base64] if image_base64 else None
        
        response, stats, wall_time = self.client.generate_response(
            model, prompt, images, options, stream=False
        )
        
        return self.metrics_analyzer.process_ollama_stats(stats, wall_time)
    
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
                        continue
        
        # Generar resumen
        return self._generate_summary(combination_results)
    
    def _print_progress(self, metrics: Dict[str, Any], 
                       current_job: int, total_jobs: int) -> None:
        """Imprime progreso de una ejecución."""
        pf_tps = metrics.get('prefill_tps', 0)
        dc_tps = metrics.get('decode_tps', 0)
        
        print(f"[{current_job}/{total_jobs}] cycle={metrics['cycle']} "
              f"run={metrics['run']} mode={metrics['mode']}")
        print(f"  ctx={metrics['context']} np={metrics['num_predict']} "
              f"temp={metrics['temperature']} seed={metrics.get('seed', 'none')}")
        
        if metrics['image_path']:
            img_name = Path(metrics['image_path']).name
            print(f"  img={img_name}")
        else:
            print(f"  modo texto puro")
            
        print(f"  prefill={metrics.get('prefill_tokens', 0)} @ "
              f"{pf_tps:.1f if pf_tps else 0} t/s | "
              f"decode={metrics.get('decode_tokens', 0)} @ "
              f"{dc_tps:.1f if dc_tps else 0} t/s")
        print(f"  wall={metrics['wall_time_s']:.2f}s\n")
    
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
                writer.writerow([
                    timestamp, self.client.host, metrics["model"], metrics["mode"],
                    metrics["prompt_hash"], metrics["prompt_length"],
                    metrics.get("image_path", ""), metrics["context"],
                    metrics["num_predict"], metrics["temperature"], 
                    metrics.get("seed", ""), metrics["cycle"], metrics["run"],
                    round(metrics["wall_time_s"], 3),
                    round(metrics.get("total_duration_s", 0), 3),
                    round(metrics.get("load_duration_s", 0), 3),
                    metrics.get("prefill_tokens", 0),
                    metrics.get("decode_tokens", 0),
                    round(metrics.get("prefill_tps", 0), 2),
                    round(metrics.get("decode_tps", 0), 2)
                ])
    
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
        
        return {
            "samples": len(valid_speeds),
            "mean": round(statistics.mean(valid_speeds), 2),
            "median": round(statistics.median(valid_speeds), 2),
            "min": round(min(valid_speeds), 2),
            "max": round(max(valid_speeds), 2),
            "stdev": round(statistics.stdev(valid_speeds), 2) if len(valid_speeds) > 1 else 0
        }
    
    def _compare_modes(self, text_stats: Dict, vision_stats: Dict) -> Dict[str, Any]:
        """Compara estadísticas entre modos texto y visión."""
        if "error" in text_stats or "error" in vision_stats:
            return {"error": "No hay suficientes datos para comparar"}
        
        if text_stats["mean"] > 0 and vision_stats["mean"] > 0:
            ratio = text_stats["mean"] / vision_stats["mean"]
            return {
                "text_vs_vision_ratio": round(ratio, 2),
                "text_faster_by_percent": round((ratio - 1) * 100, 1),
                "faster_mode": "text" if ratio > 1 else "vision"
            }
        
        return {"error": "No se pueden calcular ratios"}
