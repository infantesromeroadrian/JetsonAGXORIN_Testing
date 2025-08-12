#!/usr/bin/env python3
"""
Análisis de métricas y estadísticas para benchmarking.

Este módulo proporciona herramientas para procesar, analizar y reportar
métricas de rendimiento de los modelos de lenguaje.
"""

import json
import statistics
import time
from typing import Dict, List, Optional, Any, Union


class MetricsAnalyzer:
    """Analizador de métricas de rendimiento."""
    
    def __init__(self):
        """Inicializa el analizador de métricas."""
        self.metrics_history: List[Dict[str, Any]] = []
    
    @staticmethod
    def nanoseconds_to_seconds(nanoseconds: Optional[Union[int, float]]) -> Optional[float]:
        """
        Convierte nanosegundos a segundos.
        
        Args:
            nanoseconds: Valor en nanosegundos
            
        Returns:
            float: Valor en segundos o None si la entrada es None
        """
        if nanoseconds is None:
            return None
        
        try:
            return float(nanoseconds) / 1e9
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def safe_division(numerator: Optional[float], 
                     denominator: Optional[float]) -> Optional[float]:
        """
        División segura que maneja None y división por cero.
        
        Args:
            numerator: Numerador
            denominator: Denominador
            
        Returns:
            float: Resultado de la división o None
        """
        if numerator is None or denominator is None or denominator == 0:
            return None
        
        try:
            return float(numerator) / float(denominator)
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    
    def process_ollama_stats(self, stats: Dict[str, Any], 
                           wall_time: float) -> Dict[str, Any]:
        """
        Procesa estadísticas crudas de Ollama en métricas estructuradas.
        
        Args:
            stats: Estadísticas devueltas por Ollama
            wall_time: Tiempo transcurrido en segundos
            
        Returns:
            Dict con métricas procesadas
        """
        # Convertir duraciones de nanosegundos a segundos
        total_duration = self.nanoseconds_to_seconds(stats.get("total_duration"))
        eval_duration = self.nanoseconds_to_seconds(stats.get("eval_duration"))
        prompt_eval_duration = self.nanoseconds_to_seconds(stats.get("prompt_eval_duration"))
        load_duration = self.nanoseconds_to_seconds(stats.get("load_duration"))
        
        # Obtener contadores de tokens
        eval_count = stats.get("eval_count")
        prompt_eval_count = stats.get("prompt_eval_count")
        
        # Calcular tokens por segundo
        prefill_tps = self.safe_division(prompt_eval_count, prompt_eval_duration)
        decode_tps = self.safe_division(eval_count, eval_duration)
        
        return {
            "wall_time_s": round(wall_time, 3),
            "total_duration_s": total_duration,
            "load_duration_s": load_duration,
            "prefill_tokens": prompt_eval_count,
            "decode_tokens": eval_count,
            "prefill_duration_s": prompt_eval_duration,
            "decode_duration_s": eval_duration,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "total_tokens": (prompt_eval_count or 0) + (eval_count or 0),
            "timestamp": time.time()
        }
    
    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Añade métricas al historial.
        
        Args:
            metrics: Métricas a añadir
        """
        self.metrics_history.append(metrics.copy())
    
    def get_summary_stats(self, mode_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas resumidas del historial.
        
        Args:
            mode_filter: Filtrar por modo ("text", "vision", etc.)
            
        Returns:
            Dict con estadísticas resumidas
        """
        filtered_metrics = self.metrics_history
        
        if mode_filter:
            filtered_metrics = [m for m in self.metrics_history 
                              if m.get("mode") == mode_filter]
        
        if not filtered_metrics:
            return {"error": "No hay métricas disponibles"}
        
        # Extraer valores numéricos válidos
        decode_tps_values = [m["decode_tps"] for m in filtered_metrics 
                            if m.get("decode_tps") is not None]
        prefill_tps_values = [m["prefill_tps"] for m in filtered_metrics 
                             if m.get("prefill_tps") is not None]
        wall_times = [m["wall_time_s"] for m in filtered_metrics 
                     if m.get("wall_time_s") is not None]
        
        summary = {
            "total_runs": len(filtered_metrics),
            "mode": mode_filter or "all"
        }
        
        # Estadísticas de decode_tps
        if decode_tps_values:
            summary["decode_tps"] = {
                "mean": round(statistics.mean(decode_tps_values), 2),
                "median": round(statistics.median(decode_tps_values), 2),
                "min": round(min(decode_tps_values), 2),
                "max": round(max(decode_tps_values), 2)
            }
            
            if len(decode_tps_values) > 1:
                summary["decode_tps"]["stdev"] = round(
                    statistics.stdev(decode_tps_values), 2
                )
        
        # Estadísticas de prefill_tps
        if prefill_tps_values:
            summary["prefill_tps"] = {
                "mean": round(statistics.mean(prefill_tps_values), 2),
                "median": round(statistics.median(prefill_tps_values), 2),
                "min": round(min(prefill_tps_values), 2),
                "max": round(max(prefill_tps_values), 2)
            }
        
        # Estadísticas de tiempo de pared
        if wall_times:
            summary["wall_time_s"] = {
                "mean": round(statistics.mean(wall_times), 3),
                "median": round(statistics.median(wall_times), 3),
                "min": round(min(wall_times), 3),
                "max": round(max(wall_times), 3)
            }
        
        return summary
    
    def compare_modes(self, mode1: str, mode2: str) -> Dict[str, Any]:
        """
        Compara métricas entre dos modos diferentes.
        
        Args:
            mode1: Primer modo a comparar
            mode2: Segundo modo a comparar
            
        Returns:
            Dict con comparación detallada
        """
        stats1 = self.get_summary_stats(mode1)
        stats2 = self.get_summary_stats(mode2)
        
        if "error" in stats1 or "error" in stats2:
            return {"error": "No hay suficientes datos para comparar"}
        
        comparison = {
            "mode1": mode1,
            "mode2": mode2,
            "mode1_stats": stats1,
            "mode2_stats": stats2
        }
        
        # Calcular factores de diferencia
        if ("decode_tps" in stats1 and "decode_tps" in stats2 and
            stats1["decode_tps"]["mean"] > 0 and stats2["decode_tps"]["mean"] > 0):
            
            ratio = stats1["decode_tps"]["mean"] / stats2["decode_tps"]["mean"]
            comparison["decode_tps_ratio"] = {
                f"{mode1}_vs_{mode2}": round(ratio, 2),
                "faster_mode": mode1 if ratio > 1 else mode2,
                "speed_difference_percent": round(abs(ratio - 1) * 100, 1)
            }
        
        return comparison
    
    def save_to_jsonl(self, filepath: str, 
                     additional_fields: Optional[Dict[str, Any]] = None) -> bool:
        """
        Guarda métricas en formato JSONL.
        
        Args:
            filepath: Ruta del archivo
            additional_fields: Campos adicionales a incluir
            
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                for metrics in self.metrics_history:
                    record = metrics.copy()
                    
                    if additional_fields:
                        record.update(additional_fields)
                    
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            return True
        except Exception as e:
            print(f"Error guardando métricas en {filepath}: {e}")
            return False
    
    def print_formatted_metrics(self, metrics: Dict[str, Any], 
                               title: Optional[str] = None) -> None:
        """
        Imprime métricas de forma formateada.
        
        Args:
            metrics: Métricas a imprimir
            title: Título opcional
        """
        if title:
            print(f"\n{'='*60}")
            print(f"{title}")
            print('='*60)
        
        # Información básica
        print(f"⏱️  Tiempo total: {metrics.get('wall_time_s', 'n/a')} s")
        
        if metrics.get('prefill_tokens'):
            prefill_tps = metrics.get('prefill_tps', 'n/a')
            print(f"📤 Prefill: {metrics['prefill_tokens']} tokens @ "
                  f"{prefill_tps:.1f if isinstance(prefill_tps, (int, float)) else prefill_tps} t/s")
        
        if metrics.get('decode_tokens'):
            decode_tps = metrics.get('decode_tps', 'n/a')
            print(f"📥 Decode: {metrics['decode_tokens']} tokens @ "
                  f"{decode_tps:.1f if isinstance(decode_tps, (int, float)) else decode_tps} t/s")
        
        if metrics.get('mode'):
            print(f"🔧 Modo: {metrics['mode']}")
        
        print()


# Funciones de conveniencia para uso directo
def quick_analyze(stats: Dict[str, Any], wall_time: float) -> Dict[str, Any]:
    """
    Análisis rápido de estadísticas de Ollama.
    
    Args:
        stats: Estadísticas de Ollama
        wall_time: Tiempo transcurrido
        
    Returns:
        Dict con métricas procesadas
    """
    analyzer = MetricsAnalyzer()
    return analyzer.process_ollama_stats(stats, wall_time)
