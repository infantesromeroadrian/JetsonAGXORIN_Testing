#!/usr/bin/env python3
"""
Script de Testing para el modelo Llama 3.2 Vision 11B en Jetson AGX Orin.

Este script evalúa el rendimiento del modelo multimodal llama3.2-vision:11b
que puede procesar tanto texto como imágenes.
"""

import argparse
import json
import time
import sys
import statistics
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import requests


def wait_for_server(host: str, tries: int = 10, delay: float = 1.0) -> bool:
    """
    Espera a que el servidor Ollama esté disponible.
    
    Args:
        host: URL del servidor Ollama
        tries: Número de intentos
        delay: Retraso entre intentos en segundos
        
    Returns:
        bool: True si el servidor responde
    """
    url = f"{host}/api/version"
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                return True
        except Exception:
            time.sleep(delay)
    return False


def encode_image_to_base64(image_path: str) -> str:
    """
    Codifica una imagen a base64 para enviar a Ollama.
    
    Args:
        image_path: Ruta a la imagen
        
    Returns:
        str: Imagen codificada en base64
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error leyendo imagen {image_path}: {e}", file=sys.stderr)
        return ""


def ns_to_s(ns: Optional[int]) -> Optional[float]:
    """
    Convierte nanosegundos a segundos.
    
    Args:
        ns: Nanosegundos
        
    Returns:
        float: Segundos o None si la entrada es None
    """
    return ns / 1e9 if isinstance(ns, (int, float)) else None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """
    División segura que maneja None y división por cero.
    
    Args:
        a: Numerador
        b: Denominador
        
    Returns:
        float: Resultado de la división o None
    """
    return (a / b) if (a is not None and b) else None


def gen_once(
    host: str,
    model: str,
    prompt: str,
    images: Optional[List[str]] = None,
    options: Optional[Dict] = None,
    stream: bool = False,
    timeout: int = 600
) -> Tuple[str, Dict, float]:
    """
    Genera una respuesta del modelo una vez.
    
    Args:
        host: URL del servidor Ollama
        model: Nombre del modelo
        prompt: Texto de entrada
        images: Lista de imágenes codificadas en base64 (opcional)
        options: Opciones de generación
        stream: Si hacer streaming de la respuesta
        timeout: Timeout en segundos
        
    Returns:
        Tuple de (respuesta, estadísticas, tiempo_transcurrido)
    """
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    
    if options:
        payload["options"] = options
    
    if images:
        payload["images"] = images

    t0 = time.perf_counter()
    
    if stream:
        text = []
        stats = {}
        with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "response" in data:
                    chunk = data["response"]
                    print(chunk, end="", flush=True)
                    text.append(chunk)
                if data.get("done"):
                    stats = data
                    break
        print()  # Nueva línea al final del stream
        elapsed = time.perf_counter() - t0
        return "".join(text), stats, elapsed
    else:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        elapsed = time.perf_counter() - t0
        return data.get("response", ""), data, elapsed


def summarize(stats: Dict, elapsed: float) -> Dict[str, Any]:
    """
    Resume las estadísticas de una ejecución.
    
    Args:
        stats: Estadísticas devueltas por Ollama
        elapsed: Tiempo transcurrido en segundos
        
    Returns:
        Dict con métricas resumidas
    """
    td = ns_to_s(stats.get("total_duration"))
    ed = ns_to_s(stats.get("eval_duration"))
    ped = ns_to_s(stats.get("prompt_eval_duration"))
    ld = ns_to_s(stats.get("load_duration"))
    
    eval_count = stats.get("eval_count")
    prompt_eval_count = stats.get("prompt_eval_count")
    
    return {
        "wall_s": elapsed,
        "total_s": td,
        "load_s": ld,
        "prefill_tokens": prompt_eval_count,
        "decode_tokens": eval_count,
        "prefill_tps": safe_div(prompt_eval_count, ped) if ped else None,
        "decode_tps": safe_div(eval_count, ed) if ed else None,
    }


def test_text_only(
    host: str,
    model: str,
    prompt: str,
    options: Dict,
    runs: int,
    stream: bool,
    out_file: Optional[str]
) -> List[Dict]:
    """
    Ejecuta test solo con texto.
    
    Args:
        host: URL del servidor Ollama
        model: Nombre del modelo
        prompt: Texto de entrada
        options: Opciones de generación
        runs: Número de repeticiones
        stream: Si hacer streaming
        out_file: Archivo de salida para métricas
        
    Returns:
        Lista de métricas por ejecución
    """
    print("\n=== TEST SOLO TEXTO ===")
    print(f"Prompt: {prompt[:100]}...")
    
    metrics = []
    for i in range(runs):
        print(f"\n>> Run {i+1}/{runs} (texto)")
        text, stats, elapsed = gen_once(
            host, model, prompt, None, options, stream=stream
        )
        
        if not stream:
            print(f"Respuesta: {text[:200]}...")
        
        m = summarize(stats, elapsed)
        metrics.append({**m, "mode": "text", "run": i+1})
        
        pf_tps = f"{m['prefill_tps']:.1f}" if m['prefill_tps'] else "n/a"
        dc_tps = f"{m['decode_tps']:.1f}" if m['decode_tps'] else "n/a"
        print(f"[stats] wall={m['wall_s']:.2f}s | prefill={m['prefill_tokens']} tok @ {pf_tps} t/s | "
              f"decode={m['decode_tokens']} tok @ {dc_tps} t/s")
        
        if out_file:
            rec = {"ts": time.time(), "model": model, "mode": "text", **m}
            with open(out_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
    
    return metrics


def test_vision(
    host: str,
    model: str,
    prompt: str,
    image_path: str,
    options: Dict,
    runs: int,
    stream: bool,
    out_file: Optional[str]
) -> List[Dict]:
    """
    Ejecuta test con imagen y texto.
    
    Args:
        host: URL del servidor Ollama
        model: Nombre del modelo
        prompt: Texto de entrada
        image_path: Ruta a la imagen
        options: Opciones de generación
        runs: Número de repeticiones
        stream: Si hacer streaming
        out_file: Archivo de salida para métricas
        
    Returns:
        Lista de métricas por ejecución
    """
    print("\n=== TEST VISIÓN (IMAGEN + TEXTO) ===")
    print(f"Imagen: {image_path}")
    print(f"Prompt: {prompt[:100]}...")
    
    # Codificar imagen
    image_base64 = encode_image_to_base64(image_path)
    if not image_base64:
        print("Error: No se pudo cargar la imagen", file=sys.stderr)
        return []
    
    metrics = []
    for i in range(runs):
        print(f"\n>> Run {i+1}/{runs} (visión)")
        text, stats, elapsed = gen_once(
            host, model, prompt, [image_base64], options, stream=stream
        )
        
        if not stream:
            print(f"Respuesta: {text[:200]}...")
        
        m = summarize(stats, elapsed)
        metrics.append({**m, "mode": "vision", "run": i+1})
        
        pf_tps = f"{m['prefill_tps']:.1f}" if m['prefill_tps'] else "n/a"
        dc_tps = f"{m['decode_tps']:.1f}" if m['decode_tps'] else "n/a"
        
        # Nota: El procesamiento de imagen aumenta significativamente el prefill
        print(f"[stats] wall={m['wall_s']:.2f}s | prefill={m['prefill_tokens']} tok @ {pf_tps} t/s | "
              f"decode={m['decode_tokens']} tok @ {dc_tps} t/s")
        print(f"[vision] Procesamiento de imagen incluido en prefill")
        
        if out_file:
            rec = {
                "ts": time.time(),
                "model": model,
                "mode": "vision",
                "image": image_path,
                **m
            }
            with open(out_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
    
    return metrics


def print_comparison(text_metrics: List[Dict], vision_metrics: List[Dict]) -> None:
    """
    Imprime comparación entre modos texto y visión.
    
    Args:
        text_metrics: Métricas del modo texto
        vision_metrics: Métricas del modo visión
    """
    if not text_metrics or not vision_metrics:
        return
    
    print("\n" + "="*60)
    print("COMPARACIÓN TEXTO vs VISIÓN")
    print("="*60)
    
    # Calcular promedios
    text_decode_tps = [m["decode_tps"] for m in text_metrics if m["decode_tps"]]
    vision_decode_tps = [m["decode_tps"] for m in vision_metrics if m["decode_tps"]]
    
    text_wall = [m["wall_s"] for m in text_metrics]
    vision_wall = [m["wall_s"] for m in vision_metrics]
    
    if text_decode_tps:
        print(f"\nModo TEXTO:")
        print(f"  Decode TPS promedio: {statistics.mean(text_decode_tps):.1f} t/s")
        print(f"  Tiempo promedio: {statistics.mean(text_wall):.2f} s")
    
    if vision_decode_tps:
        print(f"\nModo VISIÓN:")
        print(f"  Decode TPS promedio: {statistics.mean(vision_decode_tps):.1f} t/s")
        print(f"  Tiempo promedio: {statistics.mean(vision_wall):.2f} s")
        print(f"  Overhead por imagen: ~{statistics.mean(vision_wall) - statistics.mean(text_wall):.2f} s")
    
    if text_decode_tps and vision_decode_tps:
        ratio = statistics.mean(text_decode_tps) / statistics.mean(vision_decode_tps)
        print(f"\nFactor de velocidad texto/visión: {ratio:.2f}x")


def main():
    """Función principal del script de testing."""
    ap = argparse.ArgumentParser(
        description="Benchmark de Llama 3.2 Vision 11B en Jetson AGX Orin"
    )
    ap.add_argument(
        "--host",
        default="http://localhost:11434",
        help="URL del servidor Ollama"
    )
    ap.add_argument(
        "--model",
        default="llama3.2-vision:11b",
        help="Modelo a usar (default: llama3.2-vision:11b)"
    )
    ap.add_argument(
        "--prompt",
        default="Describe en detalle qué ves en esta imagen. Sé específico con colores, objetos y su disposición.",
        help="Prompt para evaluación"
    )
    ap.add_argument(
        "--text-prompt",
        default="Resume en 5 líneas las capacidades del modelo Llama 3.2 Vision. Responde en español.",
        help="Prompt para test solo texto"
    )
    ap.add_argument(
        "--image",
        help="Ruta a imagen para test de visión"
    )
    ap.add_argument(
        "-n", "--runs",
        type=int,
        default=3,
        help="Número de repeticiones por modo"
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Mostrar salida token a token"
    )
    ap.add_argument(
        "--ctx",
        type=int,
        default=4096,
        help="Tamaño de ventana de contexto (default: 4096 para visión)"
    )
    ap.add_argument(
        "--temp",
        type=float,
        default=0.4,
        help="Temperature"
    )
    ap.add_argument(
        "--num-predict",
        type=int,
        default=256,
        help="Número máximo de tokens a generar"
    )
    ap.add_argument(
        "--test-mode",
        choices=["both", "text", "vision"],
        default="both",
        help="Modo de test: both, text, o vision"
    )
    ap.add_argument(
        "--out",
        help="Archivo JSONL para guardar métricas"
    )
    ap.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Omitir warmup inicial"
    )
    
    args = ap.parse_args()
    
    # Verificar servidor
    if not wait_for_server(args.host):
        print(
            f"No conecta con {args.host}. Arranca el servicio: sudo systemctl start ollama",
            file=sys.stderr
        )
        sys.exit(1)
    
    # Configurar opciones
    options = {
        "num_ctx": args.ctx,
        "temperature": args.temp,
        "num_predict": args.num_predict
    }
    
    # Warmup
    if not args.skip_warmup:
        print("Ejecutando warmup...")
        try:
            gen_once(args.host, args.model, "Hola.", None, options, stream=False)
            print("Warmup completado ✓\n")
        except Exception as e:
            print(f"Warmup error (continuamos): {e}", file=sys.stderr)
    
    # Ejecutar tests según el modo
    text_metrics = []
    vision_metrics = []
    
    if args.test_mode in ["both", "text"]:
        text_metrics = test_text_only(
            args.host, args.model, args.text_prompt,
            options, args.runs, args.stream, args.out
        )
    
    if args.test_mode in ["both", "vision"]:
        if not args.image:
            # Usar imagen de ejemplo si no se proporciona
            default_image = Path(__file__).parent.parent / "assets" / "test_image.jpg"
            if default_image.exists():
                args.image = str(default_image)
                print(f"Usando imagen de ejemplo: {args.image}")
            else:
                print(
                    "Error: Se requiere una imagen para test de visión. "
                    "Usa --image <ruta> o coloca test_image.jpg en assets/",
                    file=sys.stderr
                )
                if args.test_mode == "vision":
                    sys.exit(1)
        
        if args.image:
            vision_metrics = test_vision(
                args.host, args.model, args.prompt, args.image,
                options, args.runs, args.stream, args.out
            )
    
    # Mostrar comparación si se ejecutaron ambos modos
    if text_metrics and vision_metrics:
        print_comparison(text_metrics, vision_metrics)
    
    # Resumen final
    all_metrics = text_metrics + vision_metrics
    if all_metrics and len(all_metrics) > 1:
        all_decode_tps = [m["decode_tps"] for m in all_metrics if m["decode_tps"]]
        if all_decode_tps:
            print(f"\n📊 Promedio global decode_tps ({len(all_decode_tps)} runs): "
                  f"{statistics.mean(all_decode_tps):.1f} t/s")


if __name__ == "__main__":
    main()
