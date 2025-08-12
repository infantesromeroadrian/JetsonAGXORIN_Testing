#!/usr/bin/env python3
"""
Script de Barrido Paramétrico para Llama 3.2 Vision 11B en Jetson AGX Orin.

Este script ejecuta múltiples pruebas sistemáticas del modelo llama3.2-vision:11b
variando parámetros y comparando rendimiento entre modo texto y visión.
"""

import argparse
import json
import time
import sys
import statistics
import itertools
import csv
import hashlib
import os
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
        delay: Retraso entre intentos
        
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
    Codifica una imagen a base64.
    
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
    """Convierte nanosegundos a segundos."""
    return ns / 1e9 if isinstance(ns, (int, float)) else None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """División segura."""
    return (a / b) if (a is not None and b) else None


def gen_once(
    host: str,
    model: str,
    prompt: str,
    images: Optional[List[str]] = None,
    options: Optional[Dict] = None,
    stream: bool = False,
    timeout: int = 600,
    seed: Optional[int] = None
) -> Tuple[str, Dict, float]:
    """
    Genera una respuesta del modelo.
    
    Args:
        host: URL del servidor Ollama
        model: Nombre del modelo
        prompt: Texto de entrada
        images: Lista de imágenes en base64
        options: Opciones de generación
        stream: Si hacer streaming
        timeout: Timeout en segundos
        seed: Semilla para reproducibilidad
        
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
    if seed is not None:
        payload["seed"] = int(seed)
    if images:
        payload["images"] = images

    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    elapsed = time.perf_counter() - t0
    return data.get("response", ""), data, elapsed


def summarize(stats: Dict, elapsed: float) -> Dict[str, Any]:
    """
    Resume las estadísticas de una ejecución.
    
    Args:
        stats: Estadísticas de Ollama
        elapsed: Tiempo transcurrido
        
    Returns:
        Dict con métricas resumidas
    """
    td = ns_to_s(stats.get("total_duration"))
    ld = ns_to_s(stats.get("load_duration"))
    ed = ns_to_s(stats.get("eval_duration"))
    ped = ns_to_s(stats.get("prompt_eval_duration"))
    
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


def parse_int_list(s: str) -> List[int]:
    """Parsea una lista de enteros separados por comas."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    """Parsea una lista de floats separados por comas."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def read_prompts(prompt: Optional[str], prompt_file: Optional[str]) -> List[str]:
    """
    Lee prompts de archivo y/o argumentos.
    
    Args:
        prompt: Prompt único
        prompt_file: Archivo con prompts
        
    Returns:
        Lista de prompts
    """
    items = []
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    items.append(t)
    if prompt:
        items.append(prompt)
    
    # Prompts por defecto para visión
    if not items:
        items = [
            "Describe esta imagen en detalle.",
            "¿Qué objetos puedes identificar en esta imagen?",
            "Analiza los colores y la composición de esta imagen."
        ]
    return items


def read_images(image: Optional[str], image_dir: Optional[str]) -> List[Tuple[str, str]]:
    """
    Lee imágenes de archivo y/o directorio.
    
    Args:
        image: Ruta a imagen única
        image_dir: Directorio con imágenes
        
    Returns:
        Lista de tuplas (ruta, base64)
    """
    images = []
    
    if image and Path(image).exists():
        b64 = encode_image_to_base64(image)
        if b64:
            images.append((image, b64))
    
    if image_dir and Path(image_dir).is_dir():
        for img_path in Path(image_dir).glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                b64 = encode_image_to_base64(str(img_path))
                if b64:
                    images.append((str(img_path), b64))
    
    # Si no hay imágenes, agregar None para test solo texto
    if not images:
        images.append((None, None))
    
    return images


def short_hash(txt: str) -> str:
    """Genera un hash corto de un texto."""
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:8]


def main():
    """Función principal del script de barrido."""
    ap = argparse.ArgumentParser(
        description="Barrido paramétrico de Llama 3.2 Vision 11B"
    )
    
    # Configuración del servidor
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="llama3.2-vision:11b")
    
    # Prompts e imágenes
    ap.add_argument("--prompt", help="Prompt único")
    ap.add_argument("--prompt-file", help="Archivo con prompts")
    ap.add_argument("--image", help="Imagen única para probar")
    ap.add_argument("--image-dir", help="Directorio con imágenes")
    
    # Parámetros de barrido
    ap.add_argument("-n", "--runs", type=int, default=3,
                    help="Runs por combinación")
    ap.add_argument("--cycles", type=int, default=1,
                    help="Ciclos completos del barrido")
    ap.add_argument("--ctx", default="4096",
                    help="Lista de tamaños de contexto (ej: 2048,4096,8192)")
    ap.add_argument("--num-predict", default="128,256",
                    help="Lista de tokens a generar")
    ap.add_argument("--temp", default="0,0.4",
                    help="Lista de temperaturas")
    ap.add_argument("--seed", default="42",
                    help="Lista de semillas (vacío para omitir)")
    
    # Control de ejecución
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Pausa entre runs (segundos)")
    ap.add_argument("--warmup", action="store_true",
                    help="Hacer warmup antes de cada combinación")
    ap.add_argument("--test-mode", choices=["both", "text", "vision"],
                    default="both",
                    help="Modo de test: both, text, o vision")
    
    # Salida
    ap.add_argument("--out", help="Archivo JSONL de salida")
    ap.add_argument("--csv", help="Archivo CSV de salida")
    
    args = ap.parse_args()
    
    # Verificar servidor
    if not wait_for_server(args.host):
        print(
            f"No conecta con {args.host}. Arranca Ollama: sudo systemctl start ollama",
            file=sys.stderr
        )
        sys.exit(1)
    
    # Parsear listas de parámetros
    ctx_list = parse_int_list(args.ctx)
    np_list = parse_int_list(args.num_predict)
    temp_list = parse_float_list(args.temp)
    seed_list = [int(s) for s in args.seed.split(",") if s.strip()] if args.seed else [None]
    
    # Cargar prompts e imágenes
    prompts = read_prompts(args.prompt, args.prompt_file)
    images = read_images(args.image, args.image_dir)
    
    print(f"📊 Configuración del barrido:")
    print(f"  - Prompts: {len(prompts)}")
    print(f"  - Imágenes: {len(images)} ({'con visión' if images[0][0] else 'solo texto'})")
    print(f"  - Contextos: {ctx_list}")
    print(f"  - Num predict: {np_list}")
    print(f"  - Temperaturas: {temp_list}")
    print(f"  - Semillas: {seed_list}")
    print(f"  - Modo de test: {args.test_mode}")
    
    # Preparar CSV si es necesario
    if args.csv and not os.path.exists(args.csv):
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "ts", "host", "model", "mode", "prompt_hash", "prompt_len",
                "image_path", "ctx", "num_predict", "temp", "seed",
                "cycle", "run_idx", "wall_s", "total_s", "load_s",
                "prefill_tokens", "decode_tokens", "prefill_tps", "decode_tps"
            ])
    
    # Diccionario para agregados
    agg = {}
    
    # Generar combinaciones según el modo
    if args.test_mode == "text":
        # Solo texto, sin imágenes
        combos = list(itertools.product(
            prompts, [(None, None)], ctx_list, np_list, temp_list, seed_list
        ))
    elif args.test_mode == "vision":
        # Solo visión, requiere imágenes
        if not any(img[0] for img in images):
            print("Error: Modo visión requiere imágenes", file=sys.stderr)
            sys.exit(1)
        combos = list(itertools.product(
            prompts, images, ctx_list, np_list, temp_list, seed_list
        ))
    else:  # both
        # Todas las combinaciones
        combos = list(itertools.product(
            prompts, images, ctx_list, np_list, temp_list, seed_list
        ))
    
    total_jobs = len(combos) * args.runs * args.cycles
    print(f"\n🚀 Iniciando: {len(combos)} combinaciones × {args.runs} runs × "
          f"{args.cycles} ciclos = {total_jobs} ejecuciones\n")
    
    job_no = 0
    start_time = time.time()
    
    for cyc in range(1, args.cycles + 1):
        for prompt, (img_path, img_b64), ctx, num_predict, temp, seed in combos:
            ph = short_hash(prompt)
            options = {
                "num_ctx": ctx,
                "temperature": temp,
                "num_predict": num_predict
            }
            
            # Determinar modo
            mode = "vision" if img_path else "text"
            
            # Warmup si está habilitado
            if args.warmup:
                try:
                    gen_once(
                        args.host, args.model, "Warmup.",
                        [img_b64] if img_b64 else None,
                        options, seed=seed
                    )
                except Exception as e:
                    print(f"Warmup error: {e}", file=sys.stderr)
            
            # Clave para agregados
            key = (ph, img_path, ctx, num_predict, temp, seed)
            agg.setdefault(key, [])
            
            # Ejecutar runs
            for run_idx in range(1, args.runs + 1):
                job_no += 1
                
                try:
                    # Ejecutar generación
                    text, stats, elapsed = gen_once(
                        args.host, args.model, prompt,
                        [img_b64] if img_b64 else None,
                        options, seed=seed
                    )
                    
                    # Resumir métricas
                    m = summarize(stats, elapsed)
                    agg[key].append(m["decode_tps"])
                    
                    # Mostrar progreso
                    pf_tps = f"{m['prefill_tps']:.1f}" if m['prefill_tps'] else "n/a"
                    dc_tps = f"{m['decode_tps']:.1f}" if m['decode_tps'] else "n/a"
                    
                    print(f"[{job_no}/{total_jobs}] cyc={cyc} run={run_idx} mode={mode}")
                    print(f"  ctx={ctx} np={num_predict} temp={temp} seed={seed}")
                    print(f"  {'img=' + Path(img_path).name if img_path else 'texto puro'}")
                    print(f"  prefill={m['prefill_tokens']} @ {pf_tps} t/s | "
                          f"decode={m['decode_tokens']} @ {dc_tps} t/s")
                    print(f"  wall={m['wall_s']:.2f}s total={m['total_s'] or 0:.2f}s "
                          f"load={m['load_s'] or 0:.2f}s")
                    
                    # Guardar métricas
                    ts = time.time()
                    rec = {
                        "ts": ts,
                        "host": args.host,
                        "model": args.model,
                        "mode": mode,
                        "prompt_hash": ph,
                        "prompt_len": len(prompt),
                        "image_path": img_path,
                        "ctx": ctx,
                        "num_predict": num_predict,
                        "temp": temp,
                        "seed": seed,
                        "cycle": cyc,
                        "run_idx": run_idx,
                        **m
                    }
                    
                    if args.out:
                        with open(args.out, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                    
                    if args.csv:
                        with open(args.csv, "a", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            w.writerow([
                                ts, args.host, args.model, mode, ph, len(prompt),
                                img_path or "", ctx, num_predict, temp, seed or "",
                                cyc, run_idx,
                                round(m["wall_s"], 3),
                                round(m["total_s"] or 0, 3),
                                round(m["load_s"] or 0, 3),
                                m["prefill_tokens"],
                                m["decode_tokens"],
                                round(m["prefill_tps"] or 0, 2),
                                round(m["decode_tps"] or 0, 2)
                            ])
                    
                except Exception as e:
                    print(f"ERROR en job {job_no}: {e}", file=sys.stderr)
                    continue
                
                # Pausa entre runs
                if args.sleep > 0:
                    time.sleep(args.sleep)
    
    # Tiempo total
    total_time = time.time() - start_time
    
    # Resumen por combinación
    print("\n" + "="*60)
    print("RESUMEN POR COMBINACIÓN (decode_tps)")
    print("="*60)
    
    text_speeds = []
    vision_speeds = []
    
    for key, vals in agg.items():
        if not vals:
            continue
        
        ph, img_path, ctx, npred, temp, seed = key
        valid_vals = [v for v in vals if v]
        
        if valid_vals:
            mean_tps = statistics.mean(valid_vals)
            med_tps = statistics.median(valid_vals)
            
            mode = "vision" if img_path else "text"
            img_name = Path(img_path).name if img_path else "none"
            
            print(f"\n[{mode}] prompt={ph[:8]} img={img_name} "
                  f"ctx={ctx} np={npred} temp={temp} seed={seed}")
            print(f"  mean={mean_tps:.1f} t/s | median={med_tps:.1f} t/s | n={len(valid_vals)}")
            
            if mode == "text":
                text_speeds.extend(valid_vals)
            else:
                vision_speeds.extend(valid_vals)
    
    # Comparación global texto vs visión
    if text_speeds and vision_speeds:
        print("\n" + "="*60)
        print("COMPARACIÓN GLOBAL TEXTO vs VISIÓN")
        print("="*60)
        print(f"\nModo TEXTO ({len(text_speeds)} muestras):")
        print(f"  Media: {statistics.mean(text_speeds):.1f} t/s")
        print(f"  Mediana: {statistics.median(text_speeds):.1f} t/s")
        print(f"  Min/Max: {min(text_speeds):.1f} / {max(text_speeds):.1f} t/s")
        
        print(f"\nModo VISIÓN ({len(vision_speeds)} muestras):")
        print(f"  Media: {statistics.mean(vision_speeds):.1f} t/s")
        print(f"  Mediana: {statistics.median(vision_speeds):.1f} t/s")
        print(f"  Min/Max: {min(vision_speeds):.1f} / {max(vision_speeds):.1f} t/s")
        
        ratio = statistics.mean(text_speeds) / statistics.mean(vision_speeds)
        print(f"\n📊 Factor de velocidad texto/visión: {ratio:.2f}x")
        print(f"   (El procesamiento de texto es {ratio:.1f}x más rápido)")
    
    print(f"\n⏱️  Tiempo total de ejecución: {total_time/60:.1f} minutos")
    print(f"✅ Barrido completado: {job_no} jobs ejecutados")
    
    if args.csv:
        print(f"📄 Resultados guardados en: {args.csv}")
    if args.out:
        print(f"📄 Métricas detalladas en: {args.out}")


if __name__ == "__main__":
    main()
