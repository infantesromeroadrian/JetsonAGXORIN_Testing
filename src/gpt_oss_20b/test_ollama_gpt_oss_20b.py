#!/usr/bin/env python3
"""
Script de benchmark individual para el modelo GPT-OSS 20B
Modelo de lenguaje de código abierto con 20 mil millones de parámetros
"""
import argparse, json, time, sys, statistics
import requests
from .system_monitor import SystemMonitor, get_instant_metrics

def wait_for_server(host, tries=10, delay=1.0):
    url = f"{host}/api/version"
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=2)
            if r.ok:
                return True
        except Exception:
            time.sleep(delay)
    return False

def ns_to_s(ns):
    return ns / 1e9 if isinstance(ns, (int, float)) else None

def safe_div(a, b):
    return (a / b) if (a is not None and b) else None

def gen_once(host, model, prompt, options=None, stream=False, timeout=600):
    url = f"{host}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": stream}
    if options:
        payload["options"] = options

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
                    print(chunk, end="", flush=True)  # imprime tokens según llegan
                    text.append(chunk)
                if data.get("done"):
                    stats = data
                    break
        print()  # salto de línea al final del stream
        elapsed = time.perf_counter() - t0
        return "".join(text), stats, elapsed
    else:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        elapsed = time.perf_counter() - t0
        return data.get("response", ""), data, elapsed

def summarize(stats, elapsed, system_metrics_summary=None):
    """
    Combina métricas de Ollama con métricas del sistema para GPT-OSS 20B
    """
    td = ns_to_s(stats.get("total_duration"))
    ed = ns_to_s(stats.get("eval_duration"))
    ped = ns_to_s(stats.get("prompt_eval_duration"))
    eval_count = stats.get("eval_count")
    prompt_eval_count = stats.get("prompt_eval_count")
    
    # Métricas básicas de Ollama (existentes)
    base_metrics = {
        "wall_s": elapsed,
        "total_s": td,
        "prefill_tokens": prompt_eval_count,
        "decode_tokens": eval_count,
        "prefill_tps": safe_div(prompt_eval_count, ped) if ped else None,
        "decode_tps": safe_div(eval_count, ed) if ed else None,
    }
    
    # Añadir métricas del sistema si están disponibles
    if system_metrics_summary:
        system_metrics = {
            # CPU Metrics
            "cpu_usage_mean_percent": system_metrics_summary.get('cpu_usage_percent', {}).get('mean'),
            "cpu_usage_max_percent": system_metrics_summary.get('cpu_usage_percent', {}).get('max'),
            
            # RAM Metrics
            "ram_usage_mean_percent": system_metrics_summary.get('ram_usage_percent', {}).get('mean'),
            "ram_usage_max_percent": system_metrics_summary.get('ram_usage_percent', {}).get('max'),
            "ram_used_mean_gb": system_metrics_summary.get('ram_used_gb', {}).get('mean'),
            "ram_used_max_gb": system_metrics_summary.get('ram_used_gb', {}).get('max'),
            
            # GPU Metrics  
            "gpu_usage_mean_percent": system_metrics_summary.get('gpu_usage_percent', {}).get('mean'),
            "gpu_usage_max_percent": system_metrics_summary.get('gpu_usage_percent', {}).get('max'),
            
            # Temperature Metrics
            "cpu_temp_mean_c": system_metrics_summary.get('cpu_temperature_c', {}).get('mean'),
            "cpu_temp_max_c": system_metrics_summary.get('cpu_temperature_c', {}).get('max'),
            "gpu_temp_mean_c": system_metrics_summary.get('gpu_temperature_c', {}).get('mean'),
            "gpu_temp_max_c": system_metrics_summary.get('gpu_temperature_c', {}).get('max'),
            
            # Power Metrics
            "power_mean_watts": system_metrics_summary.get('power_consumption_watts', {}).get('mean'),
            "power_max_watts": system_metrics_summary.get('power_consumption_watts', {}).get('max'),
            
            # Monitoring Info
            "monitoring_duration_s": system_metrics_summary.get('monitoring_duration_s'),
            "monitoring_samples": system_metrics_summary.get('total_samples')
        }
        base_metrics.update(system_metrics)
    
    return base_metrics

def main():
    ap = argparse.ArgumentParser(description="Benchmark de GPT-OSS 20B en Jetson AGX Orin")
    ap.add_argument("--host", default="http://localhost:11434", help="URL del servidor Ollama")
    ap.add_argument("--model", default="gpt-oss:20b", help="Modelo a usar (default: gpt-oss:20b)")
    ap.add_argument("--prompt", default="Escribe un ensayo breve sobre las ventajas y desventajas de la inteligencia artificial en la educación moderna. Incluye ejemplos específicos y una conclusión reflexiva.",
                    help="Prompt a evaluar (por defecto: ensayo sobre IA en educación)")
    ap.add_argument("-n", "--runs", type=int, default=1, help="Número de repeticiones")
    ap.add_argument("--stream", action="store_true", help="Muestra la salida token a token")
    ap.add_argument("--ctx", type=int, default=8192, help="num_ctx (ventana de contexto, default: 8192)")
    ap.add_argument("--temp", type=float, default=0.7, help="temperature (default: 0.7 para creatividad balanceada)")
    ap.add_argument("--num-predict", type=int, default=256, help="Número máximo de tokens a generar (default: 256)")
    ap.add_argument("--out", help="Ruta para volcar métricas en JSONL (append)")
    ap.add_argument("--no-system-monitor", action="store_true", 
                    help="Desactivar monitoreo del sistema (solo métricas de Ollama)")
    ap.add_argument("--monitor-file", help="Guardar métricas detalladas del sistema en archivo JSONL")
    args = ap.parse_args()

    if not wait_for_server(args.host):
        print(f"No conecta con {args.host}. Arranca el servicio: sudo systemctl start ollama", file=sys.stderr)
        sys.exit(1)

    options = {
        "num_ctx": args.ctx, 
        "temperature": args.temp, 
        "num_predict": args.num_predict
    }
    
    # Crear monitor del sistema si está habilitado
    system_monitor = None if args.no_system_monitor else SystemMonitor()

    # Warmup (evita penalización de primer uso)
    try:
        print("Realizando warmup...")
        if system_monitor:
            system_monitor.start_monitoring()
            
        gen_once(args.host, args.model, "Hello.", options, stream=False)
        
        if system_monitor:
            system_monitor.stop_monitoring()  # limpiar métricas de warmup
            
    except Exception as e:
        print("Warmup error (continuamos):", e, file=sys.stderr)

    metrics = []
    print(f"\n=== Iniciando {args.runs} ejecuciones de benchmark para GPT-OSS 20B ===")
    print(f"Modelo: {args.model} (20 mil millones de parámetros)")
    print(f"Contexto: {args.ctx} tokens")
    print(f"Temperature: {args.temp}")
    print(f"Max tokens: {args.num_predict}")
    print(f"Host: {args.host}\n")
    
    for i in range(args.runs):
        print(f"\n>> Run {i+1}/{args.runs}")
        
        # Capturar métricas del sistema antes de la ejecución
        pre_metrics = None
        if not args.no_system_monitor:
            pre_metrics = get_instant_metrics()
            print(f"[pre] CPU: {pre_metrics.cpu_percent:.1f}% | "
                  f"RAM: {pre_metrics.ram_used_gb:.1f}GB ({pre_metrics.ram_percent:.1f}%)")
            if pre_metrics.cpu_temp:
                print(f"[pre] CPU Temp: {pre_metrics.cpu_temp:.1f}°C")
            if pre_metrics.gpu_usage_percent:
                print(f"[pre] GPU: {pre_metrics.gpu_usage_percent:.1f}%")
        
        # Iniciar monitoreo del sistema durante la ejecución
        if system_monitor:
            system_monitor.start_monitoring()
            
        # Ejecutar el benchmark
        text, stats, elapsed = gen_once(args.host, args.model, args.prompt, options, stream=args.stream)
        
        # Detener monitoreo y obtener resumen
        system_metrics_summary = None
        if system_monitor:
            monitor_history = system_monitor.stop_monitoring()
            system_metrics_summary = system_monitor.get_metrics_summary()
            
            # Guardar métricas detalladas si se especifica
            if args.monitor_file:
                monitor_filename = f"{args.monitor_file}.run_{i+1}.jsonl"
                system_monitor.save_metrics_to_file(monitor_filename)
                print(f"[info] Métricas detalladas guardadas en: {monitor_filename}")
        
        if not args.stream:
            print("="*80)
            print("RESPUESTA GENERADA:")
            print("="*80)
            print(text.strip())
            print("="*80)

        # Combinar métricas de Ollama con métricas del sistema
        m = summarize(stats, elapsed, system_metrics_summary)
        metrics.append(m)

        # Mostrar métricas básicas
        pf_tps = f"{m['prefill_tps']:.1f}" if m['prefill_tps'] else "n/a"
        dc_tps = f"{m['decode_tps']:.1f}" if m['decode_tps'] else "n/a"
        print(f"[ollama] wall={m['wall_s']:.2f}s | prefill={m['prefill_tokens']} tok @ {pf_tps} t/s | "
              f"decode={m['decode_tokens']} tok @ {dc_tps} t/s")
        
        # Mostrar métricas del sistema si están disponibles
        if system_metrics_summary and not args.no_system_monitor:
            cpu_mean = m.get('cpu_usage_mean_percent')
            cpu_max = m.get('cpu_usage_max_percent')
            ram_mean = m.get('ram_usage_mean_percent')
            ram_max = m.get('ram_usage_max_percent')
            
            if cpu_mean is not None:
                print(f"[system] CPU: {cpu_mean:.1f}% avg, {cpu_max:.1f}% max")
            if ram_mean is not None:
                print(f"[system] RAM: {ram_mean:.1f}% avg, {ram_max:.1f}% max")
                
            # Métricas de temperatura si están disponibles
            cpu_temp_mean = m.get('cpu_temp_mean_c')
            if cpu_temp_mean is not None:
                cpu_temp_max = m.get('cpu_temp_max_c')
                print(f"[system] CPU Temp: {cpu_temp_mean:.1f}°C avg, {cpu_temp_max:.1f}°C max")
                
            # Métricas de GPU si están disponibles  
            gpu_mean = m.get('gpu_usage_mean_percent')
            if gpu_mean is not None:
                gpu_max = m.get('gpu_usage_max_percent')
                print(f"[system] GPU: {gpu_mean:.1f}% avg, {gpu_max:.1f}% max")
                
            # Métricas de potencia si están disponibles
            power_mean = m.get('power_mean_watts')
            if power_mean is not None:
                power_max = m.get('power_max_watts')
                print(f"[system] Power: {power_mean:.1f}W avg, {power_max:.1f}W max")
            
            monitoring_duration = m.get('monitoring_duration_s')
            if monitoring_duration:
                print(f"[system] Monitoring: {monitoring_duration:.1f}s, {m.get('monitoring_samples')} samples")

        if args.out:
            rec = {"ts": time.time(), "model": args.model, **m}
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    if args.runs > 1:
        d_tps = [m["decode_tps"] for m in metrics if m["decode_tps"]]
        if d_tps:
            print(f"\n=== RESUMEN ESTADÍSTICO GPT-OSS 20B ===")
            print(f"Promedio decode_tps: {statistics.mean(d_tps):.1f} t/s")
            print(f"Mediana decode_tps: {statistics.median(d_tps):.1f} t/s")
            print(f"Min decode_tps: {min(d_tps):.1f} t/s")
            print(f"Max decode_tps: {max(d_tps):.1f} t/s")
            if len(d_tps) > 1:
                print(f"Desv. estándar: {statistics.stdev(d_tps):.1f} t/s")
            
            # Calcular throughput teórico basado en 20B parámetros
            if d_tps:
                avg_tps = statistics.mean(d_tps)
                # Estimación de parámetros activos por token (aproximación)
                params_per_token = 20e9  # 20 mil millones
                print(f"\n=== ANÁLISIS DE RENDIMIENTO ===")
                print(f"Parámetros del modelo: ~20B")
                print(f"Rendimiento promedio: {avg_tps:.1f} tokens/segundo")
                print(f"Procesamiento estimado: {(params_per_token * avg_tps) / 1e9:.1f}B parámetros/segundo")

if __name__ == "__main__":
    main()
