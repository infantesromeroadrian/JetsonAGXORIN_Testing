#!/usr/bin/env python3
"""
Script de barrido paramétrico para GPT-OSS 20B
Realiza múltiples pruebas con diferentes configuraciones para análisis exhaustivo
del modelo de 20 mil millones de parámetros
"""
import argparse, json, time, sys, statistics, itertools, csv, hashlib, os
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

def gen_once(host, model, prompt, options=None, stream=False, timeout=600, seed=None):
    url = f"{host}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": stream}
    if options:
        payload["options"] = options
    if seed is not None:
        payload["seed"] = int(seed)

    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    elapsed = time.perf_counter() - t0
    return data.get("response", ""), data, elapsed

def summarize(stats, elapsed, system_metrics_summary=None):
    """
    Combina métricas de Ollama con métricas del sistema para sweeps de GPT-OSS 20B
    """
    td  = ns_to_s(stats.get("total_duration"))
    ld  = ns_to_s(stats.get("load_duration"))
    ed  = ns_to_s(stats.get("eval_duration"))
    ped = ns_to_s(stats.get("prompt_eval_duration"))
    eval_count = stats.get("eval_count")
    prompt_eval_count = stats.get("prompt_eval_count")
    
    # Métricas básicas de Ollama (existentes)
    base_metrics = {
        "wall_s": elapsed,
        "total_s": td,
        "load_s": ld,
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

def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def read_prompts(prompt, prompt_file):
    items = []
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t and not t.startswith("#"):  # permitir comentarios en archivos
                    items.append(t)
    if prompt:
        items.append(prompt)
    
    # Prompts por defecto específicos para GPT-OSS 20B (diversos estilos para probar capacidades)
    if not items:
        items = [
            "Explica de manera detallada cómo funciona el aprendizaje automático y sus aplicaciones en la vida cotidiana.",
            "Escribe una historia corta de ciencia ficción ambientada en el año 2075, incluye tecnologías emergentes.",
            "Analiza las ventajas y desventajas de las energías renovables frente a los combustibles fósiles.",
            "Redacta un ensayo sobre el impacto de las redes sociales en la comunicación humana moderna.",
            "Describe el proceso de fotosíntesis en las plantas y su importancia para el ecosistema global."
        ]
    return items

def short_hash(txt):
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:8]

def main():
    ap = argparse.ArgumentParser(description="Sweep de benchmarks para GPT-OSS 20B en Jetson AGX Orin")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--prompt", help="Prompt único (se añade a los del fichero si usas --prompt-file)")
    ap.add_argument("--prompt-file", help="Fichero con prompts, uno por línea")
    ap.add_argument("-n", "--runs", type=int, default=3, help="Runs por combinación")
    ap.add_argument("--cycles", type=int, default=1, help="Repetir el barrido completo N veces")
    ap.add_argument("--ctx", default="8192", help="Lista coma: 4096,8192,16384 (default: 8192)")
    ap.add_argument("--num-predict", default="256,512", help="Lista coma: 256,512,1024 (default: 256,512)")
    ap.add_argument("--temp", default="0.3,0.7,1.0", help="Lista coma: 0.3,0.7,1.0 (explorar creatividad)")
    ap.add_argument("--seed", default="42", help="Lista coma; vacío para omitir seed fija")
    ap.add_argument("--sleep", type=float, default=1.0, help="Pausa entre runs (seg, default: 1.0)")
    ap.add_argument("--warmup", action="store_true", help="Hacer warmup antes de cada combinación")
    ap.add_argument("--out", help="JSONL de salida (append)")
    ap.add_argument("--csv", help="CSV de salida (append)")
    ap.add_argument("--no-system-monitor", action="store_true", 
                    help="Desactivar monitoreo del sistema (solo métricas de Ollama)")
    ap.add_argument("--monitor-file", help="Guardar métricas detalladas del sistema en archivos JSONL")
    args = ap.parse_args()

    if not wait_for_server(args.host):
        print(f"No conecta con {args.host}. Arranca el servicio: sudo systemctl start ollama", file=sys.stderr)
        sys.exit(1)

    ctx_list   = parse_int_list(args.ctx)
    np_list    = parse_int_list(args.num_predict)
    temp_list  = parse_float_list(args.temp)
    seed_list  = [int(s) for s in args.seed.split(",") if s.strip()] if args.seed is not None and args.seed != "" else [None]
    prompts    = read_prompts(args.prompt, args.prompt_file)

    # preparar CSV con cabeceras extendidas
    if args.csv and not os.path.exists(args.csv):
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            base_headers = [
                "ts","host","model","prompt_hash","prompt_len",
                "ctx","num_predict","temp","seed","cycle","run_idx",
                "wall_s","total_s","load_s","prefill_tokens","decode_tokens",
                "prefill_tps","decode_tps"
            ]
            
            # Añadir cabeceras de sistema si el monitoreo está habilitado
            if not args.no_system_monitor:
                system_headers = [
                    "cpu_usage_mean_percent","cpu_usage_max_percent",
                    "ram_usage_mean_percent","ram_usage_max_percent",
                    "ram_used_mean_gb","ram_used_max_gb",
                    "gpu_usage_mean_percent","gpu_usage_max_percent",
                    "cpu_temp_mean_c","cpu_temp_max_c",
                    "gpu_temp_mean_c","gpu_temp_max_c",
                    "power_mean_watts","power_max_watts",
                    "monitoring_duration_s","monitoring_samples"
                ]
                base_headers.extend(system_headers)
            
            w.writerow(base_headers)

    # agregados por combinación
    agg = {}

    combos = list(itertools.product(prompts, ctx_list, np_list, temp_list, seed_list))
    total_jobs = len(combos) * args.runs * args.cycles
    
    print("="*80)
    print("🚀 GPT-OSS 20B - BARRIDO PARAMÉTRICO EXHAUSTIVO")
    print("="*80)
    print(f"📊 Modelo: {args.model} (20 mil millones de parámetros)")
    print(f"🔢 Configuración: {len(combos)} combinaciones × {args.runs} runs × {args.cycles} ciclos")
    print(f"🎯 Total de ejecuciones: {total_jobs}")
    print(f"📝 Prompts: {len(prompts)}")
    print(f"🌡️  Temperaturas: {temp_list} (exploración de creatividad)")
    print(f"📏 Contextos: {ctx_list} tokens")
    print(f"🔤 Predicciones: {np_list} tokens")
    print(f"⏱️  Pausa entre runs: {args.sleep}s")
    print(f"💻 Host: {args.host}")
    if not args.no_system_monitor:
        print("🔍 Monitoreo del sistema: ACTIVADO")
    print("="*80)
    
    # Crear monitor del sistema si está habilitado
    system_monitor = None if args.no_system_monitor else SystemMonitor()

    job_no = 0
    start_time = time.time()
    
    for cyc in range(1, args.cycles + 1):
        for prompt, ctx, num_predict, temp, seed in combos:
            ph = short_hash(prompt)
            options = {"num_ctx": ctx, "temperature": temp, "num_predict": num_predict}

            if args.warmup:
                try:
                    print(f"🔥 Warmup para ctx={ctx}, temp={temp}...")
                    if system_monitor:
                        system_monitor.start_monitoring()
                    gen_once(args.host, args.model, "Warmup para GPT-OSS 20B.", options, seed=seed)
                    if system_monitor:
                        system_monitor.stop_monitoring()  # limpiar métricas de warmup
                except Exception as e:
                    print("⚠️  Warmup error (continuamos):", e, file=sys.stderr)

            key = (ph, ctx, num_predict, temp, seed)
            agg.setdefault(key, [])

            for run_idx in range(1, args.runs + 1):
                job_no += 1
                elapsed_time = time.time() - start_time
                eta_minutes = (elapsed_time / job_no) * (total_jobs - job_no) / 60 if job_no > 0 else 0
                
                try:
                    # Iniciar monitoreo del sistema para esta ejecución
                    if system_monitor:
                        system_monitor.start_monitoring()
                        
                    text, stats, elapsed = gen_once(args.host, args.model, prompt, options, seed=seed)
                    
                    # Detener monitoreo y obtener resumen
                    system_metrics_summary = None
                    if system_monitor:
                        monitor_history = system_monitor.stop_monitoring()
                        system_metrics_summary = system_monitor.get_metrics_summary()
                        
                        # Guardar métricas detalladas si se especifica
                        if args.monitor_file:
                            monitor_filename = f"{args.monitor_file}.gpt_oss_cyc{cyc}_run{run_idx}_ctx{ctx}_np{num_predict}_temp{temp:.1f}.jsonl"
                            system_monitor.save_metrics_to_file(monitor_filename)
                    
                    m = summarize(stats, elapsed, system_metrics_summary)
                    agg[key].append(m["decode_tps"])

                    pf_tps = f"{m['prefill_tps']:.1f}" if m['prefill_tps'] else "n/a"
                    dc_tps = f"{m['decode_tps']:.1f}" if m['decode_tps'] else "n/a"
                    
                    # Mostrar progreso con información adicional para GPT-OSS 20B
                    base_info = (f"[{job_no:4d}/{total_jobs}] cyc={cyc} run={run_idx} "
                               f"ctx={ctx} np={num_predict} temp={temp} seed={seed}")
                    
                    perf_info = (f"| prefill={m['prefill_tokens']:3d} @ {pf_tps:>6s} t/s "
                               f"| decode={m['decode_tokens']:3d} @ {dc_tps:>6s} t/s "
                               f"| wall={m['wall_s']:6.2f}s")
                    
                    # Información específica del modelo
                    if m.get('decode_tps'):
                        # Cálculo de throughput teórico (20B parámetros)
                        params_per_sec = (20e9 * m['decode_tps']) / 1e9  # GigaOps/sec
                        throughput_info = f"| {params_per_sec:.1f} GOP/s"
                    else:
                        throughput_info = "| n/a GOP/s"
                    
                    # Añadir información del sistema si está disponible
                    system_info = ""
                    if system_metrics_summary and not args.no_system_monitor:
                        cpu_mean = m.get('cpu_usage_mean_percent')
                        ram_mean = m.get('ram_usage_mean_percent')
                        
                        if cpu_mean is not None:
                            system_info += f" CPU:{cpu_mean:.1f}%"
                        if ram_mean is not None:
                            system_info += f" RAM:{ram_mean:.1f}%"
                            
                        # Añadir temperatura si está disponible
                        cpu_temp = m.get('cpu_temp_mean_c')
                        if cpu_temp is not None:
                            system_info += f" T:{cpu_temp:.1f}°C"
                            
                        # Añadir potencia si está disponible
                        power = m.get('power_mean_watts')
                        if power is not None:
                            system_info += f" P:{power:.1f}W"
                    
                    # ETA estimado
                    eta_info = f"| ETA:{eta_minutes:.1f}min" if eta_minutes > 0.1 else ""
                    
                    full_info = base_info + perf_info + throughput_info + system_info + eta_info
                    print(full_info)

                    ts = time.time()
                    rec = {
                        "ts": ts, "host": args.host, "model": args.model,
                        "prompt_hash": ph, "prompt_len": len(prompt),
                        "ctx": ctx, "num_predict": num_predict, "temp": temp, "seed": seed,
                        "cycle": cyc, "run_idx": run_idx, **m
                    }

                    if args.out:
                        with open(args.out, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                    if args.csv:
                        with open(args.csv, "a", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            base_row = [
                                ts, args.host, args.model, ph, len(prompt),
                                ctx, num_predict, temp, seed, cyc, run_idx,
                                round(m["wall_s"],3), round(m["total_s"] or 0,3), round(m["load_s"] or 0,3),
                                m["prefill_tokens"], m["decode_tokens"],
                                round(m["prefill_tps"] or 0,2), round(m["decode_tps"] or 0,2)
                            ]
                            
                            # Añadir métricas del sistema si el monitoreo está habilitado
                            if not args.no_system_monitor:
                                system_row = [
                                    round(m.get("cpu_usage_mean_percent") or 0, 2),
                                    round(m.get("cpu_usage_max_percent") or 0, 2),
                                    round(m.get("ram_usage_mean_percent") or 0, 2),
                                    round(m.get("ram_usage_max_percent") or 0, 2),
                                    round(m.get("ram_used_mean_gb") or 0, 3),
                                    round(m.get("ram_used_max_gb") or 0, 3),
                                    round(m.get("gpu_usage_mean_percent") or 0, 2) if m.get("gpu_usage_mean_percent") else None,
                                    round(m.get("gpu_usage_max_percent") or 0, 2) if m.get("gpu_usage_max_percent") else None,
                                    round(m.get("cpu_temp_mean_c") or 0, 1) if m.get("cpu_temp_mean_c") else None,
                                    round(m.get("cpu_temp_max_c") or 0, 1) if m.get("cpu_temp_max_c") else None,
                                    round(m.get("gpu_temp_mean_c") or 0, 1) if m.get("gpu_temp_mean_c") else None,
                                    round(m.get("gpu_temp_max_c") or 0, 1) if m.get("gpu_temp_max_c") else None,
                                    round(m.get("power_mean_watts") or 0, 2) if m.get("power_mean_watts") else None,
                                    round(m.get("power_max_watts") or 0, 2) if m.get("power_max_watts") else None,
                                    round(m.get("monitoring_duration_s") or 0, 2),
                                    m.get("monitoring_samples") or 0
                                ]
                                base_row.extend(system_row)
                            
                            w.writerow(base_row)
                except Exception as e:
                    print(f"❌ ERROR en cyc={cyc} run={run_idx} ctx={ctx} np={num_predict} temp={temp}: {e}", file=sys.stderr)
                if args.sleep > 0:
                    time.sleep(args.sleep)

    # resumen final detallado
    total_elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("📈 RESUMEN FINAL - GPT-OSS 20B SWEEP")
    print("="*80)
    print(f"⏱️  Tiempo total: {total_elapsed/60:.1f} minutos")
    print(f"✅ Jobs completados: {job_no}/{total_jobs}")
    print(f"📊 Promedio por job: {total_elapsed/job_no:.1f}s")
    
    print("\n📋 RESULTADOS POR COMBINACIÓN:")
    print("-" * 120)
    print(f"{'Prompt':<8} {'Ctx':<5} {'NPred':<5} {'Temp':<4} {'Seed':<6} {'Mean':<8} {'Median':<8} {'Min':<8} {'Max':<8} {'Runs':<4}")
    print("-" * 120)
    
    for key, vals in agg.items():
        if not vals: 
            continue
        ph, ctx, npred, temp, seed = key
        valid_vals = [v for v in vals if v and v > 0]
        if not valid_vals:
            continue
            
        mean_tps = statistics.mean(valid_vals)
        med_tps = statistics.median(valid_vals)
        min_tps = min(valid_vals)
        max_tps = max(valid_vals)
        
        print(f"{ph:<8} {ctx:<5} {npred:<5} {temp:<4.1f} {str(seed):<6} "
              f"{mean_tps:<8.1f} {med_tps:<8.1f} {min_tps:<8.1f} {max_tps:<8.1f} {len(valid_vals):<4}")
    
    # Análisis de rendimiento por temperatura
    temp_analysis = {}
    for key, vals in agg.items():
        if not vals:
            continue
        ph, ctx, npred, temp, seed = key
        valid_vals = [v for v in vals if v and v > 0]
        if not valid_vals:
            continue
        
        if temp not in temp_analysis:
            temp_analysis[temp] = []
        temp_analysis[temp].extend(valid_vals)
    
    if temp_analysis:
        print("\n🌡️  ANÁLISIS POR TEMPERATURA:")
        print("-" * 50)
        for temp in sorted(temp_analysis.keys()):
            vals = temp_analysis[temp]
            mean_tps = statistics.mean(vals)
            print(f"Temperature {temp:.1f}: {mean_tps:.1f} t/s promedio ({len(vals)} samples)")
    
    print("="*80)
    print("✨ GPT-OSS 20B Sweep completado exitosamente!")
    if args.csv:
        print(f"📊 Resultados CSV: {args.csv}")
    if args.out:
        print(f"📄 Métricas JSONL: {args.out}")
    print("="*80)

if __name__ == "__main__":
    main()
