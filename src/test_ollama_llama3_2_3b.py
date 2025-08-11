#!/usr/bin/env python3
import argparse, json, time, sys, statistics
import requests

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

def summarize(stats, elapsed):
    td = ns_to_s(stats.get("total_duration"))
    ed = ns_to_s(stats.get("eval_duration"))
    ped = ns_to_s(stats.get("prompt_eval_duration"))
    eval_count = stats.get("eval_count")
    prompt_eval_count = stats.get("prompt_eval_count")
    return {
        "wall_s": elapsed,
        "total_s": td,
        "prefill_tokens": prompt_eval_count,
        "decode_tokens": eval_count,
        "prefill_tps": safe_div(prompt_eval_count, ped) if ped else None,
        "decode_tps": safe_div(eval_count, ed) if ed else None,
    }

def main():
    ap = argparse.ArgumentParser(description="Benchmark sencillo de Ollama en Jetson")
    ap.add_argument("--host", default="http://localhost:11434", help="URL del servidor Ollama")
    ap.add_argument("--model", default="llama3.2:3b", help="Modelo a usar (ej: llama3.2:3b)")
    ap.add_argument("--prompt", default="Resume en 3 líneas qué es Jetson AGX Orin. Responde en español.",
                    help="Prompt a evaluar")
    ap.add_argument("-n", "--runs", type=int, default=1, help="Número de repeticiones")
    ap.add_argument("--stream", action="store_true", help="Muestra la salida token a token")
    ap.add_argument("--ctx", type=int, default=2048, help="num_ctx (ventana de contexto)")
    ap.add_argument("--temp", type=float, default=0.4, help="temperature")
    ap.add_argument("--out", help="Ruta para volcar métricas en JSONL (append)")
    args = ap.parse_args()

    if not wait_for_server(args.host):
        print(f"No conecta con {args.host}. Arranca el servicio: sudo systemctl start ollama", file=sys.stderr)
        sys.exit(1)

    options = {"num_ctx": args.ctx, "temperature": args.temp}

    # Warmup (evita penalización de primer uso)
    try:
        gen_once(args.host, args.model, "Hola.", options, stream=False)
    except Exception as e:
        print("Warmup error (continuamos):", e, file=sys.stderr)

    metrics = []
    for i in range(args.runs):
        print(f"\n>> Run {i+1}/{args.runs}")
        text, stats, elapsed = gen_once(args.host, args.model, args.prompt, options, stream=args.stream)
        if not args.stream:
            print(text.strip())

        m = summarize(stats, elapsed)
        metrics.append(m)

        pf_tps = f"{m['prefill_tps']:.1f}" if m['prefill_tps'] else "n/a"
        dc_tps = f"{m['decode_tps']:.1f}" if m['decode_tps'] else "n/a"
        print(f"[stats] wall={m['wall_s']:.2f}s | prefill={m['prefill_tokens']} tok @ {pf_tps} t/s | "
              f"decode={m['decode_tokens']} tok @ {dc_tps} t/s")

        if args.out:
            rec = {"ts": time.time(), "model": args.model, **m}
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")

    if args.runs > 1:
        d_tps = [m["decode_tps"] for m in metrics if m["decode_tps"]]
        if d_tps:
            print(f"\nPromedio decode_tps sobre {len(d_tps)} ejecuciones: {statistics.mean(d_tps):.1f} t/s")

if __name__ == "__main__":
    main()
