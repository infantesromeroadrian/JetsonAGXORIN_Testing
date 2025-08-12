#!/usr/bin/env python3
"""
Script de Barrido Paramétrico Modular para Llama 3.2 Vision 11B.

Versión refactorizada y modularizada que ejecuta barridos sistemáticos
del modelo llama3.2-vision:11b siguiendo las mejores prácticas de Python.
"""

import argparse
import statistics
import sys

# Imports de módulos locales
from .sweep_runner import SweepRunner, ParameterCombinationGenerator, PromptManager


def setup_arguments() -> argparse.ArgumentParser:
    """
    Configura y retorna el parser de argumentos para el barrido.
    
    Returns:
        ArgumentParser configurado
    """
    parser = argparse.ArgumentParser(
        description="Barrido paramétrico modular de Llama 3.2 Vision 11B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Barrido básico con imagen
  python -m llama3_2_vision_11b.sweep_ollama_llama3_2_vision_11b --image assets/test.jpg
  
  # Barrido completo variando múltiples parámetros
  python -m llama3_2_vision_11b.sweep_ollama_llama3_2_vision_11b \\
    --ctx "2048,4096" --temp "0,0.4,0.7" --runs 5 --csv results.csv
  
  # Solo modo texto con múltiples prompts
  python -m llama3_2_vision_11b.sweep_ollama_llama3_2_vision_11b \\
    --test-mode text --prompt-file prompts.txt --cycles 3
        """
    )
    
    # Configuración del servidor y modelo
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="URL del servidor Ollama (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--model", 
        default="llama3.2-vision:11b",
        help="Modelo a usar (default: llama3.2-vision:11b)"
    )
    
    # Prompts e imágenes
    parser.add_argument(
        "--prompt",
        help="Prompt único para usar en el barrido"
    )
    
    parser.add_argument(
        "--prompt-file",
        help="Archivo con lista de prompts (uno por línea)"
    )
    
    parser.add_argument(
        "--image",
        help="Imagen única para pruebas de visión"
    )
    
    parser.add_argument(
        "--image-dir", 
        help="Directorio con múltiples imágenes"
    )
    
    # Parámetros de barrido
    parser.add_argument(
        "--ctx",
        default="4096",
        help="Lista de tamaños de contexto separados por comas (default: 4096)"
    )
    
    parser.add_argument(
        "--num-predict",
        default="128,256", 
        help="Lista de tokens a generar separados por comas (default: 128,256)"
    )
    
    parser.add_argument(
        "--temp",
        default="0,0.4",
        help="Lista de temperaturas separadas por comas (default: 0,0.4)"
    )
    
    parser.add_argument(
        "--seed",
        default="42",
        help="Lista de semillas separadas por comas (vacío para aleatorio)"
    )
    
    # Control de ejecución  
    parser.add_argument(
        "-n", "--runs",
        type=int,
        default=3,
        help="Número de ejecuciones por combinación de parámetros (default: 3)"
    )
    
    parser.add_argument(
        "--cycles",
        type=int, 
        default=1,
        help="Número de ciclos completos del barrido (default: 1)"
    )
    
    parser.add_argument(
        "--test-mode",
        choices=["both", "text", "vision"],
        default="both",
        help="Modo de test: both, text, o vision (default: both)"
    )
    
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Pausa entre ejecuciones en segundos (default: 1.0)"
    )
    
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Ejecutar warmup antes de cada combinación de parámetros"
    )
    
    # Archivos de salida
    parser.add_argument(
        "--out",
        help="Archivo JSONL para guardar métricas detalladas"
    )
    
    parser.add_argument(
        "--csv",
        help="Archivo CSV para guardar resultados tabulares"
    )
    
    return parser


def print_sweep_config(prompts: list, images: list, 
                      contexts: list, num_predicts: list,
                      temperatures: list, seeds: list,
                      test_mode: str, runs: int, cycles: int) -> None:
    """
    Imprime configuración del barrido.
    
    Args:
        prompts: Lista de prompts
        images: Lista de imágenes  
        contexts: Lista de contextos
        num_predicts: Lista de tokens a predecir
        temperatures: Lista de temperaturas
        seeds: Lista de semillas
        test_mode: Modo de test
        runs: Runs por combinación
        cycles: Ciclos totales
    """
    print("📊 CONFIGURACIÓN DEL BARRIDO:")
    print(f"   Prompts: {len(prompts)}")
    print(f"   Imágenes: {len(images)} ({'con visión' if any(img[0] for img in images) else 'solo texto'})")
    print(f"   Contextos: {contexts}")
    print(f"   Tokens a generar: {num_predicts}")
    print(f"   Temperaturas: {temperatures}")
    print(f"   Semillas: {seeds}")
    print(f"   Modo de test: {test_mode}")
    print(f"   Runs por combinación: {runs}")
    print(f"   Ciclos: {cycles}")


def print_final_summary(summary: dict) -> None:
    """
    Imprime resumen final del barrido.
    
    Args:
        summary: Dict con resultados del barrido
    """
    print("\n" + "="*70)
    print("📈 RESUMEN FINAL DEL BARRIDO")
    print("="*70)
    
    print(f"⏱️  Tiempo total: {summary['elapsed_time_minutes']:.1f} minutos")
    print(f"✅ Jobs completados: {summary['completed_jobs']}/{summary['total_jobs']}")
    
    # Estadísticas por modo
    if "text_mode" in summary and "samples" in summary["text_mode"]:
        text_stats = summary["text_mode"]
        if text_stats["samples"] > 0:
            print(f"\n📝 MODO TEXTO ({text_stats['samples']} muestras):")
            print(f"   Media: {text_stats['mean']:.1f} t/s")
            print(f"   Mediana: {text_stats['median']:.1f} t/s")
            print(f"   Rango: {text_stats['min']:.1f} - {text_stats['max']:.1f} t/s")
    
    if "vision_mode" in summary and "samples" in summary["vision_mode"]:
        vision_stats = summary["vision_mode"]
        if vision_stats["samples"] > 0:
            print(f"\n🖼️  MODO VISIÓN ({vision_stats['samples']} muestras):")
            print(f"   Media: {vision_stats['mean']:.1f} t/s")
            print(f"   Mediana: {vision_stats['median']:.1f} t/s")
            print(f"   Rango: {vision_stats['min']:.1f} - {vision_stats['max']:.1f} t/s")
    
    # Comparación entre modos
    if "speed_comparison" in summary and "error" not in summary["speed_comparison"]:
        comp = summary["speed_comparison"]
        print(f"\n⚡ COMPARACIÓN TEXTO vs VISIÓN:")
        print(f"   Factor de velocidad: {comp['text_vs_vision_ratio']:.2f}x")
        print(f"   Texto es {comp['text_faster_by_percent']:.1f}% más rápido")
    
    # Top combinaciones
    if "combinations" in summary and summary["combinations"]:
        top_combos = sorted(summary["combinations"], 
                           key=lambda x: x["mean_speed"], reverse=True)[:5]
        
        print(f"\n🏆 TOP 5 COMBINACIONES (por velocidad):")
        for i, combo in enumerate(top_combos, 1):
            ph, img_path, ctx, npred, temp, seed = combo["combination"]
            img_name = img_path.split("/")[-1] if img_path else "texto"
            print(f"   {i}. {combo['mean_speed']:.1f} t/s - "
                  f"ctx={ctx} np={npred} temp={temp} img={img_name}")


def main():
    """Función principal del script de barrido refactorizado."""
    # Configurar argumentos
    parser = setup_arguments()
    args = parser.parse_args()
    
    # Inicializar el ejecutor de barridos
    print(f"🚀 Iniciando barrido paramétrico de {args.model}")
    sweep_runner = SweepRunner(host=args.host)
    
    # Verificar configuración
    if not sweep_runner.verify_setup():
        print(f"❌ Error: No se puede conectar con {args.host}", file=sys.stderr)
        print("   Ejecuta: sudo systemctl start ollama", file=sys.stderr)
        sys.exit(1)
    
    print(f"✅ Servidor Ollama disponible en {args.host}")
    
    # Parsear parámetros
    param_gen = ParameterCombinationGenerator()
    prompt_manager = PromptManager()
    
    try:
        contexts = param_gen.parse_int_list(args.ctx)
        num_predicts = param_gen.parse_int_list(args.num_predict)
        temperatures = param_gen.parse_float_list(args.temp)
        seeds = param_gen.parse_seed_list(args.seed)
    except ValueError as e:
        print(f"❌ Error parseando parámetros: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Cargar prompts e imágenes
    prompts = prompt_manager.load_prompts(args.prompt, args.prompt_file)
    images = sweep_runner.load_images(args.image, args.image_dir)
    
    # Mostrar configuración
    print_sweep_config(prompts, images, contexts, num_predicts, 
                      temperatures, seeds, args.test_mode, args.runs, args.cycles)
    
    # Validar modo visión
    if args.test_mode == "vision" and not any(img[0] for img in images):
        print("❌ Error: Modo visión requiere al menos una imagen", file=sys.stderr)
        print("   Usa --image <ruta> o --image-dir <directorio>", file=sys.stderr)
        sys.exit(1)
    
    # Ejecutar barrido
    try:
        print(f"\n🔥 Iniciando barrido...")
        summary = sweep_runner.run_sweep(
            model=args.model,
            prompts=prompts,
            images=images,
            contexts=contexts,
            num_predicts=num_predicts,
            temperatures=temperatures,
            seeds=seeds,
            test_mode=args.test_mode,
            runs_per_combo=args.runs,
            cycles=args.cycles,
            warmup=args.warmup,
            sleep_time=args.sleep,
            jsonl_output=args.out,
            csv_output=args.csv
        )
        
        # Mostrar resumen final
        print_final_summary(summary)
        
        # Mostrar archivos de salida
        if args.csv:
            print(f"\n💾 Resultados CSV: {args.csv}")
        if args.out:
            print(f"💾 Métricas JSONL: {args.out}")
        
        print("\n🎉 Barrido completado exitosamente")
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Barrido interrumpido por el usuario")
        print(f"   Jobs completados: {sweep_runner.completed_jobs}/{sweep_runner.total_jobs}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error durante el barrido: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
