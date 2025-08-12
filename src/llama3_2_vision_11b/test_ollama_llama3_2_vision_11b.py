#!/usr/bin/env python3
"""
Script de Testing Modular para Llama 3.2 Vision 11B en Jetson AGX Orin.

Versión refactorizada y modularizada que evalúa el rendimiento del modelo
multimodal llama3.2-vision:11b siguiendo las mejores prácticas de Python.
"""

import argparse
import sys
from pathlib import Path

# Imports de módulos locales
from . import VisionTestRunner
from .image_utils import ImageProcessor


def setup_arguments() -> argparse.ArgumentParser:
    """
    Configura y retorna el parser de argumentos.
    
    Returns:
        ArgumentParser configurado
    """
    parser = argparse.ArgumentParser(
        description="Benchmark modular de Llama 3.2 Vision 11B en Jetson AGX Orin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Test básico con imagen
  python -m llama3_2_vision_11b.test_ollama_llama3_2_vision_11b --image assets/test.jpg -n 3
  
  # Comparación completa texto vs visión
  python -m llama3_2_vision_11b.test_ollama_llama3_2_vision_11b --test-mode both --image assets/test.jpg
  
  # Solo modo texto
  python -m llama3_2_vision_11b.test_ollama_llama3_2_vision_11b --test-mode text -n 5
        """
    )
    
    # Configuración del servidor
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="URL del servidor Ollama (default: http://localhost:11434)"
    )
    
    # Configuración del modelo
    parser.add_argument(
        "--model",
        default="llama3.2-vision:11b",
        help="Modelo a usar (default: llama3.2-vision:11b)"
    )
    
    # Prompts
    parser.add_argument(
        "--prompt",
        default="Describe en detalle qué ves en esta imagen. Sé específico con colores, objetos y su disposición.",
        help="Prompt para evaluación con imagen"
    )
    
    parser.add_argument(
        "--text-prompt",
        default="Resume en 5 líneas las capacidades del modelo Llama 3.2 Vision. Responde en español.",
        help="Prompt para test solo texto"
    )
    
    # Imagen
    parser.add_argument(
        "--image",
        help="Ruta a imagen para test de visión"
    )
    
    # Configuración de tests
    parser.add_argument(
        "-n", "--runs",
        type=int,
        default=3,
        help="Número de repeticiones por modo (default: 3)"
    )
    
    parser.add_argument(
        "--test-mode",
        choices=["both", "text", "vision"],
        default="both",
        help="Modo de test: both, text, o vision (default: both)"
    )
    
    # Opciones del modelo
    parser.add_argument(
        "--ctx",
        type=int,
        default=4096,
        help="Tamaño de ventana de contexto (default: 4096)"
    )
    
    parser.add_argument(
        "--temp",
        type=float,
        default=0.4,
        help="Temperature (default: 0.4)"
    )
    
    parser.add_argument(
        "--num-predict",
        type=int,
        default=256,
        help="Número máximo de tokens a generar (default: 256)"
    )
    
    # Comportamiento
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Mostrar salida token por token"
    )
    
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Omitir warmup inicial"
    )
    
    # Salida
    parser.add_argument(
        "--out",
        help="Archivo JSONL para guardar métricas"
    )
    
    # Sistema de monitoreo
    parser.add_argument(
        "--no-system-monitor",
        action="store_true", 
        help="Desactivar monitoreo de métricas del sistema (CPU, RAM, GPU, temperatura)"
    )
    
    parser.add_argument(
        "--monitor-file",
        help="Archivo para guardar métricas detalladas del sistema en formato JSONL"
    )
    
    return parser


def find_image_or_exit(image_path: str, test_mode: str) -> str:
    """
    Encuentra una imagen válida o termina el programa.
    
    Args:
        image_path: Ruta proporcionada por el usuario
        test_mode: Modo de test actual
        
    Returns:
        str: Ruta de imagen válida
        
    Raises:
        SystemExit: Si no se encuentra imagen válida
    """
    if image_path:
        if ImageProcessor.validate_image_path(image_path):
            return image_path
        else:
            print(f"❌ Error: Imagen no válida: {image_path}", file=sys.stderr)
            sys.exit(1)
    
    # Buscar imagen por defecto
    default_image = ImageProcessor.find_default_image()
    if default_image:
        print(f"📷 Usando imagen por defecto: {default_image}")
        return str(default_image)
    
    # Si necesitamos imagen y no la encontramos
    if test_mode in ["vision", "both"]:
        print(
            "❌ Error: Se requiere una imagen para test de visión.\n"
            "   Usa --image <ruta> o coloca una imagen en assets/",
            file=sys.stderr
        )
        sys.exit(1)
    
    return ""


def main():
    """Función principal del script refactorizado."""
    # Configurar argumentos
    parser = setup_arguments()
    args = parser.parse_args()
    
    # Configurar opciones del modelo
    model_options = {
        "num_ctx": args.ctx,
        "temperature": args.temp,
        "num_predict": args.num_predict
    }
    
    # Inicializar el runner de tests
    print(f"🚀 Iniciando tests de {args.model}")
    enable_monitoring = not args.no_system_monitor
    test_runner = VisionTestRunner(
        host=args.host, 
        enable_system_monitoring=enable_monitoring
    )
    
    # Verificar setup
    if not test_runner.verify_setup():
        sys.exit(1)
    
    # Ejecutar warmup si no se omite
    if not args.skip_warmup:
        test_runner.run_warmup(args.model)
    
    # Encontrar imagen si es necesaria
    image_path = find_image_or_exit(args.image, args.test_mode)
    
    # Ejecutar tests según el modo seleccionado
    results = {"text_metrics": [], "vision_metrics": []}
    
    if args.test_mode in ["both", "text"]:
        print("📝 Ejecutando tests en modo TEXTO...")
        results["text_metrics"] = test_runner.run_text_only_test(
            model=args.model,
            prompt=args.text_prompt,
            options=model_options,
            runs=args.runs,
            stream=args.stream
        )
    
    if args.test_mode in ["both", "vision"]:
        print("🖼️  Ejecutando tests en modo VISIÓN...")
        results["vision_metrics"] = test_runner.run_vision_test(
            model=args.model,
            prompt=args.prompt,
            image_path=image_path,
            options=model_options,
            runs=args.runs,
            stream=args.stream
        )
    
    # Mostrar comparación si se ejecutaron ambos modos
    if args.test_mode == "both" and results["text_metrics"] and results["vision_metrics"]:
        test_runner._print_comparison(results["text_metrics"], results["vision_metrics"])
    
    # Guardar resultados si se especifica archivo de salida
    if args.out:
        additional_fields = {
            "model": args.model,
            "host": args.host,
            "test_mode": args.test_mode
        }
        
        if test_runner.save_results(args.out, additional_fields):
            print(f"💾 Métricas guardadas en: {args.out}")
        else:
            print(f"⚠️  Warning: No se pudieron guardar las métricas", file=sys.stderr)
    
    # Guardar métricas detalladas del sistema si se especifica
    if args.monitor_file and test_runner.system_monitor:
        try:
            test_runner.system_monitor.save_metrics_to_file(args.monitor_file)
            print(f"💾 Métricas del sistema guardadas en: {args.monitor_file}")
        except Exception as e:
            print(f"⚠️  Warning: Error guardando métricas del sistema: {e}", file=sys.stderr)
    
    # Mostrar resumen final
    summary = test_runner.get_summary()
    if summary["overall"].get("total_runs", 0) > 1:
        overall_stats = summary["overall"]
        if "decode_tps" in overall_stats:
            print(f"\n🎯 RESUMEN GLOBAL:")
            print(f"   Total de runs: {overall_stats['total_runs']}")
            
            # Formateo seguro de estadísticas
            decode_stats = overall_stats.get('decode_tps', {})
            mean_tps = decode_stats.get('mean')
            min_tps = decode_stats.get('min')
            max_tps = decode_stats.get('max')
            
            mean_str = f"{mean_tps:.1f}" if isinstance(mean_tps, (int, float)) else "n/a"
            min_str = f"{min_tps:.1f}" if isinstance(min_tps, (int, float)) else "n/a"
            max_str = f"{max_tps:.1f}" if isinstance(max_tps, (int, float)) else "n/a"
            
            print(f"   Velocidad promedio: {mean_str} t/s")
            print(f"   Rango: {min_str} - {max_str} t/s")
    
    print("\n✅ Tests completados")


if __name__ == "__main__":
    main()
