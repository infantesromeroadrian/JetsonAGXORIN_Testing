# 🚀 Testing y Benchmarking de NVIDIA Jetson AGX Orin para LLMs

## 📋 Descripción del Proyecto

Este proyecto tiene como objetivo evaluar y documentar el rendimiento del **NVIDIA Jetson AGX Orin Developer Kit** ejecutando modelos de lenguaje grandes (LLMs), específicamente utilizando **Ollama** como servidor de inferencia. El proyecto incluye herramientas de benchmarking, análisis comparativo con otras GPUs, y documentación técnica detallada sobre las capacidades del hardware.

### 🎯 Objetivos Principales

1. **Evaluación de Rendimiento**: Medir la velocidad de inferencia (tokens/segundo) de diferentes modelos LLM en el Jetson AGX Orin
2. **Análisis Comparativo**: Comparar el rendimiento con otras soluciones de GPU (RTX Ada 2000)
3. **Optimización de Recursos**: Identificar las mejores prácticas para ejecutar LLMs en arquitectura ARM con memoria unificada
4. **Documentación Técnica**: Proporcionar guías detalladas sobre configuración y uso del Jetson para IA

## 🏗️ Arquitectura del Proyecto

```
Testin_Jetson_AGX_ORIN/
│
├── 📁 src/                       # Código fuente principal
│   ├── test_ollama_llama3_2_3b.py         # Testing individual modelo 3B
│   ├── sweep_ollama_llama3_2_3b.py        # Barrido paramétrico modelo 3B
│   ├── test_ollama_llama3_2_vision_11b.py # Testing modelo visión 11B
│   └── sweep_ollama_llama3_2_vision_11b.py # Barrido modelo visión 11B
│
├── 📁 docs/                      # Documentación técnica
│   ├── Informe_Tecnico_Jetson_AGX_Orin.md  # Informe técnico completo
│   └── Testing_JetsonAGXORIN.docx          # Documentación adicional
│
├── 📁 data/                      # Datos de pruebas y resultados
├── 📁 models/                    # Modelos y configuraciones
├── 📁 notebooks/                 # Jupyter notebooks para análisis
├── 📁 assets/                    # Recursos multimedia
│
├── requirements.txt              # Dependencias del proyecto
└── testing_jetson_venv/         # Entorno virtual Python
```

## 💻 Especificaciones del Hardware

### NVIDIA Jetson AGX Orin Developer Kit

| Característica | Especificación |
|----------------|----------------|
| **Arquitectura** | ARM aarch64 (64-bit) |
| **GPU** | NVIDIA integrada (compartida UMA) |
| **RAM** | 64 GB LPDDR5 (61 GB utilizable) |
| **Memoria GPU** | Unificada (UMA) - comparte RAM del sistema |
| **Almacenamiento** | eMMC ~59.2 GB |
| **CUDA** | 12.6 |
| **JetPack** | 6.2.1 (L4T r36.4.4) |
| **Sistema Operativo** | Ubuntu 22.04 LTS (Jammy) |

## 🔧 Instalación y Configuración

### Requisitos Previos

1. **Jetson AGX Orin** con JetPack 6.2.1 o superior instalado
2. **Python 3.8+** instalado en el sistema
3. **Ollama** servidor de inferencia instalado y configurado

### Pasos de Instalación

```bash
# 1. Clonar el repositorio
git clone <repository-url>
cd Testin_Jetson_AGX_ORIN

# 2. Crear y activar entorno virtual
python3 -m venv testing_jetson_venv
source testing_jetson_venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación de Ollama
sudo systemctl status ollama

# 5. Descargar modelo de prueba
ollama pull llama3.2:3b
```

## 🚀 Uso

### Test Simple de Rendimiento (Modelo 3B)

```bash
# Ejecutar test básico con 5 repeticiones
python src/test_ollama_llama3_2_3b.py --model llama3.2:3b -n 5 --out metrics.jsonl

# Test con streaming (ver tokens en tiempo real)
python src/test_ollama_llama3_2_3b.py --model llama3.2:3b --stream
```

### Test con Modelo de Visión (11B)

```bash
# Test con imagen (modo visión)
python src/test_ollama_llama3_2_vision_11b.py --image assets/puerto-new-york-1068x570.webp -n 3

# Comparación texto vs visión
python src/test_ollama_llama3_2_vision_11b.py --image assets/puerto-new-york-1068x570.webp --test-mode both

# Solo modo texto (sin usar capacidades de visión)
python src/test_ollama_llama3_2_vision_11b.py --test-mode text -n 5
```

### Barrido Paramétrico Completo

```bash
# Barrido modelo 3B
python src/sweep_ollama_llama3_2_3b.py \
    --model llama3.2:3b \
    --ctx 2048,4096 \
    --num-predict 128,256,512 \
    --temp 0,0.4,0.7 \
    --runs 3 \
    --csv results.csv \
    --out metrics.jsonl

# Barrido modelo visión 11B
python src/sweep_ollama_llama3_2_vision_11b.py \
    --image assets/puerto-new-york-1068x570.webp \
    --ctx 4096,8192 \
    --num-predict 128,256 \
    --runs 3 \
    --csv vision_results.csv
```

### Parámetros Disponibles

#### `test_ollama_llama3_2_3b.py`
- `--host`: URL del servidor Ollama (default: http://localhost:11434)
- `--model`: Modelo a evaluar (default: llama3.2:3b)
- `--prompt`: Texto de entrada para la prueba
- `-n, --runs`: Número de repeticiones
- `--stream`: Mostrar salida token por token
- `--ctx`: Tamaño de ventana de contexto
- `--temp`: Temperatura de generación
- `--out`: Archivo de salida para métricas (JSONL)

#### `sweep_ollama_llama3_2_3b.py`
- Todos los parámetros anteriores, más:
- `--cycles`: Repetir el barrido completo N veces
- `--num-predict`: Lista de tokens a generar
- `--seed`: Semillas para reproducibilidad
- `--csv`: Archivo CSV de salida
- `--warmup`: Hacer calentamiento antes de cada prueba
- `--sleep`: Pausa entre ejecuciones

## 📊 Resultados y Métricas

### Rendimiento en Jetson AGX Orin

| Modelo | Tamaño | Modo | Velocidad (t/s) | RAM Usada | GPU | Estado |
|--------|--------|------|-----------------|-----------|-----|--------|
| **llama3.2:3b** | ~2 GB | Texto | 44.8 | ~5.6 GB | 90-99% | ✅ Verificado |
| **llama3.2-vision:11b** | ~7 GB | Texto | 25.4 | ~12 GB | 90-99% | ✅ Verificado |
| **llama3.2-vision:11b** | ~7 GB | Visión | 13.8 | ~15 GB | 90-99% | ✅ Verificado |

### Comparación con RTX Ada 2000 (modelo 3B)

| Hardware | Velocidad (t/s) | Factor de Aceleración |
|----------|-----------------|----------------------|
| RTX Ada 2000 | 74.65 | 1.67× |
| Jetson AGX Orin | 44.80 | 1.00× (referencia) |

### Características del Modelo de Visión (llama3.2-vision:11b)

| Métrica | Valor | Estado |
|---------|-------|--------|
| **Velocidad modo texto** | 25.4 t/s | ✅ Verificado |
| **Velocidad modo visión** | 13.8 t/s | ✅ Verificado |
| **Factor texto/visión** | 1.84× (texto 84% más rápido) | ✅ Consistente |
| **Overhead por imagen** | ~15 segundos | ✅ Medido |
| **Prefill texto** | 419-774 tokens/seg | ✅ Muy eficiente |
| **Prefill primera imagen** | 4.1 tokens/seg | ⚠️ Lento inicial |
| **Prefill imagen en caché** | 154-161 tokens/seg | ✅ Mucho mejor |
| **Factor vs modelo 3B (texto)** | 57% de velocidad | ✅ Mejor de lo esperado |
| **Factor vs modelo 3B (visión)** | 31% de velocidad | ✅ Aceptable para visión |
| **Temperatura** | 60-65°C (estable) | ✅ Normal |
| **Consumo Energético** | 31-35W en carga, 5W en reposo | ✅ Eficiente |

## 📈 Monitoreo del Sistema

### Comandos Útiles

```bash
# Monitorear uso de recursos en tiempo real
sudo tegrastats --interval 1000

# Ver memoria disponible
free -h

# Verificar almacenamiento
df -h

# Monitorear temperatura y frecuencias
watch -n 1 'cat /sys/devices/virtual/thermal/thermal_zone*/temp'
```

## 🎯 Casos de Uso Recomendados

### ✅ Ideal para:
- **Modelos de 1-3B parámetros** para máximo rendimiento (40-45 t/s)
- **Modelos de 4-8B parámetros** en INT4/FP8 con buen balance
- **Modelo de visión 11B** para análisis de imágenes (16-22 t/s)
- Aplicaciones de edge computing con IA
- Inferencia en tiempo real con restricciones de energía
- Análisis de imágenes local sin cloud (seguridad/privacidad)
- Desarrollo y prototipado de soluciones embebidas de IA

### ⚠️ Considerar con cuidado:
- **Modelos de 13B parámetros** (justo en el límite de RAM)
- Aplicaciones con contextos muy largos (>8K tokens)
- Procesamiento de múltiples imágenes en paralelo (overhead de visión)
- Primera inferencia con imágenes nuevas (latencia inicial alta)

### ❌ No recomendado:
- Modelos superiores a 13B parámetros
- Modelo llama3.2-vision:90b (excede memoria disponible)
- Entrenamiento de modelos grandes
- Aplicaciones que requieran múltiples modelos simultáneos
- Procesamiento de video en tiempo real con modelos grandes

## 🔬 Análisis Técnico

### Arquitectura de Memoria Unificada (UMA)

El Jetson AGX Orin utiliza una arquitectura de memoria unificada donde CPU y GPU comparten la misma RAM del sistema. Esto tiene implicaciones importantes:

**Ventajas:**
- No hay overhead de transferencia CPU↔GPU
- Gestión simplificada de memoria
- Ideal para cargas de trabajo mixtas

**Consideraciones:**
- La "VRAM" disponible = RAM libre del sistema
- Competencia por ancho de banda entre CPU y GPU
- Importante mantener margen de RAM libre

### Optimizaciones Recomendadas

1. **Gestión de Memoria**
   - Mantener al menos 10-15% de RAM libre
   - Evitar que el sistema use swap
   - Monitorear con `tegrastats` durante inferencia

2. **Configuración de Modelos**
   - Usar cuantización (INT4/INT8) cuando sea posible
   - Ajustar `num_ctx` según necesidades reales
   - Considerar modelos más pequeños para producción

3. **Rendimiento**
   - Hacer warmup antes de mediciones críticas
   - Usar batch processing cuando sea aplicable
   - Configurar power mode en MAXN para máximo rendimiento

## 📚 Documentación Adicional

- [Informe Técnico Completo](docs/Informe_Tecnico_Jetson_AGX_Orin.md) - Análisis detallado del hardware, pruebas de modelos 3B y 11B (visión)
- [NVIDIA Jetson Documentation](https://docs.nvidia.com/jetson/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Llama 3.2 Vision Model](https://ollama.com/library/llama3.2-vision) - Documentación del modelo multimodal

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está diseñado con fines educativos y de investigación. Consulte el archivo LICENSE para más detalles.

## 🙏 Agradecimientos

- NVIDIA por el desarrollo del Jetson AGX Orin
- Comunidad de Ollama por el servidor de inferencia
- Contribuidores y testers del proyecto

## 📧 Contacto

Para preguntas, sugerencias o colaboraciones, por favor abra un issue en el repositorio.

---

**Nota**: Este proyecto está en desarrollo activo. Las métricas y recomendaciones pueden actualizarse según nuevos hallazgos y optimizaciones.
