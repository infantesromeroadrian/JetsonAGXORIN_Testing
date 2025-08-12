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
│   ├── test_ollama_llama3_2_3b.py    # Script de testing individual
│   └── sweep_ollama_llama3_2_3b.py   # Script de barrido paramétrico
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

### Test Simple de Rendimiento

```bash
# Ejecutar test básico con 5 repeticiones
python src/test_ollama_llama3_2_3b.py --model llama3.2:3b -n 5 --out metrics.jsonl

# Test con streaming (ver tokens en tiempo real)
python src/test_ollama_llama3_2_3b.py --model llama3.2:3b --stream
```

### Barrido Paramétrico Completo

```bash
# Ejecutar barrido con múltiples configuraciones
python src/sweep_ollama_llama3_2_3b.py \
    --model llama3.2:3b \
    --ctx 2048,4096 \
    --num-predict 128,256,512 \
    --temp 0,0.4,0.7 \
    --runs 3 \
    --csv results.csv \
    --out metrics.jsonl
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

| Métrica | Valor |
|---------|-------|
| **Modelo** | llama3.2:3b (~2.0 GB) |
| **Velocidad de Decodificación** | ~44.8 tokens/seg |
| **Uso de GPU** | 90-99% @ 1.3 GHz |
| **Temperatura** | 60-61°C (estable) |
| **RAM Utilizada** | ~5.6 GB de 61 GB |
| **Consumo Energético** | 31-35W en carga, 5W en reposo |

### Comparación con RTX Ada 2000

| Hardware | Velocidad (t/s) | Factor de Aceleración |
|----------|-----------------|----------------------|
| RTX Ada 2000 | 74.65 | 1.67× |
| Jetson AGX Orin | 44.80 | 1.00× (referencia) |

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
- **Modelos de 4-8B parámetros** en INT4/FP8
- Aplicaciones de edge computing con IA
- Inferencia en tiempo real con restricciones de energía
- Desarrollo y prototipado de soluciones embebidas de IA

### ⚠️ Considerar con cuidado:
- **Modelos de 13B parámetros** (justo en el límite de RAM)
- Aplicaciones con contextos muy largos (>8K tokens)

### ❌ No recomendado:
- Modelos superiores a 13B parámetros
- Entrenamiento de modelos grandes
- Aplicaciones que requieran múltiples modelos simultáneos

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

- [Informe Técnico Completo](docs/Informe_Tecnico_Jetson_AGX_Orin.md) - Análisis detallado del hardware y pruebas
- [NVIDIA Jetson Documentation](https://docs.nvidia.com/jetson/)
- [Ollama Documentation](https://ollama.ai/docs)

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
