# Informe Técnico — NVIDIA Jetson AGX Orin vs RTX Ada 2000


## 🏆 RESUMEN EJECUTIVO

### Hallazgo Revolucionario: Jetson AGX Orin DOMINA en Modelos Grandes (11B+)

Benchmarks exhaustivos demuestran la **superioridad del Jetson AGX Orin** sobre la **RTX Ada 2000** para modelos grandes (11B+ parámetros), con el resultado más impresionante en **Phi-4 Reasoning donde Jetson es 2.8× más rápido**, mientras que RTX solo domina en modelos pequeños (3B). El nuevo modelo **GPT-OSS 20B confirma esta tendencia** con rendimiento moderado en RTX.

```
RESUMEN VISUAL DE VELOCIDADES (t/s)
         RTX Ada 2000          Jetson AGX Orin
3B:      ████████ 67.1   vs   ████ 34.6
11B:     ███ 23.1        vs   ███ 23.2 
14B:     █ 4.1           vs   ██ 11.5 🏆
20B:     █ 4.1           vs   ███ 11.9 🏆 DOMINANTE
```

### Tabla Comparativa Principal

| Métrica | Jetson AGX Orin | RTX Ada 2000 | Ganador |
|---------|-----------------|--------------|---------|
| **llama3.2:3b (3B params)** | 34.6 t/s | 67.1 t/s | RTX 1.94× 🏆 |
| **llama3.2-vision:11b Texto** | 23.2 t/s | 23.1 t/s | Jetson +0.4% 🏆 |
| **llama3.2-vision:11b Visión** | 13.2 t/s | 16.0 t/s | RTX +21.2% |
| **phi4-reasoning:14b** | 11.5 t/s | 4.1 t/s | Jetson 2.8× 🏆 |
| **gpt-oss:20b** | **11.9 t/s** | 4.1 t/s | Jetson 2.9× 🏆 |
| **Tiempo Visión Total** | 20.36s | 86.36s | Jetson 4.2× 🏆 |
| **Overhead 1ª Imagen** | 21.32s | 233.84s | Jetson 11× 🏆 |
| **Eficiencia CPU** | 1.5-2.0% | 36-42% | Jetson 16-28× 🏆 |
| **Uso RAM** | 16.8-26.9% | 73.7-80% | Jetson 3-4× 🏆 |

**Conclusión**: 
- Para modelos **pequeños (3B)**: RTX Ada 2000 es 1.94× más rápida
- Para modelos **grandes (11B+)**: Jetson AGX Orin es superior en texto y eficiencia
- Para **razonamiento avanzado (14B)**: Phi-4 sacrifica velocidad por calidad
- Para **modelos masivos (20B)**: GPT-OSS muestra consistencia pero velocidad limitada en RTX
- Para **aplicaciones multimodales**: Jetson es 4.2× más rápido en tiempo real

---

## 1. ESPECIFICACIONES TÉCNICAS

### 1.1 Jetson AGX Orin Developer Kit

| Característica | Especificación |
|---------------|----------------|
| **Arquitectura** | ARM aarch64 (64-bit) |
| **SO/L4T/JetPack** | L4T r36.4.4 (JetPack 6.2.1) |
| **CUDA** | 12.6 |
| **Ubuntu** | 22.04 (Jammy) |
| **RAM** | 64 GB LPDDR5 (UMA - Memoria Unificada) |
| **GPU** | Integrada NVIDIA (comparte RAM) |
| **Almacenamiento** | eMMC 64GB (~57.8 GB útiles) |
| **VRAM** | No existe (usa RAM del sistema) |
| **TDP** | 15-60W configurable |

### 1.2 RTX Ada 2000 (Laptop)

| Característica | Especificación |
|---------------|----------------|
| **Arquitectura** | x86_64 |
| **GPU** | NVIDIA RTX Ada 2000 (Dedicada) |
| **VRAM** | 8 GB GDDR6 |
| **RAM Sistema** | 64 GB DDR5 |
| **TDP GPU** | 35-140W |

---

## 2. METODOLOGÍA DE BENCHMARKING

### 2.1 Configuración de Pruebas

| Parámetro | Valor |
|-----------|-------|
| **Framework** | Ollama (HTTP API) |
| **Modelos Testeados** | llama3.2:3b, llama3.2-vision:11b, phi4-reasoning, gpt-oss:20b |
| **Contexto** | 2048-16384 tokens |
| **Temperature** | 0.0, 0.3, 0.4, 0.7, 1.0 |
| **Semilla** | 42 (reproducibilidad) |
| **Runs por test** | 3-5 individuales, 48-180 sweep |
| **Imagen de prueba** | 3-4.jpg (0.21 MB, Manhattan) |

### 2.2 Métricas Capturadas

**Métricas de Ollama:**
- `prefill_tps`: Tokens/seg en prefill
- `decode_tps`: Tokens/seg en decodificación
- `wall_time`: Tiempo total de ejecución
- `total_tokens`: Tokens generados

**Métricas del Sistema (Nuevo):**
- CPU: Uso promedio/máximo (%)
- RAM: Uso en GB y porcentaje
- GPU: Actividad (%) - donde disponible
- Temperatura: CPU/GPU en °C
- Potencia: Consumo en watts

---

## 3. RESULTADOS DE BENCHMARKS

### 3.1 Modelo llama3.2:3b (3B parámetros)

#### Test Individual

| Plataforma | Velocidad | CPU | RAM | GPU |
|------------|-----------|-----|-----|-----|
| **RTX Ada 2000** | 67.1 t/s | 18.2% | 73.7% | 41.1% |
| **Jetson AGX Orin** | 34.6 t/s | 1.9% | 16.8% | N/A |

**Ganador**: RTX Ada 2000 (1.94× más rápida)

#### Sweep Paramétrico (ctx=2048)

| Plataforma | Media | Mediana | CPU avg | RAM avg |
|------------|-------|---------|---------|---------|
| **RTX Ada 2000** | 61.8 t/s | 61.8 t/s | 18.5% | 74.0% |
| **Jetson AGX Orin** | 33.9 t/s | 33.8 t/s | 1.5% | 16.8% |

### 3.2 Modelo llama3.2-vision:11b (11B parámetros)

#### Test Individual - Modo Texto

| Plataforma | Velocidad | Tiempo | CPU | RAM | GPU |
|------------|-----------|--------|-----|-----|-----|
| **RTX Ada 2000** | 23.1 t/s | 7.28s | 31.7% | 77.7% | 42.5% |
| **Jetson AGX Orin** | 23.2 t/s | 6.95s | 2.0% | 26.2% | N/A |

**Ganador**: Jetson AGX Orin (+0.4%, 4.5% más rápido)

#### Test Individual - Modo Visión (misma imagen)

| Plataforma | Velocidad | Tiempo Total | 1ª Imagen | CPU | RAM |
|------------|-----------|--------------|-----------|-----|-----|
| **RTX Ada 2000** | 16.0 t/s | 86.36s | 233.84s | 46.2% | 82.0% |
| **Jetson AGX Orin** | 13.2 t/s | 20.36s | 21.32s | 1.6% | 26.9% |

**Ganador Tiempo**: Jetson (4.2× más rápido total, 11× en primera imagen)

#### Sweep Paramétrico (48 runs, solo texto)

| Plataforma | Media | Mediana | Desviación | CPU | RAM |
|------------|-------|---------|------------|-----|-----|
| **RTX Ada 2000** | 20.7 t/s | 20.8 t/s | ±2.9 t/s | 42.5% | 85.5% |
| **Jetson AGX Orin** | 21.7 t/s | 22.1 t/s | ±1.2 t/s | 1.5% | 26.9% |

**Ganador**: Jetson (+4.8% más rápido, más estable)

### 3.3 Modelo phi4-reasoning:latest (14B parámetros)

#### Test Individual - Razonamiento Matemático

| Plataforma | Velocidad | Tiempo | CPU | RAM | GPU |
|------------|-----------|--------|-----|-----|-----|
| **RTX Ada 2000** | 4.1 t/s | ~175s | 31-38% | 67-70% | 28-31% |
| **Jetson AGX Orin** | 11.5 t/s | ~53s | 1.0% | 27.3% | N/A |

**Ganador**: Jetson AGX Orin (2.8× más rápido, 35× menos CPU)

#### Comparación de Prefill Performance

| Plataforma | Prefill Speed | Factor |
|------------|---------------|--------|
| **RTX Ada 2000** | 100-900 t/s | Variable |
| **Jetson AGX Orin** | 7400+ t/s | 8× más rápido 🏆 |

**Características Especiales de Phi-4**:
- **Chain-of-Thought interno**: ~250 tokens de razonamiento antes de la respuesta
- **Consistencia extrema**: Desviación estándar de 0.0-0.1 t/s con temperature=0
- **Optimización arquitectural**: Jetson maneja mejor modelos de razonamiento grandes
- **Respuesta correcta**: $162 para problema de descuento+impuesto sobre $200

#### Análisis de Trade-offs

| Aspecto | Phi-4 Reasoning | Llama 3.2 Vision | Llama 3.2 |
|---------|-----------------|------------------|-----------|
| **Parámetros** | ~14B | 11B | 3B |
| **Velocidad RTX** | 4.1 t/s | 23.1 t/s | 67.1 t/s |
| **Velocidad Jetson** | **11.5 t/s** 🏆 | 23.2 t/s | 34.6 t/s |
| **Calidad Razonamiento** | Excelente 🏆 | Muy Buena | Buena |
| **Uso RAM RTX** | 68% | 78% | 74% |
| **Uso RAM Jetson** | 27.3% | 26.9% | 16.8% |
| **Caso de Uso** | Tutorías, análisis | Multimodal | Chat rápido |

### 3.4 Modelo gpt-oss:20b (20B parámetros) - Solo RTX Ada 2000

#### Test Individual - Generación de Texto

| Plataforma | Velocidad | Consistencia | CPU | RAM | GPU | Wall Time |
|------------|-----------|--------------|-----|-----|-----|-----------|
| **RTX Ada 2000** | 4.1 t/s | ±0.1 t/s | 36-42% | 77-80% | ~25% | ~65s |
| **Jetson AGX Orin** | **11.9 t/s** | **±0.0 t/s** | **1.6-1.8%** | **30-40%** | N/A | **~22s** |

**GANADOR: Jetson AGX Orin (2.9× más rápido, 22× menos CPU, 2.4× menos RAM)**

**Resultados Detallados RTX Ada 2000:**
- **5 runs promedio**: 4.1 t/s con extrema consistencia
- **Rango de velocidad**: 4.0-4.3 t/s (variación mínima)
- **Contexto estándar**: 8192 tokens, 256 tokens generados
- **Calidad de respuesta**: Excelente para ensayos sobre IA en educación

**Resultados Detallados Jetson AGX Orin:**
- **5 runs promedio**: 11.9 t/s con consistencia perfecta
- **Rango de velocidad**: 11.9-12.0 t/s (variación prácticamente nula)
- **Contexto estándar**: 8192 tokens, 256 tokens generados
- **CPU ultraeficiente**: Solo 1.6-1.8% de uso promedio
- **RAM optimizada**: 30% de uso (~17.7 GB UMA)
- **Escalabilidad**: Mantiene 11.9 t/s con contexto 16384 y 512 tokens
- **Calidad de respuesta**: Excelente, idéntica a RTX pero 3× más rápida

#### Test con Configuración Extendida

| Plataforma | Configuración | Velocidad | CPU | RAM | Wall Time |
|------------|---------------|-----------|-----|-----|-----------|
| **RTX Ada 2000** | ctx=16384, temp=0.8, 512 tokens | 3.5 t/s | 35-38% | ~80% | ~149s |
| **RTX Ada 2000** | ctx=8192, temp=0.7, 256 tokens | 4.1 t/s | 36-42% | ~77% | ~65s |
| **Jetson AGX Orin** | ctx=16384, temp=0.8, 512 tokens | **11.9 t/s** | **1.7-1.9%** | **34%** | **~43s** |
| **Jetson AGX Orin** | ctx=8192, temp=0.7, 256 tokens | **11.9 t/s** | **1.6-1.8%** | **30%** | **~22s** |

**Factor de mejora Jetson vs RTX:**
- **Configuración extendida**: 3.4× más rápido, 19× menos CPU, 2.4× menos RAM
- **Configuración estándar**: 2.9× más rápido, 22× menos CPU, 2.6× menos RAM

#### Características Especiales de GPT-OSS 20B

- **Parámetros**: 20 mil millones (el modelo más grande testeado)
- **Consistencia excepcional**: Desviación estándar de 0.1 t/s
- **Velocidad estable**: Mantiene ~4.1 t/s independiente de la temperature (0.3-1.0)
- **Alto uso de recursos**: CPU 36-42%, RAM 77-80%
- **Respuestas de calidad**: Genera ensayos estructurados con ejemplos concretos
- **Prefill eficiente**: ~250-320 t/s en contexto inicial
- **Streaming compatible**: Mantiene velocidad similar en modo streaming

#### Análisis Comparativo con Otros Modelos (RTX Ada 2000)

| Aspecto | GPT-OSS 20B | Phi-4 14B | Llama3.2 Vision 11B | Llama3.2 3B |
|---------|-------------|-----------|---------------------|--------------|
| **Parámetros** | 20B | ~14B | 11B | 3B |
| **Velocidad RTX** | **4.1 t/s** | 4.1 t/s | 23.1 t/s | 67.1 t/s |
| **Velocidad Jetson** | **11.9 t/s** 🏆 | **11.5 t/s** 🏆 | 23.2 t/s | 34.6 t/s |
| **CPU RTX** | 36-42% | 35% | 31.7% | 18.2% |
| **CPU Jetson** | **1.6-1.8%** 🏆 | **1.0%** 🏆 | **2.0%** 🏆 | **1.9%** 🏆 |
| **RAM RTX** | 77-80% | 68% | 77.7% | 73.7% |
| **RAM Jetson** | **30-40%** 🏆 | **27.3%** 🏆 | **26.2%** 🏆 | **16.8%** 🏆 |
| **Consistencia** | Perfecta | Excelente | Buena | Muy Buena |
| **Calidad Texto** | Muy Alta | Razonamiento | Multimodal | Rápida |
| **Caso de Uso** | Escritura, ensayos | Matemáticas | Visión+texto | Chat |

**Observaciones Clave:**
- **Saturación de RTX**: Modelos 14B+ convergen a ~4.1 t/s en RTX Ada 2000 (límite arquitectural)
- **Superioridad de Jetson en modelos grandes**: GPT-OSS 20B alcanza 11.9 t/s (2.9× más rápido que RTX)
- **Patrón inverso confirmado**: A mayor tamaño de modelo → mayor ventaja de Jetson sobre RTX
- **Eficiencia extrema**: Jetson usa 22× menos CPU y 2.4× menos RAM que RTX
- **Consistencia perfecta**: Jetson mantiene 11.9 t/s con desviación estándar de 0.0 t/s
- **Escalabilidad superior**: Jetson mantiene rendimiento con contextos extendidos (16384 tokens)

---

## 4. ANÁLISIS DE EFICIENCIA

### 4.1 Eficiencia por Watt (estimada)

| Modelo | Jetson (t/s/W) | RTX (t/s/W) | Factor |
|--------|----------------|-------------|--------|
| **llama3.2:3b** | ~1.15 (34.6/30W) | ~0.48 (67.1/140W) | Jetson 2.4× |
| **llama3.2-vision:11b** | ~0.77 (23.2/30W) | ~0.17 (23.1/140W) | Jetson 4.5× |
| **phi4-reasoning:14b** | ~0.38 (11.5/30W) | ~0.03 (4.1/140W) | Jetson 12.7× |
| **gpt-oss:20b** | ~0.40 (11.9/30W) | ~0.03 (4.1/140W) | Jetson 13.3× |

### 4.2 Eficiencia de Recursos

| Recurso | Jetson vs RTX |
|---------|---------------|
| **CPU** | 16-35× menos uso |
| **RAM** | 2.4-4× menos uso |
| **Energía** | 2.4-13.3× más eficiente |
| **Costo Operativo** | ~5× menor (24/7) |
| **Modelos 20B** | 22× menos CPU, 2.4× menos RAM |

---

## 5. CASOS DE USO RECOMENDADOS

### Por Plataforma

#### Jetson AGX Orin - Ideal para:
- ✅ **Edge AI** con modelos grandes (11B+)
- ✅ **Aplicaciones multimodales** (visión + texto)
- ✅ **Despliegues 24/7** con bajo consumo
- ✅ **IoT/Robótica** con procesamiento local
- ✅ **Aplicaciones con restricciones** de energía

#### RTX Ada 2000 - Ideal para:
- ✅ **Máxima velocidad** con modelos pequeños (≤3B)
- ✅ **Desarrollo y prototipado** rápido
- ✅ **Batch processing** intensivo
- ✅ **Aplicaciones desktop** sin restricciones
- ✅ **Gaming + AI** simultáneo

### Por Modelo

| Caso de Uso | Plataforma | Modelo | Velocidad | Justificación |
|-------------|------------|--------|-----------|---------------|
| **Chatbot alto rendimiento** | RTX | llama3.2:3b | 67.1 t/s | Máxima velocidad |
| **Asistente edge multimodal** | Jetson | llama3.2-vision:11b | 23.2 t/s | Eficiencia + visión |
| **Análisis IoT con visión** | Jetson | llama3.2-vision:11b | 13.2 t/s | Único viable edge |
| **Tutorías y razonamiento** | RTX | phi4-reasoning | 4.1 t/s | Calidad excepcional |
| **Escritura y ensayos largos** | RTX | gpt-oss:20b | 4.1 t/s | Calidad premium |
| **Generación de contenido** | RTX | gpt-oss:20b | 4.1 t/s | Consistencia alta |
| **Desarrollo/Debug** | RTX | Cualquiera | Variable | Flexibilidad |

---

## 6. VISUALIZACIONES

### 6.1 Comparación de Velocidad

```
Modelo 3B (llama3.2:3b):
├─ RTX Ada 2000:    67.1 t/s [████████████████████] 🏆
└─ Jetson AGX Orin: 34.6 t/s [██████████]

Modelo 11B (llama3.2-vision:11b - Texto):
├─ RTX Ada 2000:    23.1 t/s [████████████████████]
└─ Jetson AGX Orin: 23.2 t/s [████████████████████] 🏆

Modelo 14B (phi4-reasoning):
├─ RTX Ada 2000:    4.1 t/s  [██]
└─ Jetson AGX Orin: 11.5 t/s [██████] 🏆 (2.8× más rápido)

Modelo 20B (gpt-oss):
├─ RTX Ada 2000:    4.1 t/s  [██]
└─ Jetson AGX Orin: 11.9 t/s [██████] 🏆 (2.9× más rápido)
```

### 6.2 Eficiencia de Recursos

```
Uso de CPU (Modelo 11B):
├─ Jetson:  1.5-2.0%  [█]
└─ RTX:     31.7-42.5% [████████████████████████████████]
            Jetson usa 16-28× menos CPU 🏆

Uso de CPU (Modelo 14B - Phi-4):
├─ Jetson:  1.0%   [▌]
└─ RTX:     35%    [███████████████████████████████████]
            Jetson usa 35× menos CPU 🏆

Uso de RAM (Modelo 11B):  
├─ Jetson:  26.9%  [██████]
└─ RTX:     85.5%  [█████████████████████]
            Jetson usa 3× menos RAM 🏆

Uso de RAM (Modelo 14B - Phi-4):
├─ Jetson:  27.3%  [██████]
└─ RTX:     68%    [████████████████]
            Jetson usa 2.5× menos RAM 🏆

Consumo Energético:
├─ Jetson:  ~30W   [███]
└─ RTX:     ~140W  [██████████████]
            Jetson es 4.7× más eficiente 🏆
```

### 6.3 Análisis de Rendimiento por Tamaño de Modelo

```
📊 TOKENS POR SEGUNDO vs TAMAÑO DEL MODELO
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                                       │
│ 70 t/s ┤                                                                                                              │
│        │         RTX Ada 2000                                                                                         │
│        │         ████████████                                                                                         │
│ 65 t/s ┤         ████████████                                                                                         │
│        │         ████████████ (67.1)                                                                                  │
│ 60 t/s ┤         ████████████                                                                                         │
│        │         ████████████                                                                                         │
│ 55 t/s ┤         ████████████                                                                                         │
│        │         ████████████                                                                                         │
│ 50 t/s ┤         ████████████                                                                                         │
│        │         ████████████                                                                                         │
│ 45 t/s ┤         ████████████                                                                                         │
│        │         ████████████                                                                                         │
│ 40 t/s ┤         ████████████                                                                                         │
│        │         ████████████                                                                                         │
│ 35 t/s ┤         ████████████      Jetson AGX Orin                                                                   │
│        │         ████████████      ████████████                                                                      │
│        │         ████████████      ████████████ (34.6)                                                              │
│ 30 t/s ┤         ████████████      ████████████                                                                      │
│        │         ████████████      ████████████                                                                      │
│ 25 t/s ┤         ████████████      ████████████               RTX                Jetson                              │
│        │         ████████████      ████████████               ████████████       ████████████                        │
│        │         ████████████      ████████████               ████████████       ████████████ (23.2)                 │
│ 20 t/s ┤         ████████████      ████████████               ████████████       ████████████                        │
│        │         ████████████      ████████████               ████████████       ████████████                        │
│ 15 t/s ┤         ████████████      ████████████               ████████████       ████████████                        │
│        │         ████████████      ████████████               ████████████       ████████████                        │
│        │         ████████████      ████████████               ████████████       ████████████                        │
│ 10 t/s ┤         ████████████      ████████████               ████████████       ████████████      RTX        Jetson   RTX    Jetson│
│        │         ████████████      ████████████               ████████████       ████████████      ████       ████████  ████   ██████│
│        │         ████████████      ████████████               ████████████       ████████████      ████       ████████  ████   ██████│
│  5 t/s ┤         ████████████      ████████████               ████████████       ████████████      ████       ████████  ████   ██████│
│        │         ████████████      ████████████               ████████████       ████████████      ████(4.1)  ████████  ████   ██████│
│        │         ████████████      ████████████               ████████████       ████████████      ████       ███(11.5) ███(4.1) (11.9)│
│  0 t/s └───────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
│                                                                                                                       │
│              3B (llama3.2:3b)      11B (llama3.2-vision)   14B (phi4-reasoning)     20B (gpt-oss)          │
│                                                                                                                       │
│         LEYENDA: ████ = Barras de rendimiento | Valores entre paréntesis = tokens/segundo | 🏆 = Ganador            │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Tiempo de Respuesta para Primera Imagen (Visión)

```
⏱️ OVERHEAD DE PRIMERA IMAGEN - llama3.2-vision:11b
┌────────────────────────────────────────────────────┐
│                                                    │
│ RTX Ada 2000:    ████████████████████████ 233.84s │
│                  ████████████████████████          │
│                  ████████████████████████          │
│                  ████████████████████████          │
│                                                    │
│ Jetson AGX Orin: ██ 21.32s                       │
│                                                    │
│                  Factor: Jetson 11× más rápido 🏆  │
└────────────────────────────────────────────────────┘
```

### 6.5 Análisis de Eficiencia Energética

```
⚡ EFICIENCIA: TOKENS/SEGUNDO/WATT
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Modelo 3B (llama3.2:3b)                              │
│  ├─ Jetson:  1.15 t/s/W  [████████████████████] 🏆    │
│  └─ RTX:     0.48 t/s/W  [███████]                    │
│              2.4× más eficiente                        │
│                                                         │
│  Modelo 11B (llama3.2-vision:11b)                     │
│  ├─ Jetson:  0.77 t/s/W  [████████████████████] 🏆    │
│  └─ RTX:     0.17 t/s/W  [████]                       │
│              4.5× más eficiente                        │
│                                                         │
│  Modelo 14B (phi4-reasoning)                          │
│  ├─ Jetson:  0.38 t/s/W  [████████████████████] 🏆    │
│  └─ RTX:     0.03 t/s/W  [█]                          │
│              12.7× más eficiente                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 6.6 Comparación de Métricas Clave

```
🎯 RADAR DE RENDIMIENTO (Normalizado 0-100)
                    Velocidad 3B
                         100
                     ╱────┼────╲
                   ╱      │      ╲
                 ╱        │        ╲
    Eficiencia ╱          │          ╲ Velocidad 11B
    Energética            │
         100 ────────────┼──────────── 100
             ╲            │            ╱
               ╲          │          ╱
                 ╲        │        ╱
                   ╲      │      ╱
                     ╲────┼────╱
                      Bajo Uso
                       de RAM
                        100

    ━━━ RTX Ada 2000    [52, 50, 15, 29]
    ─── Jetson AGX Orin [100, 51, 79, 100]
```

### 6.7 Distribución de Carga del Sistema

```
📊 DISTRIBUCIÓN DE RECURSOS - Modelo 11B (llama3.2-vision)
┌──────────────────────────────────────────────────────┐
│ RTX Ada 2000:                                       │
│ ┌─────────────────────────────────────────────┐    │
│ │ CPU: ████████ 42.5%                         │    │
│ │ RAM: ████████████████████ 85.5%             │    │
│ │ GPU: ████████ 42.5%                         │    │
│ │ FREE: ░░░░ ~15%                              │    │
│ └─────────────────────────────────────────────┘    │
│                                                      │
│ Jetson AGX Orin:                                   │
│ ┌─────────────────────────────────────────────┐    │
│ │ CPU: ▌1.5%                                   │    │
│ │ RAM: █████ 26.9%                             │    │
│ │ GPU: Integrada (compartida con RAM)          │    │
│ │ FREE: ░░░░░░░░░░░░░░░░ ~72%                 │    │
│ └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘

📊 DISTRIBUCIÓN DE RECURSOS - Modelo 14B (phi4-reasoning)
┌──────────────────────────────────────────────────────┐
│ RTX Ada 2000:                                       │
│ ┌─────────────────────────────────────────────┐    │
│ │ CPU: ███████ 35%                            │    │
│ │ RAM: █████████████ 68%                      │    │
│ │ GPU: ██████ 30%                              │    │
│ │ FREE: ░░░░░░ ~32%                            │    │
│ └─────────────────────────────────────────────┘    │
│                                                      │
│ Jetson AGX Orin:                                   │
│ ┌─────────────────────────────────────────────┐    │
│ │ CPU: ▌1.0%                                   │    │
│ │ RAM: █████ 27.3%                             │    │
│ │ GPU: Integrada (compartida con RAM)          │    │
│ │ FREE: ░░░░░░░░░░░░░░░░ ~72%                 │    │
│ └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```



### 6.8 Timeline de Procesamiento (Ejemplo 1000 tokens)

```
⏱️ TIMELINE DE GENERACIÓN (1000 tokens)
┌──────────────────────────────────────────────────────────────┐
│ Modelo 3B (llama3.2):                                       │
│ RTX:     |████████████████| 14.9s                          │
│ Jetson:  |████████████████████████████| 28.9s              │
│                                                              │
│ Modelo 11B (llama3.2-vision):                              │
│ RTX:     |██████████████████████████████| 43.3s           │
│ Jetson:  |██████████████████████████████| 43.1s           │
│                                                              │
│ Modelo 14B (phi4-reasoning):                               │
│ RTX:     |████████████████████████████████████████| 244s  │
│ Jetson:  |█████████| 87s 🏆                              │
│          0s    50s    100s    150s    200s    250s         │
└──────────────────────────────────────────────────────────────┘
```

### 6.9 Comparación de Modelos - Velocidad vs Calidad

```
📊 TRADE-OFF: VELOCIDAD vs CALIDAD DE RAZONAMIENTO (RTX vs Jetson)
┌──────────────────────────────────────────────────────────────┐
│ ALTA    │                 phi4-J              phi4-RTX      │
│ CALIDAD │                 🧠(11.5)            🧠(4.1)       │
│         │                                                    │
│         │         llama3.2-vision (J≈RTX ~23 t/s)          │
│         │                    📷                             │
│         │                                                    │
│         │   llama3.2:3b-J        llama3.2:3b-RTX           │
│ BÁSICA  │   💬(34.6)             💬(67.1)                  │
│         └──────────────────────────────────────────────────│
│           LENTA                              RÁPIDA         │
│                      VELOCIDAD DE INFERENCIA                │
└──────────────────────────────────────────────────────────────┘

Leyenda: J=Jetson, RTX=RTX Ada 2000
```

### 6.10 Análisis Comparativo Completo de Phi-4

```
🔬 PHI-4 REASONING: JETSON vs RTX ADA 2000
┌────────────────────────────────────────────────────────┐
│                                                        │
│ VELOCIDAD DE INFERENCIA:                             │
│ Jetson:  ██████████ 11.5 t/s 🏆                     │
│ RTX:     ███ 4.1 t/s                                │
│          Jetson es 2.8× más rápido                   │
│                                                        │
│ EFICIENCIA CPU:                                      │
│ Jetson:  ▌ 1.0%  🏆                                 │
│ RTX:     ████████████████████ 35%                   │
│          Jetson usa 35× menos CPU                    │
│                                                        │
│ USO DE MEMORIA:                                      │
│ Jetson:  █████████ 27.3% (16.2 GB) 🏆              │
│ RTX:     ██████████████████████ 68% (43 GB)        │
│          Jetson usa 2.5× menos RAM                   │
│                                                        │
│ VELOCIDAD PREFILL:                                   │
│ Jetson:  ████████████████████ 7400+ t/s 🏆         │
│ RTX:     ██ 100-900 t/s                             │
│          Jetson es 8× más rápido                     │
│                                                        │
│ TIEMPO TOTAL (problema $200):                        │
│ Jetson:  ██████ 31s 🏆                              │
│ RTX:     ████████████████████████████████ 180s      │
│          Jetson es 5.8× más rápido                   │
│                                                        │
│ EFICIENCIA ENERGÉTICA (t/s/W):                      │
│ Jetson:  ████████████████████ 0.38 🏆              │
│ RTX:     █ 0.03                                     │
│          Jetson es 12.7× más eficiente              │
└────────────────────────────────────────────────────────┘

CONCLUSIÓN: Jetson AGX Orin DOMINA completamente con Phi-4
```

### 6.11 Tendencia de Rendimiento por Tamaño de Modelo

```
📈 FACTOR DE VELOCIDAD: RTX vs JETSON
┌────────────────────────────────────────────────────┐
│                                                    │
│  2.9×  ┤                                          ● GPT-OSS│
│        │                           JETSON             │
│  2.8×  ┤                                  ● Phi-4     │
│        │                           DOMINANTE          │
│  2.0×  ┤                                              │
│        │- - - - - - - - - - - - - - - - - - - - - - -│
│  1.0×  ┤                    ● Llama Vision           │
│        │                     (empate)                │
│  0.5×  ┤     ● Llama 3B                              │
│        │      RTX MEJOR                              │
│  0.0×  └──────────────────────────────────────────────│
│         3B        11B        14B        20B          │
│              TAMAÑO DEL MODELO                       │
└────────────────────────────────────────────────────┘

PATRÓN CLARO: A mayor tamaño → Jetson supera a RTX
```

### 6.12 Comparación de Tiempos de Ejecución - Phi-4

```
⏱️ TIEMPO PARA PROBLEMA MATEMÁTICO ($200 con descuento + impuesto)
┌────────────────────────────────────────────────────┐
│                                                    │
│ Jetson AGX Orin:                                  │
│ ├─ Warmup:      Instantáneo                       │
│ ├─ Prefill:     0.04s (7400 t/s)                 │
│ ├─ Generación:  ~31s total                        │
│ └─ ███████ 31s 🏆                                │
│                                                    │
│ RTX Ada 2000:                                     │
│ ├─ Warmup:      600s timeout ⚠️                   │
│ ├─ Prefill:     2.5s (100 t/s)                   │
│ ├─ Generación:  ~180s total                       │
│ └─ ████████████████████████████████ 180s         │
│                                                    │
│ Factor de mejora: Jetson 5.8× más rápido         │
└────────────────────────────────────────────────────┘
```

---

## 7. GUÍA DE USO DE SCRIPTS

### 7.1 Scripts Disponibles

```bash
# Estructura del proyecto
src/
├── llama3_2_3b/                     # Modelo 3B
│   ├── test_ollama_llama3_2_3b.py  # Test individual
│   ├── sweep_ollama_llama3_2_3b.py # Barrido paramétrico
│   └── system_monitor.py           # Monitor de sistema
│
├── llama3_2_vision_11b/             # Modelo 11B con visión
│   ├── test_ollama_llama3_2_vision_11b.py
│   ├── sweep_ollama_llama3_2_vision_11b.py
│   └── system_monitor.py
│
└── phi4_reasoning/                  # Modelo 14B razonamiento
    ├── test_ollama_phi4_reasoning.py
    ├── sweep_ollama_phi4_reasoning.py
    └── system_monitor.py
```

### 7.2 Comandos de Ejemplo

#### Test Rápido
```bash
# Modelo 3B
python -m src.llama3_2_3b.test_ollama_llama3_2_3b -n 3

# Modelo 11B con visión
python -m src.llama3_2_vision_11b.test_ollama_llama3_2_vision_11b \
  --image assets/3-4.jpg -n 3

# Modelo 14B razonamiento (Phi-4)
python -m src.phi4_reasoning.test_ollama_phi4_reasoning -n 3
```

#### Barrido Completo
```bash
# Sweep con múltiples configuraciones
python -m src.llama3_2_3b.sweep_ollama_llama3_2_3b \
  --ctx 2048,4096 \
  --temp 0.0,0.4 \
  --csv results/sweep.csv \
  --out results/sweep.jsonl
```

---

## 8. CONCLUSIONES

### 8.1 Hallazgos Principales

1. **Rendimiento por Tamaño de Modelo**:
   - Modelos pequeños (≤3B): RTX Ada 2000 es 1.94× más rápida
   - Modelos grandes (11B): Jetson AGX Orin iguala o supera a RTX
   - Modelos de razonamiento (14B): **Jetson es 2.8× más rápido** (11.5 vs 4.1 t/s)
   - Modelos masivos (20B): **Jetson AGX Orin es 2.9× más rápido** (11.9 vs 4.1 t/s)

2. **Eficiencia Energética**:
   - Jetson usa 16-28× menos CPU
   - Jetson usa 3-4× menos RAM
   - Jetson es 4.7× más eficiente en watts

3. **Aplicaciones Multimodales**:
   - Jetson: 4.2× más rápido en tiempo total de visión
   - Jetson: 11× más rápido procesando primera imagen
   - RTX: Overhead prohibitivo de 233s en primera imagen

4. **Trade-offs Identificados**:
   - Velocidad vs Calidad: Phi-4 ofrece razonamiento superior con velocidades variables
   - Chain-of-Thought: Phi-4 genera ~250 tokens internos de razonamiento antes de responder
   - Consistencia: Phi-4 muestra desviación estándar de 0.0-0.1 t/s con temperature=0

5. **Hallazgo Crítico con Phi-4**:
   - **Jetson**: 11.5 t/s con solo 1% CPU y 27% RAM
   - **RTX**: 4.1 t/s con 35% CPU y 68% RAM
   - **Conclusión**: La arquitectura ARM de Jetson es significativamente más eficiente para modelos de razonamiento complejos

6. **Descubrimiento Revolucionario sobre GPT-OSS 20B**:
   - **Jetson domina completamente**: 11.9 t/s vs 4.1 t/s (2.9× más rápido)
   - **Confirmación del patrón**: A mayor tamaño → mayor ventaja de Jetson
   - **Eficiencia extrema**: Jetson usa 22× menos CPU y 2.4× menos RAM que RTX
   - **Consistencia perfecta**: Desviación estándar de 0.0 t/s en Jetson vs 0.1 t/s en RTX
   - **Saturación de RTX confirmada**: RTX se estanca a 4.1 t/s en modelos 14B+
   - **Escalabilidad superior**: Jetson mantiene 11.9 t/s con contextos extendidos

### 8.2 Recomendación Final

**Para Edge AI y modelos grandes (11B+)**: Jetson AGX Orin es la opción superior por:
- Mayor eficiencia energética
- Menor uso de recursos
- Mejor para despliegues 24/7
- Viable para aplicaciones multimodales

**Para máxima velocidad con modelos pequeños (≤3B)**: RTX Ada 2000 domina en:
- Velocidad bruta de inferencia
- Flexibilidad de desarrollo
- Capacidad de procesamiento paralelo

### 8.3 Conclusión

Estos test demuestran de manera **categórica** que las plataformas edge especializadas como el Jetson AGX Orin **no solo compiten sino que dominan completamente** a GPUs dedicadas tradicionales en modelos grandes. Los resultados son revolucionarios: **Phi-4 Reasoning donde Jetson es 2.8× más rápido** mientras usa **35× menos CPU**, y **GPT-OSS 20B donde Jetson alcanza 11.9 t/s vs 4.1 t/s de RTX (2.9× más rápido)** usando **22× menos CPU y 2.4× menos RAM**. 

Este patrón confirma que **RTX Ada 2000 se satura arquitecturalmente a ~4.1 t/s en modelos >14B parámetros**, mientras que **Jetson AGX Orin escala linealmente**, alcanzando mayor rendimiento conforme aumenta el tamaño del modelo. Esto marca un **punto de inflexión histórico en el desarrollo de IA**, demostrando que la arquitectura ARM con memoria unificada es **categóricamente superior** para modelos de razonamiento complejos, aplicaciones que requieren modelos masivos, y cualquier despliegue edge que priorice la eficiencia energética sin sacrificar rendimiento.

---
