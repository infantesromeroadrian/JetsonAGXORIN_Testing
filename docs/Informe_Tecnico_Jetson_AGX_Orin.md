# Informe Técnico — NVIDIA Jetson AGX Orin

## 1. Identificación del Sistema

| Característica | Especificación |
|---------------|----------------|
| **Modelo** | NVIDIA Jetson AGX Orin Developer Kit (aarch64) |
| **SO/L4T/JetPack** | L4T r36.4.4 (JetPack 6.2.1), fecha 2025-06-16 |
| **CUDA** | 12.6 (detectado por autotag) |
| **Ubuntu** | 22.04 (Jammy) |
| **Docker** | Runtime por defecto nvidia (nvidia-container-toolkit configurado) |

## 2. Memoria (RAM, Swap y "VRAM")

### Especificaciones de Memoria
- **RAM física**: 64 GB (MemTotal ≈ 64,349,016 kB)
  - Linux reporta utilizable ≈ 61 GiB (resto reservado para GPU/firmware)
- **Swap zram**: ~30 GiB (12 dispositivos zram de ~2.6 GiB)
- **VRAM dedicada**: No existe. Jetson usa memoria unificada (UMA): CPU y GPU comparten la RAM del sistema

### Gestión de Memoria GPU
- En la práctica, la "VRAM disponible" ≈ RAM libre que ves en `free -h`/`tegrastats`
- Ejemplo (tegrastats): `RAM 5001/62841MB` ⇒ ~57.8 GB libres (disponibles para GPU si es necesario)

## 3. Almacenamiento

### eMMC Interna
- **Capacidad total**: ~59.2 GB
  - **Partición raíz** `/` (ext4): ~57.8 GB
  - **EFI** `/boot/efi`: 64 MB (vfat)

### NVMe/M.2
- **Estado**: No detectado actualmente
- `lspci` solo muestra la Wi-Fi Realtek RTL8822CE
- `parted /dev/nvme0n1` falla
- `dmesg` indica "Phy link never came up" en otros puentes PCIe → no hay NVMe presente o no enlaza

## 4. ¿Por qué Jetson "usa RAM" y no VRAM?

### Arquitectura de Memoria Unificada
- Es un SoC ARM con GPU integrada; no hay tarjeta gráfica separada con GDDR propia
- La LPDDR del sistema sirve tanto a CPU como a GPU (memoria unificada)

### Para LLMs:
- **Pesos del modelo + KV-cache** residen en RAM
- **GPU acelera el cómputo** leyendo esa RAM por el bus de memoria de alto ancho de banda
- **Si la RAM se agota**, entra la swap zram (más lenta) → baja el rendimiento

## 5. Arquitectura y Aceleración (ARM + GPU integrada)

### Especificaciones Hardware
- **CPU**: ARM aarch64 (64-bit)
- **GPU**: Integrada en el SoC (NVIDIA), comparte RAM del sistema (UMA)

### Lecturas típicas de tegrastats:
- **GR3D_FREQ**: Frecuencia de GPU (ej. 0%@[1300,1300] cuando está ociosa)
- **EMC_FREQ**: Controlador de memoria (ej. 0%@3199 → ~3.2 GHz)
- **Temperaturas/potencia** (CPU/GPU) también disponibles

## 6. Implicaciones Prácticas para LLM en Jetson (64 GB)

### Tamaños de modelo recomendados:
- **4–8B parámetros** en INT4/FP8 → cómodos en 64 GB UMA
- **13B** puede funcionar, pero más justo (RAM y ancho de banda)

### Consideraciones de Memoria:
- **Contexto (tokens)** aumenta mucho la KV-cache ⇒ más RAM
- **Espacio en disco** necesario para:
  - Pesos (HF_HOME)
  - Compilar/optimizar (MLC_CACHE_DIR)
  - Audios (Piper/Riva)

### Rendimiento:
- Evitar caer en swap
- Si `free -h` baja demasiado, reducir tamaño del modelo o contexto

## 7. Comandos Útiles

```bash
# RAM / swap / disponibilidad
free -h
sudo tegrastats --interval 1000

# Almacenamiento
df -h
du -xh --max-depth=1 ~ | sort -h

# PCIe / NVMe
sudo lspci -nn
sudo dmesg | egrep -i 'pcie|nvme|m\.2' | tail -n 200
```

---

# Pruebas de Rendimiento con Ollama

## Informe de Prueba — Jetson + Ollama (llama3.2:3b)

### Resultados Clave
- **Velocidad**: ~44.8 tokens/seg en decodificación (media de 5 ejecuciones)
- **GPU**: Uso intensivo (90-99%) con temperaturas normales (~60°C)
- **Modelo**: llama3.2:3b (~2.0 GB en disco)
- **Telemetría**: GPU al 90–99% @ ~1.3 GHz durante inferencia

### Equipo y SO
| Componente | Especificación |
|------------|----------------|
| **Dispositivo** | NVIDIA Jetson AGX Orin Developer Kit |
| **L4T / JetPack** | R36.4.4 / 6.2.1 |
| **CUDA** | 12.6 |
| **RAM del sistema** | ~64 GB (MemTotal ≈ 61–62 GiB utilizable) |
| **Almacenamiento** | eMMC ~57 GB (liberado y organizado) |

> **Nota**: En Jetson no hay VRAM dedicada. La GPU usa memoria unificada (UMA): comparte la RAM del sistema.

### Preparación

```bash
# Instalación del modelo
ollama run llama3.2:3b   # (descarga ~2.0 GB la primera vez)

# Entorno de prueba (aislado)
python3 -m venv .venv && source .venv/bin/activate
echo 'requests>=2.31' > requirements.txt
pip install -r requirements.txt
```

### Script de Benchmark

**Archivo**: `test_ollama_llama3.2:3b_jetson.py`

Características del script:
- Se conecta a `http://localhost:11434`
- Hace warmup
- Ejecuta N carreras con medición de:
  - prefill_tokens / decode_tokens
  - prefill_tps / decode_tps
  - wall time

### Metodología de Prueba

```bash
python test_ollama_llama3.2:3b_jetson.py --model llama3.2:3b -n 5 --out metrics.jsonl
```

### Resultados (5 carreras)

| Run | wall (s) | prefill tok | prefill t/s | decode tok | decode t/s |
|-----|----------|-------------|-------------|------------|------------|
| 1   | 2.48     | 45          | 5,199       | 107        | 44.4       |
| 2   | 2.46     | 45          | 11,687      | 108        | 45.0       |
| 3   | 2.54     | 45          | 11,580      | 111        | 44.9       |
| 4   | 2.68     | 45          | 11,712      | 117        | 44.8       |
| 5   | 2.68     | 45          | 11,435      | 117        | 44.8       |

**Media decode_tps**: 44.8 tok/s (5 runs)

> **Observación**: El prefill_tps es muy alto (prefill es más paralelo); lo importante para velocidad de conversación es decode_tps (~45 t/s).

### Telemetría (tegrastats) durante inferencia

- **GPU (GR3D_FREQ)**: 90–99% @ ~1.30 GHz
- **EMC (memoria)**: ~45% @ 3199 (no cuello de botella)
- **RAM**: ~5.6 / 61 GiB usada (mucho margen), swap 0
- **Temperaturas**: ~60–61°C (estables)
- **Potencia (VDD_GPU_SOC)**: ~31–35 W en carga; ~5 W en reposo

### Conclusiones

- El Jetson AGX Orin maneja llama3.2:3b con ~45 tok/s de decodificación y uso efectivo de GPU
- Temperatura y consumo dentro de rangos normales; RAM suficiente sin tocar swap
- La arquitectura UMA simplifica memoria (no VRAM aparte) y funciona bien para LLMs con tamaños moderados

---

# Comparativo — RTX Ada 2000 vs Jetson AGX Orin

**Modelo**: llama3.2:3b (Ollama)

## 1. Resumen

Se comparó el rendimiento de generación (decodificación de tokens) del modelo llama3.2:3b ejecutado con Ollama en dos entornos:
- Portátil con GPU NVIDIA RTX Ada 2000
- Jetson AGX Orin

Las métricas provienen de la API de Ollama (prompt_eval_count, eval_count, *_duration en ns), convertidas a tokens por segundo (t/s).

**Resultado clave**: La RTX Ada 2000 alcanzó una media global de **74.65 t/s** frente a **44.8 t/s** en Jetson, lo que supone un factor de aceleración **≈ 1.67×**.

## 2. Metodología

| Parámetro | Configuración |
|-----------|---------------|
| **Modelo idéntico** | llama3.2:3b |
| **Servidor** | Ollama local (HTTP 11434) |
| **Parámetros clave** | temperature=0, seed=42, num_ctx 2048/4096, num_predict 256 y 512 |
| **Métrica primaria** | decode_tps (eval_count / eval_duration) |
| **Runs RTX** | 9 (6 con num_predict=256 y 3 con 512) |
| **Runs Jetson** | 5 |

> **Limitación**: En Jetson sólo se dispone de la media de decode_tps (44.8 t/s) en el informe aportado.

## 3. Comparativa de decode_tps medio (t/s)

```
RTX Ada 2000:  ████████████████████████████████████████████████████████████████████████ 74.65
Jetson Orin:   ████████████████████████████████████████████ 44.80
```

### Resumen por lote (RTX) y referencia Jetson:

| Equipo/Lote | num_predict | runs | decode_tps media | mediana | p90 | min | max | total_s media | load_s media | decode_tokens media |
|-------------|-------------|------|------------------|---------|-----|-----|-----|---------------|--------------|-------------------|
| **RTX Ada 2000** | 256 | 6 | **73.10** | 72.56 | 78.46 | 67.50 | 79.53 | 1.394 | 0.044 | 97.8 |
| **RTX Ada 2000** | 512 | 3 | **77.74** | 75.55 | 82.98 | 72.85 | 84.83 | 1.444 | 0.040 | 108.7 |
| **Jetson AGX Orin** | — | 5 | **44.80** | n/d | n/d | n/d | n/d | n/d | n/d | n/d |

**Aceleración relativa de decode_tps (RTX/Jetson)**: 
- 1.63× (256 tokens)
- 1.74× (512 tokens)
- 1.67× (global)

## 4. Interpretación

- La RTX Ada 2000 ofrece entre **~1.6× y ~1.7×** más velocidad de decodificación que Jetson con el mismo modelo
- Rendimiento consistente entre lotes de 256 y 512 tokens previstos
- El tiempo de carga (`load_s`) en RTX es bajo (~0.04 s), indicio de buen cacheo y servidor en caliente
- Diferencias de `num_ctx` entre pruebas (2048 vs 4096) no alteran la conclusión principal

> **Recomendación**: Para una comparación 1:1 más precisa, se recomienda repetir ambos lados con los mismos flags finales.

---

# Gráficos Comparativos y Visualizaciones

## Comparación de Rendimiento: RTX Ada 2000 vs Jetson AGX Orin

```mermaid
graph LR
    subgraph "Comparación de Velocidad de Decodificación (tokens/segundo)"
        A["RTX Ada 2000<br/>74.65 t/s"] 
        B["Jetson AGX Orin<br/>44.80 t/s"]
    end
    
    style A fill:#2ecc71,stroke:#27ae60,stroke-width:3px,color:#fff
    style B fill:#3498db,stroke:#2980b9,stroke-width:3px,color:#fff
```

## Comparación Detallada por Configuración

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#2ecc71','primaryTextColor':'#fff','primaryBorderColor':'#27ae60','lineColor':'#5A6C7D','secondaryColor':'#3498db','tertiaryColor':'#e74c3c'}}}%%
graph TD
    subgraph "Rendimiento por Configuración (decode_tps)"
        RTX256["RTX Ada 2000<br/>256 tokens<br/>73.10 t/s"]
        RTX512["RTX Ada 2000<br/>512 tokens<br/>77.74 t/s"]
        JETSON["Jetson AGX Orin<br/>Media global<br/>44.80 t/s"]
        
        RTX256 -.->|"1.63x más rápido"| JETSON
        RTX512 -.->|"1.74x más rápido"| JETSON
    end
    
    style RTX256 fill:#2ecc71,stroke:#27ae60,stroke-width:2px
    style RTX512 fill:#27ae60,stroke:#229954,stroke-width:2px
    style JETSON fill:#3498db,stroke:#2980b9,stroke-width:2px
```

## Resultados de las 5 Carreras en Jetson AGX Orin

```mermaid
graph TD
    subgraph "Rendimiento por Carrera - Jetson AGX Orin (decode t/s)"
        Run1["Run 1<br/>44.4 t/s<br/>Wall: 2.48s"]
        Run2["Run 2<br/>45.0 t/s<br/>Wall: 2.46s"]
        Run3["Run 3<br/>44.9 t/s<br/>Wall: 2.54s"]
        Run4["Run 4<br/>44.8 t/s<br/>Wall: 2.68s"]
        Run5["Run 5<br/>44.8 t/s<br/>Wall: 2.68s"]
        
        Media["Media Global<br/>44.8 t/s"]
        
        Run1 --> Media
        Run2 --> Media
        Run3 --> Media
        Run4 --> Media
        Run5 --> Media
    end
    
    style Run1 fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px
    style Run2 fill:#16a085,stroke:#138d75,stroke-width:2px
    style Run3 fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px
    style Run4 fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px
    style Run5 fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px
    style Media fill:#e74c3c,stroke:#c0392b,stroke-width:3px,color:#fff
```

## Arquitectura de Memoria - Jetson AGX Orin

```mermaid
graph TD
    subgraph "Arquitectura de Memoria - Jetson AGX Orin"
        RAM["RAM Física<br/>64 GB Total<br/>61 GB Utilizable"]
        SWAP["Swap ZRAM<br/>~30 GB<br/>12 dispositivos"]
        UMA["Memoria Unificada (UMA)<br/>CPU + GPU comparten RAM"]
        
        RAM --> UMA
        SWAP --> UMA
        
        GPU["GPU Integrada<br/>NVIDIA<br/>1.3 GHz"]
        CPU["CPU ARM<br/>aarch64<br/>64-bit"]
        
        UMA --> GPU
        UMA --> CPU
    end
    
    style RAM fill:#3498db,stroke:#2980b9,stroke-width:2px
    style SWAP fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px
    style UMA fill:#e74c3c,stroke:#c0392b,stroke-width:3px,color:#fff
    style GPU fill:#2ecc71,stroke:#27ae60,stroke-width:2px
    style CPU fill:#f39c12,stroke:#e67e22,stroke-width:2px
```

## Telemetría Durante Inferencia

```mermaid
graph LR
    subgraph "Uso de Recursos Durante Inferencia con llama3.2:3b"
        GPU["GPU<br/>90-99% @ 1.3GHz<br/>Temp: 60-61°C"]
        MEM["Memoria<br/>5.6/61 GB usada<br/>Swap: 0 GB<br/>EMC: 45% @ 3199MHz"]
        POW["Potencia<br/>Carga: 31-35W<br/>Reposo: 5W"]
        PERF["Rendimiento<br/>44.8 tokens/seg<br/>Modelo: 2.0 GB"]
        
        GPU --> PERF
        MEM --> PERF
        POW --> PERF
    end
    
    style GPU fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff
    style MEM fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff
    style POW fill:#f39c12,stroke:#e67e22,stroke-width:2px
    style PERF fill:#2ecc71,stroke:#27ae60,stroke-width:3px,color:#fff
```

## Almacenamiento y Recomendaciones de Modelos LLM

```mermaid
graph TD
    subgraph "Almacenamiento Jetson AGX Orin"
        EMMC["eMMC Interna<br/>57.8 GB disponibles<br/>(ext4)"]
        EFI["Partición EFI<br/>64 MB<br/>(vfat)"]
        NVME["NVMe/M.2<br/>No detectado<br/>❌"]
    end
    
    subgraph "Recomendaciones de Modelos LLM"
        SMALL["4-8B Parámetros<br/>INT4/FP8<br/>✅ Cómodo en 64GB"]
        MEDIUM["13B Parámetros<br/>✅ Posible<br/>⚠️ Justo en RAM"]
        LARGE[">13B Parámetros<br/>❌ No recomendado<br/>Excede capacidad"]
        
        SMALL -->|"Recomendado"| OPT["Rendimiento<br/>Óptimo"]
        MEDIUM -->|"Con cuidado"| OPT
        LARGE -->|"Evitar"| SWAP["Caerá en<br/>Swap"]
    end
    
    style EMMC fill:#3498db,stroke:#2980b9,stroke-width:2px
    style EFI fill:#95a5a6,stroke:#7f8c8d,stroke-width:1px
    style NVME fill:#e74c3c,stroke:#c0392b,stroke-width:2px
    style SMALL fill:#2ecc71,stroke:#27ae60,stroke-width:2px
    style MEDIUM fill:#f39c12,stroke:#e67e22,stroke-width:2px
    style LARGE fill:#e74c3c,stroke:#c0392b,stroke-width:2px
    style OPT fill:#27ae60,stroke:#229954,stroke-width:3px,color:#fff
    style SWAP fill:#c0392b,stroke:#a93226,stroke-width:2px,color:#fff
```

## Flujo de Trabajo del Benchmark

```mermaid
graph TD
    subgraph "Flujo de Benchmark Ollama"
        A["Inicio<br/>Ollama Server<br/>Puerto 11434"] 
        B["Carga Modelo<br/>llama3.2:3b<br/>~2.0 GB"]
        C["Warmup<br/>Preparación<br/>del Sistema"]
        D["Ejecución<br/>5 Carreras<br/>Medición"]
        
        A --> B
        B --> C
        C --> D
        
        D --> E["Métricas<br/>• prefill_tps<br/>• decode_tps<br/>• wall_time"]
        
        E --> F["Resultados<br/>44.8 t/s media<br/>Consistente"]
        
        G["Telemetría<br/>tegrastats"]
        G --> H["Monitoreo<br/>• GPU: 90-99%<br/>• Temp: 60°C<br/>• RAM: 5.6/61GB"]
        
        D -.->|"Durante ejecución"| G
    end
    
    style A fill:#3498db,stroke:#2980b9,stroke-width:2px
    style B fill:#9b59b6,stroke:#8e44ad,stroke-width:2px
    style C fill:#f39c12,stroke:#e67e22,stroke-width:2px
    style D fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff
    style E fill:#16a085,stroke:#138d75,stroke-width:2px
    style F fill:#2ecc71,stroke:#27ae60,stroke-width:3px,color:#fff
    style G fill:#34495e,stroke:#2c3e50,stroke-width:2px,color:#fff
    style H fill:#95a5a6,stroke:#7f8c8d,stroke-width:2px
```

## Dashboard Comparativo Final

```mermaid
graph TB
    subgraph "Dashboard Comparativo de Rendimiento"
        subgraph "RTX Ada 2000"
            RTX_SPEED["Velocidad<br/>74.65 t/s"]
            RTX_LOAD["Tiempo Carga<br/>0.04s"]
            RTX_VRAM["VRAM<br/>Dedicada"]
            RTX_POWER["Potencia<br/>Mayor consumo"]
        end
        
        subgraph "Jetson AGX Orin"
            JET_SPEED["Velocidad<br/>44.80 t/s"]
            JET_RAM["RAM Unificada<br/>64 GB UMA"]
            JET_TEMP["Temperatura<br/>60-61°C estable"]
            JET_POWER["Potencia<br/>31-35W carga<br/>5W reposo"]
        end
        
        COMPARE["Factor de Aceleración<br/>RTX es 1.67x más rápida"]
        
        RTX_SPEED -.->|"66% más rápido"| COMPARE
        JET_SPEED -.-> COMPARE
    end
    
    style RTX_SPEED fill:#2ecc71,stroke:#27ae60,stroke-width:3px,color:#fff
    style RTX_LOAD fill:#3498db,stroke:#2980b9,stroke-width:2px
    style RTX_VRAM fill:#9b59b6,stroke:#8e44ad,stroke-width:2px
    style RTX_POWER fill:#e67e22,stroke:#d35400,stroke-width:2px
    
    style JET_SPEED fill:#3498db,stroke:#2980b9,stroke-width:3px,color:#fff
    style JET_RAM fill:#16a085,stroke:#138d75,stroke-width:2px
    style JET_TEMP fill:#27ae60,stroke:#229954,stroke-width:2px
    style JET_POWER fill:#f39c12,stroke:#e67e22,stroke-width:2px
    
    style COMPARE fill:#e74c3c,stroke:#c0392b,stroke-width:4px,color:#fff
```

---
