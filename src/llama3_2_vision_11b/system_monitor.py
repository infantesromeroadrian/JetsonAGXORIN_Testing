#!/usr/bin/env python3
"""
Sistema de monitoreo de recursos del sistema para benchmarks de LLM multimodales
Captura métricas de CPU, GPU, RAM, temperatura durante las ejecuciones de modelos de visión
"""

import psutil
import time
import threading
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import subprocess
import os

@dataclass
class SystemMetrics:
    """Clase para almacenar las métricas del sistema en un momento dado"""
    timestamp: float
    
    # CPU Metrics
    cpu_percent: float
    cpu_freq_current: float
    cpu_freq_max: float
    cpu_count_logical: int
    cpu_count_physical: int
    
    # Memory Metrics  
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float
    
    # GPU Metrics (Jetson specific)
    gpu_usage_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    
    # Temperature Metrics
    cpu_temp: Optional[float] = None
    gpu_temp: Optional[float] = None
    
    # Power Metrics (Jetson specific)
    power_consumption_watts: Optional[float] = None
    
    # Disk I/O
    disk_read_mb: Optional[float] = None
    disk_write_mb: Optional[float] = None


class SystemMonitor:
    """Monitor continuo de recursos del sistema para modelos multimodales"""
    
    def __init__(self):
        self.is_monitoring = False
        self.metrics_history = []
        self.monitor_thread = None
        self.monitoring_interval = 0.5  # segundos entre mediciones
        
    def _get_jetson_gpu_info(self) -> Dict[str, Optional[float]]:
        """Obtiene información específica de GPU para Jetson usando tegrastats"""
        gpu_info = {
            'gpu_usage_percent': None,
            'gpu_memory_used_mb': None, 
            'gpu_memory_total_mb': None,
            'gpu_memory_percent': None
        }
        
        try:
            # Intentar leer tegrastats para GPU
            result = subprocess.run(['tegrastats', '--interval', '100'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                # Parsear salida de tegrastats (formato específico de Jetson)
                # Ejemplo: GR3D_FREQ 76%@1377000000
                lines = result.stdout.strip().split('\n')
                if lines:
                    line = lines[-1]  # última línea
                    if 'GR3D_FREQ' in line:
                        # Extraer porcentaje de uso de GPU
                        parts = line.split('GR3D_FREQ')[1].strip()
                        if '%' in parts:
                            gpu_usage = float(parts.split('%')[0].strip())
                            gpu_info['gpu_usage_percent'] = gpu_usage
                            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, 
                FileNotFoundError, ValueError):
            # Si tegrastats no está disponible, intentar nvidia-smi como fallback
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=2)
                
                if result.returncode == 0:
                    line = result.stdout.strip()
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_info['gpu_usage_percent'] = float(parts[0])
                        gpu_info['gpu_memory_used_mb'] = float(parts[1])
                        gpu_info['gpu_memory_total_mb'] = float(parts[2])
                        gpu_info['gpu_memory_percent'] = (
                            gpu_info['gpu_memory_used_mb'] / gpu_info['gpu_memory_total_mb'] * 100
                        )
            except (subprocess.TimeoutExpired, subprocess.SubprocessError,
                    FileNotFoundError, ValueError):
                pass
                
        return gpu_info
    
    def _get_temperature_info(self) -> Dict[str, Optional[float]]:
        """Obtiene información de temperatura del sistema"""
        temps = {
            'cpu_temp': None,
            'gpu_temp': None
        }
        
        try:
            # Intentar leer temperaturas usando psutil
            if hasattr(psutil, 'sensors_temperatures'):
                temp_info = psutil.sensors_temperatures()
                
                # CPU temperature
                for name, entries in temp_info.items():
                    if 'cpu' in name.lower() or 'core' in name.lower():
                        if entries:
                            temps['cpu_temp'] = entries[0].current
                            break
                
                # GPU temperature (Jetson specific)
                for name, entries in temp_info.items():
                    if 'gpu' in name.lower() or 'thermal' in name.lower():
                        if entries:
                            temps['gpu_temp'] = entries[0].current
                            break
                            
        except (AttributeError, OSError):
            # Fallback: intentar leer desde /sys/class/thermal (Linux)
            try:
                thermal_zones = [f for f in os.listdir('/sys/class/thermal/') 
                               if f.startswith('thermal_zone')]
                
                for zone in thermal_zones[:2]:  # primeras 2 zonas térmicas
                    temp_file = f'/sys/class/thermal/{zone}/temp'
                    if os.path.exists(temp_file):
                        with open(temp_file, 'r') as f:
                            temp_millic = int(f.read().strip())
                            temp_celsius = temp_millic / 1000.0
                            
                        if temps['cpu_temp'] is None:
                            temps['cpu_temp'] = temp_celsius
                        elif temps['gpu_temp'] is None:
                            temps['gpu_temp'] = temp_celsius
                            
            except (OSError, ValueError):
                pass
                
        return temps
    
    def _get_power_info(self) -> Optional[float]:
        """Obtiene información de consumo de energía (específico para Jetson)"""
        try:
            # Jetson específico: leer desde INA3221 power monitors
            power_paths = [
                '/sys/bus/i2c/drivers/ina3221x/0-0040/iio_device/in_power0_input',
                '/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power0_input'
            ]
            
            total_power = 0.0
            power_readings = 0
            
            for path in power_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        power_mw = float(f.read().strip())
                        total_power += power_mw / 1000.0  # convertir a watts
                        power_readings += 1
                        
            return total_power if power_readings > 0 else None
            
        except (OSError, ValueError):
            return None
    
    def _capture_single_measurement(self) -> SystemMetrics:
        """Captura una medición única de todas las métricas del sistema"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        cpu_count_physical = psutil.cpu_count(logical=False)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        ram_total_gb = memory.total / (1024**3)
        ram_used_gb = memory.used / (1024**3) 
        ram_available_gb = memory.available / (1024**3)
        ram_percent = memory.percent
        
        # GPU metrics (Jetson specific)
        gpu_info = self._get_jetson_gpu_info()
        
        # Temperature metrics
        temp_info = self._get_temperature_info()
        
        # Power metrics
        power_watts = self._get_power_info()
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / (1024**2) if disk_io else None
        disk_write_mb = disk_io.write_bytes / (1024**2) if disk_io else None
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            cpu_freq_current=cpu_freq.current if cpu_freq else 0,
            cpu_freq_max=cpu_freq.max if cpu_freq else 0,
            cpu_count_logical=cpu_count,
            cpu_count_physical=cpu_count_physical,
            ram_total_gb=ram_total_gb,
            ram_used_gb=ram_used_gb,
            ram_available_gb=ram_available_gb,
            ram_percent=ram_percent,
            gpu_usage_percent=gpu_info['gpu_usage_percent'],
            gpu_memory_used_mb=gpu_info['gpu_memory_used_mb'],
            gpu_memory_total_mb=gpu_info['gpu_memory_total_mb'],
            gpu_memory_percent=gpu_info['gpu_memory_percent'],
            cpu_temp=temp_info['cpu_temp'],
            gpu_temp=temp_info['gpu_temp'],
            power_consumption_watts=power_watts,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb
        )
    
    def _monitoring_loop(self):
        """Bucle principal de monitoreo en hilo separado"""
        while self.is_monitoring:
            try:
                metrics = self._capture_single_measurement()
                self.metrics_history.append(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"Error en monitoreo: {e}")
                time.sleep(self.monitoring_interval)
    
    def start_monitoring(self):
        """Inicia el monitoreo continuo en un hilo separado"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.metrics_history.clear()
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Detiene el monitoreo y retorna las métricas capturadas"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)
        return self.metrics_history.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Calcula estadísticas resumidas de las métricas capturadas"""
        if not self.metrics_history:
            return {}
        
        # Extraer valores numéricos para estadísticas
        cpu_usage = [m.cpu_percent for m in self.metrics_history]
        ram_usage = [m.ram_percent for m in self.metrics_history] 
        ram_used = [m.ram_used_gb for m in self.metrics_history]
        
        gpu_usage = [m.gpu_usage_percent for m in self.metrics_history 
                    if m.gpu_usage_percent is not None]
        
        cpu_temps = [m.cpu_temp for m in self.metrics_history 
                    if m.cpu_temp is not None]
        gpu_temps = [m.gpu_temp for m in self.metrics_history 
                    if m.gpu_temp is not None]
        
        power_readings = [m.power_consumption_watts for m in self.metrics_history
                         if m.power_consumption_watts is not None]
        
        def safe_stats(values):
            if not values:
                return {'min': None, 'max': None, 'mean': None, 'count': 0}
            return {
                'min': min(values),
                'max': max(values), 
                'mean': sum(values) / len(values),
                'count': len(values)
            }
        
        return {
            'monitoring_duration_s': (
                self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp
            ) if len(self.metrics_history) > 1 else 0,
            'total_samples': len(self.metrics_history),
            'cpu_usage_percent': safe_stats(cpu_usage),
            'ram_usage_percent': safe_stats(ram_usage),
            'ram_used_gb': safe_stats(ram_used),
            'gpu_usage_percent': safe_stats(gpu_usage),
            'cpu_temperature_c': safe_stats(cpu_temps),
            'gpu_temperature_c': safe_stats(gpu_temps),
            'power_consumption_watts': safe_stats(power_readings)
        }
    
    def save_metrics_to_file(self, filepath: str):
        """Guarda todas las métricas a un archivo JSON Lines"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for metric in self.metrics_history:
                f.write(json.dumps(asdict(metric)) + '\n')


# Funciones de conveniencia para usar con los scripts de visión
def create_monitor() -> SystemMonitor:
    """Crea una nueva instancia del monitor de sistema"""
    return SystemMonitor()

def get_instant_metrics() -> SystemMetrics:
    """Obtiene una medición instantánea del sistema"""
    monitor = SystemMonitor()
    return monitor._capture_single_measurement()


if __name__ == "__main__":
    # Ejemplo de uso del monitor para modelos de visión
    print("=== Test del Sistema de Monitoreo - Modelos Multimodales ===")
    
    monitor = SystemMonitor()
    
    # Medición instantánea
    instant = get_instant_metrics()
    print(f"\nMedición instantánea:")
    # Formateo seguro de todas las métricas
    cpu_str = f"{instant.cpu_percent:.1f}" if isinstance(instant.cpu_percent, (int, float)) else "n/a"
    freq_str = f"{instant.cpu_freq_current:.0f}" if isinstance(instant.cpu_freq_current, (int, float)) else "n/a"
    ram_used_str = f"{instant.ram_used_gb:.1f}" if isinstance(instant.ram_used_gb, (int, float)) else "n/a"
    ram_total_str = f"{instant.ram_total_gb:.1f}" if isinstance(instant.ram_total_gb, (int, float)) else "n/a"
    ram_pct_str = f"{instant.ram_percent:.1f}" if isinstance(instant.ram_percent, (int, float)) else "n/a"
    
    print(f"CPU: {cpu_str}% @ {freq_str}MHz")
    print(f"RAM: {ram_used_str}/{ram_total_str}GB ({ram_pct_str}%)")
    
    if instant.gpu_usage_percent is not None and isinstance(instant.gpu_usage_percent, (int, float)):
        gpu_str = f"{instant.gpu_usage_percent:.1f}"
        print(f"GPU: {gpu_str}%")
    if instant.cpu_temp is not None and isinstance(instant.cpu_temp, (int, float)):
        cpu_temp_str = f"{instant.cpu_temp:.1f}"
        print(f"CPU Temp: {cpu_temp_str}°C")
    if instant.gpu_temp is not None and isinstance(instant.gpu_temp, (int, float)):
        gpu_temp_str = f"{instant.gpu_temp:.1f}"
        print(f"GPU Temp: {gpu_temp_str}°C")
    if instant.power_consumption_watts is not None and isinstance(instant.power_consumption_watts, (int, float)):
        power_str = f"{instant.power_consumption_watts:.1f}"
        print(f"Power: {power_str}W")
    
    # Monitoreo continuo por unos segundos
    print("\n=== Iniciando monitoreo por 5 segundos ===")
    monitor.start_monitoring()
    time.sleep(5)
    history = monitor.stop_monitoring()
    
    summary = monitor.get_metrics_summary()
    print(f"\nResumen del monitoreo:")
    # Formateo seguro del resumen
    duration = summary.get('monitoring_duration_s')
    duration_str = f"{duration:.1f}" if isinstance(duration, (int, float)) else "n/a"
    print(f"Duración: {duration_str}s")
    print(f"Muestras: {summary.get('total_samples', 0)}")
    
    cpu_stats = summary.get('cpu_usage_percent', {})
    if cpu_stats.get('count', 0) > 0:
        cpu_mean = cpu_stats.get('mean')
        cpu_min = cpu_stats.get('min')
        cpu_max = cpu_stats.get('max')
        if cpu_mean is not None and isinstance(cpu_mean, (int, float)):
            cpu_mean_str = f"{cpu_mean:.1f}"
            cpu_min_str = f"{cpu_min:.1f}" if isinstance(cpu_min, (int, float)) else "n/a"
            cpu_max_str = f"{cpu_max:.1f}" if isinstance(cpu_max, (int, float)) else "n/a"
            print(f"CPU: {cpu_mean_str}% (min={cpu_min_str}%, max={cpu_max_str}%)")
    
    ram_stats = summary.get('ram_usage_percent', {}) 
    if ram_stats.get('count', 0) > 0:
        ram_mean = ram_stats.get('mean')
        ram_min = ram_stats.get('min')
        ram_max = ram_stats.get('max')
        if ram_mean is not None and isinstance(ram_mean, (int, float)):
            ram_mean_str = f"{ram_mean:.1f}"
            ram_min_str = f"{ram_min:.1f}" if isinstance(ram_min, (int, float)) else "n/a"
            ram_max_str = f"{ram_max:.1f}" if isinstance(ram_max, (int, float)) else "n/a"
            print(f"RAM: {ram_mean_str}% (min={ram_min_str}%, max={ram_max_str}%)")
