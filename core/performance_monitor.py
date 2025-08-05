# -*- coding: utf-8 -*-

import psutil
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
import json

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("GPUtil not available, GPU monitoring disabled")

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """系统指标数据模型"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    gpu_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

@dataclass
class TrainingMetrics:
    """训练指标数据模型"""
    timestamp: float
    episode: int
    step: int
    reward: float
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    algorithm_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

class PerformanceMonitor:
    """
    性能监控系统
    监控系统资源使用情况和训练指标
    """
    
    def __init__(self, monitor_interval: float = 1.0, history_size: int = 1000):
        """
        初始化性能监控器
        
        Args:
            monitor_interval: 监控间隔（秒）
            history_size: 历史数据保存数量
        """
        self.monitor_interval = monitor_interval
        self.history_size = history_size
        
        # 数据存储
        self.system_metrics_history = deque(maxlen=history_size)
        self.training_metrics_history = deque(maxlen=history_size)
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.start_time = None
        
        # 警告阈值
        self.cpu_warning_threshold = 90.0  # CPU使用率警告阈值
        self.memory_warning_threshold = 90.0  # 内存使用率警告阈值
        self.gpu_warning_threshold = 90.0  # GPU使用率警告阈值
        
        # 回调函数
        self.warning_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # 当前算法和任务信息
        self.current_algorithm = None
        self.current_task_type = None
        
        logger.info("PerformanceMonitor initialized")
    
    def initialize(self, algorithm_name: str, task_type: str) -> None:
        """
        初始化监控器
        
        Args:
            algorithm_name: 算法名称
            task_type: 任务类型
        """
        self.current_algorithm = algorithm_name
        self.current_task_type = task_type
        self.start_time = time.time()
        
        logger.info(f"Performance monitoring initialized for {algorithm_name} on {task_type}")
    
    def start_monitoring(self) -> None:
        """开始监控"""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Performance monitoring stopped")
    
    def record_step(self, training_info: Dict[str, Any]) -> None:
        """
        记录训练步骤指标
        
        Args:
            training_info: 训练信息字典
        """
        try:
            metrics = TrainingMetrics(
                timestamp=time.time(),
                episode=training_info.get('episode', 0),
                step=training_info.get('step', 0),
                reward=training_info.get('reward', 0.0),
                loss=training_info.get('loss'),
                learning_rate=training_info.get('learning_rate'),
                algorithm_metrics=training_info.get('algorithm_metrics')
            )
            
            self.training_metrics_history.append(metrics)
            
        except Exception as e:
            logger.error(f"Failed to record training step: {e}")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # 检查警告条件
                self._check_warnings(system_metrics)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU和内存指标
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU指标
        gpu_metrics = None
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_metrics = {
                        'count': len(gpus),
                        'gpus': []
                    }
                    
                    for i, gpu in enumerate(gpus):
                        gpu_info = {
                            'id': gpu.id,
                            'name': gpu.name,
                            'load': gpu.load * 100,  # 转换为百分比
                            'memory_used': gpu.memoryUsed,
                            'memory_total': gpu.memoryTotal,
                            'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            'temperature': gpu.temperature
                        }
                        gpu_metrics['gpus'].append(gpu_info)
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_usage_percent=disk.percent,
            gpu_metrics=gpu_metrics
        )
    
    def _check_warnings(self, metrics: SystemMetrics) -> None:
        """检查警告条件"""
        warnings = []
        
        # CPU警告
        if metrics.cpu_percent > self.cpu_warning_threshold:
            warnings.append({
                'type': 'cpu_high',
                'message': f"High CPU usage: {metrics.cpu_percent:.1f}%",
                'value': metrics.cpu_percent,
                'threshold': self.cpu_warning_threshold
            })
        
        # 内存警告
        if metrics.memory_percent > self.memory_warning_threshold:
            warnings.append({
                'type': 'memory_high',
                'message': f"High memory usage: {metrics.memory_percent:.1f}%",
                'value': metrics.memory_percent,
                'threshold': self.memory_warning_threshold
            })
        
        # GPU警告
        if metrics.gpu_metrics and GPU_AVAILABLE:
            for gpu in metrics.gpu_metrics['gpus']:
                if gpu['load'] > self.gpu_warning_threshold:
                    warnings.append({
                        'type': 'gpu_high',
                        'message': f"High GPU {gpu['id']} usage: {gpu['load']:.1f}%",
                        'value': gpu['load'],
                        'threshold': self.gpu_warning_threshold,
                        'gpu_id': gpu['id']
                    })
                
                if gpu['memory_percent'] > self.memory_warning_threshold:
                    warnings.append({
                        'type': 'gpu_memory_high',
                        'message': f"High GPU {gpu['id']} memory: {gpu['memory_percent']:.1f}%",
                        'value': gpu['memory_percent'],
                        'threshold': self.memory_warning_threshold,
                        'gpu_id': gpu['id']
                    })
        
        # 触发警告回调
        for warning in warnings:
            logger.warning(warning['message'])
            for callback in self.warning_callbacks:
                try:
                    callback(warning['type'], warning)
                except Exception as e:
                    logger.error(f"Error in warning callback: {e}")
    
    def add_warning_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        添加警告回调函数
        
        Args:
            callback: 回调函数，接收警告类型和详细信息
        """
        self.warning_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        获取当前指标
        
        Returns:
            当前指标字典
        """
        current_system = None
        current_training = None
        
        if self.system_metrics_history:
            current_system = self.system_metrics_history[-1].to_dict()
        
        if self.training_metrics_history:
            current_training = self.training_metrics_history[-1].to_dict()
        
        return {
            'system': current_system,
            'training': current_training,
            'monitoring_duration': time.time() - self.start_time if self.start_time else 0,
            'algorithm': self.current_algorithm,
            'task_type': self.current_task_type
        }
    
    def get_statistics(self, last_n_minutes: int = 10) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            last_n_minutes: 统计最近N分钟的数据
            
        Returns:
            统计信息字典
        """
        cutoff_time = time.time() - (last_n_minutes * 60)
        
        # 过滤最近的系统指标
        recent_system_metrics = [
            m for m in self.system_metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        # 过滤最近的训练指标
        recent_training_metrics = [
            m for m in self.training_metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        stats = {
            'time_range_minutes': last_n_minutes,
            'system_stats': {},
            'training_stats': {},
            'recommendations': []
        }
        
        # 系统统计
        if recent_system_metrics:
            cpu_values = [m.cpu_percent for m in recent_system_metrics]
            memory_values = [m.memory_percent for m in recent_system_metrics]
            
            stats['system_stats'] = {
                'cpu': {
                    'avg': sum(cpu_values) / len(cpu_values),
                    'max': max(cpu_values),
                    'min': min(cpu_values)
                },
                'memory': {
                    'avg': sum(memory_values) / len(memory_values),
                    'max': max(memory_values),
                    'min': min(memory_values)
                },
                'sample_count': len(recent_system_metrics)
            }
            
            # GPU统计
            if recent_system_metrics[0].gpu_metrics:
                gpu_loads = []
                gpu_memory = []
                
                for m in recent_system_metrics:
                    if m.gpu_metrics and m.gpu_metrics['gpus']:
                        for gpu in m.gpu_metrics['gpus']:
                            gpu_loads.append(gpu['load'])
                            gpu_memory.append(gpu['memory_percent'])
                
                if gpu_loads:
                    stats['system_stats']['gpu'] = {
                        'load_avg': sum(gpu_loads) / len(gpu_loads),
                        'load_max': max(gpu_loads),
                        'memory_avg': sum(gpu_memory) / len(gpu_memory),
                        'memory_max': max(gpu_memory)
                    }
        
        # 训练统计
        if recent_training_metrics:
            rewards = [m.reward for m in recent_training_metrics]
            losses = [m.loss for m in recent_training_metrics if m.loss is not None]
            
            stats['training_stats'] = {
                'reward': {
                    'avg': sum(rewards) / len(rewards),
                    'max': max(rewards),
                    'min': min(rewards)
                },
                'sample_count': len(recent_training_metrics)
            }
            
            if losses:
                stats['training_stats']['loss'] = {
                    'avg': sum(losses) / len(losses),
                    'max': max(losses),
                    'min': min(losses)
                }
        
        # 生成建议
        stats['recommendations'] = self._generate_recommendations(stats)
        
        return stats
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """
        生成性能优化建议
        
        Args:
            stats: 统计信息
            
        Returns:
            建议列表
        """
        recommendations = []
        
        system_stats = stats.get('system_stats', {})
        
        # CPU建议
        cpu_stats = system_stats.get('cpu', {})
        if cpu_stats.get('avg', 0) > 80:
            recommendations.append("High CPU usage detected. Consider reducing batch size or using fewer parallel processes.")
        
        # 内存建议
        memory_stats = system_stats.get('memory', {})
        if memory_stats.get('avg', 0) > 85:
            recommendations.append("High memory usage detected. Consider reducing buffer size or batch size.")
        
        # GPU建议
        gpu_stats = system_stats.get('gpu', {})
        if gpu_stats:
            if gpu_stats.get('load_avg', 0) < 50:
                recommendations.append("Low GPU utilization. Consider increasing batch size for better GPU usage.")
            elif gpu_stats.get('memory_avg', 0) > 90:
                recommendations.append("High GPU memory usage. Consider reducing model size or batch size.")
        
        # 训练建议
        training_stats = stats.get('training_stats', {})
        reward_stats = training_stats.get('reward', {})
        if reward_stats and reward_stats.get('max', 0) - reward_stats.get('min', 0) < 0.1:
            recommendations.append("Low reward variance. Training may have converged or need hyperparameter adjustment.")
        
        return recommendations
    
    def export_metrics(self, file_path: str, format: str = 'json') -> None:
        """
        导出指标数据
        
        Args:
            file_path: 文件路径
            format: 导出格式 ('json' 或 'csv')
        """
        try:
            if format.lower() == 'json':
                data = {
                    'system_metrics': [m.to_dict() for m in self.system_metrics_history],
                    'training_metrics': [m.to_dict() for m in self.training_metrics_history],
                    'export_time': time.time(),
                    'algorithm': self.current_algorithm,
                    'task_type': self.current_task_type
                }
                
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format.lower() == 'csv':
                import pandas as pd
                
                # 系统指标
                system_df = pd.DataFrame([m.to_dict() for m in self.system_metrics_history])
                system_df.to_csv(file_path.replace('.csv', '_system.csv'), index=False)
                
                # 训练指标
                training_df = pd.DataFrame([m.to_dict() for m in self.training_metrics_history])
                training_df.to_csv(file_path.replace('.csv', '_training.csv'), index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Metrics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
    
    def clear_history(self) -> None:
        """清空历史数据"""
        self.system_metrics_history.clear()
        self.training_metrics_history.clear()
        logger.info("Performance monitoring history cleared")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        获取监控状态
        
        Returns:
            监控状态字典
        """
        return {
            'is_monitoring': self.is_monitoring,
            'monitor_interval': self.monitor_interval,
            'history_size': self.history_size,
            'system_metrics_count': len(self.system_metrics_history),
            'training_metrics_count': len(self.training_metrics_history),
            'start_time': self.start_time,
            'current_algorithm': self.current_algorithm,
            'current_task_type': self.current_task_type,
            'gpu_available': GPU_AVAILABLE
        }