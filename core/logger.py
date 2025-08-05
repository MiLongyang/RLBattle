# -*- coding: utf-8 -*-

import logging
import os
import sys
import time
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json

class AlgorithmLogger:
    """
    算法框架专用日志记录器
    提供分级日志记录和格式化功能
    """
    
    def __init__(self, name: str = "AlgorithmFramework", log_dir: str = "./logs"):
        self.name = name
        self.log_dir = log_dir
        self.logger = logging.getLogger(name)
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 防止重复添加处理器
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self):
        """设置日志记录器"""
        self.logger.setLevel(logging.DEBUG)
        
        # 创建格式化器
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器 - 详细日志
        file_handler = RotatingFileHandler(
            os.path.join(self.log_dir, f"{self.name}_detailed.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # 文件处理器 - 错误日志
        error_handler = RotatingFileHandler(
            os.path.join(self.log_dir, f"{self.name}_errors.log"),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(error_handler)
        
        # 训练日志处理器（按天轮转）
        training_handler = TimedRotatingFileHandler(
            os.path.join(self.log_dir, f"{self.name}_training.log"),
            when='midnight',
            interval=1,
            backupCount=7
        )
        training_handler.setLevel(logging.INFO)
        training_handler.setFormatter(simple_formatter)
        self.logger.addHandler(training_handler)
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """记录调试信息"""
        if extra_data:
            message = f"{message} | Extra: {json.dumps(extra_data, default=str)}"
        self.logger.debug(message)
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """记录一般信息"""
        if extra_data:
            message = f"{message} | Extra: {json.dumps(extra_data, default=str)}"
        self.logger.info(message)
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """记录警告信息"""
        if extra_data:
            message = f"{message} | Extra: {json.dumps(extra_data, default=str)}"
        self.logger.warning(message)
    
    def error(self, message: str, exception: Optional[Exception] = None, 
              extra_data: Optional[Dict[str, Any]] = None):
        """记录错误信息"""
        if extra_data:
            message = f"{message} | Extra: {json.dumps(extra_data, default=str)}"
        
        if exception:
            self.logger.error(message, exc_info=exception)
        else:
            self.logger.error(message)
    
    def critical(self, message: str, exception: Optional[Exception] = None,
                 extra_data: Optional[Dict[str, Any]] = None):
        """记录严重错误信息"""
        if extra_data:
            message = f"{message} | Extra: {json.dumps(extra_data, default=str)}"
        
        if exception:
            self.logger.critical(message, exc_info=exception)
        else:
            self.logger.critical(message)
    
    def log_training_start(self, algorithm_name: str, task_type: str, config: Dict[str, Any]):
        """记录训练开始"""
        self.info(f"Training started: {algorithm_name} on {task_type}", {
            'algorithm': algorithm_name,
            'task_type': task_type,
            'config': config,
            'timestamp': time.time()
        })
    
    def log_training_episode(self, episode: int, reward: float, metrics: Dict[str, Any]):
        """记录训练回合"""
        self.info(f"Episode {episode} completed: reward={reward:.2f}", {
            'episode': episode,
            'reward': reward,
            'metrics': metrics,
            'timestamp': time.time()
        })
    
    def log_training_end(self, total_episodes: int, total_time: float, final_metrics: Dict[str, Any]):
        """记录训练结束"""
        self.info(f"Training completed: {total_episodes} episodes in {total_time:.2f}s", {
            'total_episodes': total_episodes,
            'total_time': total_time,
            'final_metrics': final_metrics,
            'timestamp': time.time()
        })
    
    def log_model_save(self, algorithm_name: str, episode: str, save_path: str, save_time: float):
        """记录模型保存"""
        self.info(f"Model saved: {algorithm_name} episode {episode} in {save_time:.2f}s", {
            'algorithm': algorithm_name,
            'episode': episode,
            'save_path': save_path,
            'save_time': save_time,
            'timestamp': time.time()
        })
    
    def log_model_load(self, algorithm_name: str, episode: str, load_path: str, load_time: float):
        """记录模型加载"""
        self.info(f"Model loaded: {algorithm_name} episode {episode} in {load_time:.2f}s", {
            'algorithm': algorithm_name,
            'episode': episode,
            'load_path': load_path,
            'load_time': load_time,
            'timestamp': time.time()
        })
    
    def log_evaluation_start(self, algorithm_name: str, task_type: str, eval_episodes: int):
        """记录评估开始"""
        self.info(f"Evaluation started: {algorithm_name} on {task_type} for {eval_episodes} episodes", {
            'algorithm': algorithm_name,
            'task_type': task_type,
            'eval_episodes': eval_episodes,
            'timestamp': time.time()
        })
    
    def log_evaluation_end(self, algorithm_name: str, avg_reward: float, success_rate: float, eval_time: float):
        """记录评估结束"""
        self.info(f"Evaluation completed: {algorithm_name} avg_reward={avg_reward:.2f} success_rate={success_rate:.2%}", {
            'algorithm': algorithm_name,
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'eval_time': eval_time,
            'timestamp': time.time()
        })
    
    def log_performance_warning(self, metric_name: str, current_value: float, threshold: float):
        """记录性能警告"""
        self.warning(f"Performance warning: {metric_name}={current_value:.2f} exceeds threshold {threshold:.2f}", {
            'metric_name': metric_name,
            'current_value': current_value,
            'threshold': threshold,
            'timestamp': time.time()
        })
    
    def log_algorithm_error(self, algorithm_name: str, error_type: str, error_message: str, 
                           context: Optional[Dict[str, Any]] = None):
        """记录算法错误"""
        self.error(f"Algorithm error in {algorithm_name}: {error_type} - {error_message}", 
                  extra_data={
                      'algorithm': algorithm_name,
                      'error_type': error_type,
                      'error_message': error_message,
                      'context': context or {},
                      'timestamp': time.time()
                  })
    
    def set_level(self, level: str):
        """设置日志级别"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        if level.upper() in level_map:
            self.logger.setLevel(level_map[level.upper()])
            for handler in self.logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                    handler.setLevel(level_map[level.upper()])
        else:
            self.warning(f"Invalid log level: {level}")
    
    def add_file_handler(self, filename: str, level: str = 'INFO', max_bytes: int = 10*1024*1024):
        """添加额外的文件处理器"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        if level.upper() not in level_map:
            self.warning(f"Invalid log level for file handler: {level}")
            return
        
        file_path = os.path.join(self.log_dir, filename)
        handler = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=3)
        handler.setLevel(level_map[level.upper()])
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.info(f"Added file handler: {file_path}")
    
    def get_log_files(self) -> Dict[str, str]:
        """获取日志文件路径"""
        return {
            'detailed': os.path.join(self.log_dir, f"{self.name}_detailed.log"),
            'errors': os.path.join(self.log_dir, f"{self.name}_errors.log"),
            'training': os.path.join(self.log_dir, f"{self.name}_training.log")
        }
    
    def cleanup_old_logs(self, days_to_keep: int = 7):
        """清理旧日志文件"""
        import glob
        import time
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        log_pattern = os.path.join(self.log_dir, f"{self.name}_*.log*")
        log_files = glob.glob(log_pattern)
        
        cleaned_count = 0
        for log_file in log_files:
            try:
                if os.path.getmtime(log_file) < cutoff_time:
                    os.remove(log_file)
                    cleaned_count += 1
            except OSError as e:
                self.warning(f"Failed to remove old log file {log_file}: {e}")
        
        if cleaned_count > 0:
            self.info(f"Cleaned up {cleaned_count} old log files")


# 全局日志实例
_global_logger = None

def get_logger(name: str = "AlgorithmFramework", log_dir: str = "./logs") -> AlgorithmLogger:
    """获取全局日志实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AlgorithmLogger(name, log_dir)
    return _global_logger

def setup_logging(name: str = "AlgorithmFramework", log_dir: str = "./logs", 
                 level: str = "INFO") -> AlgorithmLogger:
    """设置日志系统"""
    logger = get_logger(name, log_dir)
    logger.set_level(level)
    return logger