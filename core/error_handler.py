# -*- coding: utf-8 -*-

import functools
import traceback
import time
from typing import Callable, Any, Dict, Optional, Type, Union
from .exceptions import AlgorithmFrameworkException
from .logger import get_logger

logger = get_logger()

class ErrorHandler:
    """
    统一错误处理器
    提供异常捕获、记录和恢复建议
    """
    
    def __init__(self):
        self.error_counts = {}
        self.last_errors = {}
        self.recovery_strategies = {}
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """设置默认恢复策略"""
        self.recovery_strategies = {
            'AlgorithmNotFoundError': self._handle_algorithm_not_found,
            'AlgorithmInitializationError': self._handle_algorithm_init_error,
            'ModelLoadError': self._handle_model_load_error,
            'ModelSaveError': self._handle_model_save_error,
            'ConfigValidationError': self._handle_config_error,
            'TrainingError': self._handle_training_error,
            'ResourceExhaustionError': self._handle_resource_error,
            'NetworkError': self._handle_network_error,
            'EnvironmentError': self._handle_environment_error
        }
    
    def handle_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理异常
        
        Args:
            exception: 异常对象
            context: 上下文信息
            
        Returns:
            处理结果字典
        """
        exception_type = type(exception).__name__
        
        # 更新错误计数
        self.error_counts[exception_type] = self.error_counts.get(exception_type, 0) + 1
        self.last_errors[exception_type] = {
            'exception': exception,
            'timestamp': time.time(),
            'context': context or {}
        }
        
        # 记录异常
        self._log_exception(exception, context)
        
        # 获取恢复建议
        recovery_info = self._get_recovery_info(exception, context)
        
        return {
            'exception_type': exception_type,
            'message': str(exception),
            'timestamp': time.time(),
            'context': context or {},
            'recovery_info': recovery_info,
            'error_count': self.error_counts[exception_type]
        }
    
    def _log_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None):
        """记录异常信息"""
        exception_type = type(exception).__name__
        
        if isinstance(exception, AlgorithmFrameworkException):
            # 框架异常，记录详细信息
            logger.error(f"Framework exception: {exception_type}", exception, {
                'error_code': exception.error_code,
                'details': exception.details,
                'context': context or {}
            })
        else:
            # 其他异常
            logger.error(f"Unexpected exception: {exception_type}", exception, {
                'context': context or {},
                'traceback': traceback.format_exc()
            })
    
    def _get_recovery_info(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """获取恢复信息"""
        exception_type = type(exception).__name__
        
        # 使用注册的恢复策略
        if exception_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[exception_type](exception, context)
            except Exception as e:
                logger.warning(f"Failed to get recovery info for {exception_type}: {e}")
        
        # 默认恢复信息
        return {
            'can_recover': False,
            'suggestions': [
                "Check the error message and context for more details",
                "Verify your configuration and input parameters",
                "Consult the documentation or contact support"
            ],
            'retry_recommended': False
        }
    
    def _handle_algorithm_not_found(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理算法未找到错误"""
        if hasattr(exception, 'details'):
            available_algorithms = exception.details.get('available_algorithms', [])
            suggestions = [
                f"Available algorithms: {', '.join(available_algorithms)}",
                "Check the algorithm name spelling",
                "Ensure the algorithm is properly registered"
            ]
        else:
            suggestions = [
                "Check the algorithm name spelling",
                "Verify the algorithm is installed and registered"
            ]
        
        return {
            'can_recover': True,
            'suggestions': suggestions,
            'retry_recommended': True,
            'recovery_actions': ['check_algorithm_name', 'list_available_algorithms']
        }
    
    def _handle_algorithm_init_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理算法初始化错误"""
        suggestions = [
            "Check algorithm configuration parameters",
            "Verify environment information is correct",
            "Ensure sufficient system resources",
            "Check device availability (CPU/GPU)"
        ]
        
        return {
            'can_recover': True,
            'suggestions': suggestions,
            'retry_recommended': True,
            'recovery_actions': ['validate_config', 'check_resources', 'verify_device']
        }
    
    def _handle_model_load_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理模型加载错误"""
        suggestions = [
            "Verify the model file path exists",
            "Check if the model file is corrupted",
            "Ensure the model was saved with the same algorithm",
            "Try loading a different episode/checkpoint"
        ]
        
        return {
            'can_recover': True,
            'suggestions': suggestions,
            'retry_recommended': True,
            'recovery_actions': ['check_file_exists', 'verify_model_format', 'try_different_episode']
        }
    
    def _handle_model_save_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理模型保存错误"""
        suggestions = [
            "Check disk space availability",
            "Verify write permissions for the save directory",
            "Ensure the save path is valid",
            "Try saving to a different location"
        ]
        
        return {
            'can_recover': True,
            'suggestions': suggestions,
            'retry_recommended': True,
            'recovery_actions': ['check_disk_space', 'verify_permissions', 'try_different_path']
        }
    
    def _handle_config_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理配置错误"""
        suggestions = [
            "Check configuration parameter values and types",
            "Refer to the algorithm documentation for valid parameters",
            "Use default configuration as a starting point",
            "Validate configuration against schema"
        ]
        
        return {
            'can_recover': True,
            'suggestions': suggestions,
            'retry_recommended': True,
            'recovery_actions': ['validate_config', 'use_defaults', 'check_documentation']
        }
    
    def _handle_training_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理训练错误"""
        suggestions = [
            "Check training data and environment state",
            "Reduce batch size or learning rate",
            "Verify network architecture compatibility",
            "Monitor system resources during training"
        ]
        
        return {
            'can_recover': True,
            'suggestions': suggestions,
            'retry_recommended': True,
            'recovery_actions': ['check_data', 'adjust_hyperparameters', 'monitor_resources']
        }
    
    def _handle_resource_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理资源不足错误"""
        suggestions = [
            "Reduce batch size or buffer size",
            "Close other applications to free memory",
            "Use CPU instead of GPU if GPU memory is insufficient",
            "Consider using gradient accumulation"
        ]
        
        return {
            'can_recover': True,
            'suggestions': suggestions,
            'retry_recommended': True,
            'recovery_actions': ['reduce_batch_size', 'free_memory', 'switch_device']
        }
    
    def _handle_network_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理网络错误"""
        suggestions = [
            "Check network architecture definition",
            "Verify input/output dimensions match",
            "Ensure proper layer initialization",
            "Check for gradient flow issues"
        ]
        
        return {
            'can_recover': True,
            'suggestions': suggestions,
            'retry_recommended': True,
            'recovery_actions': ['check_architecture', 'verify_dimensions', 'debug_gradients']
        }
    
    def _handle_environment_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理环境错误"""
        suggestions = [
            "Check environment configuration",
            "Verify environment dependencies are installed",
            "Reset environment state",
            "Check environment compatibility with algorithm"
        ]
        
        return {
            'can_recover': True,
            'suggestions': suggestions,
            'retry_recommended': True,
            'recovery_actions': ['check_env_config', 'reset_environment', 'verify_compatibility']
        }
    
    def register_recovery_strategy(self, exception_type: str, strategy_func: Callable):
        """注册自定义恢复策略"""
        self.recovery_strategies[exception_type] = strategy_func
        logger.info(f"Registered recovery strategy for {exception_type}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        return {
            'error_counts': self.error_counts.copy(),
            'total_errors': sum(self.error_counts.values()),
            'unique_error_types': len(self.error_counts),
            'last_errors': {k: {
                'message': str(v['exception']),
                'timestamp': v['timestamp'],
                'context': v['context']
            } for k, v in self.last_errors.items()}
        }
    
    def clear_error_history(self):
        """清空错误历史"""
        self.error_counts.clear()
        self.last_errors.clear()
        logger.info("Error history cleared")


# 全局错误处理器实例
_global_error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    return _global_error_handler

def handle_algorithm_error(func: Callable) -> Callable:
    """算法异常处理装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AlgorithmFrameworkException as e:
            error_info = _global_error_handler.handle_exception(e, {
                'function': func.__name__,
                'args': str(args)[:200],  # 限制长度
                'kwargs': str(kwargs)[:200]
            })
            logger.error(f"Algorithm framework error in {func.__name__}: {e}")
            raise
        except Exception as e:
            error_info = _global_error_handler.handle_exception(e, {
                'function': func.__name__,
                'args': str(args)[:200],
                'kwargs': str(kwargs)[:200]
            })
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            # 将普通异常包装为框架异常
            raise AlgorithmFrameworkException(f"Unexpected error in {func.__name__}: {e}")
    return wrapper

def handle_training_error(func: Callable) -> Callable:
    """训练异常处理装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                'function': func.__name__,
                'training_context': True
            }
            
            # 尝试从参数中提取训练信息
            if args and hasattr(args[0], 'current_episode'):
                context['episode'] = args[0].current_episode
            if args and hasattr(args[0], 'training_step'):
                context['step'] = args[0].training_step
            
            error_info = _global_error_handler.handle_exception(e, context)
            logger.error(f"Training error in {func.__name__}: {e}")
            raise
    return wrapper

def safe_execute(func: Callable, *args, default_return=None, **kwargs) -> Any:
    """安全执行函数，捕获异常并返回默认值"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_info = _global_error_handler.handle_exception(e, {
            'function': func.__name__ if hasattr(func, '__name__') else str(func),
            'safe_execution': True
        })
        logger.warning(f"Safe execution failed for {func}: {e}")
        return default_return