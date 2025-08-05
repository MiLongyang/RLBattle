# -*- coding: utf-8 -*-

"""
算法框架专用异常类
定义了框架中可能出现的各种异常情况
"""

class AlgorithmFrameworkException(Exception):
    """算法框架基础异常类"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'exception_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }


class AlgorithmNotFoundError(AlgorithmFrameworkException):
    """算法未找到异常"""
    def __init__(self, algorithm_name: str, available_algorithms: list = None):
        message = f"Algorithm '{algorithm_name}' not found or not registered"
        details = {
            'algorithm_name': algorithm_name,
            'available_algorithms': available_algorithms or []
        }
        super().__init__(message, 'ALG_NOT_FOUND', details)


class AlgorithmInitializationError(AlgorithmFrameworkException):
    """算法初始化异常"""
    def __init__(self, algorithm_name: str, reason: str, config: dict = None):
        message = f"Failed to initialize algorithm '{algorithm_name}': {reason}"
        details = {
            'algorithm_name': algorithm_name,
            'reason': reason,
            'config': config
        }
        super().__init__(message, 'ALG_INIT_FAILED', details)


class IncompatibleAlgorithmError(AlgorithmFrameworkException):
    """算法不兼容异常"""
    def __init__(self, algorithm_name: str, task_type: str, reason: str = None):
        message = f"Algorithm '{algorithm_name}' is not compatible with task '{task_type}'"
        if reason:
            message += f": {reason}"
        details = {
            'algorithm_name': algorithm_name,
            'task_type': task_type,
            'reason': reason
        }
        super().__init__(message, 'ALG_INCOMPATIBLE', details)


class ModelLoadError(AlgorithmFrameworkException):
    """模型加载异常"""
    def __init__(self, model_path: str, episode: str, reason: str):
        message = f"Failed to load model from '{model_path}' episode '{episode}': {reason}"
        details = {
            'model_path': model_path,
            'episode': episode,
            'reason': reason
        }
        super().__init__(message, 'MODEL_LOAD_FAILED', details)


class ModelSaveError(AlgorithmFrameworkException):
    """模型保存异常"""
    def __init__(self, model_path: str, episode: str, reason: str):
        message = f"Failed to save model to '{model_path}' episode '{episode}': {reason}"
        details = {
            'model_path': model_path,
            'episode': episode,
            'reason': reason
        }
        super().__init__(message, 'MODEL_SAVE_FAILED', details)


class ModelCorruptionError(AlgorithmFrameworkException):
    """模型文件损坏异常"""
    def __init__(self, model_path: str, corruption_details: str):
        message = f"Model file corrupted at '{model_path}': {corruption_details}"
        details = {
            'model_path': model_path,
            'corruption_details': corruption_details
        }
        super().__init__(message, 'MODEL_CORRUPTED', details)


class ConfigValidationError(AlgorithmFrameworkException):
    """配置参数验证失败异常"""
    def __init__(self, config_key: str, config_value, validation_error: str):
        message = f"Configuration validation failed for '{config_key}': {validation_error}"
        details = {
            'config_key': config_key,
            'config_value': config_value,
            'validation_error': validation_error
        }
        super().__init__(message, 'CONFIG_INVALID', details)


class MissingConfigError(AlgorithmFrameworkException):
    """缺少必要配置参数异常"""
    def __init__(self, missing_keys: list, algorithm_name: str = None):
        message = f"Missing required configuration keys: {missing_keys}"
        if algorithm_name:
            message += f" for algorithm '{algorithm_name}'"
        details = {
            'missing_keys': missing_keys,
            'algorithm_name': algorithm_name
        }
        super().__init__(message, 'CONFIG_MISSING', details)


class TrainingError(AlgorithmFrameworkException):
    """训练过程异常"""
    def __init__(self, episode: int, step: int, reason: str, algorithm_name: str = None):
        message = f"Training failed at episode {episode}, step {step}: {reason}"
        details = {
            'episode': episode,
            'step': step,
            'reason': reason,
            'algorithm_name': algorithm_name
        }
        super().__init__(message, 'TRAINING_FAILED', details)


class EvaluationError(AlgorithmFrameworkException):
    """评估过程异常"""
    def __init__(self, episode: int, reason: str, algorithm_name: str = None):
        message = f"Evaluation failed at episode {episode}: {reason}"
        details = {
            'episode': episode,
            'reason': reason,
            'algorithm_name': algorithm_name
        }
        super().__init__(message, 'EVALUATION_FAILED', details)


class PerformanceWarning(AlgorithmFrameworkException):
    """性能指标超出预期范围警告"""
    def __init__(self, metric_name: str, current_value: float, threshold: float, 
                 recommendation: str = None):
        message = f"Performance warning: {metric_name} = {current_value:.2f} exceeds threshold {threshold:.2f}"
        details = {
            'metric_name': metric_name,
            'current_value': current_value,
            'threshold': threshold,
            'recommendation': recommendation
        }
        super().__init__(message, 'PERFORMANCE_WARNING', details)


class ResourceExhaustionError(AlgorithmFrameworkException):
    """系统资源不足异常"""
    def __init__(self, resource_type: str, current_usage: float, limit: float, 
                 suggestion: str = None):
        message = f"Resource exhaustion: {resource_type} usage {current_usage:.1f}% exceeds limit {limit:.1f}%"
        details = {
            'resource_type': resource_type,
            'current_usage': current_usage,
            'limit': limit,
            'suggestion': suggestion
        }
        super().__init__(message, 'RESOURCE_EXHAUSTED', details)


class EnvironmentError(AlgorithmFrameworkException):
    """环境相关异常"""
    def __init__(self, env_name: str, reason: str, env_info: dict = None):
        message = f"Environment error in '{env_name}': {reason}"
        details = {
            'env_name': env_name,
            'reason': reason,
            'env_info': env_info or {}
        }
        super().__init__(message, 'ENV_ERROR', details)


class NetworkError(AlgorithmFrameworkException):
    """网络相关异常"""
    def __init__(self, network_name: str, layer_info: str, reason: str):
        message = f"Network error in '{network_name}' at {layer_info}: {reason}"
        details = {
            'network_name': network_name,
            'layer_info': layer_info,
            'reason': reason
        }
        super().__init__(message, 'NETWORK_ERROR', details)


class DataValidationError(AlgorithmFrameworkException):
    """数据验证异常"""
    def __init__(self, data_type: str, validation_error: str, expected_format: str = None):
        message = f"Data validation failed for {data_type}: {validation_error}"
        details = {
            'data_type': data_type,
            'validation_error': validation_error,
            'expected_format': expected_format
        }
        super().__init__(message, 'DATA_INVALID', details)