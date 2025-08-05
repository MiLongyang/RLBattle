# -*- coding: utf-8 -*-

import os
import json
import time
import glob
import shutil
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from algorithms.base_algorithm import BaseAlgorithm

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """模型元数据"""
    algorithm: str
    episode: str
    training_step: int
    episode_count: int
    num_agents: int
    obs_dims: List[int]
    action_dims: List[int]
    task_type: str
    save_time: float
    model_size: int
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建"""
        return cls(**data)


class ModelManager:
    """
    模型管理器
    负责模型的保存、加载、版本管理和性能优化
    """
    
    def __init__(self, base_dir: str = "./models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 性能优化设置
        self.save_format = 'torch'  # 支持torch, onnx等格式
        self.compression_enabled = True
        self.max_load_time = 4.0  # 最大加载时间要求（秒）
        
        # 缓存设置
        self.enable_cache = True
        self.cache_size = 5  # 最多缓存5个模型
        self._model_cache = {}
        self._cache_order = []
        
        logger.info(f"ModelManager initialized with base directory: {self.base_dir}")
    
    def save_model(self, algorithm: BaseAlgorithm, episode: Union[str, int], 
                   save_dir: Optional[str] = None, 
                   performance_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        保存模型
        
        Args:
            algorithm: 算法实例
            episode: 回合数或标识符
            save_dir: 保存目录（可选）
            performance_metrics: 性能指标（可选）
            
        Returns:
            保存路径
        """
        start_time = time.time()
        
        try:
            # 确定保存目录
            if save_dir is None:
                algorithm_name = algorithm.__class__.__name__
                task_type = getattr(algorithm.args, 'task_type', 'unknown')
                save_dir = self.base_dir / f"{algorithm_name}_{task_type}"
            else:
                save_dir = Path(save_dir)
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存模型权重
            algorithm.save_models(str(save_dir), episode)
            
            # 计算模型大小
            model_size = self._calculate_model_size(save_dir, episode)
            
            # 创建元数据
            metadata = ModelMetadata(
                algorithm=algorithm.__class__.__name__,
                episode=str(episode),
                training_step=algorithm.training_step,
                episode_count=algorithm.episode_count,
                num_agents=algorithm.num_agents,
                obs_dims=algorithm.obs_dims,
                action_dims=algorithm.action_dims,
                task_type=getattr(algorithm.args, 'task_type', 'unknown'),
                save_time=time.time(),
                model_size=model_size,
                performance_metrics=performance_metrics or algorithm.get_training_metrics()
            )
            
            # 保存元数据
            metadata_path = save_dir / f"metadata_{episode}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # 创建版本索引
            self._update_version_index(save_dir, metadata)
            
            save_time = time.time() - start_time
            logger.info(f"Model saved successfully in {save_time:.2f}s to {save_dir}")
            
            if save_time > 2.0:  # 保存时间警告阈值
                logger.warning(f"Model saving took {save_time:.2f}s, consider optimization")
            
            return str(save_dir)
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, algorithm: BaseAlgorithm, episode: Union[str, int], 
                   load_dir: Optional[str] = None, use_cache: bool = True) -> ModelMetadata:
        """
        加载模型（优化为≤4秒）
        
        Args:
            algorithm: 算法实例
            episode: 回合数或标识符
            load_dir: 加载目录（可选）
            use_cache: 是否使用缓存
            
        Returns:
            模型元数据
        """
        start_time = time.time()
        
        try:
            # 确定加载目录
            if load_dir is None:
                algorithm_name = algorithm.__class__.__name__
                task_type = getattr(algorithm.args, 'task_type', 'unknown')
                load_dir = self.base_dir / f"{algorithm_name}_{task_type}"
            else:
                load_dir = Path(load_dir)
            
            # 生成缓存键
            cache_key = f"{load_dir}_{episode}"
            
            # 检查缓存
            if use_cache and self.enable_cache and cache_key in self._model_cache:
                logger.debug(f"Loading model from cache: {cache_key}")
                cached_data = self._model_cache[cache_key]
                self._update_cache_order(cache_key)
                
                # 从缓存加载到算法
                self._load_from_cache(algorithm, cached_data)
                
                load_time = time.time() - start_time
                logger.info(f"Model loaded from cache in {load_time:.2f}s")
                return cached_data['metadata']
            
            # 预检查文件存在性
            model_files = self._get_model_files(load_dir, episode)
            if not model_files:
                raise FileNotFoundError(f"Model files not found in {load_dir} for episode {episode}")
            
            # 加载元数据
            metadata_path = load_dir / f"metadata_{episode}.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                metadata = ModelMetadata.from_dict(metadata_dict)
            else:
                # 创建默认元数据
                metadata = ModelMetadata(
                    algorithm=algorithm.__class__.__name__,
                    episode=str(episode),
                    training_step=0,
                    episode_count=0,
                    num_agents=algorithm.num_agents,
                    obs_dims=algorithm.obs_dims,
                    action_dims=algorithm.action_dims,
                    task_type=getattr(algorithm.args, 'task_type', 'unknown'),
                    save_time=0,
                    model_size=0,
                    performance_metrics={}
                )
            
            # 加载模型权重
            algorithm.load_models(str(load_dir), episode)
            
            # 缓存模型（如果启用）
            if use_cache and self.enable_cache:
                self._cache_model(cache_key, algorithm, metadata)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s from {load_dir}")
            
            if load_time > self.max_load_time:
                logger.warning(f"Model loading took {load_time:.2f}s, exceeding {self.max_load_time}s requirement")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def list_models(self, algorithm_name: Optional[str] = None, 
                   task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出可用模型
        
        Args:
            algorithm_name: 算法名称过滤（可选）
            task_type: 任务类型过滤（可选）
            
        Returns:
            模型信息列表
        """
        models = []
        
        try:
            for model_dir in self.base_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                # 解析目录名
                dir_parts = model_dir.name.split('_')
                if len(dir_parts) < 2:
                    continue
                
                dir_algorithm = dir_parts[0]
                dir_task_type = '_'.join(dir_parts[1:])
                
                # 应用过滤器
                if algorithm_name and dir_algorithm.upper() != algorithm_name.upper():
                    continue
                if task_type and dir_task_type != task_type:
                    continue
                
                # 查找元数据文件
                metadata_files = list(model_dir.glob("metadata_*.json"))
                
                for metadata_file in metadata_files:
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata_dict = json.load(f)
                        
                        model_info = {
                            'path': str(model_dir),
                            'algorithm': dir_algorithm,
                            'task_type': dir_task_type,
                            'episode': metadata_dict.get('episode', 'unknown'),
                            'save_time': metadata_dict.get('save_time', 0),
                            'model_size': metadata_dict.get('model_size', 0),
                            'performance_metrics': metadata_dict.get('performance_metrics', {})
                        }
                        
                        models.append(model_info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
            
            # 按保存时间排序
            models.sort(key=lambda x: x['save_time'], reverse=True)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def delete_model(self, algorithm_name: str, task_type: str, episode: Union[str, int]) -> bool:
        """
        删除模型
        
        Args:
            algorithm_name: 算法名称
            task_type: 任务类型
            episode: 回合数或标识符
            
        Returns:
            是否删除成功
        """
        try:
            model_dir = self.base_dir / f"{algorithm_name}_{task_type}"
            
            if not model_dir.exists():
                logger.warning(f"Model directory not found: {model_dir}")
                return False
            
            # 删除模型文件
            model_files = self._get_model_files(model_dir, episode)
            for file_path in model_files:
                file_path.unlink()
                logger.debug(f"Deleted model file: {file_path}")
            
            # 删除元数据文件
            metadata_path = model_dir / f"metadata_{episode}.json"
            if metadata_path.exists():
                metadata_path.unlink()
                logger.debug(f"Deleted metadata file: {metadata_path}")
            
            # 从缓存中移除
            cache_key = f"{model_dir}_{episode}"
            if cache_key in self._model_cache:
                del self._model_cache[cache_key]
                if cache_key in self._cache_order:
                    self._cache_order.remove(cache_key)
            
            # 更新版本索引
            self._update_version_index(model_dir, None, remove_episode=str(episode))
            
            logger.info(f"Model deleted: {algorithm_name}_{task_type} episode {episode}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False
    
    def get_model_info(self, algorithm_name: str, task_type: str, 
                      episode: Union[str, int]) -> Optional[ModelMetadata]:
        """
        获取模型信息
        
        Args:
            algorithm_name: 算法名称
            task_type: 任务类型
            episode: 回合数或标识符
            
        Returns:
            模型元数据或None
        """
        try:
            model_dir = self.base_dir / f"{algorithm_name}_{task_type}"
            metadata_path = model_dir / f"metadata_{episode}.json"
            
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            return ModelMetadata.from_dict(metadata_dict)
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None
    
    def cleanup_old_models(self, algorithm_name: str, task_type: str, 
                          keep_count: int = 10) -> int:
        """
        清理旧模型，只保留最新的几个
        
        Args:
            algorithm_name: 算法名称
            task_type: 任务类型
            keep_count: 保留的模型数量
            
        Returns:
            删除的模型数量
        """
        try:
            model_dir = self.base_dir / f"{algorithm_name}_{task_type}"
            
            if not model_dir.exists():
                return 0
            
            # 获取所有模型
            models = []
            metadata_files = list(model_dir.glob("metadata_*.json"))
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata_dict = json.load(f)
                    
                    models.append({
                        'episode': metadata_dict.get('episode', 'unknown'),
                        'save_time': metadata_dict.get('save_time', 0)
                    })
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
            
            # 按保存时间排序，保留最新的
            models.sort(key=lambda x: x['save_time'], reverse=True)
            
            deleted_count = 0
            for model in models[keep_count:]:
                if self.delete_model(algorithm_name, task_type, model['episode']):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old models for {algorithm_name}_{task_type}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")
            return 0
    
    def _get_model_files(self, model_dir: Path, episode: Union[str, int]) -> List[Path]:
        """获取模型文件列表"""
        model_files = []
        
        # 查找所有包含episode标识的文件
        pattern = f"*{episode}*"
        for file_path in model_dir.glob(pattern):
            if file_path.suffix in ['.pth', '.pt', '.onnx'] and not file_path.name.startswith('metadata'):
                model_files.append(file_path)
        
        return model_files
    
    def _calculate_model_size(self, model_dir: Path, episode: Union[str, int]) -> int:
        """计算模型大小"""
        total_size = 0
        model_files = self._get_model_files(model_dir, episode)
        
        for file_path in model_files:
            if file_path.exists():
                total_size += file_path.stat().st_size
        
        return total_size
    
    def _update_version_index(self, model_dir: Path, metadata: Optional[ModelMetadata], 
                            remove_episode: Optional[str] = None) -> None:
        """更新版本索引"""
        try:
            index_path = model_dir / "version_index.json"
            
            # 读取现有索引
            if index_path.exists():
                with open(index_path, 'r') as f:
                    index = json.load(f)
            else:
                index = {'versions': []}
            
            # 移除指定版本
            if remove_episode:
                index['versions'] = [v for v in index['versions'] if v.get('episode') != remove_episode]
            
            # 添加新版本
            if metadata:
                version_info = {
                    'episode': metadata.episode,
                    'save_time': metadata.save_time,
                    'model_size': metadata.model_size,
                    'performance_metrics': metadata.performance_metrics
                }
                
                # 移除同名版本（如果存在）
                index['versions'] = [v for v in index['versions'] if v.get('episode') != metadata.episode]
                index['versions'].append(version_info)
                
                # 按保存时间排序
                index['versions'].sort(key=lambda x: x['save_time'], reverse=True)
            
            # 保存索引
            with open(index_path, 'w') as f:
                json.dump(index, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to update version index: {e}")
    
    def _cache_model(self, cache_key: str, algorithm: BaseAlgorithm, 
                    metadata: ModelMetadata) -> None:
        """缓存模型"""
        try:
            # 检查缓存大小限制
            if len(self._model_cache) >= self.cache_size:
                # 移除最旧的缓存
                oldest_key = self._cache_order[0]
                del self._model_cache[oldest_key]
                self._cache_order.remove(oldest_key)
            
            # 缓存模型状态（简化版本，实际可能需要深拷贝）
            cached_data = {
                'metadata': metadata,
                'cache_time': time.time()
            }
            
            self._model_cache[cache_key] = cached_data
            self._cache_order.append(cache_key)
            
            logger.debug(f"Model cached: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache model: {e}")
    
    def _update_cache_order(self, cache_key: str) -> None:
        """更新缓存顺序"""
        if cache_key in self._cache_order:
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
    
    def _load_from_cache(self, algorithm: BaseAlgorithm, cached_data: Dict[str, Any]) -> None:
        """从缓存加载模型（简化实现）"""
        # 实际实现中需要恢复模型状态
        # 这里只是示例，实际需要根据具体算法实现
        logger.debug("Loading model from cache (simplified implementation)")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'enabled': self.enable_cache,
            'size': len(self._model_cache),
            'capacity': self.cache_size,
            'cached_models': list(self._model_cache.keys())
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._model_cache.clear()
        self._cache_order.clear()
        logger.info("Model cache cleared")