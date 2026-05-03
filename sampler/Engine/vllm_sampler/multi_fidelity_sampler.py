import csv
import os
import time
import itertools
import json
import subprocess
import asyncio
import yaml
from typing import Dict, List, Tuple, Any
import numpy as np
from pyDOE import lhs
from sampler import main as run_sampler
import argparse
from dataclasses import asdict


class VLLMMultiFidelitySampler:
    def __init__(self, config_path: str, run_id: int = 1):
        """
        初始化 vLLM 多保真度采样器
        
        Args:
            config_path: vLLM 保真度配置文件路径
            run_id: 运行 ID，用于区分不同的实验轮次
        """
        self.config_path = config_path
        self.run_id = run_id
        self.config = self.load_config()
        
        # 系统配置
        self.sys_name = "vllm"
        self.sample_size = 1000  # 配置采样数量
        
        # 目录设置
        self.data_dir = os.path.join('results', f'{self.sys_name}')
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.config_samples_path = os.path.join(self.data_dir, f'{self.sys_name}_lhs_configs.csv')
        self.fidelity_file_path = os.path.join(self.data_dir, f'{self.sys_name}_fidelities.csv')
        self.log_path = os.path.join(self.data_dir, 'evaluated_configs', f'run_{run_id}')
        
        os.makedirs(os.path.dirname(self.config_samples_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.fidelity_file_path), exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        # vLLM 配置参数定义
        self.vllm_knobs_info = self.get_vllm_knobs_info()
        
    def load_config(self) -> Dict:
        """加载 YAML 配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_vllm_knobs_info(self) -> Dict[str, Dict]:
        """定义 vLLM 配置参数的搜索空间"""
        return {
            'tensor_parallel_size': {
                'type': 'enum',
                'enum_values': [1, 2]  # 根据可用 GPU 数量调整
            },
            'pipeline_parallel_size': {
                'type': 'enum', 
                'enum_values': [1, 2, 4]
            },
            'block_size': {
                'type': 'enum',
                'enum_values': [8, 16, 32, 64]
            },
            'max_num_batched_tokens': {
                'type': 'enum',
                'enum_values': [2048, 4096, 8192, 16384]
            },
            'max_num_seqs': {
                'type': 'enum',
                'enum_values': [64, 128, 256, 512]
            },
            'scheduler_delay_factor': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0
            },
            'enable_chunked_prefill': {
                'type': 'enum',
                'enum_values': ['True', 'False']
            },
            'enable_prefix_caching': {
                'type': 'enum',
                'enum_values': ['True', 'False']
            },
            'disable_custom_all_reduce': {
                'type': 'enum',
                'enum_values': ['True', 'False']  
            },
            'use_v2_block_manager': {
                'type': 'enum',
                'enum_values': ['True', 'False']
            }
        }
    
    def get_fidelity_factors_info(self) -> Dict[str, Dict]:
        """从配置文件中提取保真度因子信息"""
        fidelity_factors = {}
        
        # 数据集子集大小
        if 'subset_size_or_ratio' in self.config:
            fidelity_factors['subset_ratio'] = {
                'type': 'enum',
                'enum_values': self.config['subset_size_or_ratio']
            }
        
        # 前缀共享比例
        if 'shared_prefix_ratio' in self.config:
            fidelity_factors['shared_prefix_ratio'] = {
                'type': 'enum', 
                'enum_values': self.config['shared_prefix_ratio']
            }
        
        # 并发级别
        if 'concurrency_levels' in self.config:
            fidelity_factors['concurrency_level'] = {
                'type': 'enum',
                'enum_values': self.config['concurrency_levels']
            }
        
        # 请求到达率
        if 'request_pattern' in self.config and 'arrival_rate' in self.config['request_pattern']:
            fidelity_factors['arrival_rate'] = {
                'type': 'enum',
                'enum_values': self.config['request_pattern']['arrival_rate']
            }
        
        return fidelity_factors
    
    def sampling_and_evaluate(self):
        """主要的采样和评估流程"""
        print(f"开始 vLLM 多保真度采样评估 (Run {self.run_id})")
        
        # 1. 采样 vLLM 配置
        configs = self.sample_configs_by_lhs(self.sample_size)
        print(f"生成了 {len(configs)} 个配置样本")
        
        # 2. 生成保真度组合
        fidelities = self.read_or_generate_fidelities()
        print(f"生成了 {len(fidelities)} 个保真度组合")
        
        # 3. 对每个保真度和配置组合进行评估
        total_evaluations = len(configs) * len(fidelities)
        completed_evaluations = 0
        
        for fidelity in fidelities:
            log_file = self.generate_data_file_name(fidelity)
            log_file_path = os.path.join(self.log_path, log_file)
            
            # 读取已有的评估结果，避免重复计算
            existing_configs = self.read_existing_configs(log_file_path)
            
            for config in configs:
                config_tuple = tuple(config.items())
                if config_tuple in existing_configs:
                    completed_evaluations += 1
                    continue
                
                print(f"评估进度: {completed_evaluations + 1}/{total_evaluations}")
                print(f"配置: {config}")
                print(f"保真度: {fidelity}")
                
                try:
                    self.evaluate_config(config, fidelity, log_file_path)
                    print("✓ 评估完成")
                except Exception as e:
                    print(f"✗ 评估失败: {e}")
                
                completed_evaluations += 1
        
        print(f"所有评估完成！总计: {completed_evaluations}/{total_evaluations}")
    
    def evaluate_config(self, config: Dict, fidelity: Dict, log_file_path: str):
        """在指定保真度下评估一个 vLLM 配置"""
        start_time = time.time()
        
        try:
            # 1. 启动 vLLM 服务器
            server_process = self.start_vllm_server(config)
            
            # 2. 等待服务器启动
            self.wait_for_server_ready()
            
            # 3. 运行基准测试
            metrics = self.run_benchmark(fidelity)
            
            # 4. 记录结果
            evaluated_time = time.time() - start_time
            self.logging_data(log_file_path, config, fidelity, metrics, evaluated_time)
            
        finally:
            # 5. 清理：停止服务器
            if 'server_process' in locals():
                self.stop_vllm_server(server_process)
    
    def start_vllm_server(self, config: Dict) -> subprocess.Popen:
        """启动 vLLM 服务器"""
        # 构建服务器启动命令
        cmd = [
            'bash', './run_server.sh',
            './models/Llama-2-7b-hf',  # model_path
            '8004',  # port
            str(config['tensor_parallel_size']),
            str(config['pipeline_parallel_size']),
            str(config['max_num_seqs']),
            str(config['max_num_batched_tokens']),
            str(config['scheduler_delay_factor']),
            str(config['block_size']),
            config['enable_chunked_prefill'],
            config['enable_prefix_caching'],
            config['disable_custom_all_reduce'],
            config['use_v2_block_manager']
        ]
        
        # 启动服务器进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        return process
    
    def wait_for_server_ready(self, timeout: int = 120):
        """等待服务器准备就绪"""
        import socket
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', 8004))
                sock.close()
                
                if result == 0:
                    print("vLLM 服务器已启动")
                    time.sleep(5)  # 额外等待确保完全就绪
                    return
            except:
                pass
            
            time.sleep(2)
        
        raise RuntimeError(f"vLLM 服务器在 {timeout} 秒内未能启动")
    
    def run_benchmark(self, fidelity: Dict) -> Dict:
        """运行基准测试并返回性能指标"""
        # 构建 sampler 参数
        args = argparse.Namespace(
            backend='vllm',
            pressure_test=False,
            max_concurrent_requests=fidelity.get('concurrency_level', 1),
            base_url=None,
            host='localhost',
            port='8004',
            endpoint='/generate',
            dataset_name='sharegpt',
            dataset_path='../datasets/sg_90k_part1.json',
            model='Llama-2-7b-hf',
            tokenizer='./models/Llama-2-7b-hf',
            best_of=1,
            use_beam_search=False,
            num_prompts=int(1000 * fidelity.get('subset_ratio', 1.0)),  # 根据子集比例调整
            sharegpt_output_len=None,
            request_rate=fidelity.get('arrival_rate', 10.0),
            seed=42,
            trust_remote_code=False,
            disable_tqdm=True,
            save_result=False,
            result_dir=None,
            ignore_eos=True,
            # vLLM 配置参数（用于标记）
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            block_size=16,
            max_num_batched_tokens=4096,
            max_num_seqs=256,
            scheduler_delay_factor=0.0,
            enable_chunked_prefill="False",
            enable_prefix_caching="False", 
            disable_custom_all_reduce="False",
            use_v2_block_manager="False"
        )
        
        # 运行采样器
        try:
            from sampler import main as run_sampler
            metrics = run_sampler(args)
            return metrics
        except Exception as e:
            print(f"基准测试运行失败: {e}")
            # 返回默认指标
            return {
                'request_throughput': 0.0,
                'input_throughput': 0.0,
                'output_throughput': 0.0,
                'p95_latency_ms': float('inf'),
                'mean_ttft_ms': float('inf'),
                'mean_tpot_ms': float('inf'),
                'completed': 0,
                'duration': 0.0
            }
    
    def stop_vllm_server(self, process: subprocess.Popen):
        """停止 vLLM 服务器"""
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        print("vLLM 服务器已停止")
    
    def sample_configs_by_lhs(self, num_samples: int) -> List[Dict]:
        """使用拉丁超立方采样生成 vLLM 配置"""
        if os.path.exists(self.config_samples_path):
            existing_configs = self.read_csv_configs(self.config_samples_path)
            if len(existing_configs) >= num_samples:
                return existing_configs[:num_samples]
        else:
            existing_configs = []
        
        additional_samples = num_samples - len(existing_configs)
        if additional_samples <= 0:
            return existing_configs
        
        # 只对数值型参数使用 LHS
        numeric_params = {k: v for k, v in self.vllm_knobs_info.items() if v['type'] == 'float'}
        enum_params = {k: v for k, v in self.vllm_knobs_info.items() if v['type'] == 'enum'}
        
        new_configs = []
        
        if numeric_params:
            # 对数值参数使用 LHS
            num_numeric = len(numeric_params)
            lhs_sample = lhs(num_numeric, samples=additional_samples)
            
            for i in range(additional_samples):
                config = {}
                
                # 处理数值参数
                for j, (key, val) in enumerate(numeric_params.items()):
                    range_width = val['max'] - val['min']
                    config[key] = lhs_sample[i][j] * range_width + val['min']
                
                # 处理枚举参数（随机选择）
                for key, val in enum_params.items():
                    config[key] = np.random.choice(val['enum_values'])
                
                new_configs.append(config)
        else:
            # 如果没有数值参数，只处理枚举参数
            for i in range(additional_samples):
                config = {}
                for key, val in enum_params.items():
                    config[key] = np.random.choice(val['enum_values'])
                new_configs.append(config)
        
        all_configs = existing_configs + new_configs
        self.write_csv_configs(all_configs, self.config_samples_path)
        return all_configs
    
    def read_or_generate_fidelities(self) -> List[Dict]:
        """读取或生成保真度组合"""
        if os.path.exists(self.fidelity_file_path):
            return self.read_csv_fidelities(self.fidelity_file_path)
        
        fidelities = self.generate_fidelity_samples()
        self.write_csv_fidelities(fidelities)
        return fidelities
    
    def generate_fidelity_samples(self) -> List[Dict]:
        """生成保真度样本组合"""
        fidelity_factors = self.get_fidelity_factors_info()
        
        if not fidelity_factors:
            # 如果没有保真度因子，返回默认配置
            return [{}]
        
        fidelity_keys = list(fidelity_factors.keys())
        fidelity_values = [factor['enum_values'] for factor in fidelity_factors.values()]
        
        # 生成所有组合
        combinations = list(itertools.product(*fidelity_values))
        return [dict(zip(fidelity_keys, combo)) for combo in combinations]
    
    def read_csv_configs(self, file_path: str) -> List[Dict]:
        """读取 CSV 格式的配置文件"""
        configs = []
        with open(file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                config = {}
                for key, value in row.items():
                    if key in self.vllm_knobs_info:
                        param_info = self.vllm_knobs_info[key]
                        if param_info['type'] == 'float':
                            config[key] = float(value)
                        elif param_info['type'] == 'integer':
                            config[key] = int(value)
                        else:
                            config[key] = value
                    else:
                        config[key] = value
                configs.append(config)
        return configs
    
    def read_csv_fidelities(self, file_path: str) -> List[Dict]:
        """读取 CSV 格式的保真度数据"""
        with open(file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            return [dict(row) for row in reader]
    
    def write_csv_configs(self, configs: List[Dict], file_path: str):
        """写入配置到 CSV 文件"""
        if not configs:
            return
        
        with open(file_path, 'w', newline='') as file:
            fieldnames = list(self.vllm_knobs_info.keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(configs)
    
    def write_csv_fidelities(self, fidelities: List[Dict]):
        """写入保真度数据到 CSV 文件"""
        if not fidelities:
            return
        
        with open(self.fidelity_file_path, 'w', newline='') as file:
            fieldnames = list(fidelities[0].keys()) if fidelities else []
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fidelities)
    
    def read_existing_configs(self, file_path: str) -> set:
        """读取已评估的配置，避免重复计算"""
        existing_configs = set()
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader, None)  # 跳过头部
                for row in reader:
                    if len(row) >= len(self.vllm_knobs_info):
                        config_keys = list(self.vllm_knobs_info.keys())
                        config_values = row[:len(config_keys)]
                        existing_configs.add(tuple(zip(config_keys, config_values)))
        return existing_configs
    
    def logging_data(self, log_file_path: str, config: Dict, fidelity: Dict, 
                     metrics: Dict, evaluated_time: float):
        """记录配置和性能评估数据"""

        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # 构建头部：配置参数 + 保真度参数 + 性能指标 + 元数据
                header = (list(self.vllm_knobs_info.keys()) + 
                         list(fidelity.keys()) +
                         ['request_throughput', 'input_throughput', 'output_throughput', 
                          'p95_latency_ms', 'mean_ttft_ms', 'mean_tpot_ms', 
                          'completed', 'duration', 'evaluated_time'])
                writer.writerow(header)
        
        with open(log_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            
            row = ([config[param] for param in self.vllm_knobs_info.keys()] +
                   [fidelity.get(param, '') for param in fidelity.keys()] +
                   [metrics.get('request_throughput', 0),
                    metrics.get('input_throughput', 0),
                    metrics.get('output_throughput', 0),
                    metrics.get('p95_latency_ms', 0),
                    metrics.get('mean_ttft_ms', 0),
                    metrics.get('mean_tpot_ms', 0),
                    metrics.get('completed', 0),
                    metrics.get('duration', 0),
                    evaluated_time])
            writer.writerow(row)
    
    @staticmethod
    def generate_data_file_name(fidelity: Dict) -> str:
        """生成数据文件名"""
        parts = []
        for key, value in fidelity.items():
            parts.append(f"{key}_{value}")
        
        if not parts:
            return "default_fidelity.csv"
        
        return "_".join(parts) + ".csv"


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM 多保真度采样器")
    parser.add_argument("--config", type=str, 
                       default="./config/vllm_fidelity_config.yaml",
                       help="保真度配置文件路径")
    parser.add_argument("--run-id", type=int, default=1,
                       help="运行 ID")
    
    args = parser.parse_args()
    
    sampler = VLLMMultiFidelitySampler(
        config_path=args.config,
        run_id=args.run_id
    )
    
    sampler.sampling_and_evaluate()