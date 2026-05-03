import os
import csv
import itertools
import json
from typing import List, Dict, Any
from utils.server_utils import *

from pyDOE import lhs
import numpy as np

import subprocess
import time
import threading

from datetime import datetime

class VLLMMultiFidelitySampler:
    def __init__(self, run_id: int = 1):
        """
        初始化 vLLM 多保真度采样器
        
        Args:
            run_id: 运行 ID, 用于区分不同的实验轮次
        """
        self.run_id = run_id
        self.sys_name = "vllm"
        self.sample_size = 1000
        self.sampling_params_space = SAMPLING_PARAMS_SPACE
        self.vllm_config_space = VLLM_CONFIG_SPACE

        # 目录设置
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join('results', f'{self.sys_name}')
        self.config_dir = os.path.join(self.base_dir, 'config')
        self.config_samples_path = os.path.join(self.config_dir, f'{self.sys_name}_lhs_configs.csv')
        self.log_dir = os.path.join(self.base_dir, 'logs')
        
        # 创建必要目录
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def get_combined_knobs_info(self):
        """合并 VLLM_CONFIG_SPACE 和 SAMPLING_PARAMS_SPACE 的参数信息"""
        knobs_info = {}
        
        # 1. Copy VLLM_CONFIG_SPACE
        for key, info in self.vllm_config_space.items():
            knobs_info[key] = info
        
        # 2. Copy SAMPLING_PARAMS_SPACE
        for key, info in self.sampling_params_space.items():
            knobs_info[key] = info
            
        return knobs_info
        
    def read_csv_configs(self, path):
        """从CSV读取配置"""
        if not os.path.exists(path):
            return []
        
        configs = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 类型转换
                config = {}
                for k, v in row.items():
                    try:
                        if v.lower() == 'true':
                            config[k] = True
                        elif v.lower() == 'false':
                            config[k] = False
                        elif '.' in v:
                            config[k] = float(v)
                        else:
                            config[k] = int(v)
                    except:
                        config[k] = v
                configs.append(config)
        return configs
    
    def write_csv_configs(self, configs, path):
        """将配置写入CSV"""
        if not configs:
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=configs[0].keys())
            writer.writeheader()
            for config in configs:
                writer.writerow(config)

    def sample_configs_by_lhs(self, num_samples: int) -> list:
        """
        使用LHS采样并通过rejection sampling确保参数满足约束条件
        """
        if os.path.exists(self.config_samples_path):
            existing_configs = self.read_csv_configs(self.config_samples_path)
            if len(existing_configs) >= num_samples:
                return [c for c in existing_configs[:num_samples] 
                    if int(c['max_num_batched_tokens']) >= int(c['max_num_seqs'])]
            additional_samples = num_samples - len(existing_configs)
        else:
            existing_configs = []
            additional_samples = num_samples

        knobs_info = self.get_combined_knobs_info()
        param_keys = list(knobs_info.keys())
        num_params = len(param_keys)
        
        new_configs = []
        attempts = 0
        max_attempts = additional_samples * 10
        
        while len(new_configs) < additional_samples and attempts < max_attempts:
            attempts += 1
            
            # LHS 采样一个配置
            lhs_sample = lhs(num_params, samples=1)[0]
            config = {}
            sampling_params = {}
            
            for j, key in enumerate(param_keys):
                info = knobs_info[key]
                val = lhs_sample[j]
                
                if key in self.sampling_params_space:
                    # 处理采样参数的逻辑
                    if info['type'] == 'float':
                        range_width = info['max'] - info['min']
                        value = val * range_width + info['min']
                        sampling_params[key] = round(value, 3)
                    elif info['type'] == 'integer':
                        range_width = info['max'] - info['min'] + 1
                        value = int(val * range_width) + info['min']
                        sampling_params[key] = value
                    elif info['type'] == 'enum':
                        possible_values = info['enum_values']
                        idx = int(val * len(possible_values))
                        idx = min(idx, len(possible_values) - 1)
                        sampling_params[key] = str(possible_values[idx])
                else:
                    # 处理系统配置参数
                    if info['type'] == 'integer':
                        range_width = info['max'] - info['min'] + 1
                        config[key] = int(val * range_width) + info['min']
                    elif info['type'] == 'float':
                        range_width = info['max'] - info['min']
                        config[key] = val * range_width + info['min']
                    elif info['type'] == 'enum':
                        possible_values = info['enum_values']
                        idx = int(val * len(possible_values))
                        idx = min(idx, len(possible_values) - 1)
                        config[key] = str(possible_values[idx])
            
            # === speculative decoding ===
            if config.get('enable_speculative_decoding') == 'True':
                method = config.get('speculative_method', 'eagle')
                
                # 确保方法和模型匹配
                if method == "eagle":
                    # EAGLE 配置
                    model = "./models/EAGLE-LLaMA3-Instruct-8B"
                    speculative_config = {
                        "model": model,
                        "draft_tensor_parallel_size": config.get('draft_tensor_parallel_size', 1),
                        "num_speculative_tokens": config.get('num_speculative_tokens', 2),
                        "method": "eagle",
                    }
                    config['speculative_model'] = model
                    
                elif method == "ngram":
                    # N-grams 配置
                    speculative_config = {
                        "method": "ngram",
                        "num_speculative_tokens": config.get('num_speculative_tokens', 5),
                        "prompt_lookup_max": config.get('prompt_lookup_max', 4),
                    }
                    config['speculative_model'] = "NGRAM_MODEL"
                    
                else:
                    # 默认使用 EAGLE
                    method = "eagle"
                    model = "./models/EAGLE-LLaMA3-Instruct-8B"
                    speculative_config = {
                        "model": model,
                        "draft_tensor_parallel_size": config.get('draft_tensor_parallel_size', 1),
                        "num_speculative_tokens": config.get('num_speculative_tokens', 2),
                        "method": "eagle",
                    }
                    config['speculative_model'] = model
                
                # 更新配置字段
                config['speculative_method'] = method
                config['speculative_config'] = json.dumps(speculative_config, separators=(',', ':'))
                
            else:
                # 不启用时：用明确的标识符表示"未使用"
                config['speculative_method'] = "disabled"
                config['speculative_model'] = "DISABLED"
                config['draft_tensor_parallel_size'] = 0
                config['num_speculative_tokens'] = 0
                config['prompt_lookup_max'] = 0
                config['speculative_config'] = "DISABLED"
            
            # 验证约束条件
            if config['max_num_batched_tokens'] >= config['max_num_seqs']:
                if sampling_params:
                    config['extra_body'] = json.dumps(sampling_params, ensure_ascii=False)
                else:
                    config['extra_body'] = "{}"
                new_configs.append(config)
        
        if len(new_configs) < additional_samples:
            print(f"Warning: Could only generate {len(new_configs)} valid configs after {max_attempts} attempts")
        
        all_configs = existing_configs + new_configs
        
        # === 确保所有配置都有相同的字段结构 ===
        if all_configs:
            # 获取第一个配置的字段作为模板
            template_fields = set(all_configs[0].keys())
            
            # 为所有配置补全字段
            for config in all_configs:
                for field in template_fields:
                    if field not in config:
                        # 为缺失字段设置默认值
                        if field == 'extra_body':
                            config[field] = "{}"
                        elif field == 'speculative_config':
                            # 根据enable_speculative_decoding的值决定
                            if config.get('enable_speculative_decoding') == 'True':
                                method = config.get('speculative_method', 'eagle')
                                if method == "eagle":
                                    spec_config = {
                                        "model": config.get('speculative_model', './models/EAGLE-LLaMA3-Instruct-8B'),
                                        "draft_tensor_parallel_size": config.get('draft_tensor_parallel_size', 1),
                                        "num_speculative_tokens": config.get('num_speculative_tokens', 2),
                                        "method": "eagle"
                                    }
                                elif method == "ngram":
                                    spec_config = {
                                        "method": "ngram",
                                        "num_speculative_tokens": config.get('num_speculative_tokens', 5),
                                        "prompt_lookup_max": config.get('prompt_lookup_max', 4),
                                    }
                                else:  # 默认eagle
                                    spec_config = {
                                        "model": './models/EAGLE-LLaMA3-Instruct-8B',
                                        "draft_tensor_parallel_size": config.get('draft_tensor_parallel_size', 1),
                                        "num_speculative_tokens": config.get('num_speculative_tokens', 2),
                                        "method": "eagle"
                                    }
                                config[field] = json.dumps(spec_config, separators=(',', ':'))
                            else:
                                config[field] = "DISABLED"
                        elif field == 'speculative_method':
                            config[field] = "disabled" if config.get('enable_speculative_decoding') != 'True' else 'eagle'
                        elif field == 'speculative_model':
                            config[field] = "DISABLED" if config.get('enable_speculative_decoding') != 'True' else './models/EAGLE-LLaMA3-Instruct-8B'
                        elif field == 'draft_tensor_parallel_size':
                            config[field] = 0 if config.get('enable_speculative_decoding') != 'True' else 1
                        elif field == 'num_speculative_tokens':
                            config[field] = 0 if config.get('enable_speculative_decoding') != 'True' else 2
                        elif field == 'prompt_lookup_max':
                            config[field] = 0 if config.get('enable_speculative_decoding') != 'True' else 4
                        elif field in ['enable_speculative_decoding', 'enable_chunked_prefill', 
                                    'enable_prefix_caching', 'disable_custom_all_reduce', 
                                    'enforce_eager', 'enable_log_requests']:
                            config[field] = 'False'
                        elif field in ['tp_size', 'pp_size']:
                            config[field] = 1
                        elif field == 'block_size':
                            config[field] = 16
                        elif field == 'max_num_seqs':
                            config[field] = 128
                        elif field == 'max_num_batched_tokens':
                            config[field] = 2048
                        elif field == 'swap_space':
                            config[field] = 0
                        elif field == 'max_seq_len_to_capture':
                            config[field] = 8192
                        elif field == 'scheduling_policy':
                            config[field] = 'fcfs'
                        else:
                            config[field] = ""
        
        self.write_csv_configs(all_configs, self.config_samples_path)
        return all_configs
    
    def generate_vllm_config_filename(self, vllm_cfg: Dict[str, Any]) -> str:
        """根据VLLM配置生成文件名"""
        factors = []
        
        # vLLM配置参数
        factors.append(f"tp{vllm_cfg.get('tp_size', 1)}")
        factors.append(f"pp{vllm_cfg.get('pp_size', 1)}")
        factors.append(f"bs{vllm_cfg.get('block_size', 16)}")
        factors.append(f"seqs{vllm_cfg.get('max_num_seqs', 128)}")
        factors.append(f"tokens{vllm_cfg.get('max_num_batched_tokens', 2048)}")
        
        # 可选参数（如果不是默认值）
        swap_space = vllm_cfg.get('swap_space', 0)
        if swap_space > 0:
            factors.append(f"swap{swap_space}")

        max_seq_len = vllm_cfg.get('max_seq_len_to_capture', 8192)
        if max_seq_len != 8192:
            factors.append(f"seqlen{max_seq_len}")
        
        scheduling = vllm_cfg.get('scheduling_policy', 'fcfs')
        if scheduling != 'fcfs':
            factors.append(f"sched{scheduling}")
        
        # === Speculative Decoding Filename ===
        if vllm_cfg.get('enable_speculative_decoding') == 'True':
            spec_config_str = vllm_cfg.get('speculative_config', '{}')
            try:
                if spec_config_str and spec_config_str != 'DISABLED':
                    spec_config = json.loads(spec_config_str)
                    method = spec_config.get('method', 'eagle')
                    
                    if method == 'eagle':
                        model_name = spec_config.get('model', '').split('/')[-1]
                        num_tokens = spec_config.get('num_speculative_tokens', 2)
                        factors.append(f"eagle-{model_name}-{num_tokens}")
                    elif method == 'ngram':
                        num_tokens = spec_config.get('num_speculative_tokens', 5)
                        lookup_max = spec_config.get('prompt_lookup_max', 4)
                        factors.append(f"ngram-{num_tokens}-{lookup_max}")
                            
            except (json.JSONDecodeError, AttributeError):
                # 如果解析失败，使用备用方案
                method = vllm_cfg.get('speculative_method', 'eagle')
                if method == 'eagle':
                    model = vllm_cfg.get('speculative_model', '').split('/')[-1]
                    num_tokens = vllm_cfg.get('num_speculative_tokens', 2)
                    if model and model != "DISABLED":
                        factors.append(f"eagle-{model}-{num_tokens}")
                elif method == 'ngram':
                    num_tokens = vllm_cfg.get('num_speculative_tokens', 5)
                    lookup_max = vllm_cfg.get('prompt_lookup_max', 4)
                    factors.append(f"ngram-{num_tokens}-{lookup_max}")
                
        # 布尔选项（只在启用时添加）
        if vllm_cfg.get('enable_chunked_prefill') == 'True':
            factors.append("chunked")
        if vllm_cfg.get('enable_prefix_caching') == 'True':
            factors.append("prefix")
        if vllm_cfg.get('disable_custom_all_reduce') == 'True':
            factors.append("noar")
        if vllm_cfg.get('enforce_eager') == 'True':
            factors.append("eager")
        if vllm_cfg.get('enable_log_requests') == 'True':
            factors.append("logreq")
        
        # 连接所有因子
        return "_".join(factors)

    def generate_fidelity_dirname(self, fidelity_cfg: Dict[str, Any]) -> str:
        """根据fidelity配置生成目录名"""
        factors = []
        
        rate = fidelity_cfg.get('request_rate', float('inf'))
        if rate == float('inf'):
            factors.append("rateinf")
        else:
            factors.append(f"rate{rate}")
        
        # burstiness
        burstiness = fidelity_cfg.get('burstiness', 1.0)
        factors.append(f"burst{burstiness}")
        
        # concurrency
        max_concurrency = fidelity_cfg.get('max_concurrency')
        factors.append(f"conc{max_concurrency}")
        
        # repeat_count 
        repeat_count = fidelity_cfg.get('repeat_count', 1)
        factors.append(f"repeat{repeat_count}")
        
        # num_prompts
        factors.append(f"prompts{fidelity_cfg.get('num_prompts', 100)}")
        
        return "_".join(factors)

    def generate_output_filepath(self, vllm_cfg: Dict[str, Any], fidelity_cfg: Dict[str, Any], 
                                results_dir: str, config_idx: int, fidelity_idx: int) -> str:
        """生成完整的输出文件路径"""
        vllm_filename = self.generate_vllm_config_filename(vllm_cfg)
        fidelity_dirname = self.generate_fidelity_dirname(fidelity_cfg)
        
        # 创建目录
        fidelity_dir = os.path.join(results_dir, fidelity_dirname)
        os.makedirs(fidelity_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{vllm_filename}_config{config_idx}_fidelity{fidelity_idx}_{timestamp}.json"
        
        return os.path.join(fidelity_dir, filename)

# 1. 定义参数空间（可根据实际情况扩展/修改）
VLLM_CONFIG_SPACE = {
    "tp_size": {
        "type": "integer",
        "min": 1,
        "max": 1
    },
    "pp_size": {
        "type": "integer",
        "min": 1,
        "max": 1
    },
    "block_size": {
        "type": "enum",
        "enum_values": [16, 32, 64, 128]
    },
    "max_num_seqs": {
        "type": "integer",
        "min": 64,
        "max": 4096
    },
    "max_num_batched_tokens": {
        "type": "integer",
        "min": 64,
        "max": 8192
    },
    "enable_chunked_prefill": {
        "type": "enum",
        "enum_values": ["False", "True"]
    },
    "enable_prefix_caching": {
        "type": "enum",
        "enum_values": ["False", "True"]
    },
    "disable_custom_all_reduce": {
        "type": "enum",
        "enum_values": ["False", "True"]
    },
    "swap_space": {
        "type": "integer",
        "min": 2,
        "max": 16  # GB
    },
    "max_seq_len_to_capture": {
        "type": "integer", 
        "min": 512,
        "max": 8192
    },
    "enforce_eager": {
        "type": "enum",
        "enum_values": ["False", "True"]
    },
    "scheduling_policy": {
        "type": "enum",
        "enum_values": ["fcfs", "priority"]
    },
    "enable_log_requests": {
        "type": "enum",
        "enum_values": ["False", "True"]
    },
    # === 修改后的Speculative Decoding Config ===
    "enable_speculative_decoding": {
        "type": "enum",
        "enum_values": ["False", "True"]
    },
    "speculative_method": {
        "type": "enum",
        "enum_values": [
            "eagle",             # EAGLE based draft models
            "ngram",             # N-grams speculating
            "disabled"           # 禁用时的占位符
        ]
    },
    "speculative_model": {
        "type": "enum",
        "enum_values": [
            "./models/EAGLE-LLaMA3-Instruct-8B",       # EAGLE model
            "NGRAM_MODEL",                             # N-grams不需要具体模型路径
            "DISABLED"                                 # 禁用时的占位符
        ]
    },
    "draft_tensor_parallel_size": {
        "type": "integer",
        "min": 1,
        "max": 1
    },
    "num_speculative_tokens": {
        "type": "integer",
        "min": 2,
        "max": 8
    },
    # === 新增N-grams相关参数 ===
    "prompt_lookup_max": {
        "type": "integer",
        "min": 2,
        "max": 8
    },
}

SAMPLING_PARAMS_SPACE = {
    "temperature": {
        "type": "float",
        "min": 0.0,
        "max": 1.0
    },
    "top_k": {
        "type": "integer",
        "min": 1,
        "max": 100
    },
    "min_p": {
        "type": "float",
        "min": 0.1,
        "max": 1.0
    },
    "repetition_penalty": {
        "type": "float",
        "min": 1.0,
        "max": 2.0
    },
    "length_penalty": {
        "type": "float",
        "min": 0.5,
        "max": 2.0
    }
}

FIDELITY_SPACE = {
    "num_prompts": [50, 100, 200],  # 简化选项
    "max_concurrency": [4, 8, 16, 32],
    "request_rate": [5.0, 10.0, 15.0],
    "burstiness": [0.5, 1.0, 2.0],
    "repeat_count": [1, 2],
}

# 2. 生成所有参数组合
def get_param_combinations(param_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_space.keys())
    values = list(param_space.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combos]


def main():
    # 固定参数
    model_path = "./models/llama3-8B-instruct/"
    dataset_path = "../datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    dataset_name = "sharegpt"
    model_id = "llama3-8B-instruct"
    results_dir = "./results"
    tokenizer_path = model_path
    port = "8000"
    gpus_to_use = "0"
    
    # 创建采样器实例
    sampler = VLLMMultiFidelitySampler()
    
    # 1. 使用LHS对VLLM配置和采样参数空间进行联合采样
    num_lhs_samples = 10000
    combined_configs = sampler.sample_configs_by_lhs(num_lhs_samples)
    print(f"Generated {len(combined_configs)} combined configs using LHS")
    
    # 2. 使用笛卡尔积生成所有工作负载参数组合
    fidelity_configs = get_param_combinations(FIDELITY_SPACE)
    print(f"Generated {len(fidelity_configs)} fidelity configs using Cartesian product")
    
    # 计算总实验次数
    total = len(combined_configs) * len(fidelity_configs)
    print(f"Total experiments to run: {total}")
    
    # 日志文件
    log_csv = os.path.join(results_dir, "multi_fidelity_benchmark_log.csv")
    os.makedirs(results_dir, exist_ok=True)
    
    # === Log.csv Initialization ===
    if not combined_configs or not fidelity_configs:
        print("Error: No configs generated!")
        return
    
    # 构建完整的header
    vllm_keys = list(combined_configs[0].keys())
    fidelity_keys = list(fidelity_configs[0].keys())
    header = vllm_keys + fidelity_keys + ["exit_code", "timestamp"]
    
    print(f"CSV Header: {header}")
    
    # 检查已有日志，加载已完成的测试
    completed_tests = set()
    csv_needs_header = True
    
    if os.path.exists(log_csv):
        print(f"Found existing log file: {log_csv}")
        try:
            with open(log_csv, 'r', newline='') as f:
                reader = csv.reader(f)
                try:
                    existing_header = next(reader)
                    
                    # 检查header是否匹配（忽略timestamp字段差异）
                    expected_core_header = vllm_keys + fidelity_keys + ["exit_code"]
                    existing_core_header = existing_header[:-1] if len(existing_header) > len(expected_core_header) else existing_header
                    
                    if existing_core_header == expected_core_header:
                        print("Header matched, loading completed tests...")
                        csv_needs_header = False
                        
                        for row in reader:
                            if len(row) >= len(expected_core_header):
                                # 构建测试ID（不包括exit_code和timestamp）
                                test_id = ','.join(row[:len(vllm_keys + fidelity_keys)])
                                completed_tests.add(test_id)

                        print(f"Completed tests: {len(completed_tests)}")
                    else:
                        print("Header mismatch, creating backup and restarting")
                        print(f"Existing header: {existing_header}")
                        print(f"Expected header: {header}")
                        
                        # 创建备份
                        import shutil
                        backup_path = log_csv + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                        shutil.move(log_csv, backup_path)
                        print(f"Original file has been backed up to: {backup_path}")

                        completed_tests = set()
                        csv_needs_header = True
                        
                except StopIteration:
                    print("CSV file is empty, re-creating")
                    csv_needs_header = True
                    
        except Exception as e:
            print(f"Error reading existing CSV file: {e}")
            # 创建备份并重新开始
            backup_path = log_csv + f'.error_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            import shutil
            shutil.move(log_csv, backup_path)
            completed_tests = set()
            csv_needs_header = True
    else:
        print(f"Creating new log file: {log_csv}")

    if csv_needs_header:
        try:
            with open(log_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                f.flush()
            print(f"CSV file initializationg failed, header has been written.")
        except Exception as e:
            print(f"Initializing CSV file failed: {e}")
            return
    
    # === 修复：改进的测试记录函数 ===
    def record_test_result(vllm_cfg, fidelity_cfg, exit_code):
        """记录单个测试结果到CSV"""
        try:
            # 构建行数据
            row = []
            
            # 添加vLLM配置数据
            for key in vllm_keys:
                value = vllm_cfg.get(key, "")
                if key == 'extra_body':
                    # 确保extra_body是字符串格式
                    if isinstance(value, dict):
                        value = json.dumps(value, separators=(',', ':'))
                    elif value is None:
                        value = ""
                row.append(str(value))
            
            # 添加fidelity配置数据
            for key in fidelity_keys:
                value = fidelity_cfg.get(key, "")
                row.append(str(value))
            
            # 添加结果数据
            row.append(str(exit_code))
            row.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
            
            # 写入CSV
            with open(log_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()
                
            print(f"✅ Result has been saved to CSV: exit_code={exit_code}")
            
            # 构建测试ID并添加到已完成集合
            test_id = ','.join(row[:len(vllm_keys + fidelity_keys)])
            completed_tests.add(test_id)
            
            return True
            
        except Exception as e:
            print(f"❌ Recording test result failed: {e}")
            return False
    
    # 3. 逐一评测 - 主循环
    config_idx = 0
    total_skipped = 0
    
    for vllm_cfg in combined_configs:
        config_idx += 1
        print(f"\n=== Testing vLLM Configuration {config_idx}/{len(combined_configs)} ===")
        print("vLLM config:", vllm_cfg)
        
        # 处理extra_body 
        extra_body_str = vllm_cfg.get("extra_body", "{}")
        try:
            extra_body = json.loads(extra_body_str) if extra_body_str and extra_body_str != "{}" else {}
        except json.JSONDecodeError:
            print(f"Warning: Invalid extra_body JSON: {extra_body_str}")
            extra_body = {}
        
        # 准备vLLM配置为字符串
        str_vllm_cfg = {}
        for k, v in vllm_cfg.items():
            if k != 'extra_body':
                if isinstance(v, bool):
                    str_vllm_cfg[k] = "True" if v else "False"
                else:
                    str_vllm_cfg[k] = str(v)
        
        # 检查当前vllm_cfg下是否有任何未完成的fidelity测试
        all_fidelities_completed = True
        
        for fidelity_cfg in fidelity_configs:
            # 构建测试ID
            test_row = []
            for key in vllm_keys:
                value = vllm_cfg.get(key, "")
                if key == 'extra_body':
                    if isinstance(value, dict):
                        value = json.dumps(value, separators=(',', ':'))
                    elif value is None or value == "":
                        value = extra_body_str
                test_row.append(str(value))
            
            for key in fidelity_keys:
                test_row.append(str(fidelity_cfg.get(key, "")))
                
            test_id = ','.join(test_row)
            
            if test_id not in completed_tests:
                all_fidelities_completed = False
                break
                
        if all_fidelities_completed:
            print(f"Skipping vLLM config {config_idx} - All fidelity tests completed")
            total_skipped += len(fidelity_configs)
            continue
        
        # === 启动前的全面清理和检查 ===
        print(f"Pre-startup cleanup and check for port {port}...")
        port_available = False

        # 尝试使用server_utils中的函数进行清理
        cleanup_vllm_processes(port)
        force_kill_port_processes(port)
        port_available = wait_for_port_release(port, max_wait_time=30)

        # 如果仍然无法释放端口，执行一次紧急清理
        if not port_available:
            print(f"WARNING: Port {port} still unavailable, attempting emergency cleanup...")
            try:
                # 排除当前进程
                current_pid = os.getpid()
                os.system(f"lsof -ti:{port} | grep -v {current_pid} | xargs -r kill -9 2>/dev/null || true")
                os.system(f"pkill -9 -f 'vllm' -v -P {current_pid} 2>/dev/null || true")
                os.system(f"pkill -9 -f 'api_server' -v -P {current_pid} 2>/dev/null || true")
                os.system(f"pkill -9 -f 'uvicorn' -v -P {current_pid} 2>/dev/null || true")
                time.sleep(5)
                
                # 最终检查
                port_available = check_port_available(port)
            except Exception as e:
                print(f"Emergency cleanup error: {e}")
                
            if not port_available:
                print(f"CRITICAL: Cannot free port {port}, skipping this configuration")
                # 记录所有fidelity测试为失败
                for fidelity_cfg in fidelity_configs:
                    record_test_result(vllm_cfg, fidelity_cfg, -2)  # -2表示端口占用
                continue
            else:
                print(f"Emergency cleanup successful for port {port}")

        print(f"Port {port} is available, proceeding with server startup...")

        
        # 启动服务器
        print(f"Starting vLLM server with config {config_idx}")
        cmd = [
            "bash", "./run_server.sh",
            model_path, port,
            str_vllm_cfg["tp_size"], str_vllm_cfg["pp_size"],
            str_vllm_cfg["max_num_seqs"], str_vllm_cfg["max_num_batched_tokens"],
            str_vllm_cfg["block_size"],
            str_vllm_cfg["enable_chunked_prefill"], str_vllm_cfg["enable_prefix_caching"],
            str_vllm_cfg["disable_custom_all_reduce"],
            str_vllm_cfg.get("swap_space", "0"),
            str_vllm_cfg.get("max_seq_len_to_capture", "8192"),
            str_vllm_cfg.get("enforce_eager", "False"),
            str_vllm_cfg.get("scheduling_policy", "fcfs"),
            str_vllm_cfg.get("enable_log_requests", "False"),
            str_vllm_cfg.get("enable_speculative_decoding", "False"),
            str_vllm_cfg.get("speculative_config", "null")
        ]
        
        # 服务器启动逻辑
        logs_dir = "./logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        server_log_file = os.path.join(logs_dir, f"server_config_{config_idx}_{timestamp}.log")
        
        print(f"Server logs will be saved to: {server_log_file}")
        
        server_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus_to_use)
        )
        
        # 服务器监控逻辑
        log_file_handle = open(server_log_file, 'w')
        log_lock = threading.Lock()
        monitor_thread_stop = threading.Event()
        
        def monitor_server_output():
            try:
                for line in server_process.stdout:
                    if monitor_thread_stop.is_set():
                        break
                    line = line.rstrip()
                    log_timestamp = datetime.now().strftime("%H:%M:%S")
                    formatted_line = f"[{log_timestamp}] [vLLM-{config_idx}] {line}"
                    print(formatted_line)
                    
                    with log_lock:
                        try:
                            log_file_handle.write(formatted_line + '\n')
                            log_file_handle.flush()
                        except ValueError:
                            break
            except Exception as e:
                error_msg = f"Error monitoring server output: {e}"
                print(error_msg)
        
        output_thread = threading.Thread(target=monitor_server_output, daemon=True)
        output_thread.start()
        
        # 等待服务器启动
        print(f"Waiting for vLLM server to start...")

        if not wait_for_server_ready(port, max_wait_time=600):
            print(f"Failed to start server on port {port}, attempting cleanup and skip...")
            
            # 记录服务器启动失败的所有fidelity测试
            for fidelity_cfg in fidelity_configs:
                record_test_result(vllm_cfg, fidelity_cfg, -1)  # -1表示服务器启动失败
            
            # 立即清理失败的服务器进程
            try:
                if server_process.poll() is None:
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        server_process.kill()
                        server_process.wait(timeout=5)
            except:
                pass
                
            # 清理相关进程和端口
            cleanup_vllm_processes(port)
            force_kill_port_processes(port)
            wait_for_port_release(port, max_wait_time=30)
            
            monitor_thread_stop.set()
            output_thread.join(timeout=5)
            try:
                log_file_handle.close()
            except:
                pass
            
            # 启动失败后等待更长时间
            print("Server startup failed, waiting before next configuration...")
            time.sleep(20)
            continue
        
        try:
            # 对该vLLM配置测试所有fidelity配置
            fidelity_idx = 0
            
            for fidelity_cfg in fidelity_configs:
                fidelity_idx += 1
                test_idx = (config_idx-1) * len(fidelity_configs) + fidelity_idx
                
                # 构建测试ID检查是否已完成
                test_row = []
                for key in vllm_keys:
                    value = vllm_cfg.get(key, "")
                    if key == 'extra_body':
                        if isinstance(value, dict):
                            value = json.dumps(value, separators=(',', ':'))
                        elif value is None or value == "":
                            value = extra_body_str
                    test_row.append(str(value))
                
                for key in fidelity_keys:
                    test_row.append(str(fidelity_cfg.get(key, "")))
                    
                test_id = ','.join(test_row)
                
                if test_id in completed_tests:
                    print(f"Skipping {test_idx}/{total} - Already completed")
                    total_skipped += 1
                    continue
                
                print(f"\n=== Running Test {test_idx}/{total} ===")
                print(f"vLLM config {config_idx}, Fidelity config {fidelity_idx}")
                print("Fidelity config:", fidelity_cfg)
                print("Sampling params:", extra_body)
                
                str_fidelity_cfg = {k: str(v) for k, v in fidelity_cfg.items()}
                
                # 生成输出文件路径
                output_filepath = sampler.generate_output_filepath(
                    vllm_cfg, fidelity_cfg, results_dir, config_idx, fidelity_idx
                )
                
                # 客户端命令构建
                client_cmd = [
                    "python3", "sampler.py",
                    "--backend", "openai",
                    "--host", "localhost", "--port", port,
                    "--model", model_id, "--tokenizer", tokenizer_path,
                    "--dataset-name", dataset_name, "--dataset-path", dataset_path,
                    "--num-prompts", str_fidelity_cfg["num_prompts"],
                    "--request-rate", str_fidelity_cfg["request_rate"],
                    "--burstiness", str_fidelity_cfg["burstiness"],
                    "--seed", "42", "--output-file", output_filepath,
                    "--max-concurrency", str_fidelity_cfg["max_concurrency"],
                    "--repeat-count", str_fidelity_cfg["repeat_count"],
                    "--vllm-config", json.dumps(vllm_cfg, separators=(',', ':'))
                ]

                if extra_body:
                    client_cmd.extend(["--extra-body", json.dumps(extra_body, separators=(',', ':'))])
                
                print("Running client:", " ".join(client_cmd))
                print(f"Results will be saved to: {output_filepath}")
                
                # 客户端执行逻辑
                client_log_file = os.path.join(logs_dir, f"client_config_{config_idx}_fidelity_{fidelity_idx}_{timestamp}.log")
                client_log_handle = open(client_log_file, 'w')
                client_lock = threading.Lock()
                client_thread_stop = threading.Event()
                
                client_process = subprocess.Popen(
                    client_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                def monitor_client_output():
                    try:
                        for line in client_process.stdout:
                            if client_thread_stop.is_set():
                                break
                            line = line.rstrip()
                            log_timestamp = datetime.now().strftime("%H:%M:%S")
                            formatted_line = f"[{log_timestamp}] [Client-{config_idx}-{fidelity_idx}] {line}"
                            print(formatted_line)
                            
                            with client_lock:
                                try:
                                    client_log_handle.write(formatted_line + '\n')
                                    client_log_handle.flush()
                                except ValueError:
                                    break
                    except Exception as e:
                        error_msg = f"Error monitoring client output: {e}"
                        print(error_msg)
                
                client_output_thread = threading.Thread(target=monitor_client_output, daemon=True)
                client_output_thread.start()
                
                # 等待客户端进程完成
                exit_code = client_process.wait()
                
                # 停止客户端监控
                client_thread_stop.set()
                client_output_thread.join(timeout=5)
                
                try:
                    with client_lock:
                        completion_timestamp = datetime.now().strftime("%H:%M:%S")
                        client_log_handle.write(f"[{completion_timestamp}] [Client-{config_idx}-{fidelity_idx}] === Client finished with exit code: {exit_code} ===\n")
                        client_log_handle.flush()
                except:
                    pass
                
                try:
                    client_log_handle.close()
                except:
                    pass
                
                print(f"Sampler finished with exit code: {exit_code}")
                print(f"Client logs saved to: {client_log_file}")
                
                if exit_code != 0:
                    print(f"Client failed! Check logs at: {client_log_file}")
                
                # 记录测试结果
                success = record_test_result(vllm_cfg, fidelity_cfg, exit_code)
                if not success:
                    print("⚠️  Warning: Saving result failed!")

                time.sleep(5)
        
        finally:
            # 清理资源
            print(f"Shutting down configuration {config_idx}...")
            monitor_thread_stop.set()
            
            # 使用改进的函数关闭服务器
            port_released = improved_server_shutdown(server_process, port, config_idx, log_file_handle, log_lock)
            output_thread.join(timeout=5)
            
            try:
                log_file_handle.close()
            except:
                pass
            
            # 最终检查 - 简化为单一流程
            if not port_released or not check_port_available(port):
                print(f"Port {port} may still be in use, performing final cleanup...")
                try:
                    current_pid = os.getpid()
                    os.system(f"lsof -ti:{port} | grep -v {current_pid} | xargs -r kill -9 2>/dev/null || true")
                    os.system(f"pkill -9 -f 'vllm' -v -P {current_pid} 2>/dev/null || true")
                    time.sleep(5)
                except Exception as e:
                    print(f"Final cleanup error: {e}")
            
            # 强制短暂等待，确保系统资源释放
            time.sleep(5)
            print("==== Preparing for next configuration ====")

if __name__ == "__main__":
    main()