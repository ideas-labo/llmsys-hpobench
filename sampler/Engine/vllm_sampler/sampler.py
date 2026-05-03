"""
vLLM Sampler Client Script

This script acts as a client to benchmark a running vLLM server instance.
It takes vLLM configuration parameters (as fidelity factors for tagging results),
workload parameters, and dataset information as input. It runs the benchmark
and saves the performance metrics along with the configuration used.

This script DOES NOT start or configure the vLLM server. It assumes the server
is already running with the specified configuration.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import AsyncGenerator, Any, Union, List, Optional, Tuple, Dict

import numpy as np
from clients.backend_request_func import (async_request_openai_completions, async_request_vllm, 
                                          RequestFuncInput, RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# --- Enhanced Benchmark Metrics with Server Metrics ---

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    p95_latency_ms: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    server_metrics_summary: Optional[Dict[str, Any]] = None
    successful_requests: int = 0
    failed_requests: int = 0
    similarity_metrics: Optional[Dict[str, Any]] = None


@dataclass 
class ServerMetricsSummary:
    """服务器指标汇总"""
    total_requests_processed: int = 0
    avg_gpu_cache_usage_change: float = 0.0
    avg_running_requests_change: float = 0.0 
    avg_waiting_requests_change: float = 0.0
    total_tokens_generated: int = 0
    total_tokens_prompted: int = 0
    metrics_collection_success_rate: float = 0.0


# 更新的函数映射，支持指标收集
ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_vllm,
    "openai": async_request_openai_completions
}


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    input_length_range: Optional[Tuple[int, int]] = None,
    return_groundtruth: bool = True,
) -> Union[List[Tuple[str, int, int]], Tuple[List[Tuple[str, int, int]], Dict[str, str]]]:
    """
    从ShareGPT数据集采样请求，同时可选返回ground truth数据
    
    Args:
        dataset_path: ShareGPT数据集路径
        num_requests: 需要采样的请求数量
        tokenizer: 分词器
        fixed_output_len: 固定输出长度, 如果为None则使用原始completion长度
        input_length_range: 输入长度范围(min_len, max_len), 如果为None则使用默认过滤逻辑
        return_groundtruth: 是否同时返回ground truth字典
    
    Returns:
        如果return_groundtruth=False:
            List[Tuple[str, int, int]]: (prompt, prompt_len, output_len)格式的请求列表
        如果return_groundtruth=True:
            Tuple[List[Tuple[str, int, int]], Dict[str, str]]: 请求列表和ground truth字典
    """
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("fixed_output_len must be None or at least 4 tokens")

    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    # 同时构建ground truth字典
    groundtruth_dict = {} if return_groundtruth else None
    
    # 设置长度过滤逻辑
    if input_length_range is not None:
        min_len, max_len = input_length_range
        assert min_len >= 0 and max_len >= min_len, "input_length_range invalid"
        use_length_range = True
    else:
        use_length_range = False

    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        
        # 应用长度过滤逻辑
        meets_criteria = False
        if use_length_range:
            # 使用指定的长度范围（用于prefix caching）
            if min_len <= prompt_len <= max_len:
                meets_criteria = True
        else:
            # 使用原有的默认过滤逻辑
            if prompt_len >= 4 and output_len >= 4:
                if prompt_len <= 1024 and prompt_len + output_len <= 2048:
                    meets_criteria = True

        if meets_criteria:
            filtered_dataset.append((prompt, prompt_len, output_len))
            if return_groundtruth:
                groundtruth_dict[prompt] = completion

    if return_groundtruth:
        print(f"\n📚 Loaded ground truth from ShareGPT")
        print(f"   Sampled requests num: {len(filtered_dataset)}")
        print(f"   ground truth num: {len(groundtruth_dict)}")
        return filtered_dataset, groundtruth_dict
    else:
        return filtered_dataset

def calculate_text_similarity(reference: str, hypothesis: str) -> float:
    """
    使用sacrebleu计算两段文本的相似度（BLEU评分）
    
    Args:
        reference: 参考文本（ground truth）
        hypothesis: 模型生成的文本
    
    Returns:
        BLEU分数（0到1之间的浮点数）
    """
    try:
        import sacrebleu
    except ImportError:
        print("SacreBLEU库未安装，请执行 'pip install sacrebleu' 安装")
        return 0.0
    
    try:
        # sacrebleu要求参考文本为列表
        reference_list = [reference]
        
        # 计算BLEU分数
        bleu_score = sacrebleu.sentence_bleu(hypothesis, reference_list).score
        
        # sacrebleu返回的是百分制分数（0到100），这里转换为0到1之间
        return bleu_score / 100.0
    except Exception as e:
        print(f"Errored when computing BLEU: {e}")
        return 0.0

def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    """Sample requests from the Sonnet dataset."""
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []

    # Ensure we have enough data
    if len(dataset) < num_requests:
        dataset = dataset * ((num_requests // len(dataset)) + 1)
    
    random.shuffle(dataset)

    for data in dataset:
        if len(filtered_dataset) >= num_requests:
            break
            
        prompt = data["prompt"]
        prompt_len = len(tokenizer(prompt).input_ids)
        
        # Apply prefix if specified
        if prefix_len > 0:
            # Take first prefix_len tokens as prefix
            prefix_tokens = tokenizer(prompt).input_ids[:prefix_len]
            prompt = tokenizer.decode(prefix_tokens)
            prompt_len = prefix_len
        
        # Adjust prompt to target input length
        if prompt_len < input_len:
            # Pad with spaces or repeat content if needed
            padding_needed = input_len - prompt_len
            prompt += " " * padding_needed
            prompt_len = input_len
        elif prompt_len > input_len:
            # Truncate to input_len
            tokens = tokenizer(prompt).input_ids[:input_len]
            prompt = tokenizer.decode(tokens)
            prompt_len = input_len
            
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def sample_default_requests(
    num_requests: int,
    input_len: int,
    output_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    """Generate default requests with specified input/output lengths."""
    requests = []
    
    base_prompts = [
        "Please explain the concept of artificial intelligence",
        "Write a story about a journey through space",
        "Describe the process of photosynthesis in plants", 
        "What are the benefits of renewable energy sources",
        "Explain how machine learning algorithms work"
    ]
    
    for i in range(num_requests):
        base_prompt = base_prompts[i % len(base_prompts)]
        
        # Adjust prompt length to match input_len
        tokens = tokenizer(base_prompt).input_ids
        if len(tokens) < input_len:
            # Extend prompt
            extension = f" Please provide a detailed explanation with examples and cover various aspects of this topic in approximately {input_len} tokens."
            extended_prompt = base_prompt + extension
            tokens = tokenizer(extended_prompt).input_ids
            
            # Fine-tune to exact length
            if len(tokens) > input_len:
                tokens = tokens[:input_len]
                prompt = tokenizer.decode(tokens)
            else:
                prompt = extended_prompt
        else:
            # Truncate to input_len
            tokens = tokens[:input_len]
            prompt = tokenizer.decode(tokens)
            
        actual_len = len(tokenizer(prompt).input_ids)
        requests.append((prompt, actual_len, output_len))
    
    return requests

def repeat_and_sort_requests(
    requests: List[Tuple[str, int, int]],
    repeat_count: int,
    sort: bool = False
) -> List[Tuple[str, int, int]]:
    """
    重复并可选地排序请求列表
    直接处理sampler的(prompt, prompt_len, output_len)格式
    """
    repeated_requests = requests * repeat_count
    if sort:
        repeated_requests.sort(key=lambda x: x[1])  # 按prompt_len排序
    else:
        random.shuffle(repeated_requests)
    
    return repeated_requests

async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a tuple of
            (prompt, prompt_len, output_len).
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    input_requests = iter(input_requests)

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}.")
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    groundtruth_dict: Optional[Dict[str, str]] = None 
) -> Tuple[BenchmarkMetrics, List[int]]:
    """Calculate comprehensive benchmark metrics including server metrics."""
    actual_output_lens: List[int] = []
    total_input = 0
    total_output = 0
    successful_requests = 0
    failed_requests = 0
    
    # Server metrics aggregation
    server_metrics_data = []
    ttfts_ms: List[float] = []
    tpots_ms: List[float] = []
    latencies_ms: List[float] = []

    # 初始化相似度分数列表
    similarity_scores: List[float] = []


    for i in range(len(outputs)):
        if outputs[i].success:
            successful_requests += 1
            # Set actual output length
            if tokenizer is not None:
                try:
                    output_len = len(tokenizer(outputs[i].generated_text).input_ids)
                except:
                    output_len = outputs[i].output_tokens or len(outputs[i].generated_text.split())
            else:
                output_len = outputs[i].output_tokens or len(outputs[i].generated_text.split())
            
            actual_output_lens.append(output_len)
            
            total_input += input_requests[i][1]  # prompt_len
            total_output += output_len
            
            # Collect performance metrics
            if outputs[i].ttft > 0:
                ttfts_ms.append(outputs[i].ttft * 1000)
            if outputs[i].itl:
                avg_tpot = sum(outputs[i].itl) / len(outputs[i].itl)
                tpots_ms.append(avg_tpot * 1000)
            if outputs[i].latency > 0:
                latencies_ms.append(outputs[i].latency * 1000)
            
            # Collect server metrics if available
            if outputs[i].server_metrics:
                server_metrics_data.append(outputs[i].server_metrics)
        else:
            failed_requests += 1
            actual_output_lens.append(0)

    # Calculate performance metrics
    completed = successful_requests
    if dur_s > 0:
        request_throughput = completed / dur_s
        input_throughput = total_input / dur_s
        output_throughput = total_output / dur_s
    else:
        request_throughput = input_throughput = output_throughput = 0

    # Latency metrics
    if latencies_ms:
        p95_latency_ms = np.percentile(latencies_ms, 95)
    else:
        p95_latency_ms = 0

    # TTFT metrics
    if ttfts_ms:
        mean_ttft_ms = np.mean(ttfts_ms)
        median_ttft_ms = np.median(ttfts_ms)
        p99_ttft_ms = np.percentile(ttfts_ms, 99)
    else:
        mean_ttft_ms = median_ttft_ms = p99_ttft_ms = 0

    # TPOT metrics
    if tpots_ms:
        mean_tpot_ms = np.mean(tpots_ms)
        median_tpot_ms = np.median(tpots_ms)
        p99_tpot_ms = np.percentile(tpots_ms, 99)
    else:
        mean_tpot_ms = median_tpot_ms = p99_tpot_ms = 0

    # Aggregate server metrics
    server_metrics_summary = aggregate_server_metrics(server_metrics_data)

    
    for i in range(len(outputs)):
        # 已有的性能指标计算保持不变
        
        # 添加相似度计算
        if groundtruth_dict and outputs[i].success:
            prompt = input_requests[i][0]
            if prompt in groundtruth_dict:
                reference_text = groundtruth_dict[prompt]
                hypothesis_text = outputs[i].generated_text
                
                # 计算相似度
                score = calculate_text_similarity(reference_text, hypothesis_text)
                similarity_scores.append(score)
    
    # 计算相似度统计指标
    # print(f"Debug: 收集到的相似度分数数量: {len(similarity_scores)}")

    similarity_metrics = None
    if similarity_scores:
        similarity_metrics = {
            "mean_similarity": np.mean(similarity_scores),
            "median_similarity": np.median(similarity_scores),
            "min_similarity": np.min(similarity_scores),
            "max_similarity": np.max(similarity_scores),
            "p90_similarity": np.percentile(similarity_scores, 90),
            "samples_compared": len(similarity_scores),
            "metric_type": "bleu"
        }

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=total_output,
        request_throughput=request_throughput,
        input_throughput=input_throughput,
        output_throughput=output_throughput,
        p95_latency_ms=p95_latency_ms,
        mean_ttft_ms=mean_ttft_ms,
        median_ttft_ms=median_ttft_ms,
        p99_ttft_ms=p99_ttft_ms,
        mean_tpot_ms=mean_tpot_ms,
        median_tpot_ms=median_tpot_ms,
        p99_tpot_ms=p99_tpot_ms,
        server_metrics_summary=server_metrics_summary,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        similarity_metrics=similarity_metrics  # Content Metric
    )

    return metrics, actual_output_lens


def aggregate_server_metrics(server_metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """优化的服务器指标聚合 - 只处理关键指标"""
    if not server_metrics_data:
        return {}
    
    # 定义需要处理的指标
    target_metrics = {
        "process_resident_memory_bytes",
        "process_cpu_seconds_total", 
        "vllm:kv_cache_usage_perc",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:time_to_first_token_seconds",
        "vllm:time_per_output_token_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:request_queue_time_seconds",
        "vllm:request_inference_time_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_decode_time_seconds",
        # Speculative Decoding 指标
        "vllm:spec_decode_draft_acceptance_rate",
        "vllm:spec_decode_efficiency", 
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_emitted_tokens_total"
    }
    
    # 初始化数据收集器
    metrics_data = {}
    for metric in target_metrics:
        metrics_data[metric] = {
            'after_values': [],      
            'changes': [],           
            'before_values': []      
        }
    
    # 收集数据
    for request_metrics in server_metrics_data:
        # 收集请求后状态
        if 'after_request' in request_metrics:
            after_data = request_metrics['after_request']
            for metric in target_metrics:
                if metric in after_data:
                    metrics_data[metric]['after_values'].append(after_data[metric])
        
        # 收集变化值
        if 'metrics_delta' in request_metrics:
            delta = request_metrics['metrics_delta']
            for metric in target_metrics:
                if metric in delta:
                    metrics_data[metric]['changes'].append(delta[metric])
        
        # 收集请求前状态（参考用）
        if 'before_request' in request_metrics:
            before_data = request_metrics['before_request']
            for metric in target_metrics:
                if metric in before_data:
                    metrics_data[metric]['before_values'].append(before_data[metric])
    
    # 生成统计摘要
    summary = {
        'total_requests_with_metrics': len(server_metrics_data),
        'metrics_collection_rate': 1.0 if server_metrics_data else 0.0,
    }
    
    for metric in target_metrics:
        data = metrics_data[metric]
        
        # 请求完成后的系统状态统计
        if data['after_values']:
            summary[f"{metric}_final_state"] = {
                'avg': np.mean(data['after_values']),
                'max': np.max(data['after_values']),
                'min': np.min(data['after_values']),
                'std': np.std(data['after_values']) if len(data['after_values']) > 1 else 0.0,
                'latest': data['after_values'][-1]
            }
        
        # 请求造成的影响统计  
        if data['changes']:
            summary[f"{metric}_impact"] = {
                'avg_change': np.mean(data['changes']),
                'max_change': np.max(data['changes']),
                'min_change': np.min(data['changes']),
                'total_change': np.sum(data['changes']),
                'std_change': np.std(data['changes']) if len(data['changes']) > 1 else 0.0
            }
    
    # 特殊计算：缓存命中率
    if (metrics_data['vllm:prefix_cache_hits_total']['changes'] and 
        metrics_data['vllm:prefix_cache_queries_total']['changes']):
        
        total_hits = np.sum(metrics_data['vllm:prefix_cache_hits_total']['changes'])
        total_queries = np.sum(metrics_data['vllm:prefix_cache_queries_total']['changes'])
        hit_rate = total_hits / total_queries if total_queries > 0 else 0.0
        
        summary['cache_performance'] = {
            'hit_rate': hit_rate,
            'total_hits': total_hits,
            'total_queries': total_queries
        }
    
    # 新增：Speculative Decoding 性能汇总
    spec_decode_metrics = [
        'vllm:spec_decode_num_accepted_tokens_total',
        'vllm:spec_decode_num_draft_tokens_total', 
        'vllm:spec_decode_num_emitted_tokens_total'
    ]
    
    # 计算推测解码效率
    if all(metrics_data[metric]['changes'] for metric in spec_decode_metrics):
        total_accepted = np.sum(metrics_data['vllm:spec_decode_num_accepted_tokens_total']['changes'])
        total_draft = np.sum(metrics_data['vllm:spec_decode_num_draft_tokens_total']['changes'])
        total_emitted = np.sum(metrics_data['vllm:spec_decode_num_emitted_tokens_total']['changes'])
        
        # 计算接受率和效率
        acceptance_rate = total_accepted / total_draft if total_draft > 0 else 0.0
        efficiency = total_emitted / total_draft if total_draft > 0 else 0.0
        
        summary['spec_decode_performance'] = {
            'total_accepted_tokens': total_accepted,
            'total_draft_tokens': total_draft,
            'total_emitted_tokens': total_emitted,
            'calculated_acceptance_rate': acceptance_rate,
            'calculated_efficiency': efficiency,
            'samples_processed': len(metrics_data['vllm:spec_decode_num_draft_tokens_total']['changes'])
        }
        
        # 添加推测解码效率趋势分析
        if metrics_data['vllm:spec_decode_draft_acceptance_rate']['after_values']:
            acceptance_rates = metrics_data['vllm:spec_decode_draft_acceptance_rate']['after_values']
            summary['spec_decode_performance']['acceptance_rate_trend'] = {
                'avg': np.mean(acceptance_rates),
                'min': np.min(acceptance_rates),
                'max': np.max(acceptance_rates),
                'std': np.std(acceptance_rates) if len(acceptance_rates) > 1 else 0.0,
                'latest': acceptance_rates[-1]
            }
            
        if metrics_data['vllm:spec_decode_efficiency']['after_values']:
            efficiency_values = metrics_data['vllm:spec_decode_efficiency']['after_values']
            summary['spec_decode_performance']['efficiency_trend'] = {
                'avg': np.mean(efficiency_values),
                'min': np.min(efficiency_values),
                'max': np.max(efficiency_values), 
                'std': np.std(efficiency_values) if len(efficiency_values) > 1 else 0.0,
                'latest': efficiency_values[-1]
            }
    
    # 延迟性能汇总
    latency_metrics = [
        'vllm:time_to_first_token_seconds',
        'vllm:time_per_output_token_seconds',
        'vllm:e2e_request_latency_seconds',
        'vllm:request_queue_time_seconds',
        'vllm:request_inference_time_seconds',
        'vllm:request_prefill_time_seconds',
        'vllm:request_decode_time_seconds'
    ]
    
    latency_summary = {}
    for lat_metric in latency_metrics:
        if metrics_data[lat_metric]['changes']:
            values_ms = np.array(metrics_data[lat_metric]['changes']) * 1000
            metric_name = lat_metric.replace('vllm:', '').replace('_seconds', '')
            latency_summary[metric_name] = {
                'avg_ms': np.mean(values_ms),
                'p50_ms': np.percentile(values_ms, 50),
                'p95_ms': np.percentile(values_ms, 95),
                'p99_ms': np.percentile(values_ms, 99)
            }
    
    if latency_summary:
        summary['latency_performance'] = latency_summary
    
    return summary


async def benchmark(
        backend: str,
        api_url: str,
        model_id: str,
        tokenizer: PreTrainedTokenizerBase,
        input_requests: List[Tuple[str, int, int]],
        best_of: int,
        request_rate: float,
        disable_tqdm: bool,
        collect_server_metrics: bool = True,
        burstiness: float = 1.0,
        extra_body: Optional[dict] = None,
        max_concurrency: Optional[int] = None,
        groundtruth_dict: Optional[Dict[str, str]] = None
):
    """增强的benchmark函数, 支持指标收集、并发控制和突发性控制"""
    if backend not in ASYNC_REQUEST_FUNCS:
         raise ValueError(f"Unknown backend: {backend}")
    request_func = ASYNC_REQUEST_FUNCS[backend]

    # 确定请求分布类型
    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Using backend: {backend}")
    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")  # 新增日志
    print(f"Number of prompts: {len(input_requests)}")
    print(f"Max concurrency: {max_concurrency if max_concurrency else 'unlimited'}")
    print(f"Server metrics collection: {'enabled' if collect_server_metrics else 'disabled'}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # 并发控制
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        """限制并发的请求函数"""
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, 
                                    pbar=pbar, 
                                    collect_metrics=collect_server_metrics)
        async with semaphore:
            return await request_func(request_func_input=request_func_input,
                                    pbar=pbar,
                                    collect_metrics=collect_server_metrics)

    benchmark_start_time = time.perf_counter()
    tasks = []

    # 更新调用，传递 burstiness 参数
    async for request in get_request(input_requests, request_rate, burstiness):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            extra_body=extra_body
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    # Ensure input_requests length matches outputs length for calculate_metrics
    if len(input_requests) != len(outputs):
         warnings.warn(
             f"Mismatch in input requests ({len(input_requests)}) and outputs ({len(outputs)}). "
             "Metrics might be inaccurate.", stacklevel=2)
         min_len = min(len(input_requests), len(outputs))
         input_requests = input_requests[:min_len]
         outputs = outputs[:min_len]

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        groundtruth_dict=groundtruth_dict  # 传递groundtruth
    )

    print(f"Successful requests: {metrics.successful_requests}")
    print(f"Benchmark duration: {benchmark_duration:.2f} s")
    print(f"Total input tokens: {metrics.total_input}")
    print(f"Total generated tokens: {metrics.total_output}")
    print(f"Request throughput: {metrics.request_throughput:.2f} requests/s")
    print(f"Input token throughput: {metrics.input_throughput:.2f} tokens/s")
    print(f"Output token throughput: {metrics.output_throughput:.2f} tokens/s")
    print(f"Mean TTFT: {metrics.mean_ttft_ms:.2f} ms")
    print(f"Median TTFT: {metrics.median_ttft_ms:.2f} ms")
    print(f"P99 TTFT: {metrics.p99_ttft_ms:.2f} ms")
    print(f"Mean TPOT: {metrics.mean_tpot_ms:.2f} ms")
    print(f"Median TPOT: {metrics.median_tpot_ms:.2f} ms")
    print(f"P99 TPOT: {metrics.p99_tpot_ms:.2f} ms")

    # Print server metrics summary
    if metrics.server_metrics_summary:
        print("\n--- Server Metrics Summary ---")
        for key, value in metrics.server_metrics_summary.items():
            print(f"{key}: {value}")

    return metrics, actual_output_lens, outputs


def save_results(results: Dict[str, Any], filename: str):
    """保存结果到JSON文件"""
    # 确保目录存在
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # 序列化并保存
    def make_serializable(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)


async def detect_server_type(base_url: str) -> Optional[str]:
    """检测服务器类型，返回合适的后端名称"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            # 检查OpenAI API端点
            try:
                async with session.get(f"{base_url}/v1/models", timeout=3.0) as response:
                    if response.status == 200:
                        print(f"✅ Detected OpenAI-compatible API at /v1/models")
                        return "openai"
            except:
                pass
            
            # 检查vLLM原生API端点
            try:
                async with session.get(f"{base_url}/health", timeout=3.0) as response:
                    if response.status == 200:
                        # 再检查是否有/generate端点
                        try:
                            async with session.post(f"{base_url}/generate", 
                                                  json={"prompt": "test", "max_tokens": 1}, 
                                                  timeout=3.0) as gen_response:
                                if gen_response.status != 404:
                                    print(f"✅ Detected vLLM native API at /generate")
                                    return "vllm"
                        except:
                            pass
            except:
                pass
                
            # 如果/health存在但/generate不存在，很可能是OpenAI API
            try:
                async with session.get(f"{base_url}/health", timeout=3.0) as response:
                    if response.status == 200:
                        print(f"✅ Found /health endpoint, assuming OpenAI-compatible API")
                        return "openai"
            except:
                pass
                
    except Exception as e:
        print(f"⚠️  Server detection failed: {e}")
    
    return None

async def run_benchmark(args):
    """运行benchmark的主函数"""
    print("Starting vLLM Sampler Benchmark...")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    
    # 构造API URL
    if hasattr(args, 'port') and args.port:
        api_url = f"http://{args.host}:{args.port}"
    else:
        api_url = args.host if args.host.startswith('http') else f"http://{args.host}"
    
    print(f"API URL: {api_url}")
    
    # 自动检测模型名称（对OpenAI后端）
    if args.backend == "openai":
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{api_url}/v1/models") as response:
                    if response.status == 200:
                        models_data = await response.json()
                        available_models = [m.get('id', 'unknown') for m in models_data.get('data', [])]
                        if available_models:
                            detected_model = available_models[0]
                            print(f"Auto-detected model: {detected_model}")
                            args.model = detected_model
        except Exception as e:
            print(f"Failed to auto-detect model: {e}")
    
    # 设置随机种子
    if hasattr(args, 'seed') and args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Initialize tokenizer
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code)

    # 选择数据集
    dataset_name = getattr(args, 'dataset_name', args.dataset)

    # 处理input_length_range参数
    input_length_range = None
    groundtruth_dict = None
    if hasattr(args, 'input_length_range') and args.input_length_range:
        try:
            input_length_range = tuple(map(int, args.input_length_range.split(':')))
            print(f"Using input length range for prefix caching: {input_length_range}")
        except ValueError:
            raise ValueError("input_length_range must be in format 'min:max' (e.g., '128:256')")

    if dataset_name is None:
        # Use default requests
        input_requests = sample_default_requests(
            args.num_prompts, args.input_len, args.output_len, tokenizer)
    elif dataset_name == "sharegpt":
        # 统一的ShareGPT采样，支持prefix caching测试
        input_requests, groundtruth_dict = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts, 
            tokenizer=tokenizer, 
            fixed_output_len=args.output_len,
            input_length_range=input_length_range,
            return_groundtruth=True  # 同时获取ground truth  
        )
        
        if input_length_range:
            print(f"📏 Input length range: {input_length_range[0]}-{input_length_range[1]} tokens")

    elif dataset_name == "sonnet":
        input_requests = sample_sonnet_requests(
            args.dataset_path, args.num_prompts, args.input_len, 
            args.output_len, args.prefix_len, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 添加repeat_count作为fidelity factor的支持
    if hasattr(args, 'repeat_count') and args.repeat_count > 1:
        print(f"🔄 Applying repeat count: {args.repeat_count}")
        input_requests = repeat_and_sort_requests(
            input_requests, 
            repeat_count=args.repeat_count,
            sort=getattr(args, 'sort_requests', False)
        )

    # 准备额外参数
    extra_body = {}
    if hasattr(args, 'extra_body') and args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError:
            print("Warning: Invalid extra_body JSON, ignoring...")

    benchmark_start_time = time.time()
    
    # 根据后端选择API URL格式
    if args.backend == "openai":
        if not api_url.endswith('/v1/completions'):
            api_url = f"{api_url}/v1/completions"
    else:  # vllm backend
        if not api_url.endswith('/generate'):
            api_url = f"{api_url}/generate"
    
    metrics, actual_output_lens, outputs = await benchmark(
        backend=args.backend,
        api_url=api_url,
        model_id=args.model,
        tokenizer=tokenizer,
        input_requests=input_requests,
        best_of=args.best_of,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
        collect_server_metrics=not getattr(args, 'disable_metrics_collection', False),
        burstiness=getattr(args, 'burstiness', 1.0), 
        extra_body=extra_body,
        max_concurrency=getattr(args, 'max_concurrency', None),
        groundtruth_dict=groundtruth_dict  # 添加这行
    )
    benchmark_end_time = time.time()

    # 打印相似度指标
    if metrics.similarity_metrics:
        print("\n--- 文本相似度指标 ---")
        print(f"Samples compared num: {metrics.similarity_metrics.get('samples_compared')}")
        print(f"Mean similarity: {metrics.similarity_metrics.get('mean_similarity'):.4f}")
        print(f"Median similarity: {metrics.similarity_metrics.get('median_similarity'):.4f}")
        print(f"P90 similarity: {metrics.similarity_metrics.get('p90_similarity'):.4f}")
        print(f"Metric type: {metrics.similarity_metrics.get('metric_type')}")
    

    # 准备结果数据 - 兼容原有格式
    results = {
        "timestamp": datetime.now().isoformat(),
        # "args": vars(args),  # 包含所有命令行参数
        "benchmark_duration_s": benchmark_end_time - benchmark_start_time,
        "metrics": asdict(metrics),
        "actual_output_lens": actual_output_lens,
        "input_requests_count": len(input_requests),
        "version_info": {
            "script_version": "2.0",
            "backend_type": args.backend,
            "metrics_collection_enabled": not getattr(args, 'disable_metrics_collection', False)
        },
        
        # VLLM config info
        "vllm_config": getattr(args, 'vllm_config_dict', {}),
        
        # Workload config info
        "workload_config": {
            "num_prompts": getattr(args, 'num_prompts', 100),
            "request_rate": getattr(args, 'request_rate', float('inf')),
            "burstiness": getattr(args, 'burstiness', 1.0),
            "max_concurrency": getattr(args, 'max_concurrency', None),
            "repeat_count": getattr(args, 'repeat_count', 1),
            "dataset_name": getattr(args, 'dataset_name', None),
            "input_length_range": getattr(args, 'input_length_range', None),
        }
    }

    # 保存详细的输出数据（可选）
    if getattr(args, 'save_detailed_outputs', False):
        detailed_outputs = []
        for i, output in enumerate(outputs):
            output_data = {
                "index": i,
                "success": output.success,
                "generated_text": output.generated_text if len(output.generated_text) < 1000 else output.generated_text[:1000] + "...",
                "latency": output.latency,
                "ttft": output.ttft,
                "output_tokens": output.output_tokens,
                "tpot": output.tpot,
                "error": output.error,
                "server_metrics_available": bool(output.server_metrics)
            }
            detailed_outputs.append(output_data)
        results["detailed_outputs"] = detailed_outputs

    if hasattr(args, 'output_file') and args.output_file:
        save_results(results, args.output_file)
        print(f"\n✅ Benchmark completed successfully!")
        print(f"📊 Results saved to: {args.output_file}")
    else:
        print(f"\n✅ Benchmark completed successfully!")

    print(f"🎯 Success rate: {metrics.successful_requests}/{len(input_requests)} ({metrics.successful_requests/len(input_requests)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM serving throughput.")
    
    # 基本参数
    parser.add_argument("--backend", type=str, default="openai", 
                       choices=list(ASYNC_REQUEST_FUNCS.keys()),
                       help="Backend to use for requests")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, help="Server port")

    parser.add_argument("--vllm-config", type=str, default="{}",
                    help="Complete vLLM configuration as JSON string")

    # 数据集参数
    parser.add_argument("--dataset", type=str, default=None,
                       choices=["sharegpt", "sonnet"], 
                       help="Dataset type. If None, uses default requests")
    parser.add_argument("--dataset-name", type=str, default=None,
                       choices=["sharegpt", "sonnet"], 
                       help="Dataset name (alternative to --dataset)")
    parser.add_argument("--dataset-path", type=str, 
                       help="Path to the dataset file")
    
    parser.add_argument("--input-length-range", type=str, default=None,
                       help="Input length range for prefix caching test, specified as 'min:max' (e.g., '128:256')")
    parser.add_argument("--output-len", type=int, default=None,
                       help="Output length")
    
    # 模型参数
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, 
                       help="Tokenizer name or path. If not specified, uses the model name")
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code for tokenizer")
    parser.add_argument("--best-of", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
    parser.add_argument("--extra-body", type=str,
                    help="Extra request body as JSON string")
   

    # fidelity factors
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                       help="Request rate (requests per second). Default is infinite")
    parser.add_argument("--burstiness", type=float, default=1.0,
        help="Burstiness factor for request generation. "
             "Default is 1.0 (Poisson process). "
             "Values < 1.0 create more bursty traffic, "
             "values > 1.0 create more uniform traffic."
    )
    parser.add_argument("--max-concurrency", type=int, default=None,
                       help="Maximum number of concurrent requests")
    parser.add_argument("--repeat-count", type=int, default=1,
                       help="Number of times to repeat each request for prefix caching test")
    
    
    # 行为控制参数
    parser.add_argument("--ignore-eos", action="store_true",
                       help="Ignore EOS token")
    parser.add_argument("--disable-tqdm", action="store_true",
                       help="Disable progress bar")
    parser.add_argument("--disable-metrics-collection", action="store_true",
                       help="Disable server metrics collection")
    

    parser.add_argument("--output-file", type=str,
                       help="Output file name. If not specified, auto-generated")

    


    args = parser.parse_args()
    
    try:
        args.vllm_config_dict = json.loads(args.vllm_config) if args.vllm_config else {}
    except json.JSONDecodeError:
        print("Warning: Invalid vllm-config JSON, using empty config")
        args.vllm_config_dict = {}
    
    # 转换字符串布尔值
    bool_attrs = ['enable_chunked_prefill', 'enable_prefix_caching', 'disable_custom_all_reduce', 'use_v2_block_manager']
    for attr in bool_attrs:
        if hasattr(args, attr):
            val = getattr(args, attr)
            if isinstance(val, str):
                setattr(args, attr, val.lower() in ['true', '1', 'yes', 'on'])
    
    # 验证参数
    dataset_name = getattr(args, 'dataset_name', args.dataset)
    if dataset_name and not args.dataset_path:
        parser.error(f"--dataset-path is required when using dataset {dataset_name}")
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        asyncio.run(run_benchmark(args))
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()