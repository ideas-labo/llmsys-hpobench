import io
import json
import os
import sys
import time
import traceback
import asyncio
import random
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, List

import aiohttp
import psutil
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    best_of: int = 1
    logprobs: Optional[int] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False
    extra_body: Optional[dict] = None
    language: Optional[str] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    gpu_hit_rate: List[float] = field(
        default_factory=list)  
    error: str = ""
    # 服务器指标
    server_metrics: Optional[Dict[str, Any]] = None


class MetricsCollector:
    """用于收集服务器指标的工具类 - 优化版，只收集关键指标"""
    
    def __init__(self, base_url: str, collect_metrics: bool = True):
        self.base_url = base_url.rstrip('/')
        self.collect_metrics = collect_metrics
        # 获取当前进程信息用于CPU时间计算
        self.process = psutil.Process(os.getpid())
        
        # 定义需要收集的关键指标
        self.target_metrics = {
            # 进程级基础指标
            "process_resident_memory_bytes",
            "process_cpu_seconds_total",
            
            # 缓存/显存健康度
            "vllm:kv_cache_usage_perc",
            "vllm:prefix_cache_queries_total", 
            "vllm:prefix_cache_hits_total",
            
            # 延迟分布
            "vllm:time_to_first_token_seconds",
            "vllm:time_per_output_token_seconds",
            "vllm:e2e_request_latency_seconds",
            
            # 各阶段耗时拆解
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
        
    async def get_metrics(self, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
        """获取服务器指标 - 只收集关键指标"""
        if not self.collect_metrics:
            return None
            
        metrics = {}
        
        # 1. 尝试从服务器端点获取指标
        server_metrics = await self._get_server_metrics(session)
        if server_metrics:
            # 只保留目标指标
            filtered_metrics = {k: v for k, v in server_metrics.items() 
                              if k in self.target_metrics}
            metrics.update(filtered_metrics)
        
        # 2. 手动收集进程级指标（如果服务器端点没有提供）
        process_metrics = self._get_process_metrics()
        if process_metrics:
            # 只添加缺失的进程指标
            for key, value in process_metrics.items():
                if key in self.target_metrics and key not in metrics:
                    metrics[key] = value
            
        return metrics if metrics else None
    
    async def _get_server_metrics(self, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
        """从服务器端点获取指标 - 优化解析"""
        try:
            # 尝试多个可能的metrics端点
            endpoints = [
                f"{self.base_url}/metrics",
                f"{self.base_url.replace('generate', 'metrics')}"
            ]
            
            for endpoint in endpoints:
                try:
                    async with session.get(endpoint, timeout=2.0) as response:
                        if response.status == 200:
                            content_type = response.headers.get('content-type', '')
                            
                            if 'application/json' in content_type:
                                return await response.json()
                            else:
                                # Prometheus格式 - 优化解析
                                text = await response.text()
                                return self._parse_prometheus_metrics_optimized(text)
                except:
                    continue
                    
        except Exception:
            pass
        
        return None
    
    def _get_process_metrics(self) -> Dict[str, Any]:
        """手动收集进程级指标 - 只收集需要的"""
        try:
            # 获取进程CPU时间
            cpu_times = self.process.cpu_times()
            
            # 计算总CPU时间（用户时间 + 系统时间）
            total_cpu_seconds = cpu_times.user + cpu_times.system
            
            process_metrics = {
                'process_cpu_seconds_total': total_cpu_seconds,
            }
            
            # 如果可用，添加内存信息
            try:
                memory_info = self.process.memory_info()
                process_metrics['process_resident_memory_bytes'] = memory_info.rss
            except:
                pass
                
            return process_metrics
            
        except Exception:
            return {}
    
    def _parse_prometheus_metrics_optimized(self, text: str) -> Dict[str, Any]:
        """优化的Prometheus指标解析 - 只解析目标指标"""
        metrics = {}
        
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                if ' ' in line:
                    parts = line.split(' ', 1)
                    metric_name = parts[0].split('{')[0]
                    
                    # 只解析我们需要的指标
                    if metric_name in self.target_metrics:
                        value = float(parts[1].split()[0])
                        metrics[metric_name] = value
            except:
                continue
        
        # 如果服务器指标中没有进程指标，添加手动收集的
        process_metrics_needed = {
            'process_cpu_seconds_total', 
            'process_resident_memory_bytes'
        }
        
        missing_metrics = process_metrics_needed - set(metrics.keys())
        if missing_metrics:
            process_metrics = self._get_process_metrics()
            for key in missing_metrics:
                if key in process_metrics:
                    metrics[key] = process_metrics[key]
            
        return metrics


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


async def fetch_stats(session, url):
    """获取 GPU 指标"""
    try:
        metrics_url = url.replace('generate', 'metrics')
        async with session.get(metrics_url, timeout=2.0) as response:
            if response.status == 200:
                return await response.text()
    except Exception:
        pass
    
    return json.dumps({
        "pending_queue_length": 0,
        "gpu_cache_usage": 0,
        "gpu_hit_rate": 0
    })


async def async_request_vllm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
    ignore_eos: bool = True,
    collect_metrics: bool = True,
    **kwargs
) -> RequestFuncOutput:
    """
    vLLM API 请求函数，支持指标收集
    """
    api_url_list = request_func_input.api_url.split(',')
    
    # 初始化指标收集器
    base_url = api_url_list[0].rsplit('/', 1)[0] if api_url_list else ""
    metrics_collector = MetricsCollector(base_url, collect_metrics)
    
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "prompt": request_func_input.prompt,
            "n": 1,
            "best_of": request_func_input.best_of,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "ignore_eos": ignore_eos,
            "stream": True,
        }

        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.server_metrics = {}

        # 收集请求前指标
        if collect_metrics:
            before_metrics = await metrics_collector.get_metrics(session)
            if before_metrics:
                output.server_metrics['before_request'] = before_metrics

        ttft = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        
        try:
            # 服务器选择逻辑
            if len(api_url_list) == 1:
                api_url = api_url_list[0]
            else:
                gpu_usages_waiting_len = await asyncio.gather(
                    *[fetch_stats(session, url) for url in api_url_list]
                )
                min_pending_queue, min_gpu_usage, min_index = sorted([
                    (json.loads(metric)["pending_queue_length"], 
                     json.loads(metric)["gpu_cache_usage"], idx) 
                    for idx, metric in enumerate(gpu_usages_waiting_len)
                ], key=lambda x: (x[0], x[1]))[0]
                api_url = api_url_list[min_index]
                
            assert api_url.endswith("generate")
            
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for data in response.content.iter_any():
                        timestamp = time.perf_counter()
                        
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp
                                
                    output.latency = time.perf_counter() - st

                    # 解析响应
                    body = data.decode("utf-8").strip("\0")
                    try:
                        output.generated_text = json.loads(
                            body)["text"][0][len(request_func_input.prompt):]
                        output.success = True
                        output.output_tokens = len(output.generated_text.split())
                    except Exception as e:
                        output.success = False
                        output.error = f"Response parsing error: {str(e)}"
                else:
                    output.success = False
                    output.error = f"HTTP {response.status}: {response.reason}"
                    
        except Exception as e:
            output.success = False
            output.error = f"Request error: {str(e)}"

        # 收集GPU指标
        try:
            gpu_usages_waiting_len = await asyncio.gather(
                *[fetch_stats(session, url) for url in api_url_list]
            )
            output.gpu_hit_rate = [
                json.loads(metric)['gpu_hit_rate'] 
                for metric in gpu_usages_waiting_len
            ]
        except Exception:
            output.gpu_hit_rate = [0.0] * len(api_url_list)

        # 收集请求后指标
        if collect_metrics:
            after_metrics = await metrics_collector.get_metrics(session)
            if after_metrics:
                output.server_metrics['after_request'] = after_metrics

        # 计算指标差异
        if ('before_request' in output.server_metrics and 
            'after_request' in output.server_metrics):
            output.server_metrics['metrics_delta'] = _calculate_metrics_delta(
                output.server_metrics['before_request'],
                output.server_metrics['after_request']
            )

        # 计算平均token间延迟
        if output.itl:
            output.tpot = sum(output.itl) / len(output.itl)

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
    collect_metrics: bool = True,
    **kwargs
) -> RequestFuncOutput:
    """
    OpenAI completions API 请求函数，支持指标收集
    """
    api_url = request_func_input.api_url
    
    # 确保API URL正确
    if not api_url.endswith(("completions", "profile")):
        base_url = api_url.rstrip('/')
        if base_url.endswith('/generate'):
            base_url = base_url.replace('/generate', '')
        if not base_url.endswith('/v1'):
            base_url += '/v1'
        api_url = f"{base_url}/completions"
    
    # 初始化指标收集器
    base_url = api_url.rsplit('/', 2)[0]
    metrics_collector = MetricsCollector(base_url, collect_metrics)
    
    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model_name if request_func_input.model_name else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.7,
            "repetition_penalty": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        
        if request_func_input.logprobs:
            payload["logprobs"] = request_func_input.logprobs
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
            
        headers = {"Content-Type": "application/json"}
        if os.environ.get('OPENAI_API_KEY'):
            headers["Authorization"] = f"Bearer {os.environ.get('OPENAI_API_KEY')}"

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.server_metrics = {}

        # 收集请求前指标
        if collect_metrics:
            before_metrics = await metrics_collector.get_metrics(session)
            if before_metrics:
                output.server_metrics['before_request'] = before_metrics

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        first_chunk_received = False
        
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        try:
                            chunk_str = chunk_bytes.decode("utf-8")
                            
                            for line in chunk_str.split('\n'):
                                line = line.strip()
                                if not line or not line.startswith("data: "):
                                    continue
                                    
                                line = line[6:]  # 移除 "data: "
                                if line == "[DONE]":
                                    break
                                
                                try:
                                    data = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                
                                timestamp = time.perf_counter()
                                
                                if choices := data.get("choices"):
                                    text = choices[0].get("text", "")
                                    
                                    if not first_chunk_received and text:
                                        first_chunk_received = True
                                        output.ttft = timestamp - st
                                    
                                    if first_chunk_received and text:
                                        output.itl.append(timestamp - most_recent_timestamp)
                                    
                                    most_recent_timestamp = timestamp
                                    generated_text += text
                                
                                elif usage := data.get("usage"):
                                    output.output_tokens = usage.get("completion_tokens", 0)
                        
                        except Exception:
                            continue
                    
                    output.generated_text = generated_text
                    output.latency = time.perf_counter() - st
                    output.success = first_chunk_received
                    
                    if output.itl:
                        output.tpot = sum(output.itl) / len(output.itl)
                    
                    if not first_chunk_received:
                        output.error = "No valid response chunks received"
                        
                else:
                    output.success = False
                    output.error = f"HTTP {response.status}: {response.reason}"
                        
        except Exception as e:
            output.success = False
            output.error = f"Request error: {str(e)}"

        # 收集请求后指标
        if collect_metrics:
            after_metrics = await metrics_collector.get_metrics(session)
            if after_metrics:
                output.server_metrics['after_request'] = after_metrics

        # 计算指标差异
        if ('before_request' in output.server_metrics and 
            'after_request' in output.server_metrics):
            output.server_metrics['metrics_delta'] = _calculate_metrics_delta(
                output.server_metrics['before_request'],
                output.server_metrics['after_request']
            )

        if pbar:
            pbar.update(1)
        return output


def _calculate_metrics_delta(before_metrics: Dict[str, Any], 
                           after_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """计算指标变化量"""
    delta = {}
    
    # 处理数值型指标
    for key in before_metrics:
        if key in after_metrics:
            try:
                before_val = before_metrics[key]
                after_val = after_metrics[key]
                
                # 处理数值
                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    delta[key] = after_val - before_val
                # 处理列表（取最后一个值）
                elif isinstance(before_val, list) and isinstance(after_val, list):
                    if before_val and after_val:
                        delta[key] = after_val[-1] - before_val[-1]
            except:
                continue
                
    return delta


# 函数映射
ASYNC_REQUEST_FUNCS = {
    # "vllm": async_request_vllm,
    "vllm": async_request_openai_completions,
    "openai": async_request_openai_completions
}

# 简化的测试函数，专门用于调试
async def debug_test():
    """调试测试 - 手动发送请求"""
    print("Debug: Manual request test...")
    
    async with aiohttp.ClientSession() as session:
        # 首先获取正确的模型名称
        try:
            async with session.get("http://localhost:8000/v1/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    available_models = [m.get('id', 'unknown') for m in models_data.get('data', [])]
                    print(f"Available models: {available_models}")
                    model_name = available_models[0] if available_models else "unknown"
                else:
                    model_name = "./models/Llama-2-7b-hf/"
        except:
            model_name = "./models/Llama-2-7b-hf/"
        
        print(f"Using model: {model_name}")
        
        # 测试基本的POST请求到OpenAI端点
        payload = {
            "model": model_name,
            "prompt": "Hello, how are you?",
            "max_tokens": 20,
            "temperature": 0.0,
            "stream": False  # 先测试非流式
        }
        
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            async with session.post(
                "http://localhost:8000/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"Response status: {response.status}")
                print(f"Response headers: {dict(response.headers)}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"Success! Response: {json.dumps(result, indent=2)}")
                else:
                    error_text = await response.text()
                    print(f"Error response: {error_text}")
                    
        except Exception as e:
            print(f"Request failed: {e}")
            
        # 测试流式请求
        print("\n" + "-"*30)
        print("Testing streaming request...")
        
        payload_stream = payload.copy()
        payload_stream["stream"] = True
        
        try:
            async with session.post(
                "http://localhost:8000/v1/completions",
                json=payload_stream,
                headers={"Content-Type": "application/json"}
            ) as response:
                print(f"Streaming response status: {response.status}")
                
                if response.status == 200:
                    print("Streaming response chunks:")
                    chunk_count = 0
                    async for chunk in response.content:
                        chunk_count += 1
                        if chunk_count <= 5:  # 只显示前5个chunk
                            print(f"  Chunk {chunk_count}: {chunk}")
                        if chunk_count >= 10:  # 限制显示数量
                            print(f"  ... (stopped after {chunk_count} chunks)")
                            break
                else:
                    error_text = await response.text()
                    print(f"Streaming error: {error_text}")
                    
        except Exception as e:
            print(f"Streaming request failed: {e}")


# 使用示例
async def example_usage():
    """使用示例 - 使用正确的模型名称"""
    
    # 首先获取可用模型
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:8000/v1/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    available_models = [m.get('id', 'unknown') for m in models_data.get('data', [])]
                    model_name = available_models[0] if available_models else "./models/Llama-2-7b-hf/"
                else:
                    model_name = "./models/Llama-2-7b-hf/"
        except:
            model_name = "./models/Llama-2-7b-hf/"
    
    print(f"Using model: {model_name}")
    print("="*50)
    
    # 测试OpenAI API - 使用正确的模型名称
    print("Testing OpenAI API...")
    openai_request = RequestFuncInput(
        prompt="Hello, how are you?",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=20,
        output_len=50,
        model=model_name  # 使用从API获取的正确模型名称
    )
    
    openai_output = await async_request_openai_completions(openai_request, collect_metrics=True)
    print(f"OpenAI - Success: {openai_output.success}")
    if openai_output.success:
        print(f"OpenAI - Generated text: {repr(openai_output.generated_text)}")
        print(f"OpenAI - TTFT: {openai_output.ttft:.3f}s")
        print(f"OpenAI - Latency: {openai_output.latency:.3f}s")
        print(f"OpenAI - Output tokens: {openai_output.output_tokens}")
        print(f"OpenAI - Token intervals: {len(openai_output.itl)} intervals")
        if openai_output.tpot > 0:
            print(f"OpenAI - Avg token latency: {openai_output.tpot:.3f}s")
    else:
        print(f"OpenAI - Error: {openai_output.error}")
    
    if openai_output.server_metrics:
        print("OpenAI - Server metrics collected:")
        for key, value in openai_output.server_metrics.items():
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} metrics")
                # 显示一些关键指标
                if key == 'metrics_delta' and value:
                    key_metrics = ['vllm:num_requests_running', 'vllm:gpu_cache_usage_perc', 
                                 'pending_queue_length', 'gpu_cache_usage']
                    for metric in key_metrics:
                        if metric in value:
                            print(f"    {metric}: {value[metric]:+.2f}")
            else:
                print(f"  {key}: available")


if __name__ == "__main__":
    import asyncio
    # 运行调试测试
    asyncio.run(debug_test())
    print("\n" + "="*50 + "\n")
    # 运行完整测试
    asyncio.run(example_usage())