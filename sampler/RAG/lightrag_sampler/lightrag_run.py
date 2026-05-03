import asyncio
import os
import inspect
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
import pandas as pd
import json
from dotenv import load_dotenv
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
from lightrag.utils import TokenTracker
import itertools
import random
import tempfile
from collections import Counter
import re
import string
import sys
import ollama

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from typing import Union
from lightrag.api import __api_version__

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./dickens"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_ollama_demo.log"))

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

token_tracker = TokenTracker()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def _ollama_model_if_cache(
        model,
        prompt,
        system_prompt=None,
        history_messages=[],
        **kwargs,
) -> Union[str, AsyncIterator[str]]:
    stream = True if kwargs.get("stream") else False

    kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None) or 300  # Default timeout 300s
    kwargs.pop("hashing_kv", None)
    api_key = kwargs.pop("api_key", None)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response = await ollama_client.chat(model=model, messages=messages, **kwargs)
        if stream:
            """cannot cache stream response and process reasoning"""

            async def inner():
                try:
                    async for chunk in response:
                        yield chunk["message"]["content"]
                except Exception as e:
                    logger.error(f"Error in stream response: {str(e)}")
                    raise
                finally:
                    try:
                        await ollama_client._client.aclose()
                        logger.debug("Successfully closed Ollama client for streaming")
                    except Exception as close_error:
                        logger.warning(f"Failed to close Ollama client: {close_error}")

            return inner()
        else:
            eval_count = response['eval_count']
            prompt_count = response['prompt_eval_count']
            total_uasge = eval_count + prompt_count
            token_counts = {
                "prompt_tokens": prompt_count,
                "completion_tokens": eval_count,
                "total_tokens": total_uasge,
            }
            token_tracker.add_usage(token_counts)
            model_response = response["message"]["content"]
            # model_response = response
            """
            If the model also wraps its thoughts in a specific tag,
            this information is not needed for the final
            response and can simply be trimmed.
            """

            return model_response
    except Exception as e:
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama client after exception")
        except Exception as close_error:
            logger.warning(
                f"Failed to close Ollama client after exception: {close_error}"
            )
        raise e
    finally:
        if not stream:
            try:
                await ollama_client._client.aclose()
                logger.debug(
                    "Successfully closed Ollama client for non-streaming response"
                )
            except Exception as close_error:
                logger.warning(
                    f"Failed to close Ollama client in finally block: {close_error}"
                )


async def ollama_model_complete(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await _ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "llama3.1:8b"),
        llm_model_max_token_size=25000,
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://192.168.110.47:11434"),
            "options": {"num_ctx": 25000},
            "timeout": int(os.getenv("TIMEOUT", "1200")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text:v1.5"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://192.168.110.47:11434"),
            ),
        ),
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        max_parallel_insert=5,
        llm_model_max_async=5
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def initialize_rag_with_config(rag_init_config):
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        # llm_model_name=os.getenv("LLM_MODEL", rag_init_config["llm_model_name"]),llama3:8b
        llm_model_name=os.getenv("LLM_MODEL", "llama3:8b"),
        llm_model_max_token_size=rag_init_config["llm_model_max_token_size"],
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://192.168.110.3:11434"),
            "options": rag_init_config["llm_model_kwargs"]["options"],
            "timeout": int(os.getenv("TIMEOUT", "2000")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://192.168.110.3:11434"),
            ),
        ),
        llm_model_max_async=rag_init_config["llm_model_max_async"],
        chunk_token_size=rag_init_config["chunk_token_size"],
        chunk_overlap_token_size=rag_init_config["chunk_overlap_token_size"],
        entity_extract_max_gleaning=rag_init_config["entity_extract_max_gleaning"],
        embedding_batch_num=rag_init_config["embedding_batch_num"],
        embedding_func_max_async=rag_init_config["embedding_func_max_async"],
        max_parallel_insert=rag_init_config["max_parallel_insert"],
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        embedding_cache_config={"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


def calculate_lexical_answer_correctness(prediction, reference):
    """
    计算基于词汇召回率的Answer Correctness (Lexical-AC)
    Lexical-AC = 匹配的参考答案词符数 / 参考答案的总词符数
    """
    if not prediction or not reference:
        return 0.0

    # 处理reference可能是列表的情况
    if isinstance(reference, list):
        reference = reference[0] if reference else ""

    import re

    def tokenize_text(text):
        """
        词符化：将文本拆分为词符，包括标点符号
        """
        # 转换为小写以便匹配
        text = text.lower()
        # 使用正则表达式分割，保留标点符号
        # 匹配字母数字序列或单个标点符号
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    # 词符化
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)

    if not ref_tokens:
        return 0.0

    # 计算匹配的参考答案词符数
    # 使用multiset方式计算，考虑词符出现次数
    from collections import Counter
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    matched_tokens = 0
    for token, ref_count in ref_counter.items():
        # 取预测中该词符出现次数和参考中出现次数的最小值
        matched_tokens += min(pred_counter.get(token, 0), ref_count)

    # 召回率计算：匹配的参考答案词符数 / 参考答案的总词符数
    lexical_ac = matched_tokens / len(ref_tokens)

    return lexical_ac


def calculate_answer_precision(prediction, reference):
    """
    计算答案精确率：匹配的词符数 / 预测答案的总词符数
    """
    if not prediction or not reference:
        return 0.0

    if isinstance(reference, list):
        reference = reference[0] if reference else ""

    import re
    from collections import Counter

    def tokenize_text(text):
        text = text.lower()
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)

    if not pred_tokens:
        return 0.0

    # 计算匹配的词符数
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    matched_tokens = 0
    for token, pred_count in pred_counter.items():
        matched_tokens += min(pred_count, ref_counter.get(token, 0))

    precision = matched_tokens / len(pred_tokens)

    return precision


def calculate_answer_f1_score(prediction, reference):
    """
    计算答案F1分数：召回率和精确率的调和平均
    """
    recall = calculate_lexical_answer_correctness(prediction, reference)
    precision = calculate_answer_precision(prediction, reference)

    if recall + precision == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_generation_metrics(predictions, references):
    """
    计算生成质量的核心指标
    """
    if not predictions or not references:
        return {
            'lexical_answer_correctness': 0.0,
            'answer_precision': 0.0,
            'answer_f1_score': 0.0
        }

    lexical_ac_scores = []
    precision_scores = []
    f1_scores = []

    print(f"\n=== 生成质量计算示例 ===")

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # 计算词汇答案正确性
        lexical_ac = calculate_lexical_answer_correctness(pred, ref)
        lexical_ac_scores.append(lexical_ac)
        precision = calculate_answer_precision(pred, ref)
        precision_scores.append(precision)

        # 计算F1分数
        f1 = calculate_answer_f1_score(pred, ref)
        f1_scores.append(f1)

        # # 显示前3个示例的详细计算
        # if i < 3:
        #     print(f"\n示例 {i+1}:")
        #     print(f"参考答案: {ref}")
        #     print(f"预测答案: {pred}")
        #     print(f"Lexical-AC (召回率): {lexical_ac:.4f}")
        #     print(f"精确率: {precision:.4f}")
        #     print(f"F1分数: {f1:.4f}")

    return {
        'lexical_answer_correctness': np.mean(lexical_ac_scores),
        'answer_precision': np.mean(precision_scores),
        'answer_f1_score': np.mean(f1_scores),
        'individual_scores': {
            'lexical_ac': lexical_ac_scores,
            'precision': precision_scores,
            'f1': f1_scores
        }
    }


def calculate_retrieval_metrics(retrieved_docs, true_context, embedding_model, question):
    """
    计算3个核心检索评估指标：MRR, NDCG, Context Similarity
    """
    metrics = {
        'mrr': 0.0,
        'ndcg': 0.0,
        'context_similarity': 0.0,
        'best_match_position': -1,
        'relevant_docs_count': 0
    }

    if not true_context or not retrieved_docs:
        return metrics

    # 计算每个检索文档与真实上下文的相似度
    true_embedding = embedding_model.embed_query(true_context)
    true_embedding = np.array(true_embedding).reshape(1, -1)

    relevance_scores = []

    for i, doc in enumerate(retrieved_docs):
        doc_embedding = embedding_model.embed_query(doc.page_content)
        doc_embedding = np.array(doc_embedding).reshape(1, -1)
        similarity = cosine_similarity(true_embedding, doc_embedding)[0][0]

        # 定义相关性阈值
        is_relevant = similarity >= 0.6
        relevance_scores.append(1 if is_relevant else 0)

    # 1. Mean Reciprocal Rank (MRR)
    for i, is_relevant in enumerate(relevance_scores):
        if is_relevant:
            metrics['mrr'] = 1.0 / (i + 1)
            metrics['best_match_position'] = i + 1
            break

    # 2. Normalized Discounted Cumulative Gain (NDCG)
    def calculate_dcg(relevance_scores):
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                dcg += rel / np.log2(i + 2)
        return dcg

    dcg = calculate_dcg(relevance_scores)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_relevance)

    if idcg > 0:
        metrics['ndcg'] = dcg / idcg

    # 3. Context Similarity (整体上下文相似度)
    retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    retrieved_embedding = embedding_model.embed_query(retrieved_context)
    retrieved_embedding = np.array(retrieved_embedding).reshape(1, -1)
    metrics['context_similarity'] = cosine_similarity(true_embedding, retrieved_embedding)[0][0]

    # 相关文档数量
    metrics['relevant_docs_count'] = sum(relevance_scores)

    return metrics


def create_temporary_dataset(fidelity):
    dataset_category, corpus_scale, domain_mixture = fidelity
    # dataset_category = os.path.join(dataset_category, "unique_contexts")
    base_dir = "data/"
    oringinal_main_domain_dir = os.path.join(base_dir, dataset_category)
    oringinal_secondary_domain_dir = os.path.join(base_dir, "tech" if dataset_category == "agri" else "agri")
    main_domain_dir = os.path.join(oringinal_main_domain_dir, "unique_contexts")
    secondary_domain_dir = os.path.join(oringinal_secondary_domain_dir, "unique_contexts")
    # 验证目录存在
    for path in [main_domain_dir, secondary_domain_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据集目录不存在: {path}")
        if len(os.listdir(path)) < 100:
            raise ValueError(f"目录 {path} 应包含至少100个文件")

    # 创建临时工作目录
    temp_dir = tempfile.mkdtemp(prefix="rag_tmp_")

    # 从主领域采样文件
    main_samples = random.sample(os.listdir(main_domain_dir), corpus_scale)
    # 从次领域采样文件
    secondary_samples = random.sample(os.listdir(secondary_domain_dir), domain_mixture)

    # 创建混合数据集
    mixed_data = []

    # 得到问题和答案
    parts = main_samples[0].split("_")
    oringinal_file_name = '_'.join(parts[:2])
    original_file_path = os.path.join(oringinal_main_domain_dir, oringinal_file_name + ".jsonl")
    with open(original_file_path, 'r') as f:
        try:
            original_data = json.load(f)
            question = original_data.get("question", "No question found")
            answer = original_data.get("answers", "No answer found")[0]
            reference = original_data.get("context", "No reference found")
        except json.JSONDecodeError:
            print(f"警告: 文件 {original_file_path} 包含无效JSON内容，已跳过")
            question = "No question found"
            answer = "No answer found"

    # 处理主领域文件
    for filename in main_samples:
        file_path = os.path.join(main_domain_dir, filename)
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                mixed_data.append(data)
            except json.JSONDecodeError:
                print(f"警告: 文件 {file_path} 包含无效JSON内容，已跳过")

    # 处理次领域文件
    for filename in secondary_samples:
        file_path = os.path.join(secondary_domain_dir, filename)
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                mixed_data.append(data)
            except json.JSONDecodeError:
                print(f"警告: 文件 {file_path} 包含无效JSON内容，已跳过")

    # 创建临时数据文件
    output_filename = f"mixed_{dataset_category}_cs{corpus_scale}_dm{domain_mixture}.txt"
    output_path = os.path.join(temp_dir, output_filename)

    # 保存混合数据
    with open(output_path, 'a', encoding='utf-8') as f:
        for data in mixed_data:
            f.write(data[0])

    return output_path, question, answer, reference


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


async def main():
    config_path = "sampling/LightRAG_LHS_minibatch.csv"
    config_info_path = "sampling/configs/LightRAG.json"
    all_configs = pd.read_csv(config_path).to_dict()
    configs_info = json.load(open(config_info_path))
    config_num = len(all_configs['llm_model_name'])

    data_path = "data/hotpot-data/hotpot-master/fidelity_data_with_lenght_rebuild_more_precise"
    files = os.listdir(data_path)

    for file in files:
        try:
            fidelity_file_path = os.path.join(data_path, file)

            print(f"\n=== 当前Fidelity: {file} ===")
            results_path = "sampling/result/25-7-21-more-precise/" + f"{file}_result.csv"
            if os.path.exists(results_path):
                continue
            with open(results_path, "w", newline="", encoding="utf-8") as csvfile:
                # 创建字段名列表（所有配置参数+评估指标）
                fieldnames = list(all_configs.keys()) + [
                    "total_time", "insert_time", "query_time", "precision", "f1_score", "recall", "insert_input",
                    "insert_output", "insert_total", "insert_calls", "query_input", "query_output", "query_total",
                    "query_calls"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

            # Clear old data files

            for i in range(config_num):
                token_tracker.reset()
                files_to_delete = [
                    "graph_chunk_entity_relation.graphml",
                    "kv_store_doc_status.json",
                    "kv_store_full_docs.json",
                    "kv_store_text_chunks.json",
                    "vdb_chunks.json",
                    "vdb_entities.json",
                    "vdb_relationships.json",
                ]

                for file in files_to_delete:
                    file_path = os.path.join(WORKING_DIR, file)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Deleting old file:: {file_path}")
                query_param = QueryParam()
                init_dict = {}
                llm_dict = {}
                neo4j_dict = {}
                nanodb_dict = {}
                for config_name, config_value_list in all_configs.items():
                    if configs_info[config_name]["scope"] == "query":  # 创建参数类设置查询参数
                        setattr(query_param, config_name, config_value_list[i])

                    elif configs_info[config_name]["scope"] == "init":  # rag初始化参数
                        init_dict[config_name] = config_value_list[i]

                    elif configs_info[config_name]["scope"] == "llm":  # llm参数（rag初始化参数的一部分）
                        real_name = config_name.split(".")[-1]
                        llm_dict[real_name] = config_value_list[i]

                    elif configs_info[config_name]["scope"] == "nanodb":  # vectordb参数
                        nanodb_dict[config_name] = config_value_list[i]

                    elif configs_info[config_name]["scope"] == "neo4j":  # neo4j部分关键参数（可能有权限问题）
                        real_name = config_name.split(".")[-1]
                        neo4j_dict[real_name] = config_value_list[i]
                    else:
                        print("error config name")
                        exit()

                init_dict["llm_model_kwargs"] = {}
                init_dict["llm_model_kwargs"]["options"] = llm_dict
                init_dict["cosine_better_than_threshold"] = nanodb_dict["cosine_threshold"]
                total_time_start = time.time()
                # Initialize RAG instance
                # rag = await initialize_rag_with_config(init_dict)
                rag = await initialize_rag()
                query_param = QueryParam(
                    user_prompt="Only the answer is needed, no overview, and keep it as concise as possible.")  # todo delete this line
                # 处理数据集
                json_data = None
                with open(fidelity_file_path, "r", encoding="utf-8") as f:
                    json_data = json.loads(f.read())[0]  # todo: the first QA-context pair

                all_context = json_data["context"]
                formed_context = []
                supporting_facts_title = json_data['supporting_facts']
                supporting_facts = []
                for ctx in all_context:
                    formed_content = "title:" + ctx[0] + "; setence:" + str(ctx[1])
                    formed_context.append(formed_content)
                    for title_num_pair in supporting_facts_title:
                        title = title_num_pair[0]
                        num = title_num_pair[1]
                        if ctx[0] == title:
                            supporting_facts.append(ctx[1][num])

                question = json_data["question"]
                answer = json_data["answer"]
                print("\n==========insert===========")
                time_start = time.time()
                await rag.ainsert(formed_context)
                time_end = time.time()
                insert_time = time_end - time_start
                insert_token_usage = token_tracker.get_usage()
                insert_input = insert_token_usage['prompt_tokens']
                insert_output = insert_token_usage['completion_tokens']
                insert_total = insert_token_usage['total_tokens']
                insert_calls = insert_token_usage['call_count']
                token_tracker.reset()
                # Perform search with configuration
                print("\n=====================")
                print(f"Query with config {i}")
                print("=====================")
                time_start = time.time()
                print("query start")
                resp, context = await rag.aquery(
                    question,
                    query_param,
                )
                print("query end")
                if inspect.isasyncgen(resp):
                    await print_stream(resp)
                else:
                    print(resp)
                time_end = time.time()
                query_time = time_end - time_start
                total_time = time_end - total_time_start

                """precision =calculate_answer_precision(resp, answer)
                recall = calculate_lexical_answer_correctness(resp, answer)
                f1_score = calculate_answer_f1_score(resp, answer)"""
                f1, precision, recall = f1_score(resp, answer)
                query_token_usage = token_tracker.get_usage()
                query_input = query_token_usage['prompt_tokens']
                query_output = query_token_usage['completion_tokens']
                query_total = query_token_usage['total_tokens']
                query_calls = query_token_usage['call_count']
                # calculate_retrieval_metrics(context, reference, rag.embedding_func, question)
                row = {
                    "total_time": total_time,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "insert_input": insert_input,
                    "insert_output": insert_output,
                    "insert_total": insert_total,
                    "insert_calls": insert_calls,
                    "query_input": query_input,
                    "query_output": query_output,
                    "query_total": query_total,
                    "query_calls": query_calls,
                    "insert_time": insert_time,
                    "query_time": query_time,
                }

                # 添加所有配置参数
                for config_name in all_configs:
                    row[config_name] = all_configs[config_name][i]

                # 追加写入结果到CSV
                with open(results_path, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(row)
                await rag.aclear_cache()
                await rag.llm_response_cache.index_done_callback()
                del rag
                # await rag.
                print(f"Configuration {i} completed and results saved.")

                time.sleep(5)

        except Exception as e:
            new_name = results_path.replace(".csv", "_error.csv")
            if not os.path.exists(new_name):
                os.rename(results_path, new_name)
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")

