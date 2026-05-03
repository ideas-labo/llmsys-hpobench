import asyncio
import os
import inspect
import logging
import logging.config
import time

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status

from dotenv import load_dotenv
import csv
import ast
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


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "llama3:8b"),
        llm_model_max_token_size=8192,
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {"num_ctx": 8192},
            "timeout": int(os.getenv("TIMEOUT", "300")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)



mode = ["naive", "local", "global", "hybrid"]
only_need_context = [True, False]
top_k = [40, 50, 60, 70, 80]
max_token_for_text_unit = [3000, 4000, 5000]
history_turns = [2, 3, 4, 5]


async def main():
    with open('configurations.csv', 'r') as infile, \
            open('results_agriculture.csv', 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['runtime_seconds']  # 添加结果列
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        # 遍历每一行配置
        for i, row in enumerate(reader):
            print(f"\n处理第 {i + 1} 行配置:")

            # 创建同名变量并赋值
            # row
            history_turns = int(row["history_turns"])
            mode = row["mode"]
            only_need_context = row["only_need_context"]
            max_token_for_text_unit = int(row["max_token_for_text_unit"])
            top_k = int(row["top_k"])

            Q = QueryParam(history_turns=history_turns, mode=mode, only_need_context=only_need_context, top_k=top_k, max_token_for_text_unit=max_token_for_text_unit)

            try:
                time1 = time.time()
                # Clear old data files
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

                # Initialize RAG instance
                rag = await initialize_rag()

                # Test embedding function
                test_text = ["This is a test string for embedding."]
                embedding = await rag.embedding_func(test_text)
                embedding_dim = embedding.shape[1]
                print("\n=======================")
                print("Test embedding function")
                print("========================")
                print(f"Test dict: {test_text}")
                print(f"Detected embedding dimension: {embedding_dim}\n\n")

                with open("/home/xzz/LightRAG/book/datasets/unique_contexts/agriculture_unique_contexts.json", "r", encoding="utf-8") as f:
                    await rag.ainsert(f.read())

                # Perform naive search
                print("\n=====================")
                print(f"run {i}")
                print("=====================")
                resp = await rag.aquery(
                    "what the story talk about?",
                    param=Q,
                )
                if inspect.isasyncgen(resp):
                    await print_stream(resp)
                else:
                    print(resp)



            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                if rag:
                    await rag.llm_response_cache.index_done_callback()
                    await rag.finalize_storages()
                time2 = time.time()
                runtime = time2 - time1
                result_row = {**row, 'runtime_seconds': runtime}
                writer.writerow(result_row)


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
