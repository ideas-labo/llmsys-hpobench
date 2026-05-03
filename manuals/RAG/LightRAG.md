# LIGHTRAG: SIMPLE AND FAST RETRIEVAL-AUGMENTED GENERATION
...

## Case 1:
### System: 
AI Component:

LLM executors

Non-AI Components:

Database: Neo4j
Vector Store: FAISS
Caching Layer: Redis 7
API Gateway: FastAPI

### Configuration:
...

Here’s the categorized parameter list in English, retaining the **Default** column as requested:

---

### **LLM-Related Parameters**  
| Knob Name                             | Type/Description                                             | Default            |
|---------------------------------------|--------------------------------------------------------------|--------------------|
| `llm_model_name`                      | Name of the LLM model                                        | `qwen2.5-coder:7b` |
| `llm_model_max_token_size`            | Maximum token limit for LLM input/output                     | `32768`            |
| `llm_model_max_async`                 | Concurrency limit for asynchronous LLM calls                 | `4`                |
| `enable_llm_cache`                    | Enable caching for LLM responses                             | `True`             |
| `enable_llm_cache_for_entity_extract` | Enable LLM caching for entity extraction tasks               | `True`             |

---

### **Non-LLM-Related Parameters**  


#### **Storage & Indexing**  
| Knob Name        | Type/Description                                   | Default                |
|------------------|----------------------------------------------------|------------------------|
| `kv_storage`     | Key-value storage class (e.g., `JsonKVStorage`)    | `JsonKVStorage`        |
| `vector_storage` | Vector storage class (e.g., `NanoVectorDBStorage`) | `NanoVectorDBStorage`  |
| `graph_storage`  | Graph storage class (e.g., `NetworkXStorage`)      | `NetworkXStorage`      |

#### **Preprocessing & Chunking**  
| Knob Name                  | Type/Description                      | Default                         |
|----------------------------|---------------------------------------|---------------------------------|
| `chunk_token_size`         | Token size for text chunking          | `1200`                          |
| `chunk_overlap_token_size` | Overlapping token size between chunks | `100`                           |

#### **Graph & Entity Extraction**  
| Knob Name                       | Type/Description                                            | Default                       |  
|---------------------------------|-------------------------------------------------------------|-------------------------------|
| `entity_extract_max_gleaning`   | Depth of entity extraction                                  | `1`                           |
| `node_embedding_algorithm`      | Graph embedding algorithm (e.g., `node2vec`)                | `node2vec`                    |
| `node2vec_params`               | Parameters for Node2Vec (walk length, window size, etc.)    | `{"dimensions": 1536, ...}`   |

#### **Embedding & Vector DB**  
| Knob Name                      | Type/Description                                      | Default                     |
|--------------------------------|-------------------------------------------------------|-----------------------------|
| `embedding_func`               | Embedding generation function (e.g., `openai_embed`)  | `openai_embed`              |
| `embedding_batch_num`          | Batch size for embedding generation                   | `32`                        |
| `vector_db_storage_cls_kwargs` | Vector DB configuration (e.g., similarity threshold)  | `{"cosine_threshold": 0.2}` |
| `embedding_func_max_async`     | Maximum number of concurrent embedding function calls | `16`                        |
---

#### **Query**

#### **QueryParam Configuration**  
| Knob Name                      | Type/Description                                                                                  | Default                                                            |
|--------------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| `mode`                         | Retrieval mode (Literal: "local", "global", "hybrid", "naive", "mix", "bypass")                   | `"global"`                                                         |
| `response_type`                | Response format (e.g., "Multiple Paragraphs", "Single Paragraph", "Bullet Points")                | `"Multiple Paragraphs"`                                            |
| `top_k`                        | Number of top items to retrieve (entities in local mode, relationships in global)                 | `int(os.getenv("TOP_K", "60"))`                                    |
| `max_token_for_text_unit`      | Max tokens per retrieved text chunk                                                               | `int(os.getenv("MAX_TOKEN_TEXT_CHUNK", "4000"))`                   |
| `max_token_for_global_context` | Max tokens for relationship descriptions in global retrieval                                      | `int(os.getenv("MAX_TOKEN_RELATION_DESC", "4000"))`                |
| `max_token_for_local_context`  | Max tokens for entity descriptions in local retrieval                                             | `int(os.getenv("MAX_TOKEN_ENTITY_DESC", "4000"))`                  |
| `history_turns`                | Number of conversation turns (user-assistant pairs) to include in context                         | `3`                                                                |


## Environment / Fidelity Factors

To ensure that evaluation results accurately reflect real-world deployments, several environmental and fidelity factors must be controlled. The table below lists the key factors considered in evaluation:

| Fidelity Factor               | Description                                                                                                                                                                                                                                                     |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Effective reference Ratio** | supporting reference/all reference [2, 12]                                                                                                                                                                                                                      |
| **question type**             | the type of question ["bridge", "comparison"] "bridge"：It means that it is necessary to connect facts by spanning multiple contextual information and derive answers. "comparison"：It means that it is necessary to compare multiple facts to draw conclusions. |
| **Question difficulty**       | The difficulty of query question ["easy", "medium", "hard"]                                                                                                                                                                                                     |  
| **Question ratio**            | The proportion of test samples                                                                                                                                                                                                                                  |  



### Performance Metrics  
When evaluating LightRAG, the authors focus on the following key metrics derived from the experimental design:  

| **Metric**                             | **Description**                                                                                                                                            |
|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Token Efficiency**                   | Quantifies token processing overhead during retrieval (e.g., LightRAG uses **<100 tokens per query** vs. GraphRAG’s **610,000 tokens** for Legal dataset). |
| **API Call Efficiency**                | Measures the number of API calls required for retrieval (e.g., LightRAG requires **1 API call per query** vs. GraphRAG’s **hundreds of API calls**).       |
| **Total time**                         | total run time cost                                                                                                                                        |
| **time to retrieve**                   | The time spent by LightRAG on retrieving entities and entity relationships and contructing KG.                                                             |
| **answear similarity**                 | The similarity between the generated text and the correct answer.                                                                                          |
| **MRR**                                | Mean reciprocal rank.                                                                                                                                      |
| **NDCG**                               | Normalized discounted cumulative gain.                                                                                                                     |
| **context similarity**                 | retrieval context similarity.                                                                                                                              |



