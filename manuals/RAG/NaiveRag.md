# NaiveRag: SIMPLE AND NAIVE RETRIEVAL-AUGMENTED GENERATION
...

## Overview: 

LLM control knob: Embedding parameters and chat parameters

RAG control knob: retriever Parameters, Preprocessing & Chunking parameter 

Database control: database

## Configuration Specification:

---

### **retriever Parameters**  
| Knob Name                   | Type/Description                                                                 | Default                          |  
|-----------------------------|----------------------------------------------------------------------------------|----------------------------------|
| `search_kwargs.k` | The number of documents to be returned | 10 |


### **Preprocessing & Chunking Parameters**  
| Knob Name                   | Type/Description                                                                 | Default                          |  
|----------------------------------|----------------------------------------------------------------------------------|----------------------------------|  
| `chunk_token_size`               | Token size for text chunking                                                     | `1200`                          |  
| `chunk_overlap_token_size`       | Overlapping token size between chunks                                            | `100`                           |  

### **Embedding Parameters**  
| Knob Name                   | Type/Description                                                                 | Default                          |  
|-----------------------------|----------------------------------------------------------------------------------|----------------------------------|  
| `model`                    | Name of the Ollama model to use (e.g., `nomic-embed-text:latest`)                                  | Required (no default)           |  
| `num_ctx`                  | Context window size for token generation                                         | `2048`                          |  
| `repeat_penalty`           | Penalty for repetition (higher = stronger penalty)                               | `1.1` (0-2)                           |  
| `temperature`              | Sampling temperature for controlling creativity                                  | `0.8` (0-1)                          |  
| `top_k` | Optional, an integer. Reduces the probability of generating nonsense. A higher value (e.g., 100) gives more diverse answers, while a lower value (e.g., 10) is more conservative. | `40` |

### **Chat Parameters**  
| Parameter Name | Type/Description | Default |
| --- | --- | --- |
| `model`                    | Name of the Ollama model to use (e.g., `llama3`)                                  | Required (no default)           |  
| `num_ctx` | Optional, an integer. Sets the size of the context window used to generate the next token. | `2048` |
| `repeat_penalty` | Optional, a float. Sets the strength of the penalty for repetitions. A higher value (e.g., 1.5) penalizes repetitions more severely, while a lower value (e.g., 0.9) is more lenient. | `1.1` (0-2) |
| `temperature` | Optional, a float. The temperature parameter of the model. Increasing the temperature makes the model answer more creatively. | `0.8` (0-1) |
| `top_k` | Optional, an integer. Reduces the probability of generating nonsense. A higher value (e.g., 100) gives more diverse answers, while a lower value (e.g., 10) is more conservative. | `40` |


---

### **Databae Choice**  

| Knob Name | Type | Default | Choice |
| --- | --- | --- | --- |
| `DATABASE_TYPE` | Name of database to use | `duckdb` |`chroma`, `faiss`, `duckdb`  | 


## Environment / Fidelity Factors

To ensure that evaluation results accurately reflect real-world deployments, several environmental and fidelity factors must be controlled. The table below lists the key factors considered in evaluation:

| Fidelity Factor                    | Description|
|------------------------------------|----------|
| `Question_ratio` | Test sample ratio: proportion of questions selected for evaluation from total question set (0-1) |
| `Question_difficulty` | Question difficulty level: 0-Easy, 1-Medium, 2-Hard (based on question length percentiles at 33% and 66%) |
| `Corpus_scale` | Corpus scale: number of additional domain corpora mixed into the main dataset (0-3) |
| `Dataset_category` | Dataset category: 0-agriculture, 1-art, 2-biography, 3-cs (different scale) |

The normalized fidelity directory and CSV name use hyphen-separated values:

```text
{question_ratio}-{corpus_scale}-{question_difficulty}-{dataset_category}
```

Example: `0.2-0-easy-agriculture`.

### UltraDomain Benchmark (with Four Sub-Datasets)

- **Sub-Datasets Composition**:  
  1. **Agriculture**  
    - **Source**: 12 documents, 2.017 million tokens  
    - **Content**: Agricultural practices (beekeeping management, crop production, disease prevention, etc.).  

  2. **CS**  
    - **Source**: 10 documents, 2.306 million tokens  
    - **Content**: Computer science (machine learning, big data processing, recommendation systems, etc.).  

  3. **Legal**  
    - **Source**: 94 documents, 5.081 million tokens  
    - **Content**: Legal domain (corporate restructuring, compliance governance, legal agreements, etc.).  

  4. **Mix**  
    - **Source**: 61 documents, 619,000 tokens  
    - **Content**: Interdisciplinary texts (literature, philosophy, history, etc.).  

---

### Performance Metrics  
When evaluating NaiveRAG, the authors focus on the following key metrics derived from the experimental design:  

## RAG System Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Semantic Similarity Metrics** | |
| `average_similarity` | Average cosine similarity between predicted answers and reference answers calculated using nomic-embed-text embedding vectors |
| **Performance Time Metrics** | |
| `chunk_time_seconds` | Time consumed for document chunking processing (seconds) |
| `build_time_seconds` | Time consumed for vector database construction and system initialization (seconds) |
| `test_time_seconds` | Time consumed for RAG question-answering evaluation testing (seconds) |
| `total_time_seconds` | Total time consumption (sum of chunking, building, and testing time) |
| **Retrieval Quality Metrics** | |
| `MRR` | Mean Reciprocal Rank, measuring the quality of the first relevant document position [0, 1] |
| `NDCG` | Normalized Discounted Cumulative Gain, a ranking quality metric considering document position weights [0, 1] |
| `Context Similarity` | Similarity between retrieved context and true context [-1, 1] |
| `avg_relevant_docs` | Average number of relevant documents in retrieval results |
| **Generation Quality Metrics** | |
| `lexical_answer_correctness` | Lexical answer correctness (recall), proportion of reference answer vocabulary matched in predictions [0, 1] |
| `answer_precision` | Answer precision, proportion of correct vocabulary in predicted answers [0, 1] |
| `answer_f1_score` | Harmonic mean of recall and precision, balancing completeness and accuracy of generated answers [0, 1] |
| **Token Usage Metrics** | |
| `total_tokens` | Total number of tokens used in the evaluation process |
| `embedding_tokens` | Number of tokens used for embedding processing |
| `llm_input_tokens` | Number of tokens used for LLM input |
| `llm_output_tokens` | Number of tokens used for LLM output generation |
| `avg_tokens_per_question` | Average number of tokens consumed per question |
| `embedding_to_llm_ratio` | Ratio of embedding tokens to LLM tokens, used to evaluate resource allocation balance |



