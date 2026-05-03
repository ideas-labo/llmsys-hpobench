#!/bin/bash

# --- Configuration ---
MODEL_PATH="./models/Llama-2-7b-hf"           # 模型路径
DATASET_PATH="../datasets/sg_90k_part1.json"  # 数据集路径 (JSON格式)
DATASET_NAME="sharegpt"                       # 数据集格式 ('sharegpt' 或 'sonnet')
MODEL_ID="Llama-2-7b-hf"                    # 模型标识符 (用于日志和文件名)
RESULTS_DIR="./results"                       # 结果保存目录
TOKENIZER_PATH=${MODEL_PATH}                  # Tokenizer 路径 (如果与模型不同则修改)

# 日志配置
LOG_DIR="./logs"                              # 日志保存目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")           # 时间戳
MAIN_LOG="${LOG_DIR}/run_sampler_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/run_sampler_error_${TIMESTAMP}.log"

# Workload Parameters
REQUEST_RATE=10.0                            # 请求速率 (req/s), 'inf' for no limit
BURSTINESS=1.0                               # 突发性系数 (1.0=泊松过程, <1.0=更突发, >1.0=更均匀)
NUM_PROMPTS=100                              # 要采样的提示数量
MAX_CONCURRENT_REQUESTS=16                   # 最大并发请求数 (设为0或空表示无限制)
# Prefix Caching 测试参数
REPEAT_COUNT=1                               # 请求重复次数 (用于prefix caching测试)


INPUT_LENGTH_RANGE=""                        # 输入长度范围 (格式: "min:max", 如 "128:256")
SORT_REQUESTS="false"                        # 是否按输入长度排序请求

PORT="8000"                                  # 服务器监听端口
GPUS_TO_USE="0"                              # 要使用的GPU ID(s), e.g., "0" or "0,1"

# --- 日志函数 ---
log_message() {
    local level="$1"
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${MAIN_LOG}"
}

log_info() {
    log_message "INFO" "$@"
}

log_warn() {
    log_message "WARN" "$@"
}

log_error() {
    log_message "ERROR" "$@"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $@" >> "${ERROR_LOG}"
}

log_config() {
    local config_num="$1"
    local config_str="$2"
    echo "=== Configuration ${config_num} ===" >> "${MAIN_LOG}"
    echo "Config String: ${config_str}" >> "${MAIN_LOG}"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> "${MAIN_LOG}"
    echo "GPU Memory Before:" >> "${MAIN_LOG}"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >> "${MAIN_LOG}"
    echo "================================" >> "${MAIN_LOG}"
}

# --- 创建日志目录 ---
create_log_dirs() {
    mkdir -p "${LOG_DIR}"
    mkdir -p "${RESULTS_DIR}"
    
    # 创建主日志文件
    log_info "Starting vLLM configuration sampling session"
    log_info "Model: ${MODEL_ID}"
    log_info "Dataset: ${DATASET_PATH}"
    log_info "Request Rate: ${REQUEST_RATE}"
    log_info "Burstiness: ${BURSTINESS}"
    log_info "Max Concurrent Requests: ${MAX_CONCURRENT_REQUESTS}"
    log_info "Results will be saved to: ${RESULTS_DIR}"
    log_info "Logs will be saved to: ${LOG_DIR}"
    
    # 记录系统信息
    log_info "System Information:"
    echo "$(date '+%Y-%m-%d %H:%M:%S') GPU Info:" >> "${MAIN_LOG}"
    nvidia-smi >> "${MAIN_LOG}" 2>&1
    echo "$(date '+%Y-%m-%d %H:%M:%S') Disk Space:" >> "${MAIN_LOG}"
    df -h >> "${MAIN_LOG}"
}

# --- 端口清理和检查函数 ---
# cleanup_port() {
#     local port=$1
#     log_info "Cleaning up port ${port}..."
    
#     # 使用lsof查找并杀死占用端口的进程
#     local pids=$(lsof -ti:${port} 2>/dev/null)
#     if [ ! -z "$pids" ]; then
#         log_warn "Found processes occupying port ${port}: $pids"
#         echo "$pids" | xargs -r kill -9 2>/dev/null
#         sleep 3
#     fi
    
#     # 额外清理vLLM进程
#     pkill -9 -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true
#     sleep 2
# }
# --- 端口清理和检查函数（Docker优化版本） ---
cleanup_port() {
    local port=$1
    log_info "Cleaning up port ${port} (Docker environment)..."
    
    # 1. 先尝试查找占用端口的进程
    local pids=$(ss -tulpn | grep ":${port} " | grep -o 'pid=[0-9]*' | cut -d= -f2 2>/dev/null | sort -u)
    
    # 备选方案：使用netstat
    if [ -z "$pids" ]; then
        pids=$(netstat -tulpn 2>/dev/null | grep ":${port} " | awk '{print $7}' | cut -d/ -f1 2>/dev/null | grep -E '^[0-9]+$' | sort -u)
    fi
    
    # 备选方案：使用lsof（如果可用）
    if [ -z "$pids" ] && command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -ti:${port} 2>/dev/null)
    fi
    
    if [ ! -z "$pids" ]; then
        log_warn "Found processes occupying port ${port}: $pids"
        for pid in $pids; do
            if [ -n "$pid" ] && [ "$pid" -gt 0 ]; then
                log_info "Killing process $pid"
                kill -TERM "$pid" 2>/dev/null || true
                sleep 1
                # 如果进程仍然存在，使用KILL信号
                if kill -0 "$pid" 2>/dev/null; then
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            fi
        done
        sleep 3
    fi
    
    # 2. 额外清理vLLM进程（Docker环境下更有效的方法）
    log_info "Cleaning up vLLM processes..."
    
    # 使用ps查找vLLM进程并终止
    local vllm_pids=$(ps aux | grep -E "(vllm\.entrypoints|api_server)" | grep -v grep | awk '{print $2}')
    if [ ! -z "$vllm_pids" ]; then
        log_warn "Found vLLM processes: $vllm_pids"
        for pid in $vllm_pids; do
            if [ -n "$pid" ] && [ "$pid" -gt 0 ]; then
                kill -TERM "$pid" 2>/dev/null || true
                sleep 1
                if kill -0 "$pid" 2>/dev/null; then
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            fi
        done
    fi
    
    # 3. 使用pkill作为备选（在某些Docker环境中可能不可用）
    pkill -f "vllm.*--port.*${port}" 2>/dev/null || true
    pkill -f "api_server.*${port}" 2>/dev/null || true
    
    sleep 2
    
    # 4. 验证清理结果
    local remaining_pids=$(ss -tulpn 2>/dev/null | grep ":${port} " | grep -o 'pid=[0-9]*' | cut -d= -f2 2>/dev/null)
    if [ ! -z "$remaining_pids" ]; then
        log_warn "Port ${port} still occupied by processes: $remaining_pids"
        return 1
    else
        log_info "Port ${port} successfully cleared"
        return 0
    fi
}


# wait_for_server() {
#     local port=$1
#     local max_wait=60
#     local count=0
    
#     log_info "Waiting for server on port ${port} to become ready..."
    
#     while [ $count -lt $max_wait ]; do
#         if lsof -i:${port} -sTCP:LISTEN -t > /dev/null 2>&1; then
#             log_info "Server on port ${port} is ready after ${count} seconds"
#             return 0
#         fi
#         sleep 2
#         count=$((count + 2))
#         if [ $((count % 10)) -eq 0 ]; then
#             log_info "Still waiting... (${count}/${max_wait}s)"
#         fi
#     done
    
#     log_error "Server on port ${port} failed to start within ${max_wait} seconds"
#     return 1
# }
wait_for_server() {
    local port=$1
    local max_wait=60
    local count=0
    
    log_info "Waiting for server on port ${port} to become ready..."
    
    while [ $count -lt $max_wait ]; do
        # Docker环境下使用多种方法检查端口
        local port_check=false
        
        # 方法1: 使用ss命令
        if ss -tulpn 2>/dev/null | grep -q ":${port} "; then
            port_check=true
        fi
        
        # 方法2: 使用netstat (备选)
        if [ "$port_check" = false ] && command -v netstat >/dev/null 2>&1; then
            if netstat -tulpn 2>/dev/null | grep -q ":${port} "; then
                port_check=true
            fi
        fi
        
        # 方法3: 使用curl测试HTTP连接
        if [ "$port_check" = false ]; then
            if curl -s --connect-timeout 2 "http://localhost:${port}/health" >/dev/null 2>&1; then
                port_check=true
            fi
        fi
        
        # 方法4: 使用lsof (如果可用)
        if [ "$port_check" = false ] && command -v lsof >/dev/null 2>&1; then
            if lsof -i:${port} -sTCP:LISTEN -t > /dev/null 2>&1; then
                port_check=true
            fi
        fi
        
        if [ "$port_check" = true ]; then
            log_info "Server on port ${port} is ready after ${count} seconds"
            return 0
        fi
        
        sleep 2
        count=$((count + 2))
        if [ $((count % 10)) -eq 0 ]; then
            log_info "Still waiting... (${count}/${max_wait}s)"
        fi
    done
    
    log_error "Server on port ${port} failed to start within ${max_wait} seconds"
    return 1
}

# cleanup_vllm_server() {
#     local port=$1
#     local server_pid=$2
    
#     log_info "Stopping vLLM server for config ${config_num}"
    
#     # 1. 首先尝试优雅地终止指定的服务器进程
#     if [ ! -z "${server_pid}" ]; then
#         log_info "Sending TERM signal to server PID: ${server_pid}"
#         kill ${server_pid} > /dev/null 2>&1
#         sleep 3
        
#         # 检查进程是否还在运行
#         if kill -0 ${server_pid} > /dev/null 2>&1; then
#             log_warn "Server process ${server_pid} still running, sending KILL signal"
#             kill -9 ${server_pid} > /dev/null 2>&1
#         fi
#     fi
    
#     # 2. 强制清理所有vLLM进程（参考benchmark_pipeline.sh的方法）
#     log_info "Force killing all vLLM processes..."
    
#     # 杀死所有OpenAI API服务器进程
#     pgrep -f "vllm.entrypoints.openai.api_server" | xargs -r kill -9 > /dev/null 2>&1
    
#     # 也清理可能的原生API服务器进程
#     pgrep -f "vllm.entrypoints.api_server" | xargs -r kill -9 > /dev/null 2>&1
    
#     # 3. 清理端口占用
#     log_info "Cleaning up port ${port}..."
#     local int_port=$((port-0))
#     echo "Cleaning port: ${int_port}"
#     lsof -t -i:${int_port} | xargs -r kill -9 > /dev/null 2>&1
    
#     # 4. 额外的端口清理（使用原有的cleanup_port逻辑）
#     local pids=$(lsof -ti:${port} 2>/dev/null)
#     if [ ! -z "$pids" ]; then
#         log_warn "Found remaining processes on port ${port}: $pids"
#         echo "$pids" | xargs -r kill -9 > /dev/null 2>&1
#     fi
    
#     # 5. 等待清理完成
#     sleep 3
    
#     # 6. 验证清理结果
#     if lsof -i:${port} -sTCP:LISTEN -t > /dev/null 2>&1; then
#         log_warn "Port ${port} still occupied after cleanup"
#     else
#         log_info "Port ${port} successfully cleared"
#     fi
# }
cleanup_vllm_server() {
    local port=$1
    local server_pid=$2
    
    log_info "Stopping vLLM server for config ${config_num} (Docker environment)"
    
    # 1. 首先尝试优雅地终止指定的服务器进程
    if [ ! -z "${server_pid}" ] && [ "${server_pid}" -gt 0 ]; then
        log_info "Sending TERM signal to server PID: ${server_pid}"
        if kill -0 "${server_pid}" 2>/dev/null; then
            kill -TERM "${server_pid}" 2>/dev/null || true
            sleep 5
            
            # 检查进程是否还在运行
            if kill -0 "${server_pid}" 2>/dev/null; then
                log_warn "Server process ${server_pid} still running, sending KILL signal"
                kill -KILL "${server_pid}" 2>/dev/null || true
                sleep 2
            else
                log_info "Server process ${server_pid} terminated gracefully"
            fi
        else
            log_info "Server process ${server_pid} already terminated"
        fi
    fi
    
    # 2. 使用ps命令强制清理所有vLLM进程（Docker环境下更可靠）
    log_info "Force killing all vLLM processes..."
    
    # 查找所有vLLM相关进程
    local vllm_pids=$(ps aux | grep -E "(vllm\.entrypoints|api_server)" | grep -v grep | awk '{print $2}')
    if [ ! -z "$vllm_pids" ]; then
        log_warn "Found vLLM processes to clean: $vllm_pids"
        for pid in $vllm_pids; do
            if [ -n "$pid" ] && [ "$pid" -gt 0 ]; then
                log_info "Terminating vLLM process: $pid"
                kill -TERM "$pid" 2>/dev/null || true
                sleep 1
                if kill -0 "$pid" 2>/dev/null; then
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            fi
        done
    fi
    
    # 3. 使用pkill作为备选（如果可用）
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    pkill -f "api_server" 2>/dev/null || true
    
    # 4. 清理端口占用
    log_info "Cleaning up port ${port}..."
    cleanup_port "${port}"
    
    # 5. 等待清理完成
    sleep 3
    
    # 6. 验证清理结果
    local remaining_processes=$(ps aux | grep -E "(vllm\.entrypoints|api_server)" | grep -v grep)
    if [ ! -z "$remaining_processes" ]; then
        log_warn "Some vLLM processes may still be running:"
        echo "$remaining_processes" >> "${ERROR_LOG}"
    else
        log_info "All vLLM processes successfully cleaned"
    fi
}

# --- vLLM Configurations to Sample ---
CONFIGURATIONS=(
    "1;1;16;128;2048;0.0;False;False;False;False"   
    # "1;1;32;128;2048;0.0;False;False;False;False"
    # "1;1;16;256;4096;0.0;False;False;False;False"
    # "1;1;16;512;8192;0.0;False;False;False;False"  
)

# --- 初始化 ---
create_log_dirs
export RES_DIR_PATH=${RESULTS_DIR}

# --- Main Sampling Loop ---
log_info "Starting vLLM configuration sampling with ${#CONFIGURATIONS[@]} configurations..."

config_num=0
for config in "${CONFIGURATIONS[@]}"; do
    config_num=$((config_num + 1))
    
    # 解析当前配置
    IFS=';' read -r tp_size pp_size block_size max_num_seqs max_num_batched_tokens scheduler_delay_factor enable_chunked_prefill enable_prefix_caching disable_custom_all_reduce use_v2_block_manager <<< "$config"
    
    log_info "Starting configuration ${config_num}/${#CONFIGURATIONS[@]}"
    log_config "${config_num}" "${config}"
    
    echo "-----------------------------------------------------"
    echo "Testing Configuration ${config_num}:"
    echo "  TP Size: ${tp_size}"
    echo "  PP Size: ${pp_size}"
    echo "  Block Size: ${block_size}"
    echo "  Max Seqs: ${max_num_seqs}"
    echo "  Max Batched Tokens: ${max_num_batched_tokens}"
    echo "  Scheduler Delay: ${scheduler_delay_factor}"
    echo "  Chunked Prefill: ${enable_chunked_prefill}"
    echo "  Prefix Caching: ${enable_prefix_caching}"
    echo "  Disable Custom AR: ${disable_custom_all_reduce}"
    echo "  Use v2 Block Mgr: ${use_v2_block_manager}"
    echo "  Repeat Count: ${REPEAT_COUNT}"
    if [ -n "${INPUT_LENGTH_RANGE}" ]; then
        echo "  Input Length Range: ${INPUT_LENGTH_RANGE}"
    fi
    echo "-----------------------------------------------------"

    # 清理端口
    cleanup_port "${PORT}"

    # 1. 启动 vLLM 服务器
    log_info "Starting vLLM server with config ${config_num}"
    SERVER_LOG="${LOG_DIR}/server_${config_num}_${TIMESTAMP}.log"
    
    CUDA_VISIBLE_DEVICES=${GPUS_TO_USE} bash ./run_server.sh \
        "${MODEL_PATH}" \
        "${PORT}" \
        "${tp_size}" \
        "${pp_size}" \
        "${max_num_seqs}" \
        "${max_num_batched_tokens}" \
        "${scheduler_delay_factor}" \
        "${block_size}" \
        "${enable_chunked_prefill}" \
        "${enable_prefix_caching}" \
        "${disable_custom_all_reduce}" \
        "${use_v2_block_manager}" \
        > "${SERVER_LOG}" 2>&1 &

    SERVER_PID=$!
    log_info "Server started with PID: ${SERVER_PID}, logs: ${SERVER_LOG}"

    # 2. 等待服务器就绪
    if ! wait_for_server "${PORT}"; then
        log_error "Server failed to start for config ${config_num}. Skipping."
        kill -9 ${SERVER_PID} > /dev/null 2>&1
        cleanup_port "${PORT}"
        continue
    fi

    # 3. 运行采样客户端 - 简化的并发控制
    log_info "Starting benchmark client for config ${config_num}"
    CLIENT_LOG="${LOG_DIR}/client_${config_num}_${TIMESTAMP}.log"
    
    # 设置并发限制参数（如果指定了数值）
    CONCURRENCY_ARGS=""
    if [ -n "${MAX_CONCURRENT_REQUESTS}" ] && [ "${MAX_CONCURRENT_REQUESTS}" -gt 0 ]; then
        CONCURRENCY_ARGS="--max-concurrency ${MAX_CONCURRENT_REQUESTS}"
    fi

        # 设置 prefix caching 相关参数
    PREFIX_CACHING_ARGS=""
    if [ "${REPEAT_COUNT}" -gt 1 ]; then
        PREFIX_CACHING_ARGS="${PREFIX_CACHING_ARGS} --repeat-count ${REPEAT_COUNT}"
    fi
    
    if [ -n "${INPUT_LENGTH_RANGE}" ]; then
        PREFIX_CACHING_ARGS="${PREFIX_CACHING_ARGS} --input-length-range ${INPUT_LENGTH_RANGE}"
        log_info "Enabling prefix caching test mode with range: ${INPUT_LENGTH_RANGE}"
    fi
    
    if [ "${SORT_REQUESTS}" = "true" ]; then
        PREFIX_CACHING_ARGS="${PREFIX_CACHING_ARGS} --sort-requests"
    fi
    
    python3 sampler.py \
        --backend openai \
        --host localhost \
        --port "${PORT}" \
        --model "${MODEL_ID}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --dataset-name "${DATASET_NAME}" \
        --dataset-path "${DATASET_PATH}" \
        --num-prompts ${NUM_PROMPTS} \
        --request-rate ${REQUEST_RATE} \
        --burstiness ${BURSTINESS} \
        --seed 42 \
        --save-result \
        --result-dir "${RESULTS_DIR}" \
        --tensor-parallel-size ${tp_size} \
        --pipeline-parallel-size ${pp_size} \
        --block-size ${block_size} \
        --max-num-seqs ${max_num_seqs} \
        --max-num-batched-tokens ${max_num_batched_tokens} \
        --scheduler-delay-factor ${scheduler_delay_factor} \
        --enable-chunked-prefill "${enable_chunked_prefill}" \
        --enable-prefix-caching "${enable_prefix_caching}" \
        --disable-custom-all-reduce "${disable_custom_all_reduce}" \
        --use-v2-block-manager "${use_v2_block_manager}" \
        ${CONCURRENCY_ARGS} \
        ${PREFIX_CACHING_ARGS} \
        > "${CLIENT_LOG}" 2>&1

    CLIENT_EXIT_CODE=$?
    
    if [ ${CLIENT_EXIT_CODE} -eq 0 ]; then
        log_info "Client completed successfully for config ${config_num}"
    else
        log_error "Client failed with exit code ${CLIENT_EXIT_CODE} for config ${config_num}"
        echo "Client logs saved to: ${CLIENT_LOG}" >> "${ERROR_LOG}"
    fi

    # 4. 停止 vLLM 服务器
    log_info "Stopping vLLM server for config ${config_num}"
    kill ${SERVER_PID} > /dev/null 2>&1
    sleep 3
    cleanup_vllm_server "${PORT}" "${SERVER_PID}"

    # 记录配置完成
    log_info "Completed configuration ${config_num}/${#CONFIGURATIONS[@]}"
    echo "GPU Memory After Config ${config_num}:" >> "${MAIN_LOG}"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits >> "${MAIN_LOG}"
    echo "---" >> "${MAIN_LOG}"

done

# --- 完成总结 ---
log_info "All configurations completed"
log_info "Results saved in: ${RESULTS_DIR}"
log_info "Logs saved in: ${LOG_DIR}"
log_info "Main log: ${MAIN_LOG}"
log_info "Error log: ${ERROR_LOG}"

# 生成日志摘要
SUMMARY_LOG="${LOG_DIR}/summary_${TIMESTAMP}.log"
echo "=== Sampling Session Summary ===" > "${SUMMARY_LOG}"
echo "Date: $(date)" >> "${SUMMARY_LOG}"
echo "Model: ${MODEL_ID}" >> "${SUMMARY_LOG}"
echo "Dataset: ${DATASET_PATH}" >> "${SUMMARY_LOG}"
echo "Total Configurations: ${#CONFIGURATIONS[@]}" >> "${SUMMARY_LOG}"
echo "Results Directory: ${RESULTS_DIR}" >> "${SUMMARY_LOG}"
echo "Log Directory: ${LOG_DIR}" >> "${SUMMARY_LOG}"
echo "" >> "${SUMMARY_LOG}"
echo "Configuration Details:" >> "${SUMMARY_LOG}"
for i in "${!CONFIGURATIONS[@]}"; do
    echo "Config $((i+1)): ${CONFIGURATIONS[$i]}" >> "${SUMMARY_LOG}"
done

echo "Sampling complete. Results saved in ${RESULTS_DIR}"
echo "Logs saved in ${LOG_DIR}"
echo "Summary: ${SUMMARY_LOG}"