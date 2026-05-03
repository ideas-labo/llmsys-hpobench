import psutil
import socket
import time
import subprocess
from datetime import datetime
import os
import signal
import requests  

def check_port_available(port, host='localhost'):
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
            return True
    except OSError:
        return False

def cleanup_vllm_processes(port):
    """清理所有vLLM相关进程"""
    print("Running cleanup script...")
    killed_pids = []
    
    # 获取当前进程PID，避免终止自身
    current_pid = os.getpid()
    print(f"Current process PID: {current_pid} (will be excluded from cleanup)")
    
    try:
        # 方法1: 使用psutil查找并终止进程
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # 跳过当前进程
                if proc.info['pid'] == current_pid:
                    print(f"Skipping current process (PID {current_pid})")
                    continue
                    
                # 检查进程名或命令行是否包含vllm相关关键词
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                process_name = proc.info['name'].lower()
                
                # 检查是否是vLLM相关进程 - 使用更精确的匹配
                is_vllm_process = False
                
                if ('vllm.entrypoints' in cmdline.lower() or 
                    'vllm/entrypoints' in cmdline.lower() or
                    'api_server.py' in cmdline.lower() or 
                    'uvicorn server:app' in cmdline.lower() or
                    ('python' in process_name and 'serve' in cmdline and 'vllm' in cmdline)):
                    is_vllm_process = True
                
                # 检查是否监听指定端口
                is_port_listener = False
                try:
                    # for conn in proc.info['connections'] or []:
                    connections = proc.connections(kind='inet')
                    for conn in connections:
                        if (conn.status == psutil.CONN_LISTEN and 
                            conn.laddr.port == int(port)):
                            is_port_listener = True
                            break
                except (psutil.AccessDenied, AttributeError):
                    # 如果无法获取连接信息，仅基于进程名判断
                    pass
                
                # 如果是vLLM进程或监听目标端口，则终止
                if is_vllm_process or is_port_listener:
                    print(f"Killing process {proc.info['pid']} ({proc.info['name']}) - vLLM: {is_vllm_process}, Port: {is_port_listener}")
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                        killed_pids.append(proc.info['pid'])
                    except psutil.TimeoutExpired:
                        print(f"Force killing process {proc.info['pid']}")
                        proc.kill()
                        killed_pids.append(proc.info['pid'])
                    except psutil.NoSuchProcess:
                        pass  # 进程已经不存在
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception as e:
                print(f"Error processing {proc.info['pid']}: {e}")
                continue
                
    except Exception as e:
        print(f"Error during psutil cleanup: {e}")
    
    # 方法2: 使用通用清理命令
    current_pid = os.getpid()
    cleanup_commands = [
        f"pkill -f 'vllm.entrypoints' -v -P {current_pid}",
        f"pkill -f 'api_server' -v -P {current_pid}", 
        f"pkill -f 'uvicorn' -v -P {current_pid}",
        f"lsof -ti:{port} | grep -v {current_pid} | xargs -r kill -TERM",
        f"sleep 2 && lsof -ti:{port} | grep -v {current_pid} | xargs -r kill -9"
    ]
    
    for cmd in cleanup_commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.stdout.strip():
                print(f"Command '{cmd}' output: {result.stdout.strip()}")
        except subprocess.TimeoutExpired:
            print(f"Command timeout: {cmd}")
        except Exception as e:
            print(f"Error running command '{cmd}': {e}")
    
    if killed_pids:
        print(f"Killed processes: {killed_pids}")
        time.sleep(5)  # 等待进程完全清理
    else:
        print("No processes needed cleanup")
    
    return len(killed_pids) > 0

def wait_for_port_release(port, max_wait=30):
    """等待端口释放"""
    print(f"Waiting for port {port} to be released...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            # 尝试绑定端口来检查是否已释放
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', int(port)))
                print(f"Port {port} is now available")
                return True
        except OSError:
            time.sleep(1)
    
    print(f"Warning: Port {port} may still be in use after {max_wait}s")
    return False

def wait_for_server_ready(port, max_wait_time=600):
    """等待服务器就绪"""
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"Server is ready on port {port}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        # 显示等待进度
        elapsed = time.time() - start_time
        remaining = max_wait_time - elapsed
        if elapsed % 5 < 1:  # 每5秒显示一次
            print(f"Server not ready yet, waiting... ({elapsed:.1f}s elapsed, {remaining:.1f}s remaining)")
        
        time.sleep(2)
    
    print(f"Server failed to start within {max_wait_time} seconds")
    return False

def force_kill_port_processes(port):
    """强制终止占用指定端口的进程 - 增强版本"""
    try:
        killed_processes = []
        
        # 方法1: 使用 psutil 查找占用端口的进程
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                # 单独获取进程的连接信息
                connections = proc.connections(kind='inet')
                for conn in connections:
                    if (hasattr(conn, 'laddr') and 
                        hasattr(conn.laddr, 'port') and 
                        conn.laddr.port == int(port)):
                        print(f"Found process {proc.info['pid']} ({proc.info['name']}) using port {port}")
                        try:
                            # 先尝试温和终止
                            proc.terminate()
                            proc.wait(timeout=3)
                            killed_processes.append(proc.info['pid'])
                            print(f"Terminated process {proc.info['pid']}")
                        except psutil.TimeoutExpired:
                            # 强制终止
                            proc.kill()
                            proc.wait(timeout=3)
                            killed_processes.append(proc.info['pid'])
                            print(f"Force killed process {proc.info['pid']}")
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
            except Exception as e:
                continue
                
        if killed_processes:
            print(f"Successfully killed {len(killed_processes)} processes using port {port}")
            time.sleep(3)  # 给系统一些时间完成清理
        else:
            print(f"No processes found using port {port}")
            
    except Exception as e:
        print(f"Error in force_kill_port_processes: {e}")

def wait_for_port_release(port, max_wait_time=60, check_interval=2):
    """等待端口释放"""
    print(f"Waiting for port {port} to be released...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        if check_port_available(port):
            print(f"Port {port} is now available")
            return True
        
        elapsed = time.time() - start_time
        remaining = max_wait_time - elapsed
        print(f"Port {port} still in use, waiting... ({elapsed:.1f}s elapsed, {remaining:.1f}s remaining)")
        time.sleep(check_interval)
    
    print(f"Timeout waiting for port {port} to be released")
    return False

def check_port_available(port, host='localhost'):
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
            return True
    except OSError:
        return False
    
def improved_server_shutdown(server_process, port, config_idx, log_file_handle, log_lock):
    """改进的服务器关闭逻辑"""
    print(f"Stopping vLLM server for config {config_idx}")
    
    # 在日志文件中记录关闭开始
    try:
        with log_lock:
            stop_timestamp = datetime.now().strftime("%H:%M:%S")
            log_file_handle.write(f"[{stop_timestamp}] [vLLM-{config_idx}] === Server shutdown initiated ===\n")
            log_file_handle.flush()
    except Exception:
        pass
    
    # 步骤1: 优雅终止主进程
    try:
        print("Attempting graceful shutdown...")
        if server_process and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
                print("Server terminated gracefully")
            except subprocess.TimeoutExpired:
                print("Graceful shutdown timeout, forcing kill...")
                server_process.kill()
                try:
                    server_process.wait(timeout=5)
                except:
                    pass
    except Exception as e:
        print(f"Error during server shutdown: {e}")
    
    # 步骤2: 清理所有相关进程 - 不关心返回值
    cleanup_vllm_processes(port)
    
    # 步骤3: 强制清理端口进程
    print(f"Force killing any processes on port {port}...")
    force_kill_port_processes(port)
    
    # 步骤4: 执行最终验证
    print(f"Verifying port {port} availability...")
    port_available = check_port_available(port)
    
    # 如果端口仍然被占用，使用更激进的方法
    if not port_available:
        print(f"Port {port} still in use, using emergency measures...")
        try:
            current_pid = os.getpid()
            # 修改所有清理命令，排除当前进程
            os.system(f"lsof -ti:{port} | grep -v {current_pid} | xargs -r kill -9")
            os.system(f"pkill -9 -f 'vllm' -v -P {current_pid}")
            os.system(f"pkill -9 -f 'uvicorn' -v -P {current_pid}")
            os.system(f"pkill -9 -f 'api_server' -v -P {current_pid}")
            time.sleep(5)
            port_available = check_port_available(port)
        except:
            pass
    
    # 最终状态
    if port_available:
        print(f"✅ Port {port} is free and ready for next configuration")
    else:
        print(f"⚠️ WARNING: Port {port} may still be occupied")
    
    # 记录结束
    try:
        with log_lock:
            completion_timestamp = datetime.now().strftime("%H:%M:%S")
            log_file_handle.write(f"[{completion_timestamp}] [vLLM-{config_idx}] === Server shutdown completed ===\n")
            log_file_handle.flush()
    except Exception:
        pass
    
    return port_available