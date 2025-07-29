#!/usr/bin/env python3

import time
import subprocess
import multiprocessing
import os

# --- 配置参数 ---
THRESHOLD = 5  # GPU 内存占用率阈值（百分比），低于此值视为空闲
CHECK_INTERVAL = 60  # 每隔5分钟检查一次（单位：秒）

# 定义每个进程要运行的命令及其对应的 GPU 分配
# 每个元组包含 (GPU_IDS, COMMAND)
# GPU_IDS 是一个字符串，用于设置 CUDA_VISIBLE_DEVICES 环境变量，例如 "0,1"
# COMMAND 是您要运行的脚本或命令
POST_COMMANDS = [
    ("1", "python textpe/run_pe_fill_in_the_blanks.py --dataset waveui --data results/waveui/openai/gpt-4o-mini/caption10240_part0.csv --voting text --output results/waveui/openai/gpt-4o-mini/pe/meta-llama/spti/text-voting/epsilon=1.0/trial1+blank0.5+temperature1.0+diverse_in_random --config textpe/configs/ordinary.yaml"),  # 进程1使用 GPU 0 和 1
]
# 注意：`--gpu 0` 等参数是示例，您需要根据 `run_job.sh` 的实际需求修改

# --- 函数定义 ---
def get_gpu_usage(device_ids):
    """
    获取所有 GPU 的内存使用率。
    返回一个列表，每个元素代表一个 GPU 的使用率百分比。
    """
    try:
        total = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits']
        ).decode().strip().split('\n')

        used = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
        ).decode().strip().split('\n')

        # 过滤掉空字符串，防止转换失败
        total = list(map(int, filter(None, total)))
        used = list(map(int, filter(None, used)))

        if not total: # 如果没有检测到GPU
            print("未检测到任何 GPU。请检查 nvidia-smi 是否正常工作。")
            return []

        usage = [(u / t) * 100 for u, t in zip(used, total)]
        usage = [usage[idx] for idx in device_ids]
        return usage

    except FileNotFoundError:
        print("错误：未找到 'nvidia-smi' 命令。请确保 NVIDIA 驱动已正确安装并添加到 PATH 中。")
        return []
    except Exception as e:
        print(f"获取 GPU 信息失败: {e}")
        return []

def run_single_job(gpu_ids, command):
    """
    在指定 GPU 上运行单个作业。
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["PYTHONPATH"] = "/data/whx/textDP/:/data/whx/textDP/DPLDM/:/data/whx/textDP/Infinity/"
    print(f"进程 {os.getpid()} 正在启动，使用 GPU: {gpu_ids}，执行命令: {command}")
    try:
        # 使用 shell=True 以便执行包含空格的命令和环境变量设置
        subprocess.run(command, shell=True, env=env, check=True)
        print(f"进程 {os.getpid()} (GPU: {gpu_ids}) 命令执行完毕。")
    except subprocess.CalledProcessError as e:
        print(f"进程 {os.getpid()} (GPU: {gpu_ids}) 命令执行失败，错误代码: {e.returncode}, 输出: {e.stderr}")
    except Exception as e:
        print(f"进程 {os.getpid()} (GPU: {gpu_ids}) 执行过程中发生错误: {e}")


def check_and_run_jobs(device_ids = [0,1,2,3,4,5]):
    """
    循环检查 GPU 状态，并在所有 GPU 空闲时启动并行作业。
    """
    while True:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n--- 检测时间: {current_time} ---")
        usages = get_gpu_usage(device_ids)

        if not usages:
            print("无法获取 GPU 信息，等待下次检查...")
            time.sleep(CHECK_INTERVAL)
            continue

        # 检查所有配置的 GPU 是否都空闲
        # 这里我们假设 POST_COMMANDS 中的 GPU_IDS 是实际存在的且可以被检查到的
        # 实际逻辑是检查所有检测到的 GPU 是否都空闲
        all_gpus_idle = all(u < THRESHOLD for u in usages)

        if all_gpus_idle:
            print(f"所有 {len(usages)} 个 GPU 占用率都低于 {THRESHOLD}%，准备并行运行后续程序。")
            processes = []
            for gpu_ids, command in POST_COMMANDS:
                p = multiprocessing.Process(target=run_single_job, args=(gpu_ids, command))
                processes.append(p)
                p.start() # 启动子进程

            # 等待所有子进程完成
            for p in processes:
                p.join()
            print("所有并行程序执行完毕。")
            break # 所有任务完成后退出循环
        else:
            print(f"当前所有 GPU 使用率: {[f'{u:.2f}%' for u in usages]}")
            print(f"有 GPU 占用率高于 {THRESHOLD}%。等待 {CHECK_INTERVAL / 60} 分钟后再次检查...\n")
            time.sleep(CHECK_INTERVAL)

# --- 主程序入口 ---
if __name__ == "__main__":
    check_and_run_jobs(device_ids=[3])