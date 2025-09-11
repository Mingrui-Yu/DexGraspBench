import multiprocessing
import os

def get_n_worker(mode="cpu"):
    """
    自动选择合适的 n_worker
    mode: "cpu" (CPU密集型) 或 "io" (IO密集型)
    """
    n_cores = multiprocessing.cpu_count()  # 获取逻辑核数
    if mode == "cpu":
        # CPU 密集型，基本和逻辑核数一致
        return n_cores
    elif mode == "io":
        # I/O 密集型，可以超过核数，比如 2x 或 3x
        return n_cores * 2
    else:
        raise ValueError("mode 必须是 'cpu' 或 'io'")

if __name__ == "__main__":
    print(f"逻辑核数: {multiprocessing.cpu_count()}")
    print(f"CPU密集型推荐 n_worker = {get_n_worker('cpu')}")
    print(f"IO密集型推荐 n_worker = {get_n_worker('io')}")
