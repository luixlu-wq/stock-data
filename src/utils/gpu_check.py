"""
GPU verification and benchmarking utilities.
"""
import time

import torch
import numpy as np

from src.utils.logger import setup_logger


def check_gpu() -> dict:
    """
    Check GPU availability and properties.

    Returns:
        Dictionary with GPU information
    """
    logger = setup_logger('gpu_check')

    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': []
    }

    if torch.cuda.is_available():
        gpu_info['device_count'] = torch.cuda.device_count()

        logger.info("=" * 50)
        logger.info("GPU INFORMATION")
        logger.info("=" * 50)
        logger.info(f"CUDA Available: Yes")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {gpu_info['device_count']}")
        logger.info("")

        for i in range(gpu_info['device_count']):
            props = torch.cuda.get_device_properties(i)

            device_info = {
                'name': props.name,
                'compute_capability': f"{props.major}.{props.minor}",
                'total_memory': f"{props.total_memory / 1024**3:.2f} GB",
                'multi_processor_count': props.multi_processor_count
            }

            gpu_info['devices'].append(device_info)

            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  Compute Capability: {props.major}.{props.minor}")
            logger.info(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"  Multi-Processor Count: {props.multi_processor_count}")
            logger.info("")

    else:
        logger.warning("=" * 50)
        logger.warning("CUDA NOT AVAILABLE")
        logger.warning("=" * 50)
        logger.warning("PyTorch will use CPU for computations")
        logger.warning("To use GPU, ensure:")
        logger.warning("  1. You have an NVIDIA GPU")
        logger.warning("  2. CUDA toolkit is installed")
        logger.warning("  3. PyTorch with CUDA support is installed")
        logger.warning("")
        logger.warning("Install PyTorch with CUDA:")
        logger.warning("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        logger.warning("")

    return gpu_info


def benchmark_gpu_vs_cpu(matrix_size: int = 5000, iterations: int = 10) -> dict:
    """
    Benchmark GPU vs CPU performance.

    Args:
        matrix_size: Size of matrices for multiplication
        iterations: Number of iterations to average

    Returns:
        Dictionary with benchmark results
    """
    logger = setup_logger('gpu_check')

    logger.info("=" * 50)
    logger.info("GPU vs CPU BENCHMARK")
    logger.info("=" * 50)
    logger.info(f"Matrix size: {matrix_size}x{matrix_size}")
    logger.info(f"Iterations: {iterations}")
    logger.info("")

    results = {
        'cpu_time': 0,
        'gpu_time': 0,
        'speedup': 0
    }

    # CPU Benchmark
    logger.info("Running CPU benchmark...")
    cpu_times = []

    for i in range(iterations):
        a = torch.randn(matrix_size, matrix_size)
        b = torch.randn(matrix_size, matrix_size)

        start = time.time()
        c = torch.mm(a, b)
        cpu_time = time.time() - start

        cpu_times.append(cpu_time)

    results['cpu_time'] = np.mean(cpu_times)
    logger.info(f"CPU Average Time: {results['cpu_time']:.4f} seconds")

    # GPU Benchmark (if available)
    if torch.cuda.is_available():
        logger.info("Running GPU benchmark...")
        gpu_times = []

        device = torch.device('cuda:0')

        # Warmup
        a_gpu = torch.randn(matrix_size, matrix_size, device=device)
        b_gpu = torch.randn(matrix_size, matrix_size, device=device)
        _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()

        for i in range(iterations):
            a_gpu = torch.randn(matrix_size, matrix_size, device=device)
            b_gpu = torch.randn(matrix_size, matrix_size, device=device)

            torch.cuda.synchronize()
            start = time.time()
            c_gpu = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start

            gpu_times.append(gpu_time)

        results['gpu_time'] = np.mean(gpu_times)
        results['speedup'] = results['cpu_time'] / results['gpu_time']

        logger.info(f"GPU Average Time: {results['gpu_time']:.4f} seconds")
        logger.info(f"Speedup: {results['speedup']:.2f}x")
        logger.info("")

    else:
        logger.warning("GPU not available, skipping GPU benchmark")
        logger.info("")

    return results


def get_recommended_device() -> torch.device:
    """
    Get recommended device (GPU if available, otherwise CPU).

    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


if __name__ == "__main__":
    # Run checks
    gpu_info = check_gpu()

    # Run benchmark if GPU available
    if gpu_info['cuda_available']:
        benchmark_gpu_vs_cpu(matrix_size=5000, iterations=10)
    else:
        print("\nSkipping benchmark (GPU not available)")

    # Show recommended device
    logger = setup_logger('gpu_check')
    device = get_recommended_device()
    logger.info(f"Recommended device for training: {device}")
