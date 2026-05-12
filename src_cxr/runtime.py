# src/runtime.py
import os
import multiprocessing
import tensorflow as tf
from tqdm import tqdm as notebook_tqdm
import numpy as np
import random
from src.config import SEED

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_tensorflow_runtime(verbose: bool = True) -> int:
    """
    TensorFlow runtime inicializálás:
    - GPU-k listázása
    - memory growth bekapcsolása
    - CPU magszám lekérdezése

    Returns
    -------
    int
        Elérhető CPU magok száma.
    """

    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_gemm=false "
        "--xla_gpu_autotune_level=2"
    )

    gpus = tf.config.list_physical_devices("GPU")
    cpu_count = multiprocessing.cpu_count()
    
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[WARN] GPU memory growth beállítási hiba: {e}")
     
#    if verbose:
#        print("Num GPUs Available:", len(gpus))

    if verbose:
        print("TF version:", tf.__version__)
        print("GPUs: ", gpus)
        print("CPU cores: ", cpu_count)

        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("TF version =", tf.__version__)
        print("Built with CUDA =", tf.test.is_built_with_cuda())
        print("GPUs =", gpus)
        print("Logical GPUs =", tf.config.list_logical_devices("GPU"))
   
        run_gpu_test()

    return cpu_count

def run_gpu_test() -> None:
    import tensorflow as tf
    import time
    
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus)
    
    if gpus:
        with tf.device("/GPU:0"):
            a = tf.random.normal((4096, 4096))
            b = tf.random.normal((4096, 4096))
    
            t0 = time.time()
            c = tf.matmul(a, b)
            _ = c.numpy()
            t1 = time.time()
    
        print("GPU matmul done in", t1 - t0, "sec")
    else:
        print("No GPU visible to TensorFlow")
