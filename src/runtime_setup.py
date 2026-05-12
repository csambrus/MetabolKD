# src/runtime_setup.py
import os
import multiprocessing
import numpy as np
import random
import tensorflow as tf
import torch
from src.config import SEED
from datetime import datetime
import traceback
import sys
from src.config import LOGS_DIR


def set_global_seed(seed: int = SEED) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Globalis seed beallitva: {seed}")
    return seed


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

def notebook_error_logger(exc_type, exc_value, exc_traceback):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = LOGS_DIR / f"notebook_error_{ts}.log"

    with open(logfile, "w", encoding="utf-8") as f:
        f.write("Jupyter Notebook Error Log\n")
        f.write("=" * 60 + "\n")
        f.write(f"Time: {ts}\n\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

    print(f"❌ Hiba történt. Log mentve ide: {logfile}")

    # normál traceback is maradjon
    traceback.print_exception(exc_type, exc_value, exc_traceback)

def setup_notebook_error_logger():

    sys.excepthook = notebook_error_logger
    print("✅ Globális hiba-logger aktív.")

    return notebook_error_logger
