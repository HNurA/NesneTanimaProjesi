import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    """
    Sonuçların tekrarlanabilir olması için rastgelelik tohumlarını sabitler.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed {seed} olarak ayarlandı.")

def create_dir(dir_path):
    """
    Verilen yolda klasör yoksa oluşturur.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Klasör oluşturuldu: {dir_path}")