#gpu_utils.py
import torch
import time
from common import print_with_time

class GpuUtils:
    def __init__(self) -> None:
        self.qtd_clear = 0

        torch.backends.cuda.matmul.allow_tf32 = False #evita usar Tensor Cores que podem introduzir imprecisões e evitar erros
        torch.backends.cudnn.allow_tf32 = False #evita usar Tensor Cores que podem introduzir imprecisões e evitar erros
        try:
            torch.set_float32_matmul_precision("medium")  # "high" pode ativar TF32 em Ampere
        except Exception:
            pass        
        
    def clear_gpu_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Sincroniza para garantir que todas as operações na GPU terminaram            
            self.qtd_clear += 1

            if self.qtd_clear % 15 == 0:
                time.sleep(0.1)  # 0.1 segundo a cada 10 limpezas para evitar sobrecarga

    def print_gpu_info(self):
        print_with_time(f"PyTorch: {torch.__version__}")
        print_with_time(f"Torch CUDA runtime: {torch.version.cuda}")
        print_with_time(f"Torch built {torch.version.git_version}")
        print_with_time(f"CUDA disponível: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_with_time(f"GPU: {torch.cuda.get_device_name(0)}")
            print_with_time(f"Driver/CUDA detectados: {torch.version.cuda} | Device count: {torch.cuda.device_count()}")