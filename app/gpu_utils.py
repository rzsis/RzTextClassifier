
import torch


class GpuUtils:
    
    def clear_gpu_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Sincroniza para garantir que todas as operações na GPU terminaram