import torch


def choose_torch_device(preferred_gpu_indices=None, allow_cpu_fallback=True):
    preferred_gpu_indices = preferred_gpu_indices or [1, 2]

    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        for gpu_index in preferred_gpu_indices:
            if gpu_index < cuda_count:
                return f"cuda:{gpu_index}"

    if allow_cpu_fallback:
        return "cpu"

    raise RuntimeError(
        f"Requested GPUs {preferred_gpu_indices}, but none are available."
    )
