import os
import sys
import types
import inspect
import torch as _torch


def fix_cuda_env():
    if os.environ.get("_FIXED_CUDNN_PATHS") == "1":
        return

    conda = os.environ.get("CONDA_PREFIX", "")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in ld.split(":") if p]
    parts = [p for p in parts if p != "/usr/local/cuda/lib64"]

    if conda:
        conda_lib = f"{conda}/lib"
        if conda_lib not in parts:
            parts.insert(0, conda_lib)

        maybe_cudnn = (
            f"{conda}/lib/python{sys.version_info.major}."
            f"{sys.version_info.minor}/site-packages/nvidia/cudnn/lib"
        )
        if os.path.isdir(maybe_cudnn) and maybe_cudnn not in parts:
            parts.insert(1, maybe_cudnn)

    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    os.environ.pop("CUDA_HOME", None)
    os.environ.pop("CUDA_PATH", None)
    os.environ["_FIXED_CUDNN_PATHS"] = "1"


def patch_torch_amp():
    try:
        from torch.amp import custom_fwd, custom_bwd  # noqa: F401
    except Exception:
        import torch.cuda.amp as cuda_amp

        if not hasattr(_torch, "amp"):
            _torch.amp = types.SimpleNamespace()
        if not hasattr(_torch.amp, "custom_fwd"):
            _torch.amp.custom_fwd = cuda_amp.custom_fwd
        if not hasattr(_torch.amp, "custom_bwd"):
            _torch.amp.custom_bwd = cuda_amp.custom_bwd

    try:
        from torch.amp import custom_fwd as _cfwd_test  # noqa: F401
        sig = str(inspect.signature(_cfwd_test))
        if "device_type" not in sig:
            raise ImportError
    except Exception:
        import torch.cuda.amp as cuda_amp

        def _make_compat_decorator(old_obj):
            def wrapper(*args, **kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k in ("cast_inputs",)}
                if args and callable(args[0]) and len(args) == 1:
                    fn = args[0]
                    try:
                        dec = old_obj(**kwargs)
                        return dec(fn)
                    except TypeError:
                        return old_obj(fn)
                try:
                    dec = old_obj(**kwargs)
                    return dec
                except TypeError:
                    return old_obj
            return wrapper

        if not hasattr(_torch, "amp"):
            _torch.amp = types.SimpleNamespace()

        _torch.amp.custom_fwd = _make_compat_decorator(cuda_amp.custom_fwd)
        _torch.amp.custom_bwd = _make_compat_decorator(cuda_amp.custom_bwd)


def setup_runtime_patches():
    fix_cuda_env()
    patch_torch_amp()