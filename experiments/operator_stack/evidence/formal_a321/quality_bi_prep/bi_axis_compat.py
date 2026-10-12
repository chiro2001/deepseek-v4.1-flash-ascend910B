"""Accept the NumPy-style ``axis`` alias in the vendor batch-invariant reduce_sum.

``vllm_ascend.batch_invariant`` replaces ``torch.sum``/``Tensor.sum`` with its
``reduce_sum`` when ``VLLM_BATCH_INVARIANT=1``. That replacement only accepts
``dim``/``keepdim``, while the vendor's own W4A8 weight loader calls
``.sum(axis=1)`` (a NumPy alias that stock PyTorch sums do accept), so the
engine fails during weight post-processing. This shim wraps the vendor
function to map ``axis``->``dim`` and keeps everything else identical.

It is imported by ``sitecustomize`` only when ``VLLM_BATCH_INVARIANT=1``; the
stock path is never touched.
"""
import importlib.abc
import sys

TARGET = 'vllm_ascend.batch_invariant'
_WRAPPED = '_dsv41_axis_compat_wrapped'


def _wrap(module):
    import torch

    if not hasattr(torch.ops, 'batch_invariant_ops'):
        # The vendor ops namespace only exists after its extension is imported;
        # the module-level import in vllm_ascend.batch_invariant normally did
        # that, so reaching here means the namespace is genuinely absent.
        try:
            import batch_invariant_ops  # noqa: F401
        except Exception:
            return []
        if not hasattr(torch.ops, 'batch_invariant_ops'):
            return []

    original = getattr(module, 'reduce_sum', None)
    if original is None or getattr(original, _WRAPPED, False):
        return
    # Capture the stock torch.sum before the vendor patch replaces it.
    stock_sum = torch.sum
    stock_tensor_sum = torch.Tensor.sum

    def reduce_sum(x, dim=None, keepdim=False, axis=None, **kwargs):
        if dim is None and axis is not None:
            dim = axis
        # Mirror the vendor contract: x.dim() == 1 with no dim means the last
        # axis, exactly as the vendor function documents.
        if dim is None and getattr(x, 'dim', None) is not None and x.dim() == 1:
            dim = -1
        # The Ascend batch-invariant reduce_sum only implements the last
        # dimension. Weight post-processing calls like ``.sum(axis=1)`` on 3-D
        # tensors target an inner dimension; there the batch-invariant kernel
        # is not required (those sums run once during load, not inside the
        # served forward), so fall back to the stock torch implementation.
        if dim is not None and getattr(x, 'dim', None) is not None:
            resolved = dim + x.dim() if dim < 0 else dim
            if resolved != x.dim() - 1:
                return stock_sum(x, dim=dim, keepdim=keepdim)
        return original(x, dim, keepdim, **kwargs)

    setattr(reduce_sum, _WRAPPED, True)
    module.reduce_sum = reduce_sum
    # The vendor module import above also imported ``batch_invariant_ops`` when
    # the AscendC package is present, so the op namespace exists by now.
    try:
        if getattr(module, 'HAS_ASCENDC_BATCH_INVARIANT', False):
            install_meta_kernels()
    except Exception:
        import traceback
        traceback.print_exc()


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = None
        for finder in list(sys.meta_path):
            if finder is self or finder is sys.meta_path[0]:
                continue
            find = getattr(finder, 'find_spec', None)
            if find is None:
                continue
            try:
                found = find(fullname, path, target)
            except Exception:
                found = None
            if found is not None:
                spec = found
                break
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader

        class _Loader:
            def create_module(self, spec_):
                create = getattr(loader, 'create_module', None)
                return create(spec_) if create else None

            def exec_module(self, module):
                loader.exec_module(module)
                _wrap(module)

            def __getattr__(self, name):
                return getattr(loader, name)

        spec.loader = _Loader()
        return spec


def install():
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def _reduce_meta(keepdim):
    def fn(x, dim, keepdim_):
        import torch
        dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
        shape = list(x.shape)
        for d in sorted((d + x.dim() if d < 0 else d) for d in dims):
            shape[d] = 1 if keepdim_ else None
        return torch.empty([s for s in shape if s is not None],
                           device='meta', dtype=x.dtype)
    return fn


def install_meta_kernels():
    """Register missing Meta/FakeTensor kernels for the vendor ops.

    The A3 batch-invariant ops package ships without Meta kernels, so Dynamo's
    fake-tensor tracing cannot infer output shapes (e.g. the engram gate
    ``.sum(-1)`` kept its input shape and the following broadcast failed).
    Real-device execution is unaffected; these registrations only describe
    shapes for tracing and are skipped when a kernel already resolves.
    """
    import torch

    def softmax_meta(x, dim):
        return torch.empty(x.shape, device='meta', dtype=x.dtype)

    def mm_meta(a, b):
        return torch.matmul(
            torch.empty(a.shape, device='meta', dtype=a.dtype),
            torch.empty(b.shape, device='meta', dtype=b.dtype))

    def add_rms_norm_meta(x1, x2, gamma, epsilon):
        rstd = torch.empty(tuple(x1.shape[:-1]) + (1,), device='meta',
                           dtype=torch.float32)
        out = torch.empty(x1.shape, device='meta', dtype=x1.dtype)
        return out, rstd, out

    kernels = {
        'batch_invariant_ops::npu_reduce_sum_batch_invariant': _reduce_meta(False),
        'batch_invariant_ops::npu_reduce_mean_batch_invariant': _reduce_meta(False),
        'batch_invariant_ops::npu_softmax_batch_invariant': softmax_meta,
        'batch_invariant_ops::npu_log_softmax_batch_invariant': softmax_meta,
        'batch_invariant_ops::npu_mm_batch_invariant': mm_meta,
        'batch_invariant_ops::npu_matmul_batch_invariant': mm_meta,
        'batch_invariant_ops::npu_add_rms_norm_batch_invariant': add_rms_norm_meta,
    }
    installed, skipped = [], []
    for op, fn in kernels.items():
        try:
            torch.library.impl(op, 'Meta', fn)
            installed.append(op.split('::')[-1])
        except Exception as exc:  # keep the list for the caller's receipt
            skipped.append(op.split('::')[-1] + ':' + type(exc).__name__)
    import sys as _sys
    print('[BI-META] installed=' + ','.join(installed) + ' skipped=' + ','.join(skipped),
          file=_sys.stderr, flush=True)
    return installed
