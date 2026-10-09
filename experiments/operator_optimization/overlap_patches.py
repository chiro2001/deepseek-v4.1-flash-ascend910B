"""Toggle the existing V4.1 Q/KV pipeline before capture and on bank switch."""
IMPLS = []


def initialize(model):
    global IMPLS
    seen = set()
    for module in model.modules():
        wrapper = getattr(module, "dsa_attn", None)
        attention = getattr(wrapper, "dsa_attn", None)
        impl = getattr(attention, "impl", None)
        if impl is not None and hasattr(impl, "multistream_dsv4_dsa_overlap") and id(impl) not in seen:
            seen.add(id(impl))
            IMPLS.append(impl)
    assert len(IMPLS) == 40, len(IMPLS)
    for impl in IMPLS:
        assert all(hasattr(impl, name) for name in ["cv_wq_a", "cv_wkv", "cv_wq_b"])
    set_enabled(False)


def set_enabled(enabled):
    for impl in IMPLS:
        impl.multistream_dsv4_dsa_overlap = enabled
    return {"enabled": enabled, "layers": len(IMPLS)}
