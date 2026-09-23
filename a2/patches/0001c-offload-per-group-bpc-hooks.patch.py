# SPDX-License-Identifier: Apache-2.0
# [SWA_pergroup] 运行期补丁（不动镜像）：
#   ① `build_offloading_config()` —— 让 `blocks_per_chunk` 支持 **per-group（dict）**；
#   ② `CPUOffloadingSpec.get_manager()` —— unit 模式下换成 "每 key 占 bpc_g 个 unit" 的 manager；
#   ③ 只读日志（池子实际 unit 数 / 每组的 bpc）。
#
# 口径（见 logs/021 §2）：
#   * `blocks_per_chunk` 仍是**标量** ⇒ 一切照旧（`cache.blocks_per_chunk` 不变，用镜像内 manager）。
#   * `blocks_per_chunk` 是 **dict** ⇒
#       - `cache.blocks_per_chunk = 1`（池的"一格" = 1 个 GPU block；池容量 `num_blocks` 从此按 **unit** 计）；
#       - 逐组 bpc 解析后写进 `extra_config["blocks_per_chunk_by_group"]`
#         （调度侧/worker 侧各自算出，只依赖 `kv_cache_config` ⇒ 两侧逐字一致）；
#       - manager 换成 `PerGroupBPCManager`。
#     支持的键：整数 group id / `"swa"`（滑窗组）/ `"full"`（其余）/ `"default"`（兜底，缺省 = 8）。

from __future__ import annotations

import os
import traceback
from typing import Any

try:  # [A2-OFFLOAD] 同上：容器内走包路径
    from pgp_manager import BPC_BY_GROUP_KEY, PerGroupBPCManager, bpc_map_from_extra  # noqa: E402
except ImportError:  # pragma: no cover - 单卡 tiny 的 PYTHONPATH 路径走上面那条
    from vllm.v1.kv_offload.cpu.pgp_manager import (  # noqa: E402
        BPC_BY_GROUP_KEY,
        PerGroupBPCManager,
        bpc_map_from_extra,
    )


def _unwrap_spec(kv_cache_spec):
    """unwrap `UniformTypeKVCacheSpecs`：判"是不是滑窗"只看代表 spec。"""
    members = getattr(kv_cache_spec, "kv_cache_specs", None)
    if members:
        return next(iter(members.values()))
    return kv_cache_spec


def _group_kinds(kv_cache_config) -> list[str]:
    from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec

    kinds: list[str] = []
    for group in kv_cache_config.kv_cache_groups:
        spec = _unwrap_spec(group.kv_cache_spec)
        if isinstance(spec, SlidingWindowSpec):
            kinds.append("swa")
        elif isinstance(spec, MambaSpec):
            kinds.append("mamba")
        else:
            kinds.append("full")
    return kinds


def resolve_per_group_bpc(
    user_cfg: Any, kv_cache_config
) -> tuple[int | None, dict[int, int] | None]:
    """返回 `(标量 bpc 或 None, per-group map 或 None)`。"""
    if user_cfg is None or isinstance(user_cfg, int):
        return user_cfg, None
    if not isinstance(user_cfg, dict):
        raise ValueError(
            "[SWA_pergroup] blocks_per_chunk 必须是整数或 dict，得到 "
            f"{type(user_cfg).__name__}"
        )

    from vllm.v1.kv_cache_interface import MambaSpec

    kinds = _group_kinds(kv_cache_config)
    for group in kv_cache_config.kv_cache_groups:
        spec = _unwrap_spec(group.kv_cache_spec)
        if isinstance(spec, MambaSpec) and getattr(
            spec, "mamba_cache_mode", None
        ) == "align":
            raise ValueError(
                "[SWA_pergroup] 单位池模式不支持 MambaSpec(align)："
                "`resolve_mamba_align_size()` 假定全局 blocks_per_chunk"
            )

    base = user_cfg.get("default", user_cfg.get("full", 8))
    per_kind = {
        "swa": int(user_cfg.get("swa", base)),
        "full": int(user_cfg.get("full", base)),
    }
    bpc_by_group: dict[int, int] = {}
    for idx, kind in enumerate(kinds):
        if idx in user_cfg:
            value = user_cfg[idx]
        elif str(idx) in user_cfg:
            value = user_cfg[str(idx)]
        else:
            value = per_kind.get(kind, int(base))
        value = int(value)
        if value <= 0:
            raise ValueError(
                f"[SWA_pergroup] group {idx} 的 blocks_per_chunk={value} 必须 > 0"
            )
        bpc_by_group[idx] = value
    return 1, bpc_by_group


def install_offload_config_hook(module) -> None:
    """①  `blocks_per_chunk` 支持 dict（per-group）。"""
    orig_build = module.build_offloading_config
    if getattr(orig_build, "_pgp_patched", False):
        return

    def build_offloading_config(vllm_config, kv_cache_config):
        ktc = vllm_config.kv_transfer_config
        assert ktc is not None
        extra = ktc.kv_connector_extra_config
        if not extra:
            return orig_build(vllm_config, kv_cache_config)
        # 幂等：第一次调用时备份用户值（之后 `blocks_per_chunk` 会被改写成 1）。
        if "_pgp_user_blocks_per_chunk" in extra:
            user_cfg = extra["_pgp_user_blocks_per_chunk"]
        else:
            user_cfg = extra.get("blocks_per_chunk")
            extra["_pgp_user_blocks_per_chunk"] = user_cfg

        scalar_bpc, bpc_by_group = resolve_per_group_bpc(user_cfg, kv_cache_config)
        if bpc_by_group is None:
            return orig_build(vllm_config, kv_cache_config)

        extra[BPC_BY_GROUP_KEY] = {str(k): int(v) for k, v in bpc_by_group.items()}
        extra["blocks_per_chunk"] = int(scalar_bpc or 1)
        cfg = orig_build(vllm_config, kv_cache_config)
        try:
            print(
                "[SWA_pergroup] unit 模式生效：cache.blocks_per_chunk=%s "
                "per-group bpc=%s kinds=%s"
                % (
                    cfg.cache.blocks_per_chunk,
                    {k: v for k, v in sorted(bpc_by_group.items())},
                    _group_kinds(kv_cache_config),
                ),
                flush=True,
            )
        except Exception:
            print("[SWA_pergroup] 配置日志失败:\n" + traceback.format_exc(), flush=True)
        return cfg

    build_offloading_config._pgp_patched = True
    module.build_offloading_config = build_offloading_config
    print("[SWA_pergroup] build_offloading_config 钩子已装", flush=True)


def install_spec_hook(module) -> None:
    """② `CPUOffloadingSpec.get_manager()`：unit 模式换 manager；③ 池子日志。"""
    cls = module.CPUOffloadingSpec
    if getattr(cls.get_manager, "_pgp_patched", False):
        return
    orig_get_manager = cls.get_manager
    orig_init = cls.__init__

    def get_manager(self):
        bpc_map = bpc_map_from_extra(self.extra_config)
        if bpc_map and self._manager is None:
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))
            self._manager = PerGroupBPCManager(
                num_blocks=self.num_blocks,
                bpc_by_group=bpc_map,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
            print(
                "[SWA_pergroup] PerGroupBPCManager 生效：num_units=%d bpc=%s"
                % (self.num_blocks, {k: v for k, v in sorted(bpc_map.items())}),
                flush=True,
            )
            return self._manager
        return orig_get_manager(self)

    get_manager._pgp_patched = True
    cls.get_manager = get_manager

    def __init__(self, config, *args, **kwargs):
        orig_init(self, config, *args, **kwargs)
        try:
            bpc_map = bpc_map_from_extra(self.extra_config)
            parallel = getattr(config, "parallel", None)
            print(
                "[SWA_pergroup] CPU 卸载池: num_units=%s kv_bytes_per_unit=%s "
                "cpu_page_size_per_worker=%s replicated_layout=%s "
                "blocks_per_chunk=%s per_group=%s cpu_bytes_to_use=%s "
                "worker_kv_bytes_per_block=%s world_size=%s"
                % (
                    getattr(self, "num_blocks", None),
                    getattr(self, "kv_bytes_per_chunk", None),
                    getattr(self, "cpu_page_size_per_worker", None),
                    getattr(self, "replicated_layout", None),
                    getattr(self, "blocks_per_chunk", None),
                    None
                    if bpc_map is None
                    else {k: v for k, v in sorted(bpc_map.items())},
                    getattr(self, "extra_config", {}).get("cpu_bytes_to_use"),
                    getattr(config, "worker_kv_bytes_per_block", None),
                    getattr(parallel, "world_size", None),
                ),
                flush=True,
            )
            if bpc_map is not None and int(self.blocks_per_chunk) != 1:
                print(
                    "[SWA_pergroup] !!! unit 模式下 blocks_per_chunk != 1：%s"
                    % self.blocks_per_chunk,
                    flush=True,
                )
        except Exception:
            print("[SWA_pergroup] 池日志失败:\n" + traceback.format_exc(), flush=True)

    cls.__init__ = __init__
    print(
        "[SWA_pergroup] CPUOffloadingSpec 钩子已装（get_manager + 池日志）", flush=True
    )


def install_connector_check_hook(module) -> None:
    """保险：确认 connector 绑定的 `build_offloading_config` 是打过钩子的那份。"""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
            config as _cfg,
        )

        if getattr(_cfg.build_offloading_config, "_pgp_patched", False):
            module.build_offloading_config = _cfg.build_offloading_config
        else:
            print(
                "[SWA_pergroup] !!! connector 里的 build_offloading_config 不是补丁版",
                flush=True,
            )
    except Exception:
        print("[SWA_pergroup] connector 钩子失败:\n" + traceback.format_exc(), flush=True)


def log_env_banner() -> None:
    print(
        "[SWA_pergroup] hooks loaded (pid=%s) PGP_BPC=%s"
        % (os.getpid(), os.environ.get("PGP_BPC", "<unset>")),
        flush=True,
    )


def install_worker_pool_log(module) -> None:
    """worker 侧（NPUOffloadingWorker）**物理**池字节：每张 canonical 张量的 page × 行数。"""
    cls = module.NPUOffloadingWorker
    if getattr(cls.__init__, "_pgp_patched", False):
        return
    orig_init = cls.__init__

    def __init__(self, kv_caches, blocks_per_chunk, num_cpu_blocks, *args, **kwargs):
        orig_init(self, kv_caches, blocks_per_chunk, num_cpu_blocks, *args, **kwargs)
        try:
            pages = [int(t.page_size_bytes) for t in kv_caches.tensors]
            sum_page = sum(pages)
            host_bytes = sum(num_cpu_blocks * p * blocks_per_chunk for p in pages)
            print(
                "[SWA_pergroup] worker 物理池: tensors=%d bpc=%d units=%d "
                "sum_page_bytes=%d host_bytes=%d pages=%s"
                % (
                    len(pages),
                    blocks_per_chunk,
                    num_cpu_blocks,
                    sum_page,
                    host_bytes,
                    pages if len(pages) <= 20 else pages[:20] + ["..."],
                ),
                flush=True,
            )
        except Exception:
            print(
                "[SWA_pergroup] worker 池日志失败:\n" + traceback.format_exc(),
                flush=True,
            )

    __init__._pgp_patched = True
    cls.__init__ = __init__
    print("[SWA_pergroup] NPUOffloadingWorker 池日志钩子已装", flush=True)
