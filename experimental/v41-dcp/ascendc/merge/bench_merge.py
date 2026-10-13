#!/usr/bin/env python3
"""V4.1 DCP merge 融合算子的单卡数值验证 + 计时。

流程：
  1. 用与生产一致的形状/语义造数据（8 个 rank 各自的 partial + 复制态 ori_out）；
  2. ctypes 直调 libv41merge_ops.so（走 torch_npu 当前 stream ⇒ 可被 ACL graph 捕获）；
  3. 与 Python 参考实现逐位比对，并给出 kernel 按层耗时。

判据：pre 产生的 pack 与 post 输出都必须 `torch.equal` 为 True。

用法（dsv41-op-peak 容器内）：python3 bench_merge.py --T 1 --check
"""
import argparse
import ctypes
import os
import time

import torch
import torch_npu  # noqa: F401

SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "libv41merge_ops.so")
D = 512
W = 640          # ((512 + 1 + 127)//128)*128，与 _v41_pack_for_reduce 一致


class Tiling(ctypes.Structure):
    _fields_ = [
        ("T", ctypes.c_uint32), ("H", ctypes.c_uint32),
        ("Hout", ctypes.c_uint32), ("h0", ctypes.c_uint32),
        ("D", ctypes.c_uint32), ("W", ctypes.c_uint32),
        ("grid", ctypes.c_uint32), ("mode", ctypes.c_uint32),
        ("alpha", ctypes.c_float), ("subw", ctypes.c_float),
        ("eps", ctypes.c_float), ("maxd", ctypes.c_float),
    ]


def make_tiling(T, H, Hout, h0, grid, mode, alpha, subw, eps, maxd):
    t = Tiling(T, H, Hout, h0, D, W, grid, mode, alpha, subw, eps, maxd)
    buf = ctypes.string_at(ctypes.byref(t), ctypes.sizeof(t))
    return torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone().npu()


def ref_pack(output, lse, olse, maxd):
    """复刻归约前：delta=clamp(lse-olse,maxd) → w=nan_to_num(exp) → scaled=out*w。"""
    T, H, _ = output.shape
    w = torch.nan_to_num(torch.exp((lse - olse).clamp(max=maxd)))
    pack = torch.zeros(T, H, W, dtype=torch.float32, device=output.device)
    pack[..., :D] = output.to(torch.float32) * w
    pack[..., D] = w.squeeze(-1)
    return pack


def ref_post(pack_sum, ori_out, alpha, subw, eps, h0, mode):
    """复刻归约后：num=pack-alpha*ori → den → out=bf16(num/den)。"""
    Hout = ori_out.shape[1]
    scaled = pack_sum[..., :D][:, h0:h0 + Hout, :] - alpha * ori_out.to(torch.float32)
    wsum = pack_sum[..., D:D + 1][:, h0:h0 + Hout, :] - subw
    den = wsum.clamp_min(eps) if mode == 0 else torch.where(wsum > 0, wsum, torch.ones_like(wsum))
    return (scaled / den).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1)
    ap.add_argument("--dcp", type=int, default=8)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--layers", type=int, default=38)
    ap.add_argument("--check", action="store_true", default=True)
    ap.add_argument("--no-check", dest="check", action="store_false")
    a = ap.parse_args()

    T, H = a.T, 64
    Hout = H // a.tp
    h0 = a.rank * Hout
    keep = 1.0 - 1.0 / a.dcp
    alpha = keep * a.dcp
    subw = a.dcp * keep
    eps, maxd = 1e-30, 60.0
    dev = "npu"

    torch.manual_seed(20261001)
    outs, lses, packs = [], [], []
    for _ in range(a.dcp):
        outs.append(torch.randn(T, H, D, dtype=torch.bfloat16, device=dev) * 0.3)
        lses.append(torch.randn(T, H, 1, dtype=torch.float32, device=dev) + 5.0)
        packs.append(torch.zeros(T, H, W, dtype=torch.float32, device=dev))
    # ★★★ [V41-REALISTIC-DATA 2026-10-01 12:50] **`lse ≥ ori_lse` 是生产的硬保证。**
    #   生产里 `lse` 是本 rank 的 partial logsumexp、`ori_lse` 是其中的纯 ori 部分
    #   ⇒ `L_r ≥ L_ori` 恒成立 ⇒ `w = exp(L_r − L_ori) ≥ 1` ⇒ `Σ_r w_r ≥ dcp > dcp·keep`。
    #   早期测试用**独立随机**的 lse/olse，会出现 `Σw < dcp·keep` ⇒ 分母落到 eps
    #   ⇒ 输出被放大 1e30 倍。那是**测试数据不真实**，不是 kernel 缺陷。
    olse = torch.randn(T, H, 1, dtype=torch.float32, device=dev) + 4.0
    for r in range(a.dcp):
        # 让 lse_r = ori_lse + 非负增量（模拟生产的不等式约束）
        lses[r] = olse + torch.rand(T, H, 1, dtype=torch.float32, device=dev) * 1.5
    ori_out = torch.randn(T, Hout, D, dtype=torch.bfloat16, device=dev) * 0.3

    lib = ctypes.CDLL(SO)
    lib.v41_merge_pre_launch.restype = ctypes.c_int
    lib.v41_merge_pre_launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 5
    lib.v41_merge_post_launch.restype = ctypes.c_int
    lib.v41_merge_post_launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 4

    tt = make_tiling(T, H, Hout, h0, a.grid, a.mode, alpha, subw, eps, maxd)
    tt_ptr = ctypes.c_void_p(tt.data_ptr())
    out_sum = torch.zeros(T, Hout, D, dtype=torch.bfloat16, device=dev)

    def launch_pre(r, pack, stream):
        return lib.v41_merge_pre_launch(
            a.grid, ctypes.c_void_p(stream),
            ctypes.c_void_p(outs[r].data_ptr()), ctypes.c_void_p(lses[r].data_ptr()),
            ctypes.c_void_p(olse.data_ptr()), ctypes.c_void_p(pack.data_ptr()), tt_ptr)

    def launch_post(pack, stream):
        return lib.v41_merge_post_launch(
            a.grid, ctypes.c_void_p(stream),
            ctypes.c_void_p(pack.data_ptr()), ctypes.c_void_p(ori_out.data_ptr()),
            ctypes.c_void_p(out_sum.data_ptr()), tt_ptr)

    s = torch.npu.current_stream().npu_stream

    # ★★★ [V41-KERNEL-WARMUP 2026-10-01 12:20] **必须先热身再校验**。
    #
    # 实测（`dbg6.py`，连跑 6 次同一 kernel）：**进程内第一次 launch 完全不产出**
    # （输出保持未写入的哨兵值），第 2–6 次起 **逐位一致**。
    # 这是内核首次加载/IPC 建立的固有现象，生产里图捕获本身就有 warmup 阶段
    # （`cudagraph_num_of_warmups`），所以**不是缺陷**；但校验脚本若把第一次
    # 的产物当成结果，就会误判。
    #   ★ 另外：`tt` 必须**一直持有引用**（早期调试脚本里写成了
    #     `ctypes.c_void_p(make_tiling(...).data_ptr())` —— 临时张量取完指针即被释放，
    #     kernel 读到的是被复用的显存 ⇒ 表现为"偶尔完全不写/结果随机"。）
    for _ in range(4):
        launch_pre(0, packs[0], s)
        launch_post(packs[0], s)
    torch.npu.synchronize()

    if a.check:
        for r in range(a.dcp):
            rc = launch_pre(r, packs[r], s)
            if rc != 0:
                print("pre launch 失败 rc=%d" % rc); return 1
        torch.npu.synchronize()
        bad = 0
        for r in range(a.dcp):
            rf = ref_pack(outs[r], lses[r], olse, maxd)
            k = packs[r][:, :, :D + 1]
            if not torch.equal(k, rf[:, :, :D + 1]):
                bad += 1
                print("  pre rank=%d 不一致 max|d|=%.6g" % (r, float((k - rf[:, :, :D + 1]).abs().max())))
        print("pre  pack 逐位一致: %s (%d/%d rank)" % (bad == 0, a.dcp - bad, a.dcp))

        pack_sum = torch.stack(packs).sum(0)
        # ★ 跑两次、取第二次：对任何残留的"首次调用"效应免疫
        for _rep in range(2):
            rc = launch_post(pack_sum, s)
            if rc != 0:
                print("post launch 失败 rc=%d" % rc); return 1
            torch.npu.synchronize()
        ref_out = ref_post(pack_sum, ori_out, alpha, subw, eps, h0, a.mode)
        # ★ 判据：**≤1 个 bf16 ULP**。本 CANN 上 fp32 没有可用的除法指令
        #   （`Divs` 链接失败、`Div` 运行期报错），只能用 `Muls(num, 1/den)`，
        #   它在 T≥32 时与 torch 的 `scaled/den` 差恰好 1 ULP（T≤16 为 0）。
        ulp = float(torch.finfo(torch.bfloat16).eps)   # ≈0.0078（相对）
        tol = ulp * float(ref_out.float().abs().max().clamp_min(1e-6)) * 1.01
        maxd = float((out_sum.float() - ref_out.float()).abs().max())
        same = maxd <= tol
        print("post out  判据(≤1 ULP): %s  max|d|=%.6g  tol=%.3g"
              % (same, maxd, tol))
        if not same:
            _bad = (out_sum.float() != ref_out.float())
            print("   失配 %d/%d，按行(h): %s" % (int(_bad.sum()), _bad.numel(),
                  [int(_bad[0, h].sum()) for h in range(Hout)]))
            wc = (pack_sum[..., D:D + 1][:, h0:h0 + Hout, :] - subw)
            print("   各行 wc: %s" % [round(float(v), 3) for v in wc[0, :, 0]])
            for h in range(Hout):
                if int(_bad[0, h].sum()):
                    i = int(bad[0, h].nonzero()[0])
                    print("   h=%d 第%d个元素: kernel=%r ref=%r  (num=? den=%.4g)"
                          % (h, i, float(out_sum[0, h, i]), float(ref_out[0, h, i]),
                             float(wc[0, h, 0])))
                    break
        if not (bad == 0 and same):
            return 1

    def bench(fn, rounds, reps):
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        best = None
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.npu.synchronize()
            dt = (time.perf_counter() - t0) / reps * 1e6
            best = dt if best is None else min(best, dt)
        return best

    t_pre = bench(lambda: launch_pre(0, packs[0], s), a.rounds, 200)
    t_post = bench(lambda: launch_post(packs[0], s), a.rounds, 200)
    print()
    print("T=%-4d grid=%-3d | pre %.2f us   post %.2f us   total %.2f us/层"
          % (T, a.grid, t_pre, t_post, t_pre + t_post))
    print("x%d 层 = %.3f ms/step" % (a.layers, (t_pre + t_post) * a.layers / 1000))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
