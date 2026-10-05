import os, sys, time, statistics
import torch, torch.distributed as dist, torch_npu  # noqa
torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
dist.init_process_group(backend="hccl", rank=int(os.environ.get("RANK", "0")), world_size=int(os.environ.get("WORLD_SIZE", "1")))
rank = dist.get_rank()
shapes = [("61KB", (30, 1024)), ("491KB", (240, 1024)), ("2MB", (1024, 1024))]
for label, sh in shapes:
    x = torch.randn(*sh, dtype=torch.bfloat16, device="npu")
    nb = x.numel() * 2
    for _ in range(50):
        dist.all_reduce(x)
    torch.npu.synchronize()
    rounds = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(50):
            dist.all_reduce(x)
        torch.npu.synchronize()
        rounds.append((time.perf_counter() - t0) / 50 * 1e6)
    if rank == 0:
        print("%-7s %8d B  rounds_us=%s  median=%.1f  BW=%.1f GB/s" % (
            label, nb, [round(r, 1) for r in rounds], statistics.median(rounds), nb / (statistics.median(rounds) / 1e6) / 1e9), flush=True)
dist.destroy_process_group()
