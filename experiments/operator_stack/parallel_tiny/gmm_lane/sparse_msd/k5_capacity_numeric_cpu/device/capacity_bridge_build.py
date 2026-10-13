"""Generate and compile the K5 capacity device bridge for one profile.

Reuses the frozen v5 probe bridge text (borrowed/v5_probe_bridge.cpp) through
bridge_factory, so borrowed context/stream handling, the five 32B pointer
lists, the 72B/16B HostArgs payloads, canaries and cleanup stay identical.
Two additions are applied on top, both asserted unique:

  * the physical-device declaration accepts 4 (this lane's window);
  * one planner-only entry is appended for the illegal-count negative case
    (planner launch + frame D2H only, no GMM math launch).

The bridge still owns no device/context/stream initialization.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parent
SPARSE = LINE.parent
sys.path.insert(0, str(SPARSE))

from k5_capacity_numeric_cpu import bridge_factory as BF  # noqa: E402
from k5_capacity_numeric_cpu import profile as KP  # noqa: E402

SDK = Path('/usr/local/Ascend/cann-9.1.0')
FROZEN = Path('/work/gmm_lane_sparse_msd_build_20261011_v3_repro')
LOADER = Path('/work/gmm_lane_sparse_msd_runtime_loader_v2_cpu_20261011')
LAUNCH = Path('/work/gmm_lane_sparse_msd_launch_args_v3_cpu_20261011')
FROZEN_SHA = {'loader_v2': 'b2e6c3033f372a0542d96f9aa2053f230617052aacd2b37bd484a5293b4cf884',
              'launch_v3': '5af9db3a35f3cc652bb7643b71f64fb621c70e66e89cd6896b0d0c49787571f5'}
PHYSICAL_NEEDLE = 'std::string(declared)!="7"'
LABEL_NEEDLE = 'check("rtGetDeviceIndexByPhyId(6)",'
PROBE_NEEDLE = '    bool synchronized = true;\n};'
PROBE_FIELDS = '''    bool synchronized = true;
    // K5 capacity device lane: bound state for the timed launch-only entries.
    uint64_t bound_x{}, bound_weight{}, bound_assist{}, bound_scale{}, bound_counts{}, bound_token{};
    uint32_t bound_sparse{};
    Gmm2MsdArgsV3 bound_args{};
    Gmm2PlannerArgsV3 bound_planner{};
    bool bound{false};
};
'''
PLANNER_ONLY = '''
// K5 capacity device lane: planner-only negative-case entry. It enqueues the
// frozen planner on the borrowed stream and reads back only the frame; no GMM
// math, no weight access and no timing claim is involved.
extern "C" int k5_probe_planner_only(void *owner, uint64_t counts, uint32_t *frame) {
    Probe *p = static_cast<Probe *>(owner);
    if (!p || !counts || !frame || !p->library) return -910815;
    try {
        aclrtContext current_context{};
        check("aclrtGetCurrentContext:planner_only", aclrtGetCurrentContext(&current_context));
        check("same_borrowed_context_guard:planner_only", current_context == p->borrowed_context ? 0 : -910812);
        check("memset:planner_only_frame", aclrtMemset(reinterpret_cast<void *>(p->workspace_base + SYSTEM), FRAME_BYTES, 0, FRAME_BYTES));
        Gmm2PlannerArgsV3 planner{counts, p->workspace_base + SYSTEM};
        ++p->planner_launches;
        check("HostArgs:planner_only", gmm2_launch_v3_host_args(p->functions.planner, p->borrowed_stream, &planner, 16, 1, 0));
        check("sync:planner_only", aclrtSynchronizeStreamWithTimeout(p->borrowed_stream, 10000));
        check("D2H:planner_only_frame", aclrtMemcpy(frame, FRAME_BYTES, reinterpret_cast<void *>(planner.frame), FRAME_BYTES, ACL_MEMCPY_DEVICE_TO_HOST));
        return 0;
    } catch (Failure failure) { return failure.code; }
}
'''
TIMED_ENTRIES = '''
// K5 capacity device lane: timed launch entries. bind() validates the frozen
// contract, packs the five 32B single-tensor lists, resets canaries and the
// private workspace, all outside any timed window. launch_only() enqueues only
// the planner (sparse) plus the selected GMM kernel and never syncs or reads
// back, so a caller event pair measures the operator window only.
extern "C" int k5_probe_bind(void *owner, uint64_t x, uint64_t weight, uint64_t assist, uint64_t scale,
                             uint64_t counts, uint64_t token_scale, uint32_t sparse) {
    Probe *p = static_cast<Probe *>(owner);
    if (!p || !x || !weight || !assist || !scale || !counts || !token_scale || sparse > 1 ||
        !p->workspace_allocation || !p->y_allocation || p->bound) return -910818;
    try {
        aclrtContext current_context{};
        check("aclrtGetCurrentContext:bind", aclrtGetCurrentContext(&current_context));
        check("same_borrowed_context_guard:bind", current_context == p->borrowed_context ? 0 : -910812);
        Gmm2SparseBuffers b{};
        b.x = {x, uint64_t(PROFILE.capacity) * PROFILE.k};
        b.weight = {weight, uint64_t(PROFILE.experts) * PROFILE.k * PROFILE.n / 2};
        b.assist = {assist, uint64_t(PROFILE.experts) * PROFILE.n * 4};
        b.scale = {scale, uint64_t(PROFILE.experts) * PROFILE.n * 8};
        b.counts = {counts, uint64_t(PROFILE.experts) * 8};
        b.token_scale = {token_scale, uint64_t(PROFILE.capacity) * 4};
        b.y = {addr(p->y_allocation) + 128, Y_BYTES};
        b.x_list = {addr(p->lists[0]), 32}; b.weight_list = {addr(p->lists[1]), 32};
        b.assist_list = {addr(p->lists[2]), 32}; b.scale_list = {addr(p->lists[3]), 32};
        b.y_list = {addr(p->lists[4]), 32};
        b.workspace = {p->workspace_base, WORKSPACE};
        b.ffts_address = p->ffts; b.row_capacity = PROFILE.capacity; b.experts = PROFILE.experts;
        b.k = PROFILE.k; b.n = PROFILE.n; b.group_list_type = 1;
        check("frozen_buffer_contract:bind", capacity_validate(&b, &PROFILE));
        std::array<uint64_t, 20> words{};
        check("frozen_single_tensor_list_pack:bind", capacity_pack_lists(&b, &PROFILE, words.data()));
        for (unsigned i = 0; i < 5; ++i) check("H2D:single_tensor_list32:bind",
            aclrtMemcpy(p->lists[i], 32, words.data() + 4 * i, 32, ACL_MEMCPY_HOST_TO_DEVICE));
        check("memset:bind_workspace_canaries", aclrtMemset(p->workspace_allocation, M1_WORKSPACE_ALLOCATION_BYTES, 0xa5, M1_WORKSPACE_ALLOCATION_BYTES));
        check("memset:bind_workspace", aclrtMemset(reinterpret_cast<void *>(b.workspace.address), WORKSPACE, 0, WORKSPACE));
        check("memset:bind_y_canaries", aclrtMemset(p->y_allocation, Y_BYTES + 256, 0xa5, Y_BYTES + 256));
        p->bound_x = x; p->bound_weight = weight; p->bound_assist = assist; p->bound_scale = scale;
        p->bound_counts = counts; p->bound_token = token_scale; p->bound_sparse = sparse;
        p->bound_args = Gmm2MsdArgsV3{p->ffts, b.x_list.address, b.weight_list.address, b.assist_list.address,
                                      b.scale_list.address, counts, token_scale, b.y_list.address, b.workspace.address};
        p->bound_planner = Gmm2PlannerArgsV3{counts, b.workspace.address + SYSTEM};
        p->bound = true;
        return 0;
    } catch (Failure failure) { return failure.code; }
}

extern "C" int k5_probe_launch_only(void *owner) {
    Probe *p = static_cast<Probe *>(owner);
    if (!p || !p->bound) return -910819;
    try {
        if (p->bound_sparse) {
            ++p->planner_launches;
            check("HostArgs:timed_planner_included", gmm2_launch_v3_host_args(p->functions.planner, p->borrowed_stream, &p->bound_planner, 16, 1, 0));
            ++p->candidate_launches;
        } else ++p->native_launches;
        check(p->bound_sparse ? "HostArgs:timed_candidate" : "HostArgs:timed_native",
              gmm2_launch_v3_host_args(p->bound_sparse ? p->functions.candidate : p->functions.native,
                                       p->borrowed_stream, &p->bound_args, 72, 24, 1));
        p->synchronized = false;
        return 0;
    } catch (Failure failure) { return failure.code; }
}

extern "C" int k5_probe_batch_finish(void *owner) {
    Probe *p = static_cast<Probe *>(owner);
    if (!p) return -910820;
    try {
        check("sync:timed_batch", aclrtSynchronizeStreamWithTimeout(p->borrowed_stream, 30000));
        p->synchronized = true;
        check_canaries(p);
        p->bound = false;
        return 0;
    } catch (Failure failure) { return failure.code; }
}
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--family', choices=['target', 'draft'], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert 'gmm_lane_sparse_msd' in str(root) and root.exists()
    receipt_path = root / 'prepare_receipt.json'
    prepare = json.loads(receipt_path.read_text())
    assert prepare['family'] == args.family and prepare['CPU_only'] and prepare['NPU_launches'] == 0
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == os.environ.get('ASCEND_VISIBLE_DEVICES') == ''
    for label, library, expected in [('loader_v2', LOADER / 'build/libgmm2_sparse_msd_loader_v2.so', FROZEN_SHA['loader_v2']),
                                     ('launch_v3', LAUNCH / 'build/libgmm2_launch_args_v3.so', FROZEN_SHA['launch_v3'])]:
        assert sha(library) == expected, 'Frozen ' + label + ' dependency changed'
    profile = KP.bind(args.family, (48 if args.family == 'target' else 16, 2304, 2560),
                      'original_CPU_transposed_INT8', 2304, 'k5_capacity_device:' + str(LINE))
    workspace_bytes = prepare['native_workspace_bytes']
    source, geometry, workspace, receipt = BF.generate(profile, workspace_bytes, root,
                                                       'new_geometry_kernel_compiler_metadata')
    assert source.count(PHYSICAL_NEEDLE) == 1, 'Physical declaration set changed'
    source = source.replace(PHYSICAL_NEEDLE, PHYSICAL_NEEDLE + ' && std::string(declared)!="4"')
    assert source.count(LABEL_NEEDLE) == 1, 'Mapping-stage label changed'
    source = source.replace(LABEL_NEEDLE, 'check("rtGetDeviceIndexByPhyId(expected)",')
    assert source.count(PROBE_NEEDLE) == 1, 'Probe struct layout changed'
    source = source.replace(PROBE_NEEDLE, PROBE_FIELDS)
    assert source.count('extern "C" int m1_probe_close(void **owner)') == 1
    source = source + PLANNER_ONLY + TIMED_ENTRIES
    bridge = root / 'bridge'
    assert not bridge.exists()
    bridge.mkdir()
    (bridge / 'probe_bridge.cpp').write_text(source)
    (bridge / 'capacity_geometry.h').write_text(geometry)
    (bridge / 'workspace_layout.h').write_text(workspace)
    command = ['g++', '-std=c++17', '-O2', '-shared', '-fPIC', str(bridge / 'probe_bridge.cpp'),
               str(LINE / 'capacity_adapter.cpp'),
               '-I', str(bridge), '-I', str(LINE), '-I', str(LOADER / 'source/runtime_loader_v2'),
               '-I', str(LAUNCH / 'source'), '-I', str(SDK / 'include'),
               '-I', str(SDK / 'aarch64-linux/pkg_inc/runtime'), '-I', str(SDK / 'aarch64-linux/pkg_inc'),
               '-L', str(FROZEN), '-lgmm2_sparse_msd_probe',
               '-L', str(LOADER / 'build'), '-lgmm2_sparse_msd_loader_v2',
               '-L', str(LAUNCH / 'build'), '-lgmm2_launch_args_v3',
               '-L', str(SDK / 'lib64'), '-lascendcl', '-lruntime', '-Wl,--no-undefined',
               '-Wl,-rpath,' + ':'.join(map(str, [FROZEN, LOADER / 'build', LAUNCH / 'build'])),
               '-o', str(root / f'libk5_capacity_{args.family}.so')]
    log = root / 'logs' / ('bridge_build_' + args.family + '.log')
    assert not log.exists()
    with log.open('wb') as output:
        result = subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
    built = root / f'libk5_capacity_{args.family}.so'
    record = {'schema': 'k5_capacity_bridge_build_v1', 'family': args.family, 'CPU_only': True,
              'NPU_launches': 0, 'command': command, 'exit_code': result.returncode, 'log': str(log),
              'profile': profile.to_dict(), 'native_workspace_bytes': workspace_bytes,
              'capacity_workspace_bytes': workspace_bytes + profile.frame_bytes,
              'bridge_generation_receipt': receipt,
              'physical_declaration_accepted': ['6', '2', '3', '7', '4'],
              'planner_only_entry': 'k5_probe_planner_only',
              'shared_bytes': built.stat().st_size if built.exists() else None,
              'shared_sha256': sha(built) if built.exists() else None,
              'source_sha256': {name: sha(bridge / name) for name in
                                ['probe_bridge.cpp', 'capacity_geometry.h', 'workspace_layout.h']},
              'adapter_sha256': sha(LINE / 'capacity_adapter.cpp'),
              'actual_numeric_passed': False, 'performance_receipt_granted': False,
              'formal_receipt_granted': False}
    (root / 'bridge_receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'family': args.family, 'exit_code': result.returncode,
                      'shared_sha256': record['shared_sha256'],
                      'capacity_workspace_bytes': record['capacity_workspace_bytes']}), flush=True)
    raise SystemExit(0 if result.returncode == 0 else 1)


if __name__ == '__main__':
    main()
