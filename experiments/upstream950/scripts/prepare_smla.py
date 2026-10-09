"""Clone installed CANN source locally, rename one op, patch only CSA addresses."""
import hashlib
import json
import shutil
from pathlib import Path

source = Path('/vllm-workspace/vllm-ascend/csrc')
root = Path('/work/build/smla')
tree = root/'csrc'
root.mkdir(parents=True,exist_ok=True)
if not tree.exists():
    def ignore(path, names):
        excluded = {'.git','__pycache__'}
        if Path(path) == source:
            excluded |= {'build','build_out','output'}
        return [name for name in names if name in excluded]
    shutil.copytree(source,tree,symlinks=False,
                    ignore=ignore)
original = source/'attention/sparse_flash_mla'
target = tree/'attention/up950_sparse_flash_mla'
assert not target.exists(), 'Private op already prepared; inspect its manifest before regenerating'
shutil.copytree(original,target,symlinks=False)
original_hashes = {str(p.relative_to(original)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in original.rglob('*') if p.is_file()}
# A private, unmodified control isolates source/toolchain/wrapper differences.
control = tree/'attention/up950_base_sparse_flash_mla'
assert not control.exists()
shutil.copytree(original,control,symlinks=False)
for path in sorted(control.rglob('*'),key=lambda p:len(p.parts),reverse=True):
    if path.is_file():
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            continue
        content = content.replace('SparseFlashMla','Up950BaseSparseFlashMla')
        content = content.replace('sparse_flash_mla','up950_base_sparse_flash_mla')
        content = content.replace('SMLA','UP950BASE_SMLA')
        content = content.replace('SPARSE_FLASH_MLA','UP950BASE_SPARSE_FLASH_MLA')
        content = content.replace('sparse_mla_checker','up950_base_sparse_mla_checker')
        content = content.replace('GetOptionalStorageShape','Up950BaseGetOptionalStorageShape')
        content = content.replace('SPARSE_MLA_COMMON_CHECKER','UP950BASE_COMMON_CHECKER')
        content = content.replace('add_sparse_mla_common_checker_sources','add_up950_base_common_checker_sources')
        if path.name.endswith('torch_adpt.h'):
            content = content.replace('namespace vllm_ascend {',
                                      'namespace up950_base_adapter {\nusing namespace vllm_ascend;')
        path.write_text(content)
    if 'sparse_flash_mla' in path.name:
        path.rename(path.with_name(path.name.replace('sparse_flash_mla','up950_base_sparse_flash_mla')))
for path in sorted(target.rglob('*'),key=lambda p:len(p.parts),reverse=True):
    if path.is_file():
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            continue
        content = content.replace('SparseFlashMla','Up950SparseFlashMla')
        content = content.replace('sparse_flash_mla','up950_sparse_flash_mla')
        content = content.replace('SMLA','UP950PREFETCH_SMLA')
        content = content.replace('SPARSE_FLASH_MLA','UP950PREFETCH_SPARSE_FLASH_MLA')
        content = content.replace('sparse_mla_checker','up950_prefetch_sparse_mla_checker')
        content = content.replace('GetOptionalStorageShape','Up950GetOptionalStorageShape')
        content = content.replace('SPARSE_MLA_COMMON_CHECKER','UP950PREFETCH_COMMON_CHECKER')
        content = content.replace('add_sparse_mla_common_checker_sources','add_up950_prefetch_common_checker_sources')
        if path.name.endswith('torch_adpt.h'):
            content = content.replace('namespace vllm_ascend {',
                                      'namespace up950_prefetch_adapter {\nusing namespace vllm_ascend;')
        path.write_text(content)
    if 'sparse_flash_mla' in path.name:
        path.rename(path.with_name(path.name.replace('sparse_flash_mla','up950_sparse_flash_mla')))

path = target/'op_kernel/arch22/up950_sparse_flash_mla_csa_block_vector.h'
text = path.read_text()
assert text.count('    uint32_t mergeMte3Idx = 0;') == 1
text = text.replace('    uint32_t mergeMte3Idx = 0;', '''    // The first 384 int32 slots of v0ValidSizeBuff are unused by the
    // original kernel; Vec2 broadcast scratch begins exactly at slot 384.
    // Cache only the current AIV's half-tile. No extra UB allocation.
    static constexpr uint32_t ADDRESS_CACHE_CAPACITY = 384;
    bool indexCacheReady_ = false;
    bool pageCacheReady_ = false;
    int64_t cachedIndexStart_ = 0;
    int64_t cachedIndexCount_ = 0;
    uint32_t cachedPageOffset_ = 0;
    uint32_t cachedPageCount_ = 0;
    uint32_t mergeMte3Idx = 0;''')
old = '''    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>(cmpS2Offset % constInfo.sparseBlockSize);'''
new = '''    int32_t logicalBlock =
        (indexCacheReady_ && topkGmIdx >= cachedIndexStart_ &&
         topkGmIdx < cachedIndexStart_ + cachedIndexCount_) ?
            v0ValidSizeUb_.GetValue(static_cast<uint32_t>(topkGmIdx - cachedIndexStart_)) :
            topkGm_.GetValue(topkGmBaseOffset + topkGmIdx);
    realS2Idx = static_cast<int64_t>(logicalBlock) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>(cmpS2Offset % constInfo.sparseBlockSize);'''
assert text.count(old) == 1
text = text.replace(old,new)
old = '''            cmpBlockTableGm_.GetValue(runInfo.bIdx * constInfo.cmpMaxBlockNumPerBatch + blkTableIdx) *'''
new = '''            (pageCacheReady_ && blkTableIdx < cachedPageCount_ ?
                v0ValidSizeUb_.GetValue(cachedPageOffset_ + static_cast<uint32_t>(blkTableIdx)) :
                cmpBlockTableGm_.GetValue(runInfo.bIdx * constInfo.cmpMaxBlockNumPerBatch + blkTableIdx)) *'''
assert text.count(old) == 1
text = text.replace(old,new)
old = '''    // 处理两个基本块
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;'''
new = '''    // Preload the exact local index interval. Every address is still
    // checked against the original visible-length and sparse-count guards.
    indexCacheReady_ = false;
    pageCacheReady_ = false;
    cachedIndexStart_ = s2GmStartOffset / constInfo.sparseBlockSize;
    int64_t indexEnd = CeilDiv(s2GmLimit, static_cast<int64_t>(constInfo.sparseBlockSize));
    if (indexEnd > constInfo.sparseBlockCount) {
        indexEnd = constInfo.sparseBlockCount;
    }
    cachedIndexCount_ = indexEnd - cachedIndexStart_;
    if (cachedIndexCount_ > 0 && cachedIndexCount_ <= ADDRESS_CACHE_CAPACITY) {
        uint32_t alignedIndices = static_cast<uint32_t>(UP950PREFETCH_SMLAAlign(cachedIndexCount_, 8));
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(cachedIndexCount_) * sizeof(int32_t);
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPadExtParams<int32_t> pad{true, 0,
            static_cast<uint8_t>(alignedIndices - cachedIndexCount_), -1};
        DataCopyPad(v0ValidSizeUb_, topkGm_[topkGmBaseOffset + cachedIndexStart_], params, pad);
        indexCacheReady_ = true;
        if constexpr (KV_LAYOUT_T == UP950PREFETCH_SMLA_LAYOUT::PA_BBND) {
            cachedPageOffset_ = alignedIndices;
            cachedPageCount_ = constInfo.cmpMaxBlockNumPerBatch;
            uint32_t alignedPages = static_cast<uint32_t>(UP950PREFETCH_SMLAAlign(cachedPageCount_, 8));
            if (cachedPageCount_ > 0 && alignedIndices + alignedPages <= ADDRESS_CACHE_CAPACITY) {
                params.blockLen = cachedPageCount_ * sizeof(int32_t);
                pad.rightPadding = static_cast<uint8_t>(alignedPages - cachedPageCount_);
                DataCopyPad(v0ValidSizeUb_[cachedPageOffset_],
                            cmpBlockTableGm_[runInfo.bIdx * constInfo.cmpMaxBlockNumPerBatch], params, pad);
                pageCacheReady_ = true;
            }
        }
        event_t mte2ToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
        WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
    }
    // 处理两个基本块
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;'''
assert text.count(old) == 1
text = text.replace(old,new)
path.write_text(text)
manifest = {'original_root':str(original),'original_files':original_hashes,
            'private_op':'Up950SparseFlashMla','private_dir':str(target),
            'control_op':'Up950BaseSparseFlashMla','control_dir':str(control),
            'algorithm_patch':'current-AIV index and optional page-table UB preload',
            'extra_ub_bytes':0,'reserved_scratch_bytes':384*4,
            'modified_header_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest | {'original_files':len(original_hashes)}))
