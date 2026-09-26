# DeepSeek-V4.1-Flash —— **A3 单机 8+8 PD 分离 + CED 形态** 工作镜像
#
# 这是什么：把本仓库里"已验证可跑"的那套 A3 CED-PD 部署形态，
#   烘焙成「官方基础镜像 + 我们的工作层」。
#   原先这些文件靠在 `serve_a2.sh` 里用 `-v` 挂载；本镜像把它们**装到真实路径**，
#   起服时可以完全不用 `-v`（`PATCH_MODE=baked`）。
#
# 与 GitHub 侧脚本的一致性：payload 清单见同目录 `PAYLOAD.md`（唯一事实源）；
#   两边都由那份清单派生，`verify_consistency.sh` 逐文件比对 md5。
#
# 用法（由 deploy/a3-ced-pd/build_image.sh 自动调用）：
#   docker build --build-arg BASE_IMAGE=<基础镜像> -t local/dsv41-a3-ced-pd:v1 .
#
# ⚠️ 本镜像**不含模型权重**（273 GB，需现场准备，用 MODEL=<路径> 传入）。
#
# ⚠️ 续行铁律（照抄 A2 Dockerfile 的教训，别改回去）：
#   ① RUN 的续行链里，除最后一行外每一行都必须以 `\` 结尾；
#   ② **绝对不要在 `\` 前面或行内写 `#` 注释** —— Docker 先把续行拼成一整行
#      再交给 shell，行内 `#` 会把后面所有内容都注释掉。
#      解释性文字写在 RUN **外面**，或用 echo 输出。

ARG BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3
FROM ${BASE_IMAGE}

# FROM 之前声明的 ARG 在 FROM 之后不可见，必须重新声明。
ARG BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-a3
ARG ASCEND_PKG=/vllm-workspace/vllm-ascend/vllm_ascend
ARG VLLM_ROOT=/vllm-workspace/vllm
ARG REPO_REV=unknown
ENV ASCEND_PKG=${ASCEND_PKG}
ENV VLLM_ROOT=${VLLM_ROOT}
ENV BASE_IMAGE_TAG=${BASE_IMAGE}
ENV DSV41_DEPLOY_FORM=a3-ced-pd

SHELL ["/bin/bash", "-lc"]
USER root

# ---------- 1) payload 落地到 /tmp/bake（由 build_payload.sh 组装）----------
COPY payload/ascend/ /tmp/bake/ascend/
COPY payload/ced/ /tmp/bake/ced/
COPY payload/draft/ /tmp/bake/draft/
COPY payload/guard/ /tmp/bake/guard/
COPY payload/patches/ /tmp/bake/patches/
COPY payload/scripts/ /tmp/bake/scripts/
# payload 清单也打进镜像：事后可以对着它核对"这层到底装了哪些文件"，
# 不需要回头找仓库（清单本身逐字节派生自仓库源文件）。
COPY payload/PAYLOAD.sha256 /opt/dsv41/deploy/PAYLOAD.sha256

# ---------- 2) vllm-ascend 补丁件落位 + 备份 + 逐文件编译校验 ----------
# 备份成 .cedpdorig（原图镜像里已有的同名文件才备份）。
# 逐文件 py_compile：语法错的补丁件必须在**构建期**就拦下来，不能留到起服。
RUN set -euo pipefail; \
    inst() { \
      local tgt="${ASCEND_PKG}/$2"; \
      test -f "/tmp/bake/ascend/$1" || { echo "[build] MISSING payload $1"; exit 22; }; \
      test -f "$tgt" || { echo "[build] MISSING target $tgt"; exit 21; }; \
      cp -f "$tgt" "$tgt.cedpdorig"; \
      cp -f "/tmp/bake/ascend/$1" "$tgt"; \
      python3 -m py_compile "$tgt"; \
      echo "[build] inst  $2"; \
    }; \
    newf() { \
      local tgt="${ASCEND_PKG}/$2"; \
      cp -f "/tmp/bake/ascend/$1" "$tgt"; \
      python3 -m py_compile "$tgt"; \
      echo "[build] newf  $2"; \
    }; \
    inst engram_hbm.py             models/deepseek_v41/engram_hbm.py; \
    inst engram_hash.py            models/deepseek_v41/engram_hash.py; \
    inst engram_gate.py            models/deepseek_v41/engram_gate.py; \
    inst model.py                  models/deepseek_v41/model.py; \
    inst indexer.py                models/deepseek_v41/indexer.py; \
    inst ascend_forward_context.py ascend_forward_context.py; \
    inst rope_dsv4.py              ops/rope_dsv4.py; \
    inst block_table.py            worker/block_table.py; \
    inst token_dispatcher.py       ops/fused_moe/token_dispatcher.py; \
    newf engram_jit_kernel.py      models/deepseek_v41/engram_jit_kernel.py; \
    newf engram_plan_kernel.py     models/deepseek_v41/engram_plan_kernel.py; \
    newf engram_device_index.py    models/deepseek_v41/engram_device_index.py; \
    newf engram_graph.py           models/deepseek_v41/engram_graph.py

# ---------- 3) CED 件落位（连接器 + dsa_v41）----------
RUN set -euo pipefail; \
    C=/tmp/bake/ced; \
    cp -f "$C/mooncake_hybrid_connector.py" \
          "${ASCEND_PKG}/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py"; \
    cp -f "$C/dsa_v41.py" "${ASCEND_PKG}/attention/dsa_v41.py"; \
    python3 -m py_compile "${ASCEND_PKG}/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py"; \
    python3 -m py_compile "${ASCEND_PKG}/attention/dsa_v41.py"; \
    echo "[build] CED connector + dsa_v41 installed"

# ---------- 4) draft 版暂存（PATCH_MODE=baked + DRAFT_GRAPH=1 时由启动器装到 live tree）----------
# serve_a2.sh 的 baked 分支会从 /opt/dsv41/patches/draft/ 拷这三个文件，
# 并做硬门断言（必须能在 dspark_proposer.py 里 grep 到 DSPARK_GRAPH_CAPTURE_METADATA）。
RUN set -euo pipefail; \
    mkdir -p /opt/dsv41/patches/draft /opt/dsv41/patches/files; \
    cp -f /tmp/bake/draft/dsa_v1.py            /opt/dsv41/patches/draft/; \
    cp -f /tmp/bake/draft/dspark_proposer.py   /opt/dsv41/patches/draft/; \
    cp -f /tmp/bake/draft/llm_base_proposer.py /opt/dsv41/patches/draft/; \
    cp -f /tmp/bake/ascend/indexer.py          /opt/dsv41/patches/files/; \
    cp -f /tmp/bake/ascend/token_dispatcher.py /opt/dsv41/patches/files/token_dispatcher_moemask.py; \
    ls -la /opt/dsv41/patches/draft/; \
    grep -c DSPARK_GRAPH_CAPTURE_METADATA /opt/dsv41/patches/draft/dspark_proposer.py

# ---------- 5) vLLM core：admission gate **构建期就打**（baked 模式不再现场打）----------
RUN set -euo pipefail; \
    cd "${VLLM_ROOT}"; \
    mkdir -p /opt/dsv41; \
    cp -f /tmp/bake/patches/admission_gate.patch /opt/dsv41/admission_gate.patch; \
    P=/opt/dsv41/admission_gate.patch; \
    if git apply --check "$P" 2>/dev/null; then \
      git apply "$P"; echo "[build] admission gate applied"; \
    elif git apply --reverse --check "$P" 2>/dev/null; then \
      echo "[build] admission gate already present in base"; \
    else \
      echo "[build] FATAL: admission gate 无法应用（基底 vLLM 版本不同）"; exit 23; \
    fi; \
    n=$(grep -c admission_gate vllm/v1/core/sched/scheduler.py || true); \
    test "${n:-0}" -ge 1 || { echo "[build] FATAL: live tree 里找不到 admission gate"; exit 24; }; \
    echo "[build] admission gate live hits=$n"

# ---------- 6) CED core 补丁、请求边界护栏与起服脚本到位 ----------
# 三个 CED 补丁**故意不在这里 apply**：它们各自带基线 sha256 硬门，
# 留到运行期打 → 基底版本不同会拒绝起服，而不是静默跑错版本。
RUN set -euo pipefail; \
    mkdir -p /opt/dsv41/scripts /opt/dsv41/results; \
    cp -f /tmp/bake/scripts/*.sh /opt/dsv41/scripts/; \
    chmod +x /opt/dsv41/scripts/*.sh; \
    cp -f /tmp/bake/patches/core_scheduler_replay.patch      /opt/dsv41/ced_scheduler_replay.patch; \
    cp -f /tmp/bake/patches/core_scheduler_prefill_hit.patch /opt/dsv41/ced_scheduler_prefill_hit.patch; \
    cp -f /tmp/bake/patches/core_model_runner_prompt_tail.patch /opt/dsv41/ced_runner_prompt_tail.patch; \
    for f in /opt/dsv41/ced_scheduler_replay.patch /opt/dsv41/ced_scheduler_prefill_hit.patch /opt/dsv41/ced_runner_prompt_tail.patch; do \
      test -s "$f" || { echo "[build] FATAL: missing $f"; exit 25; }; \
    done; \
    mkdir -p /opt/dsv41/guards; \
    cp -f /tmp/bake/guard/v41_decode_guard.py /opt/dsv41/guards/v41_decode_guard.py; \
    test -s /opt/dsv41/guards/v41_decode_guard.py || { echo "[build] FATAL: missing decode guard"; exit 26; }; \
    grep -q "def decode_guard" /opt/dsv41/guards/v41_decode_guard.py || { echo "[build] FATAL: decode guard 里没有 decode_guard"; exit 27; }; \
    ls -la /opt/dsv41/scripts/

# ---------- 7) 源码基线指纹（可复算：告诉别人这层是在哪个 commit 上切的）----------
RUN set -euo pipefail; \
    { echo "deploy_form=a3-ced-pd"; \
      echo "repo_rev=${REPO_REV}"; \
      echo "base_image=${BASE_IMAGE_TAG}"; \
      echo "ascend_pkg=${ASCEND_PKG}"; \
      echo "vllm_root=${VLLM_ROOT}"; \
      echo "built_at=$(date -Is)"; \
      echo "python=$(python3 -V 2>&1)"; \
      echo "--- repo_shipped (本层携带的补丁件) ---"; \
      for f in \
        models/deepseek_v41/engram_hbm.py \
        models/deepseek_v41/engram_hash.py \
        models/deepseek_v41/engram_gate.py \
        models/deepseek_v41/engram_jit_kernel.py \
        models/deepseek_v41/engram_plan_kernel.py \
        models/deepseek_v41/engram_device_index.py \
        models/deepseek_v41/engram_graph.py \
        models/deepseek_v41/model.py \
        models/deepseek_v41/indexer.py \
        ascend_forward_context.py \
        ops/rope_dsv4.py \
        worker/block_table.py \
        ops/fused_moe/token_dispatcher.py \
        attention/dsa_v41.py \
        distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py ; do \
        echo "$(md5sum ${ASCEND_PKG}/${f} | cut -d' ' -f1)  ${f}"; \
      done; \
      echo "--- core_patch_baseline (运行期补丁的 sha256 硬门) ---"; \
      echo "$(sha256sum ${VLLM_ROOT}/vllm/v1/core/sched/scheduler.py | cut -d' ' -f1)  vllm/v1/core/sched/scheduler.py"; \
      echo "$(sha256sum ${ASCEND_PKG}/worker/model_runner_v1.py | cut -d' ' -f1)  vllm_ascend/worker/model_runner_v1.py"; \
    } > /opt/dsv41/BUILD_INFO.txt; \
    cat /opt/dsv41/BUILD_INFO.txt

# ---------- 8) 构建期冒烟（不需要 NPU）----------
# ⚠️ 冒烟脚本从外部 COPY 进来，**不要**在 Dockerfile 里写 heredoc：
#    `RUN ...; \` + `python3 - <<'PY'` 会被 Dockerfile 解析器截断成
#    `unknown instruction: IMPORT`（已踩）。只有 BuildKit 的 heredoc 语法才支持。
COPY checks/build_smoke.py /opt/dsv41/deploy/build_smoke.py
RUN set -euo pipefail; \
    python3 /opt/dsv41/deploy/build_smoke.py

WORKDIR /workspace
