# DeepSeek-V4.1-Flash A2 测试镜像 v3：把全部**已验证**的代码修补烘焙进镜像
#
# 用法（由 scripts/build_image.sh 自动调用，一般不用手写）：
#   docker build --build-arg BASE_IMAGE=<基础镜像> \
#                --build-arg ASCEND_PKG=/vllm-workspace/vllm-ascend/vllm_ascend \
#                --build-arg VLLM_ROOT=/vllm-workspace/vllm -t dsv41-a2:v8 .
#
# 注意：本镜像**不含模型权重**（273 GB，需现场准备）。
ARG BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler
FROM ${BASE_IMAGE}

# FROM 之前声明的 ARG 在 FROM 之后不可见，必须重新声明。
ARG BASE_IMAGE=quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler
ARG ASCEND_PKG=/vllm-workspace/vllm-ascend/vllm_ascend
ARG VLLM_ROOT=/vllm-workspace/vllm
ARG SKIP_PGO=0
ENV ASCEND_PKG=${ASCEND_PKG}
ENV VLLM_ROOT=${VLLM_ROOT}
ENV BASE_IMAGE_TAG=${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]
USER root

# ---------- 1) 已验证补丁（整文件覆盖 + 新增 sidecar）----------
COPY patches/files/engram_hbm.py            /tmp/bake/engram_hbm.py
COPY patches/files/engram_hash.py           /tmp/bake/engram_hash.py
COPY patches/files/engram_jit_kernel.py     /tmp/bake/engram_jit_kernel.py
COPY patches/files/engram_plan_kernel.py    /tmp/bake/engram_plan_kernel.py
COPY patches/files/engram_device_index.py   /tmp/bake/engram_device_index.py
COPY patches/files/engram_graph.py          /tmp/bake/engram_graph.py
COPY patches/files/engram_gate.py           /tmp/bake/engram_gate.py
COPY patches/files/model.py                 /tmp/bake/model.py
COPY patches/files/ascend_forward_context.py /tmp/bake/ascend_forward_context.py
COPY patches/files/dsa_v1.py                /tmp/bake/dsa_v1.py
COPY patches/files/indexer.py               /tmp/bake/indexer.py
COPY patches/files/token_dispatcher_moemask.py /tmp/bake/token_dispatcher.py
COPY patches/files/rope_dsv4.py             /tmp/bake/rope_dsv4.py
COPY patches/files/block_table.py           /tmp/bake/block_table.py
COPY patches/files/draft/                    /opt/dsv41/patches/draft/
COPY patches/files/token_dispatcher_moezero.py /opt/dsv41/patches/files/token_dispatcher_moezero.py
COPY patches/files/indexer.py               /opt/dsv41/patches/files/indexer.py

# ---------- 2) 落位 + 备份 + 逐文件编译校验 ----------
#
# ⚠️ 续行铁律（本文件踩过坑，别改回去）：
#   ① RUN 的续行链中，**除最后一行外每一行都必须以 `\` 结尾**；
#   ② **绝对不要在 `\` 前面或行内写 `#` 注释** —— Docker 先把续行拼成一整行再交给
#      shell，行内 `#` 会把**后面所有内容**都注释掉（包括还没执行的命令）；
#      而如果把 `\` 写在注释后面，`\` 本身也在注释里 ⇒ 续行失效，
#      RUN 指令在此处**提前结束**，后面的 `local`/`test` 会被当成 Dockerfile
#      指令解析并报 "unknown instruction"。
#   ③ 解释性文字请写在 RUN **外面**（像本段这样），或用 `echo` 输出。
#
# inst <src_in_tmp> <target_rel>       覆盖已有文件（先备份成 .a2orig）
# newf <src_in_tmp> <target_rel>       新增文件（无备份）
RUN set -euo pipefail; \
    inst() { \
      local tgt="${ASCEND_PKG}/$2"; \
      test -f "/tmp/bake/$1" || { echo "[build] MISSING payload $1"; exit 22; }; \
      test -f "$tgt"         || { echo "[build] MISSING target $tgt"; exit 21; }; \
      cp -f "$tgt" "$tgt.a2orig"; \
      cp -f "/tmp/bake/$1" "$tgt"; \
      python3 -m py_compile "$tgt"; \
      echo "[build] installed $2"; \
    }; \
    newf() { \
      local tgt="${ASCEND_PKG}/$2"; \
      cp -f "/tmp/bake/$1" "$tgt"; \
      python3 -m py_compile "$tgt"; \
      echo "[build] installed(new) $2"; \
    }; \
    inst engram_hbm.py                models/deepseek_v41/engram_hbm.py; \
    inst engram_hash.py               models/deepseek_v41/engram_hash.py; \
    inst engram_gate.py               models/deepseek_v41/engram_gate.py; \
    inst model.py                     models/deepseek_v41/model.py; \
    inst indexer.py                   models/deepseek_v41/indexer.py; \
    inst ascend_forward_context.py    ascend_forward_context.py; \
    inst dsa_v1.py                    attention/dsa_v1.py; \
    inst token_dispatcher.py          ops/fused_moe/token_dispatcher.py; \
    inst rope_dsv4.py                 ops/rope_dsv4.py; \
    inst block_table.py               worker/block_table.py; \
    newf engram_jit_kernel.py         models/deepseek_v41/engram_jit_kernel.py; \
    newf engram_plan_kernel.py        models/deepseek_v41/engram_plan_kernel.py; \
    newf engram_device_index.py       models/deepseek_v41/engram_device_index.py; \
    newf engram_graph.py              models/deepseek_v41/engram_graph.py; \
    rm -rf /tmp/bake

# ---------- 3) vLLM core：admission gate（prefill 不饿死 decode）----------
COPY patches/admission_gate.patch /opt/dsv41/admission_gate.patch
RUN set -euo pipefail; \
    cd "${VLLM_ROOT}"; \
    if git apply --check /opt/dsv41/admission_gate.patch 2>/dev/null; then \
      git apply /opt/dsv41/admission_gate.patch; echo "[build] admission gate applied"; \
    elif git apply --reverse --check /opt/dsv41/admission_gate.patch 2>/dev/null; then \
      echo "[build] admission gate already applied"; \
    else \
      echo "[build] WARNING: admission gate 无法应用（基础镜像版本可能不同）"; \
      echo "[build]          服务仍可运行，但请把 VLLM_ADMISSION_GATE=0 传下去"; \
      cp /opt/dsv41/admission_gate.patch /opt/dsv41/admission_gate.patch.NOT_APPLIED; \
    fi

# ---------- 4) 运行脚本 + PGO 产物 ----------
COPY scripts/serve_a2.sh  /opt/dsv41/scripts/serve_a2.sh
COPY scripts/serve_v2.sh  /opt/dsv41/scripts/serve_v2.sh
COPY scripts/run_test.sh  /opt/dsv41/scripts/run_test.sh
RUN chmod +x /opt/dsv41/scripts/*.sh; mkdir -p /opt/dsv41/results /opt/dsv41/pgo
COPY optim/pgo/ /opt/dsv41/pgo/

# ---------- 5) 构建指纹 ----------
RUN set -euo pipefail; \
    { echo "base_image=${BASE_IMAGE_TAG}"; \
      echo "ascend_pkg=${ASCEND_PKG}"; \
      echo "vllm_root=${VLLM_ROOT}"; \
      echo "built_at=$(date -Is)"; \
      echo "python=$(python3 -V 2>&1)"; \
      echo "pgo_python_md5=$(md5sum /opt/dsv41/pgo/python3 2>/dev/null | cut -d' ' -f1)"; \
      echo "pgo_libpython_md5=$(md5sum /opt/dsv41/pgo/libpython3.12.so.1.0 2>/dev/null | cut -d' ' -f1)"; \
      for f in \
        models/deepseek_v41/engram_hbm.py \
        models/deepseek_v41/engram_hash.py \
        models/deepseek_v41/engram_jit_kernel.py \
        models/deepseek_v41/engram_plan_kernel.py \
        models/deepseek_v41/engram_device_index.py \
        models/deepseek_v41/engram_graph.py \
        models/deepseek_v41/engram_gate.py \
        models/deepseek_v41/model.py \
        models/deepseek_v41/indexer.py \
        ascend_forward_context.py \
        attention/dsa_v1.py \
        ops/fused_moe/token_dispatcher.py \
        ops/rope_dsv4.py \
        worker/block_table.py ; do \
        echo "$(md5sum ${ASCEND_PKG}/${f} | cut -d' ' -f1)  ${f}"; \
      done; \
    } > /opt/dsv41/BUILD_INFO.txt; \
    cat /opt/dsv41/BUILD_INFO.txt

WORKDIR /workspace
