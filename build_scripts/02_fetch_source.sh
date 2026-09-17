#!/bin/bash
# 下载 CPython 3.12.13 源码（容器内 /work/src）
# 主源：华为云镜像（国内快）；交叉校验源：阿里云镜像
# 期望 sha256（官方 3.12.13 tarball）在下载后由两镜像互相印证
set -euo pipefail
cd /work/src

URL_HUAWEI="https://mirrors.huaweicloud.com/python/3.12.13/Python-3.12.13.tgz"
URL_ALIYUN="https://mirrors.aliyun.com/python-release/source/Python-3.12.13.tgz"

# 之前 python.org 的残缺下载，直接丢弃（避免混淆）
rm -f Python-3.12.13.tgz.part /work/src/Python-3.12.13.tgz.tmp

echo "[1/4] 华为云下载（curl -C - 支持断点续传）"
for attempt in 1 2 3; do
    if curl -fSL -C - --retry 3 --retry-delay 2 --connect-timeout 15 \
            -o Python-3.12.13.tgz "$URL_HUAWEI"; then
        break
    fi
    echo "  第 $attempt 次失败，重试 ..."
    sleep 2
done

echo "[2/4] 完整性校验"
ls -l Python-3.12.13.tgz
SIZE=$(stat -c %s Python-3.12.13.tgz)
echo "size_bytes=$SIZE"
# 期望 27262093 字节（华为云 Content-Length），容忍 0 差异
if [ "$SIZE" != "27262093" ]; then
    echo "!! 大小与镜像 Content-Length 不符，中止"
    exit 1
fi
tar -tzf Python-3.12.13.tgz > /dev/null && echo "tar OK"

echo "[3/4] 交叉校验 sha256（阿里云同文件）"
curl -fSL -C - --retry 3 --connect-timeout 15 -o /tmp/Python-3.12.13.aliyun.tgz "$URL_ALIYUN"
sha256sum Python-3.12.13.tgz /tmp/Python-3.12.13.aliyun.tgz | tee /work/logs/source_sha256.txt
if [ "$(sha256sum < Python-3.12.13.tgz | cut -d' ' -f1)" != "$(sha256sum < /tmp/Python-3.12.13.aliyun.tgz | cut -d' ' -f1)" ]; then
    echo "!! 两源 sha256 不一致，中止"
    exit 1
fi
echo "sha256 一致 -> OK"
rm -f /tmp/Python-3.12.13.aliyun.tgz

echo "[4/4] 解压"
rm -rf cpython-3.12.13
tar xzf Python-3.12.13.tgz
mv Python-3.12.13 cpython-3.12.13
ls -ld /work/src/cpython-3.12.13
grep -E '^#define PY_(MAJOR|MINOR|MICRO)_VERSION|^#define PY_RELEASE_LEVEL|^#define PY_RELEASE_SERIAL' /work/src/cpython-3.12.13/Include/patchlevel.h
