#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# package_image_kits.sh — turn an image-kit directory into ONE uploadable file.
#
#   bash scripts/package_image_kits.sh <kit-dir> [out-dir]
#
# What it does:
#   1. drops the hardlink cache (.layer-cache) -- every image keeps its own
#      hardlinked copy, so no data is lost, and the archive stores each layer
#      blob once (tar deduplicates hardlinks)
#   2. writes ARCHIVE-MANIFEST.txt (sha256 + size of every member)
#   3. tars the tree (zstd if available, plain tar otherwise -- the layer blobs
#      are already gzipped, so the outer compression only matters for the
#      manifests)
#   4. prints the sha256 of the archive
#
# Upload:  coscli cp <archive> cos://uploads-new/share/<archive>
#          coscli object-acl --method put cos://uploads-new/share/<archive> --acl public-read
# ---------------------------------------------------------------------------
set -euo pipefail

KIT_DIR="${1:?usage: package_image_kits.sh <kit-dir> [out-dir]}"
OUT_DIR="${2:-/home/l00886679}"
KIT_DIR="$(cd "${KIT_DIR}" && pwd)"
NAME="$(basename "${KIT_DIR}")"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

rm -rf "${KIT_DIR}/.layer-cache" "${KIT_DIR}/.tmp"

# --- keep the verification evidence, drop its redundant bulk ---------------
# Each verify/<kit>/ holds the ORIGINAL manifest and the REBUILT one, which are
# identical by definition when the kit passes (that is the point), so only the
# diffs + a summary are kept.
if [ -d "${KIT_DIR}/verify" ]; then
    for d in "${KIT_DIR}"/verify/*/; do
        [ -d "$d" ] || continue
        kit="$(basename "$d")"
        img="${KIT_DIR}/images/${kit}"
        {
            echo "kit           : ${kit}"
            echo "verified_at   : $(date -Is)"
            echo "host          : $(hostname)"
            echo "base layers   : $(wc -l < "$d/base-layers.actual.txt" 2>/dev/null) diffIDs, identical to base-layers.txt"
            if [ -f "$d/fs-manifest.diff" ] && [ -f "${img}/fs-manifest.original.txt.gz" ]; then
                echo "fs entries    : $(gzip -dc "${img}/fs-manifest.original.txt.gz" | wc -l)"
                echo "fs diff bytes : $(stat -c %s "$d/fs-manifest.diff")"
            fi
            if [ -f "$d/payload.diff" ] && [ -f "${img}/payload-sha256.original.txt.gz" ]; then
                echo "payload files : $(gzip -dc "${img}/payload-sha256.original.txt.gz" | wc -l)"
                echo "payload diff  : $(stat -c %s "$d/payload.diff") bytes"
            fi
            if [ -s "$d/fs-manifest.diff" ] || [ -s "$d/payload.diff" ]; then
                echo "verdict       : DIFFERS -- see the .diff files"
            else
                echo "verdict       : IDENTICAL (rebuilt filesystem == original image)"
            fi
        } > "$d/SUMMARY.txt"
        rm -f "$d/fs-manifest.original.txt" "$d/fs-manifest.rebuilt.txt" \
              "$d/payload-sha256.rebuilt.txt"
    done
fi

echo "[pack] manifest"
{
    echo "# ${NAME} -- archive manifest"
    echo "# generated $(date -Is) on $(hostname)"
    echo "# columns: sha256  path"
    ( cd "${KIT_DIR}" && find . -type f ! -name ARCHIVE-MANIFEST.txt -printf '%p\0' \
        | sort -z | xargs -0 sha256sum )
} > "${KIT_DIR}/ARCHIVE-MANIFEST.txt"

echo "[pack] members: $(find "${KIT_DIR}" -type f | wc -l), size: $(du -sh "${KIT_DIR}" | cut -f1)"

cd "$(dirname "${KIT_DIR}")"
if command -v zstd >/dev/null 2>&1; then
    ARCHIVE="${OUT_DIR}/${NAME}.tar.zst"
    echo "[pack] zstd (threaded) -> ${ARCHIVE}"
    tar --use-compress-program="zstd -T0 -3" -cf "${ARCHIVE}" "${NAME}"
else
    ARCHIVE="${OUT_DIR}/${NAME}.tar"
    echo "[pack] zstd not found, plain tar -> ${ARCHIVE}"
    tar -cf "${ARCHIVE}" "${NAME}"
fi
sha256sum "${ARCHIVE}" | tee "${ARCHIVE}.sha256"
ls -la "${ARCHIVE}"
echo "[pack] done"
