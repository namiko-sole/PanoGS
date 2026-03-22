#!/usr/bin/env bash
set -euo pipefail

# Batch wrapper for adaptive two-stage room splitting.
# Requires: conda env "PanoGS" and colmap_spliter/split_rooms.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPLIT_PY="${SCRIPT_DIR}/split_rooms.py"

if [[ ! -f "${SPLIT_PY}" ]]; then
  echo "[ERROR] Cannot find split script: ${SPLIT_PY}" >&2
  exit 1
fi

INPUT=""
INPUT_LIST=""
OUTPUT_ROOT=""
TARGET_ROOMS="0"
EXPORT_SUBMODELS="1"
SHARED_Q="0.70"
SIM_Q="0.85"
SHARED_FLOOR="5"
SHARED_CAP="200"
MIN_ROOM_SIZE="5"
MERGE_SCORE_THRESHOLD="0.03"
BOUNDARY_PURITY_THRESHOLD="0.65"
SPARSE_REL="sparse/0"
CONDA_ENV="PanoGS"

usage() {
  cat <<EOF
Usage:
  bash colmap_spliter/adaptive_split.sh --input <dataset_root> --output-root <output_root> [options]
  bash colmap_spliter/adaptive_split.sh --input-list <list.txt> --output-root <output_root> [options]

Required:
  --output-root <dir>            Output root for split results
  --input <dataset_root>         Single dataset root (contains sparse/0, images)
    or
  --input-list <list.txt>        Text file with one dataset root per line

Optional:
  --target-rooms <int>           Target room count, 0 disables hard target (default: 0)
  --no-export-submodels          Disable colmap submodel export
  --shared-quantile <float>      Adaptive shared-point quantile (default: 0.70)
  --sim-quantile <float>         Adaptive similarity quantile (default: 0.85)
  --shared-floor <int>           Lower bound for adaptive min shared points (default: 5)
  --shared-cap <int>             Upper bound for adaptive min shared points (default: 200)
  --min-room-size <int>          Merge tiny components before stage-2 (default: 5)
  --merge-score-threshold <f>    Stage-2 stop threshold when target-rooms=0 (default: 0.03)
  --boundary-purity-threshold <f>Boundary camera threshold (default: 0.65)
  --sparse <rel_path>            Relative sparse path under input root (default: sparse/0)
  --conda-env <name>             Conda env name (default: PanoGS)
  -h, --help                     Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"; shift 2 ;;
    --input-list)
      INPUT_LIST="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --target-rooms)
      TARGET_ROOMS="$2"; shift 2 ;;
    --no-export-submodels)
      EXPORT_SUBMODELS="0"; shift ;;
    --shared-quantile)
      SHARED_Q="$2"; shift 2 ;;
    --sim-quantile)
      SIM_Q="$2"; shift 2 ;;
    --shared-floor)
      SHARED_FLOOR="$2"; shift 2 ;;
    --shared-cap)
      SHARED_CAP="$2"; shift 2 ;;
    --min-room-size)
      MIN_ROOM_SIZE="$2"; shift 2 ;;
    --merge-score-threshold)
      MERGE_SCORE_THRESHOLD="$2"; shift 2 ;;
    --boundary-purity-threshold)
      BOUNDARY_PURITY_THRESHOLD="$2"; shift 2 ;;
    --sparse)
      SPARSE_REL="$2"; shift 2 ;;
    --conda-env)
      CONDA_ENV="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "[ERROR] --output-root is required" >&2
  usage
  exit 1
fi

if [[ -n "${INPUT}" && -n "${INPUT_LIST}" ]]; then
  echo "[ERROR] Use either --input or --input-list, not both" >&2
  exit 1
fi
if [[ -z "${INPUT}" && -z "${INPUT_LIST}" ]]; then
  echo "[ERROR] One of --input or --input-list is required" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

DATASETS=()
if [[ -n "${INPUT}" ]]; then
  DATASETS+=("${INPUT}")
else
  if [[ ! -f "${INPUT_LIST}" ]]; then
    echo "[ERROR] input list not found: ${INPUT_LIST}" >&2
    exit 1
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(echo "${line}" | xargs)"
    if [[ -n "${line}" ]]; then
      DATASETS+=("${line}")
    fi
  done < "${INPUT_LIST}"
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "[ERROR] No valid datasets found" >&2
  exit 1
fi

echo "[INFO] datasets: ${#DATASETS[@]}"
echo "[INFO] output root: ${OUTPUT_ROOT}"

for ds in "${DATASETS[@]}"; do
  if [[ ! -d "${ds}" ]]; then
    echo "[WARN] skip missing dataset: ${ds}" >&2
    continue
  fi
  if [[ ! -f "${ds}/${SPARSE_REL}/images.bin" ]]; then
    echo "[WARN] skip (missing images.bin): ${ds}/${SPARSE_REL}/images.bin" >&2
    continue
  fi

  ds_name="$(basename "${ds}")"
  out_dir="${OUTPUT_ROOT}/${ds_name}_auto_adaptive"

  cmd=(
    conda run -n "${CONDA_ENV}" python "${SPLIT_PY}"
    --input "${ds}"
    --sparse "${SPARSE_REL}"
    --output "${out_dir}"
    --auto-adaptive
    --target-rooms "${TARGET_ROOMS}"
    --shared-quantile "${SHARED_Q}"
    --sim-quantile "${SIM_Q}"
    --shared-floor "${SHARED_FLOOR}"
    --shared-cap "${SHARED_CAP}"
    --min-room-size "${MIN_ROOM_SIZE}"
    --merge-score-threshold "${MERGE_SCORE_THRESHOLD}"
    --boundary-purity-threshold "${BOUNDARY_PURITY_THRESHOLD}"
  )

  if [[ "${EXPORT_SUBMODELS}" == "0" ]]; then
    cmd+=(--no-export-submodels)
  fi

  echo "[RUN] ${ds}"
  echo "[OUT] ${out_dir}"
  "${cmd[@]}"

  summary="${out_dir}/room_split_summary.json"
  if [[ -f "${summary}" ]]; then
    echo "[OK] summary: ${summary}"
  else
    echo "[WARN] missing summary: ${summary}" >&2
  fi
  echo

done

echo "[DONE] adaptive split batch finished"
