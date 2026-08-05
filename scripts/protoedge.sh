#!/usr/bin/env bash
# =============================================================================
# edge_simulate.sh — Jetson Orin Nano edge constraint simulator
#
# Restricts your Ubuntu + NVIDIA machine to match Jetson Orin Nano hardware:
#   CPU  : 4 cores @ ~1.5 GHz  (Orin Nano has 6 Cortex-A78AE, we use 4)
#   RAM  : 4 GB soft limit      (Orin Nano has 8 GB shared, ~4 GB for app)
#   GPU  : 10 W power cap       (Orin Nano 10W TDP mode)
#   GPU  : FP16 forced via env  (TRT FP16 default on Orin)
#
# Usage:
#   chmod +x edge_simulate.sh
#   ./edge_simulate.sh [--profile orin_nano|orin_nx|jetson_nano] [--dry-run]
#
# Requirements: taskset, nvidia-smi (with admin rights for power cap),
#               systemd (for cgroup RAM limit), python3
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PROFILE="orin_nano"
DRY_RUN=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/protoedge.py"
LOG_DIR="${SCRIPT_DIR}/edge_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --script)  TARGET_SCRIPT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Device profiles ────────────────────────────────────────────────────────────
# Each profile defines: CPU_CORES  RAM_MB  GPU_POWER_W  LABEL
declare -A PROFILES_CORES=( [orin_nano]=4   [orin_nx]=8    [jetson_nano]=4  )
declare -A PROFILES_RAM=(   [orin_nano]=4096 [orin_nx]=8192 [jetson_nano]=2048 )
declare -A PROFILES_POWER=( [orin_nano]=10  [orin_nx]=20   [jetson_nano]=10 )
declare -A PROFILES_LABEL=( [orin_nano]="Jetson Orin Nano (10W)" [orin_nx]="Jetson Orin NX (20W)" [jetson_nano]="Jetson Nano (10W)" )

CPU_CORES=${PROFILES_CORES[$PROFILE]}
RAM_MB=${PROFILES_RAM[$PROFILE]}
GPU_POWER_W=${PROFILES_POWER[$PROFILE]}
LABEL=${PROFILES_LABEL[$PROFILE]}

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'
CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'

info()  { echo -e "${CYN}[edge]${RST} $*"; }
ok()    { echo -e "${GRN}[ok]${RST}   $*"; }
warn()  { echo -e "${YEL}[warn]${RST} $*"; }
error() { echo -e "${RED}[err]${RST}  $*" >&2; }
sep()   { echo -e "${BLD}────────────────────────────────────────────────────${RST}"; }

# ── Helpers ────────────────────────────────────────────────────────────────────

get_total_cores() { nproc; }

# Build a CPU mask string "0-N" capped at available cores
build_cpu_mask() {
    local requested=$1
    local available
    available=$(get_total_cores)
    local use=$(( requested < available ? requested : available ))
    if [[ $use -le 1 ]]; then
        echo "0"
    else
        echo "0-$(( use - 1 ))"
    fi
}

# Cap GPU power via nvidia-smi (requires sudo or persistence mode enabled)
set_gpu_power_limit() {
    local watts=$1
    if ! command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi not found — skipping GPU power cap"
        return
    fi

    GPU_ID=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1)
    CURRENT_LIMIT=$(nvidia-smi -i "$GPU_ID" --query-gpu=power.limit --format=csv,noheader,nounits | tr -d ' ')
    DEFAULT_LIMIT=$(nvidia-smi -i "$GPU_ID" --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
    MIN_LIMIT=$(nvidia-smi -i "$GPU_ID" --query-gpu=power.min_limit --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo "10")

    info "GPU current limit: ${CURRENT_LIMIT}W | default: ${DEFAULT_LIMIT}W | requested cap: ${watts}W"

    # Clamp to minimum supported
    local effective_watts=$watts
    if (( $(echo "$watts < $MIN_LIMIT" | bc -l) )); then
        warn "Requested ${watts}W is below GPU minimum (${MIN_LIMIT}W). Using ${MIN_LIMIT}W instead."
        effective_watts=$MIN_LIMIT
    fi

    if sudo nvidia-smi -i "$GPU_ID" -pl "$effective_watts" &>/dev/null; then
        ok "GPU power capped to ${effective_watts}W (simulating ${LABEL})"
        POWER_CAP_SET=true
    else
        warn "Could not set GPU power limit (needs sudo). Run: sudo nvidia-smi -pl ${effective_watts}"
        warn "Continuing without GPU power cap — inference times will be faster than real Jetson."
        POWER_CAP_SET=false
    fi
}

# Restore GPU power limit to default on exit
restore_gpu_power() {
    if [[ "${POWER_CAP_SET:-false}" == true ]] && command -v nvidia-smi &>/dev/null; then
        GPU_ID=$(nvidia-smi --query-gpu=index --format=csv,noheader | head -1)
        DEFAULT=$(nvidia-smi -i "$GPU_ID" --query-gpu=power.default_limit --format=csv,noheader,nounits | tr -d ' ')
        sudo nvidia-smi -i "$GPU_ID" -pl "$DEFAULT" &>/dev/null && ok "GPU power limit restored to ${DEFAULT}W"
    fi
}
trap restore_gpu_power EXIT

# ── Pre-flight checks ──────────────────────────────────────────────────────────
sep
echo -e "${BLD}  Edge Device Simulation: ${LABEL}${RST}"
sep
info "Profile   : ${PROFILE}"
info "CPU cores : ${CPU_CORES} (of $(get_total_cores) available)"
info "RAM limit : ${RAM_MB} MB"
info "GPU power : ${GPU_POWER_W} W"
info "Script    : ${TARGET_SCRIPT}"
sep

if [[ ! -f "$TARGET_SCRIPT" ]]; then
    error "Target script not found: ${TARGET_SCRIPT}"
    error "Place proto_net_edge.py in the same directory, or pass --script <path>"
    exit 1
fi

if $DRY_RUN; then
    warn "DRY RUN — showing what would be executed, not running."
fi

mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/edge_run_${PROFILE}_${TIMESTAMP}.log"

# ── Build CPU affinity mask ───────────────────────────────────────────────────
CPU_MASK=$(build_cpu_mask "$CPU_CORES")
info "CPU affinity mask: ${CPU_MASK}"

# ── Apply GPU power cap ────────────────────────────────────────────────────────
POWER_CAP_SET=false
if ! $DRY_RUN; then
    set_gpu_power_limit "$GPU_POWER_W"
fi

# ── Set environment variables to simulate edge constraints ────────────────────
# OMP/MKL threads match CPU core count
# PYTORCH_CUDA_ALLOC_CONF mimics the smaller unified memory pool on Orin
export OMP_NUM_THREADS=$CPU_CORES
export MKL_NUM_THREADS=$CPU_CORES
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8"
export CUDA_VISIBLE_DEVICES=0        # single GPU only (Orin has 1)
export EDGE_PROFILE="$PROFILE"       # read by proto_net_edge.py
export EDGE_RAM_MB="$RAM_MB"
export EDGE_GPU_POWER_W="$GPU_POWER_W"

# ── Build the final command ────────────────────────────────────────────────────
# systemd-run  → cgroup with RAM hard limit (MemoryMax)
# taskset      → CPU core pinning
# python3      → your script

RAM_BYTES=$(( RAM_MB * 1024 * 1024 ))

CMD_CORE="taskset -c ${CPU_MASK} python3 ${TARGET_SCRIPT}"

# systemd-run wraps everything in a transient cgroup slice
# --scope           = run in calling shell's session (no service unit needed)
# MemoryMax         = hard RAM ceiling (OOM-kills if exceeded, like real edge)
# MemorySwapMax=0   = no swap (Jetson has no swap by default)
# CPUQuota          = optionally throttle CPU speed (100% of N cores)
CMD_FULL="systemd-run \
    --scope \
    --user \
    -p MemoryMax=${RAM_BYTES} \
    -p MemorySwapMax=0 \
    -p CPUQuota=$(( CPU_CORES * 100 ))% \
    -- ${CMD_CORE}"

sep
info "Final command:"
echo "  ${CMD_FULL}"
sep

if $DRY_RUN; then
    warn "Dry run complete. Remove --dry-run to execute."
    exit 0
fi

# ── Snapshot system state before run ──────────────────────────────────────────
{
    echo "=== EDGE SIMULATION RUN ==="
    echo "Timestamp : ${TIMESTAMP}"
    echo "Profile   : ${PROFILE} | ${LABEL}"
    echo "CPU mask  : ${CPU_MASK} (${CPU_CORES} cores)"
    echo "RAM limit : ${RAM_MB} MB"
    echo "GPU power : ${GPU_POWER_W} W (cap set: ${POWER_CAP_SET})"
    echo "OMP/MKL   : ${OMP_NUM_THREADS} threads"
    echo "ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
    echo ""
    echo "=== nvidia-smi snapshot ==="
    nvidia-smi 2>/dev/null || echo "nvidia-smi unavailable"
    echo ""
    echo "=== run output ==="
} | tee "$LOGFILE"

# ── Execute ───────────────────────────────────────────────────────────────────
ok "Starting edge-simulated run... (logs → ${LOGFILE})"
sep

eval "$CMD_FULL" 2>&1 | tee -a "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]}
sep
if [[ $EXIT_CODE -eq 0 ]]; then
    ok "Run complete. Full log saved to: ${LOGFILE}"
else
    error "Script exited with code ${EXIT_CODE}. Check log: ${LOGFILE}"
fi

exit $EXIT_CODE