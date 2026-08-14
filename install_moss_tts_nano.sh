#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "Install failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

ROOT_DIR="${AI_OV_ROOT:-$(pwd)}"
ENV_DIR="$ROOT_DIR/.runtime/moss_tts_env"
SRC_DIR="$ROOT_DIR/.runtime/src/MOSS-TTS-Nano"
MODEL_DIR="$ROOT_DIR/models/tts"
OUTPUT_DIR="$ROOT_DIR/.runtime/tts"
SERVE_SCRIPT="$ROOT_DIR/.runtime/moss_tts_serve.sh"
WEB_STREAMING_PATCH="$ROOT_DIR/patches/moss-tts-nano-web-streaming-voice.patch"
YANGMI_PROMPT_AUDIO="$SRC_DIR/assets/audio/zh_11.wav"
CPU_THREADS="${MOSS_CPU_THREADS:-4}"
PORT="${MOSS_TTS_PORT:-18083}"
RUN_SMOKE="${RUN_SMOKE:-1}"
INSTALL_WETEXT="${INSTALL_WETEXT:-1}"

log() {
  printf '[moss-install] %s\n' "$*"
}

die() {
  printf '[moss-install] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required but was not found in PATH"
}

require_path() {
  local path="$1"
  local message="$2"
  [ -e "$path" ] || die "$message: $path"
}

apply_web_streaming_patch() {
  if git -C "$SRC_DIR" apply --check "$WEB_STREAMING_PATCH"; then
    git -C "$SRC_DIR" apply "$WEB_STREAMING_PATCH"
    log "Applied built-in voice Web streaming patch"
    return
  fi
  if git -C "$SRC_DIR" apply --reverse --check "$WEB_STREAMING_PATCH"; then
    log "Built-in voice Web streaming patch is already applied"
    return
  fi
  die "MOSS Web streaming patch does not apply cleanly: $WEB_STREAMING_PATCH"
}

log "Project root: $ROOT_DIR"

require_command git
require_command python3
require_path "$WEB_STREAMING_PATCH" "Missing repository-owned MOSS Web streaming patch"

require_path "$MODEL_DIR/MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json" \
  "Missing MOSS-TTS-Nano ONNX model manifest"
require_path "$MODEL_DIR/MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json" \
  "Missing MOSS-Audio-Tokenizer-Nano ONNX codec metadata"

mkdir -p "$ROOT_DIR/.runtime/src" "$OUTPUT_DIR"

if [ ! -d "$SRC_DIR/.git" ]; then
  log "Cloning OpenMOSS/MOSS-TTS-Nano into $SRC_DIR"
  git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS-Nano.git "$SRC_DIR"
else
  log "Updating existing MOSS source tree at $SRC_DIR"
  if git -C "$SRC_DIR" diff --quiet && git -C "$SRC_DIR" diff --cached --quiet; then
    git -C "$SRC_DIR" pull --ff-only
  else
    log "Skipping MOSS source update because tracked local changes are present"
  fi
fi

apply_web_streaming_patch

log "Creating/updating Python venv at $ENV_DIR"
python3 -m venv "$ENV_DIR"
PY="$ENV_DIR/bin/python"
PIP=("$PY" -m pip)

log "Upgrading pip tooling"
"${PIP[@]}" install -U pip setuptools wheel

log "Installing CPU-only torch and torchaudio"
"${PIP[@]}" install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.7.0" \
  "torchaudio==2.7.0"

log "Installing MOSS ONNX runtime dependencies"
"${PIP[@]}" install \
  "numpy>=1.24" \
  "fastapi>=0.110.0" \
  "python-multipart>=0.0.9" \
  "sentencepiece>=0.1.99" \
  "transformers==4.57.1" \
  "uvicorn>=0.29.0" \
  "onnxruntime>=1.20.0" \
  "huggingface_hub>=0.23.0" \
  "soundfile"

if [ "$INSTALL_WETEXT" = "1" ]; then
  log "Installing WeTextProcessing for the official web server warmup path"
  if ! "${PIP[@]}" install "WeTextProcessing>=1.0.4.1"; then
    log "Direct WeTextProcessing install failed; trying pynini first, then retrying"
    "${PIP[@]}" install "pynini==2.1.6.post1"
    "${PIP[@]}" install "WeTextProcessing>=1.0.4.1"
  fi
else
  log "Skipping WeTextProcessing because INSTALL_WETEXT=$INSTALL_WETEXT"
  log "Note: one-shot ONNX generation works without WeText, but official serve warmup may fail."
fi

log "Installing MOSS-TTS-Nano CLI from source without touching qwen35_env"
"${PIP[@]}" install -e "$SRC_DIR" --no-deps

log "Verifying installed packages"
"$PY" - <<'PY'
import onnxruntime
import torch
import torchaudio
import transformers

print("onnxruntime", onnxruntime.__version__)
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("transformers", transformers.__version__)
PY

"$ENV_DIR/bin/moss-tts-nano" --help >/dev/null

log "Writing service helper: $SERVE_SCRIPT"
cat > "$SERVE_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="$ENV_DIR"
MODEL_DIR="\${MOSS_MODEL_DIR:-$MODEL_DIR}"
OUTPUT_DIR="\${MOSS_OUTPUT_DIR:-$OUTPUT_DIR}"
CPU_THREADS="\${MOSS_CPU_THREADS:-$CPU_THREADS}"
PORT="\${MOSS_TTS_PORT:-$PORT}"

exec "\$ENV_DIR/bin/moss-tts-nano" serve \\
  --backend onnx \\
  --onnx-model-dir "\$MODEL_DIR" \\
  --output-dir "\$OUTPUT_DIR" \\
  --host 127.0.0.1 \\
  --port "\$PORT" \\
  --execution-provider cpu \\
  --cpu-threads "\$CPU_THREADS" \\
  --max-new-frames 375
EOF
chmod +x "$SERVE_SCRIPT"

if [ "$RUN_SMOKE" = "1" ]; then
  log "Running ONNX CPU smoke generation"
  require_path "$YANGMI_PROMPT_AUDIO" "Missing MOSS demo Yangmi prompt audio"
  "$ENV_DIR/bin/moss-tts-nano" generate \
    --backend onnx \
    --onnx-model-dir "$MODEL_DIR" \
    --voice Xiaoyu \
    --prompt-speech "$YANGMI_PROMPT_AUDIO" \
    --text "机械臂已停止，请保持安全距离。" \
    --output "$OUTPUT_DIR/moss_nano_smoke.wav" \
    --execution-provider cpu \
    --cpu-threads "$CPU_THREADS" \
    --sample-mode fixed \
    --realtime-streaming-decode 1 \
    --max-new-frames 180

  log "Smoke WAV: $OUTPUT_DIR/moss_nano_smoke.wav"
else
  log "Skipping smoke generation because RUN_SMOKE=$RUN_SMOKE"
fi

log "MOSS-TTS-Nano ONNX CPU environment is ready"
log "Start server: $SERVE_SCRIPT"
log "One-shot example:"
log "$ENV_DIR/bin/moss-tts-nano generate --backend onnx --onnx-model-dir $MODEL_DIR --voice Xiaoyu --prompt-speech $YANGMI_PROMPT_AUDIO --realtime-streaming-decode 1 --text '机械臂已停止，请保持安全距离。' --output $OUTPUT_DIR/test.wav"
