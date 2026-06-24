"""Modal vLLM OpenAI-compatible endpoint for the context-budget A13 sweep.

Adapted from inference-release-lab/serving/vllm_modal.py. The key difference:
the model name is baked into the image env at deploy time so it survives
Modal's re-import of this module *inside the container* — the upstream script
read the model from a local env var only, so containers silently fell back to
the default model. Here ``CBL_MODEL`` / ``CBL_SERVED_MODEL_NAME`` are read
locally at deploy time and injected via ``image.env(...)``.

Deploy 3B and 7B as separate apps:

    # one-time: a secret providing VLLM_API_KEY
    modal secret create --force cbl-vllm-api-key --from-json /tmp/cbl_secret.json

    CBL_MODEL=Qwen/Qwen2.5-3B-Instruct CBL_SERVED_MODEL_NAME=qwen2.5-3b-instruct \
    CBL_MODAL_APP=cbl-vllm-3b modal deploy serving/vllm_modal.py

    CBL_MODEL=Qwen/Qwen2.5-7B-Instruct CBL_SERVED_MODEL_NAME=qwen2.5-7b-instruct \
    CBL_MODAL_APP=cbl-vllm-7b modal deploy serving/vllm_modal.py

Stop every session:

    modal app stop --yes cbl-vllm-3b
    modal app stop --yes cbl-vllm-7b
"""

from __future__ import annotations

import os
import subprocess

import modal


MODEL = os.environ.get("CBL_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SERVED_MODEL_NAME = os.environ.get("CBL_SERVED_MODEL_NAME", "qwen2.5-7b-instruct")
APP_NAME = os.environ.get("CBL_MODAL_APP", "cbl-vllm")
VLLM_VERSION = os.environ.get("CBL_VLLM_VERSION", "0.11.0")
TRANSFORMERS_VERSION = os.environ.get("CBL_TRANSFORMERS_VERSION", "4.57.1")
SECRET_NAME = os.environ.get("CBL_MODAL_SECRET", "cbl-vllm-api-key")
PORT = int(os.environ.get("CBL_VLLM_PORT", "8000"))
MAX_MODEL_LEN = int(os.environ.get("CBL_MAX_MODEL_LEN", "8192"))
MAX_NUM_SEQS = int(os.environ.get("CBL_MAX_NUM_SEQS", "64"))
MAX_INPUTS = int(os.environ.get("CBL_MAX_INPUTS", "64"))
GPU_MEMORY_UTILIZATION = os.environ.get("CBL_GPU_MEMORY_UTILIZATION", "0.92")
SCALEDOWN_SECONDS = int(os.environ.get("CBL_SCALEDOWN_SECONDS", "120"))


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        "huggingface_hub[hf_transfer]",
    )
    # Bake config into the image env so the container re-import sees the same
    # values the deploy command was given.
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "CBL_MODEL": MODEL,
            "CBL_SERVED_MODEL_NAME": SERVED_MODEL_NAME,
        }
    )
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L4",
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=3600,
    scaledown_window=SCALEDOWN_SECONDS,
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=PORT, startup_timeout=1800)
def serve():
    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        raise RuntimeError("Modal Secret must provide VLLM_API_KEY.")
    model = os.environ["CBL_MODEL"]
    served = os.environ["CBL_SERVED_MODEL_NAME"]
    cmd = [
        "vllm",
        "serve",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--api-key",
        api_key,
        "--served-model-name",
        served,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--enable-prefix-caching",
    ]
    safe_cmd = list(cmd)
    safe_cmd[safe_cmd.index("--api-key") + 1] = "<set>"
    print(" ".join(safe_cmd))
    subprocess.Popen(cmd)
