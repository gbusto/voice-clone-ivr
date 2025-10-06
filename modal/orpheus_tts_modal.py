#!/usr/bin/env python3
import modal
import os
import io
import json
from typing import Optional, Dict, Any

app = modal.App("orpheus-tts-modal")

model_volume = modal.Volume.from_name("orpheus-models", create_if_missing=True)
training_volume = modal.Volume.from_name("orpheus-training", create_if_missing=True)

orpheus_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-runtime-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    )
    .env(
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .pip_install(
        [
            "torch==2.4.0",
            "torchvision==0.19.0",
            "transformers",
            "datasets",
            "accelerate",
            "huggingface_hub",
            "numpy",
            "safetensors",
            "snac",
            "orpheus-speech",
        ],
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(["vllm==0.7.3"])  # enforce recommended vLLM version
    .run_commands([
        "mkdir -p /models",
        "mkdir -p /training",
        "python -c 'import torch, vllm, transformers, huggingface_hub'",
    ])
)


@app.function(
    image=orpheus_image,
    gpu="A100",
    timeout=60 * 60 * 15,
    volumes={"/models": model_volume, "/training": training_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_models(
    repo_id: str = "canopylabs/orpheus-3b-0.1-ft",
    local_dir: str = "/models/orpheus-3b-0.1-ft",
) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    # token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    # # partially print the token (last 4 chars) to verify that it's not empty
    # print(f"token: {token[-4:]}")
    os.makedirs(local_dir, exist_ok=True)

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        # token=token,
        ignore_patterns=["*.msgpack", "*.pt", "*.bin.index.json"],
        resume_download=True,
    )

    model_volume.commit()
    return {"status": "ok", "path": path}


MODEL = None


def _load_orpheus(model_path: str):
    global MODEL
    if MODEL is not None:
        return MODEL
    from orpheus_tts import OrpheusModel
    import torch

    resolved = model_path if os.path.isdir(model_path) else "canopylabs/orpheus-3b-0.1-ft"
    MODEL = OrpheusModel(model_name=resolved, dtype=torch.bfloat16)
    return MODEL


@app.function(
    image=orpheus_image,
    gpu="A100-40GB",
    volumes={"/models": model_volume, "/training": training_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=2,
)
@modal.fastapi_endpoint(method="POST")
def tts_generate(request: Dict[str, Any]):
    body = request if isinstance(request, dict) else json.loads(request)
    prompt = body.get("prompt")
    voice = body.get("voice", "tara")
    model_dir = body.get("model_dir", "/models/orpheus-3b-0.1-ft")
    temperature = float(body.get("temperature", 0.6))
    top_p = float(body.get("top_p", 0.8))
    max_tokens = int(body.get("max_tokens", 1200))
    repetition_penalty = float(body.get("repetition_penalty", 1.3))

    if not prompt:
        return {"error": "prompt is required"}, 400

    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        download_models.call()

    model = _load_orpheus(model_dir)

    import base64
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        for chunk in model.generate_speech(
            prompt=prompt,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        ):
            if isinstance(chunk, (bytes, bytearray)):
                wf.writeframes(chunk)

    # Default to JSON base64 response for broad compatibility
    wav_bytes = buf.getvalue()
    if str(body.get("format", "json")).lower() == "wav":
        # FastAPI path: return raw bytes with audio/wav
        from fastapi import Response as FastAPIResponse
        return FastAPIResponse(content=wav_bytes, media_type="audio/wav")

    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return {"audio_b64": b64, "content_type": "audio/wav", "voice": voice}


@app.function(
    image=orpheus_image,
    gpu="A100",
    timeout=60 * 60 * 15,
    volumes={"/models": model_volume, "/training": training_volume},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
)
@modal.fastapi_endpoint(method="POST")
def finetune_endpoint(request: Dict[str, Any]):
    body = request if isinstance(request, dict) else json.loads(request)
    dataset_id = body.get("dataset_id")
    if not dataset_id:
        return {"error": "dataset_id is required"}, 400
    base_model_path = body.get("base_model_path", "/models/orpheus-3b-0.1-ft")
    output_dir_name = body.get("output_dir_name", "orpheus-finetune-output")
    epochs = int(body.get("epochs", 1))
    batch_size = int(body.get("batch_size", 1))
    learning_rate = float(body.get("learning_rate", 1e-5))
    save_steps = int(body.get("save_steps", 200))
    run_name = body.get("run_name")
    project_name = body.get("project_name", "orpheus-tts-finetune")

    # Ensure base model present
    if not os.path.exists(base_model_path) or not os.listdir(base_model_path):
        download_models.call()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
    import wandb

    tok = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path)

    ds = load_dataset(dataset_id, split="train")

    wandb.init(project=project_name, name=run_name or output_dir_name)

    out_dir = f"/training/{output_dir_name}"
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=1,
        bf16=True,
        output_dir=out_dir,
        report_to="wandb",
        save_steps=save_steps,
        remove_unused_columns=True,
        learning_rate=learning_rate,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()

    training_volume.commit()
    model_volume.commit()

    return {"status": "completed", "output_dir": out_dir}


# @app.local_entrypoint()
# def main(
#     action: str = "serve",
#     prompt: str = "Hello from Orpheus!",
#     voice: str = "tara",
#     dataset_id: Optional[str] = None,
# ):
#     if action == "download":
#         res = download_models.remote()
#         print(json.dumps(res, indent=2))
#     elif action == "serve":
#         print("POST to this endpoint to synthesize speech:")
#         print(f"https://modal.com/apps/{app.name}")
#     elif action == "finetune":
#         if not dataset_id:
#             raise SystemExit("dataset_id is required for finetune")
#         res = finetune_orpheus.remote(dataset_id=dataset_id)
#         print(json.dumps(res, indent=2))
#     else:
#         raise SystemExit("Unknown action. Use: download | serve | finetune")s