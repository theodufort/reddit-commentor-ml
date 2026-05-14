import json
import os
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTConfig, SFTTrainer

load_dotenv()
login(token=os.environ["HF_TOKEN"])

#HF_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HF_MODEL_NAME = "Qwen/Qwen3.5-9B"
OUTPUT_DIR = "./reddit-lora"

MAX_SEQ_LENGTH = 1024

# Training parameters - adjust these values to experiment with training behavior.
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.075
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
EVAL_STRATEGY = "steps"
EVAL_STEPS = 500
SAVE_STRATEGY = "steps"
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 3
LOAD_BEST_MODEL_AT_END = True
REPORT_TO = "none"
SEED = 42
PER_DEVICE_BATCH_SIZE = {"gpu_fast": 4, "gpu_legacy": 4, "cpu": 2}
GRADIENT_ACCUMULATION_STEPS = {"gpu_fast": 4, "gpu_legacy": 4, "cpu": 8}

LORA_R = 16
LORA_ALPHA = LORA_R * 2 # LORA_R or LORA_R * 2
LORA_DROPOUT = 0.0
LORA_BIAS = "none"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

LORA_CONFIG = dict(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    target_modules=LORA_TARGET_MODULES,
)


def _pick_best_cuda_device() -> int | None:
    """Return the CUDA device index with the highest compute capability, or None."""
    n = torch.cuda.device_count()
    if n == 0:
        return None
    best = max(range(n), key=lambda i: torch.cuda.get_device_capability(i))
    name = torch.cuda.get_device_name(best)
    major, minor = torch.cuda.get_device_capability(best)
    print(f"Selected GPU {best}: {name} (sm_{major}{minor})")
    return best


CUDA_DEVICE: int | None = _pick_best_cuda_device() if torch.cuda.is_available() else None


def _gpu_mode() -> str:
    if CUDA_DEVICE is None:
        return "cpu"
    major, _ = torch.cuda.get_device_capability(CUDA_DEVICE)
    return "gpu_fast" if major >= 7 else "gpu_legacy"


GPU_MODE = _gpu_mode()
DEVICE_MAP = {"": CUDA_DEVICE} if CUDA_DEVICE is not None else {"": "cpu"}


def load_model_and_tokenizer():
    if GPU_MODE == "gpu_fast":
        # Restrict Unsloth to the chosen GPU via CUDA_VISIBLE_DEVICES.
        # Must be set before unsloth initialises its CUDA context.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=f"unsloth/{HF_MODEL_NAME.split('/')[-1]}",
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            **LORA_CONFIG,
            use_gradient_checkpointing="unsloth",
        )
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    elif GPU_MODE == "gpu_legacy":
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            dtype=torch.float16,
            device_map=DEVICE_MAP,
        )
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA_CONFIG))
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    else:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=DEVICE_MAP,
        )
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA_CONFIG))
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    labels = {"gpu_fast": "GPU (Unsloth/QLoRA)", "gpu_legacy": "GPU (HF/LoRA fp16)", "cpu": "CPU (HF/LoRA)"}
    print(f"Training on {labels[GPU_MODE]}")

    model, tokenizer = load_model_and_tokenizer()

    dataset = DatasetDict({
        "train": Dataset.from_list(json.loads(Path("data/train.json").read_text())),
        "validation": Dataset.from_list(json.loads(Path("data/val.json").read_text())),
    })

    def format_chat(example):
        return {"text": tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )}

    dataset = dataset.map(format_chat)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE[GPU_MODE],
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS[GPU_MODE],
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            warmup_steps=max(
                1,
                int(
                    (dataset["train"].num_rows * NUM_TRAIN_EPOCHS)
                    / (PER_DEVICE_BATCH_SIZE[GPU_MODE] * GRADIENT_ACCUMULATION_STEPS[GPU_MODE])
                    * WARMUP_RATIO
                ),
            ),
            weight_decay=WEIGHT_DECAY,
            bf16=GPU_MODE == "gpu_fast",
            fp16=GPU_MODE == "gpu_legacy",
            use_cpu=GPU_MODE == "cpu",
            logging_steps=LOGGING_STEPS,
            eval_strategy=EVAL_STRATEGY,
            eval_steps=EVAL_STEPS,
            save_strategy=SAVE_STRATEGY,
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
            report_to=REPORT_TO,
            seed=SEED,
        ),
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}-final")
    print(f"Model saved to {OUTPUT_DIR}-final")


if __name__ == "__main__":
    main()
