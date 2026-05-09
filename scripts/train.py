import json
import os
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

load_dotenv()
login(token=os.environ["HF_TOKEN"])

HF_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./reddit-lora"
MAX_SEQ_LENGTH = 2048

LORA_CONFIG = dict(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def _gpu_mode() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    major, _ = torch.cuda.get_device_capability()
    # sm_70+ supports bitsandbytes 4-bit quantization and Unsloth
    return "gpu_fast" if major >= 7 else "gpu_legacy"


GPU_MODE = _gpu_mode()


def load_model_and_tokenizer():
    if GPU_MODE == "gpu_fast":
        from unsloth import FastLanguageModel
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
    elif GPU_MODE == "gpu_legacy":
        # sm_61 (GTX 1070): no bitsandbytes 4-bit, use fp16 LoRA directly on CUDA
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA_CONFIG))
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    else:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, torch_dtype=torch.float32)
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
            num_train_epochs=3,
            per_device_train_batch_size=4 if GPU_MODE == "gpu_fast" else (2 if GPU_MODE == "gpu_legacy" else 1),
            gradient_accumulation_steps=4 if GPU_MODE == "gpu_fast" else (8 if GPU_MODE == "gpu_legacy" else 16),
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=100,
            weight_decay=0.01,
            fp16=GPU_MODE != "cpu",
            use_cpu=GPU_MODE == "cpu",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="none",
            seed=42,
        ),
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}-final")
    print(f"Model saved to {OUTPUT_DIR}-final")


if __name__ == "__main__":
    main()
