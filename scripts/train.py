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

def _gpu_is_compatible() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 7  # Unsloth/modern PyTorch requires sm_70+

HAS_GPU = _gpu_is_compatible()


def load_model_and_tokenizer():
    if HAS_GPU:
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
    else:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME, dtype=torch.float32)
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA_CONFIG))
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    print(f"Training on {'GPU (Unsloth/QLoRA)' if HAS_GPU else 'CPU (HF/LoRA)'}")

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
            per_device_train_batch_size=4 if HAS_GPU else 1,
            gradient_accumulation_steps=4 if HAS_GPU else 16,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=100,
            weight_decay=0.01,
            fp16=HAS_GPU,
            use_cpu=not HAS_GPU,
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
