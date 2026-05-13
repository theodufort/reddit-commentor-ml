from unsloth import FastLanguageModel

MODEL_PATH = "./reddit-lora-final"
OUTPUT_PATH = "reddit-model-gguf"
QUANTIZATION = "q4_k_m"  # options: q4_k_m, q5_k_m, q8_0, f16


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

    model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit",)
    print(f"GGUF exported to {OUTPUT_PATH}/")


if __name__ == "__main__":
    main()
