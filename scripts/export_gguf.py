from unsloth import FastLanguageModel

MODEL_PATH = "./reddit-lora-final"
OUTPUT_PATH = "reddit-model-gguf"
QUANTIZATION = "q4_k_m"  # options: q4_k_m, q5_k_m, q8_0, f16


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model.save_pretrained_gguf(OUTPUT_PATH, tokenizer, quantization_method=QUANTIZATION)
    print(f"GGUF exported to {OUTPUT_PATH}/")


if __name__ == "__main__":
    main()
