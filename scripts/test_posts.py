"""
Test fine-tuned model on Reddit posts.
Usage: python test_posts.py [--model ./reddit-lora]
"""

import argparse
from unsloth import FastLanguageModel

SYSTEM_PROMPT = "You are a helpful Reddit commenter."

TEST_POSTS = [
    {
        "title": "Need help with Supabase auth",
        "selftext": "I'm trying to set up email auth with Supabase but keep getting a 400 error when calling signInWithPassword. I've enabled email auth in the dashboard. What am I missing?",
        "subreddit": "supabase",
    },
    {
        "title": "RLS policies are confusing",
        "selftext": "I've been trying to set up row level security for my app but I can't figure out why my select policy isn't working. I have a simple policy that checks auth.uid() = user_id but it returns no rows even when I'm logged in.",
        "subreddit": "supabase",
    },
    {
        "title": "Supabase vs Firebase in 2026?",
        "selftext": "Starting a new project and trying to decide between Supabase and Firebase. I like the idea of Postgres but I'm worried about the learning curve. Anyone who's used both have thoughts?",
        "subreddit": "supabase",
    },
]


def format_post(post: dict) -> str:
    return f"Subreddit: r/{post['subreddit']}\nTitle: {post['title']}\n\n{post['selftext']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./reddit-lora")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--system", type=str, default=SYSTEM_PROMPT)
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.\n")

    for i, post in enumerate(TEST_POSTS, 1):
        print("=" * 60)
        print(f"POST {i}: {post['title']}")
        print(f"r/{post['subreddit']}")
        print("-" * 60)
        print(post["selftext"])
        print("-" * 60)

        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": format_post(post)},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.9,
            do_sample=args.temperature > 0,
        )

        response = tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        print(f"\nRESPONSE:\n{response}\n")


if __name__ == "__main__":
    main()