import json
import random
from pathlib import Path

from reddit_ml.db.client import conn
from reddit_ml.db.queries.comments import get_comments
from reddit_ml.processing.cleaning import clean_comment, quality_filter
from reddit_ml.processing.formatting import format_example

SUBREDDIT = "Supabase"
OUTPUT_DIR = Path("data")
TRAIN_SPLIT = 0.95


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    with conn.cursor() as cur:
        rows = get_comments(cur, subreddit=SUBREDDIT, batch_size=50_000)

    examples = []
    for body_raw, title, selftext, subreddit in rows:
        body = clean_comment(body_raw)
        if not quality_filter(body):
            continue
        examples.append(format_example(
            subreddit=subreddit,
            post_title=title,
            post_body=selftext,
            comment_body=body,
        ))

    random.shuffle(examples)
    split = int(len(examples) * TRAIN_SPLIT)
    train, val = examples[:split], examples[split:]

    (OUTPUT_DIR / "train.json").write_text(json.dumps(train, indent=2))
    (OUTPUT_DIR / "val.json").write_text(json.dumps(val, indent=2))
    print(f"Train: {len(train)} | Val: {len(val)}")


if __name__ == "__main__":
    main()
