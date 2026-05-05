from pathlib import Path

from reddit_ml.data.loader import load_comments, load_posts
from reddit_ml.db.client import conn
from reddit_ml.db.queries.comments import insert_comments
from reddit_ml.db.queries.posts import insert_posts

COMMENTS_PATH = Path("C:/Users/admin/Downloads/r_Supabase_comments.jsonl")
POSTS_PATH = Path("C:/Users/admin/Downloads/r_Supabase_posts.jsonl")


def main():
    with conn.cursor() as cur:
        posts = load_posts(POSTS_PATH)
        insert_posts(cur, posts)
        print(f"Inserted {len(posts)} posts")

        comments = load_comments(COMMENTS_PATH)
        insert_comments(cur, comments)
        print(f"Inserted {len(comments)} comments")

    conn.commit()


if __name__ == "__main__":
    main()
