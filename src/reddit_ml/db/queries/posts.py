from datetime import datetime, timezone

from psycopg2.extensions import cursor

from reddit_ml.data.types import Post


def insert_posts(cur: cursor, posts: list[Post]) -> None:
    cur.executemany(
        """
        INSERT INTO public.reddit_posts (id, created_at, url, subreddit, author, selftext, title, ups, downs)
        VALUES (%(id)s, %(created_at)s, %(url)s, %(subreddit)s, %(author)s, %(selftext)s, %(title)s, %(ups)s, 0)
        ON CONFLICT DO NOTHING
        """,
        [
            {
                "id": p.id,
                "created_at": datetime.fromtimestamp(p.created_utc, tz=timezone.utc),
                "url": p.url,
                "subreddit": p.subreddit,
                "author": p.author,
                "selftext": p.selftext,
                "title": p.title,
                "ups": p.ups,
            }
            for p in posts
        ],
    )
