from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from psycopg2.extensions import cursor

if TYPE_CHECKING:
    from reddit_ml.data.types import Comment


def insert_comments(cur: cursor, comments: list[Comment]) -> None:
    cur.executemany(
        """
        INSERT INTO public.reddit_comments (id, post_id, permalink, body, ups, downs, created_at, author)
        SELECT %(id)s, %(post_id)s, %(permalink)s, %(body)s, %(ups)s, 0, %(created_at)s, %(author)s
        WHERE EXISTS (SELECT 1 FROM public.reddit_posts WHERE id = %(post_id)s)
        ON CONFLICT (id) DO NOTHING
        """,
        [
            {
                "id": c.id,
                "post_id": c.link_id.removeprefix("t3_"),
                "permalink": c.permalink,
                "body": c.body,
                "ups": c.ups,
                "created_at": datetime.fromtimestamp(c.created_utc, tz=timezone.utc),
                "author": c.author,
            }
            for c in comments
        ],
    )


def get_comments(cur: cursor, subreddit: str, min_body_len: int = 50, batch_size: int = 10000) -> list[tuple]:
    cur.execute(
        """
        SELECT rc.body, rp.title, rp.selftext, rp.subreddit
        FROM public.reddit_comments AS rc
        INNER JOIN public.reddit_posts AS rp ON rc.post_id = rp.id
        WHERE lower(rp.subreddit) = lower(%(subreddit)s)
          AND CHAR_LENGTH(rc.body) > %(min_body_len)s and rc.body not like '%%http%%' and rc.body not like '%%vote%%'
        LIMIT %(batch_size)s
        """,
        {"subreddit": subreddit, "min_body_len": min_body_len, "batch_size": batch_size},
    )
    return cur.fetchall()
