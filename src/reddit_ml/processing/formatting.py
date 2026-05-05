SYSTEM_PROMPT = (
    "You are a helpful and knowledgeable Reddit commenter. "
    "Write informative, natural-sounding replies that are genuinely useful. "
    "Match the tone of the subreddit — be conversational but substantive. "
    "Avoid being preachy, robotic, or overly formal."
)


def format_example(subreddit: str, post_title: str, post_body: str, comment_body: str) -> dict:
    user_parts = [f"[r/{subreddit}]", f"Post: {post_title}"]
    if post_body:
        user_parts.append(post_body[:300])

    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_parts)},
            {"role": "assistant", "content": comment_body},
        ]
    }
