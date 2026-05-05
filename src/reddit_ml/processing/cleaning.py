import re


def clean_comment(text: str) -> str:
    text = re.sub(r"&gt;.*?\n", "", text)
    text = re.sub(r"https?://\S+", "[link]", text)
    text = re.sub(r"/u/\w+", "[user]", text)
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^edit:.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


_LOW_EFFORT = re.compile(
    r"^(this|lol|same|f$|rip|nice|yep|nah)|came here to say|underrated comment|take my upvote",
    re.IGNORECASE,
)


def quality_filter(body: str) -> bool:
    if len(body) < 80 or len(body) > 1500:
        return False
    if body.count(".") + body.count("!") + body.count("?") < 2:
        return False
    if body in ("[deleted]", "[removed]"):
        return False
    if _LOW_EFFORT.search(body):
        return False
    return True
