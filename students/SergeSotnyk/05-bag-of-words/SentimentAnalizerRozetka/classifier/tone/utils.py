from typing import List


def stars_to_sentiment(stars):
    res = []
    for s in stars:
        if s < 3:
            res.append('neg')
        elif s < 5:
            res.append('ind')  # indifferent
        else:
            res.append('pos')
    return res


def find_classes(label: str, sentiments: List[str]) -> List[bool]:
    return [s == label for s in sentiments]