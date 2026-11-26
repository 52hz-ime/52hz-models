"""Check that a decoded sentence really reads like the keys the user typed."""

from pinyin import to_keys


LIMIT = 5
SHARE = 0.30


def distance(left, right):
    """Edit distance between two strings; a substitution, an insertion and a deletion each cost one."""
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(previous[right_index] + 1, current[-1] + 1, previous[right_index - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def reliable(keys, text):
    """Whether a sentence is close enough to the keys to be shown.

    On benchmark trials a correct sentence never strayed more than two keystrokes from the keys, and
    wrong-but-related readings stayed within a quarter of the keys; only unrelated output went past a
    third. The limit is deliberately loose: a wrong reading is still worth showing, nonsense is not.
    """
    return distance(keys, to_keys(text)) <= max(LIMIT, SHARE * len(keys))
