"""Check whether a decoded sentence reads like the keys it came from, character by character."""

from pinyin import units_of


FLOOR, SHARE = 2, 0.3
# A character whose reading is this many keystrokes from the keys is one a repair should look at.
FAR = 2
# A character with no typed letters of its own: more than any typo, so that it alone busts a short budget.
UNTYPED = 4
UNREACHABLE = 1 << 20


def distance(left, right):
    """Optimal string alignment distance: an adjacent transposition costs one."""
    if left == right:
        return 0
    previous2, previous = None, list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(previous[right_index] + 1, current[-1] + 1, previous[right_index - 1] + (left_char != right_char),
                               previous2[right_index - 2] + 1 if left_index > 1 and right_index > 1
                               and left_char == right[right_index - 2] and left[left_index - 2] == right_char else UNREACHABLE))
        previous2, previous = previous, current
    return previous[-1]


def fit(piece, readings):
    """The cost of reading the typed letters as one character: nothing when they spell one of its readings."""
    return [0] if piece in readings else [distance(piece, reading) for reading in readings]


def budget(units):
    """How many syllables of a sentence may be mistyped: at least two, about a third of them above that."""
    return max(FLOOR, int(SHARE * sum(1 for unit in units if unit[2]) + 0.5))


def align(keys, text):
    """Match every character of the text with the keys it should have been typed as.

    A Chinese character takes the letters of its syllable. One keystroke error in them costs one, and so
    does a transposition across the boundary of two neighbouring syllables; anything further costs its
    distance, so a character that has nothing to do with the keys is never mistaken for a few typos.
    Any other character takes itself. A typed letter that no character claims also costs one.

    Among alignments that cost the same, the one with fewer characters FAR or more keystrokes from the
    keys is kept, then the one that edits fewer characters, then the one that leaves fewer stretches of
    unclaimed keys. So a character is only reported as far from the keys when no equally cheap alignment
    reads it within one keystroke, and a word the text left out shows as one whole stretch between two
    characters, not as letters taken by its neighbours or split around one of them.

    Returns a report: each character with the letters it was matched to and its cost, the letters nobody
    claimed, in the order of the keys, and where they sit (`gaps`: one entry per stretch, with the number of
    characters that come before it), the total cost, the budget, and whether the total is within it.
    `doubts` lists the characters whose reading is two or more keystrokes from the keys, the ones a repair
    should look at.
    Returns None when the text has no character at all.
    """
    units = units_of(text)
    count, length = len(units), len(keys)
    if not count:
        return None
    unreachable = (UNREACHABLE,) * 4
    # best[row][column][after a skipped letter] is (cost, characters far from the keys, characters edited,
    # stretches of unclaimed keys).
    best = [[[unreachable, unreachable] for _ in range(length + 1)] for _ in range(count + 1)]
    back = [[[None, None] for _ in range(length + 1)] for _ in range(count + 1)]
    best[0][0][0] = (0, 0, 0, 0)

    def relax(row, column, after_skip, value, trace):
        if value < best[row][column][after_skip]:
            best[row][column][after_skip] = value
            back[row][column][after_skip] = trace

    for row in range(count + 1):
        for column in range(length + 1):
            for after_skip in (0, 1):
                base, far, edited, gaps = best[row][column][after_skip]
                if base >= UNREACHABLE:
                    continue
                if column < length:
                    relax(row, column + 1, 1, (base + 1, far, edited, gaps + (not after_skip)), (row, column, after_skip, None))
                if row == count:
                    continue
                readings, chinese = units[row][1], units[row][2]
                reading = readings[0]
                # A character with several readings may be typed as any of them, and they are not all the same length.
                sizes = sorted({size for other in readings for size in (len(other) - 1, len(other), len(other) + 1)} | {0}) \
                    if chinese else (0, len(reading))
                for size in sizes:
                    end = column + size
                    if size < 0 or end > length:
                        continue
                    piece = keys[column:end]
                    cost = UNTYPED if size == 0 and chinese else min(fit(piece, readings))
                    relax(row + 1, end, 0, (base + cost, far + (cost >= FAR), edited + bool(cost), gaps), (row, column, after_skip, piece))
                if chinese and row + 1 < count and units[row + 1][2]:
                    pair = reading + units[row + 1][1][0]
                    end = column + len(pair)
                    if end <= length:
                        piece = keys[column:end]
                        cost = 0 if piece == pair else distance(piece, pair)
                        if cost <= 1:
                            relax(row + 2, end, 0, (base + cost, far, edited + bool(cost), gaps), (row, column, after_skip, piece))

    last = min((0, 1), key=lambda after_skip: best[count][length][after_skip])
    total = best[count][length][last][0]
    if total >= UNREACHABLE:
        return None
    typed, costs, skipped = [""] * count, [0] * count, []
    row, column, after_skip = count, length, last
    while (row, column) != (0, 0):
        previous_row, previous_column, previous_skip, piece = back[row][column][after_skip]
        if piece is None:
            skipped.append((row, previous_column, keys[previous_column]))
        elif row - previous_row == 2:
            # Two syllables taken together: the error sits in the half that differs from its reading.
            first, second = units[previous_row][1][0], units[previous_row + 1][1][0]
            typed[previous_row], typed[previous_row + 1] = piece[:len(first)], piece[len(first):]
            if piece != first + second:
                at_fault = previous_row if fit(typed[previous_row], units[previous_row][1])[0] else previous_row + 1
                costs[at_fault] = 1
        else:
            typed[previous_row] = piece
            costs[previous_row] = UNTYPED if not piece and units[previous_row][2] else min(fit(piece, units[previous_row][1]))
        row, column, after_skip = previous_row, previous_column, previous_skip
    skipped.reverse()                                        # the order of the keys
    gaps = []
    for before, position, letter in skipped:
        if gaps and (gaps[-1]["before"], gaps[-1]["at"]) == (before, position - 1):
            gaps[-1]["letters"], gaps[-1]["at"] = gaps[-1]["letters"] + letter, position
        else:
            gaps.append({"before": before, "at": position, "letters": letter})
    for gap in gaps:
        del gap["at"]
    return {"units": [{"char": unit[0], "reading": closest(typed[index], unit[1]), "typed": typed[index], "cost": costs[index]}
                      for index, unit in enumerate(units)],
            "skipped": [letter for _, _, letter in skipped], "gaps": gaps,
            "cost": total, "budget": budget(units), "consistent": total <= budget(units),
            "flagged": [index for index in range(count) if costs[index]],
            "doubts": [index for index in range(count) if costs[index] >= FAR or not typed[index]]}

def closest(piece, readings):
    """The reading of a character that its typed letters spell best."""
    return min(readings, key=lambda reading: (distance(piece, reading), readings.index(reading)))


def summary(report):
    """The part of a report worth keeping next to a reply."""
    return {"cost": report["cost"], "budget": report["budget"], "doubts": len(report["doubts"]), "flagged": len(report["flagged"]),
            "skipped": len(report["skipped"])}
