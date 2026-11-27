"""Check whether a decoded sentence reads like the keys it came from, character by character."""

from pinyin import units_of


FLOOR, SHARE = 2, 0.3
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

    Returns a report: each character with the letters it was matched to and its cost, the letters nobody
    claimed, the total cost, the budget, and whether the total is within it. `doubts` lists the characters
    whose reading is two or more keystrokes from the keys, the ones a repair should look at.
    Returns None when the text has no character at all.
    """
    units = units_of(text)
    count, length = len(units), len(keys)
    if not count:
        return None
    best = [[UNREACHABLE] * (length + 1) for _ in range(count + 1)]
    back = [[None] * (length + 1) for _ in range(count + 1)]
    best[0][0] = 0

    def relax(row, column, cost, trace):
        if cost < best[row][column]:
            best[row][column] = cost
            back[row][column] = trace

    for row in range(count + 1):
        for column in range(length + 1):
            base = best[row][column]
            if base >= UNREACHABLE:
                continue
            if column < length:
                relax(row, column + 1, base + 1, (row, column, None))
            if row == count:
                continue
            readings, chinese = units[row][1], units[row][2]
            reading = readings[0]
            sizes = (0, len(reading) - 1, len(reading), len(reading) + 1) if chinese else (0, len(reading))
            for size in sizes:
                end = column + size
                if size < 0 or end > length:
                    continue
                piece = keys[column:end]
                cost = UNTYPED if size == 0 and chinese else min(fit(piece, readings))
                relax(row + 1, end, base + cost, (row, column, piece))
            if chinese and row + 1 < count and units[row + 1][2]:
                pair = reading + units[row + 1][1][0]
                end = column + len(pair)
                if end <= length:
                    piece = keys[column:end]
                    cost = 0 if piece == pair else distance(piece, pair)
                    if cost <= 1:
                        relax(row + 2, end, base + cost, (row, column, piece))

    total = best[count][length]
    if total >= UNREACHABLE:
        return None
    typed, costs, skipped = [""] * count, [0] * count, []
    row, column = count, length
    while (row, column) != (0, 0):
        previous_row, previous_column, piece = back[row][column]
        if piece is None:
            skipped.append(keys[previous_column])
        elif row - previous_row == 2:
            # Two syllables taken together: the error sits in the half that differs from its reading.
            first, second = units[previous_row][1][0], units[previous_row + 1][1][0]
            typed[previous_row], typed[previous_row + 1] = piece[:len(first)], piece[len(first):]
            if piece != first + second:
                at_fault = previous_row if fit(typed[previous_row], units[previous_row][1])[0] else previous_row + 1
                costs[at_fault] = 1
        else:
            typed[previous_row], costs[previous_row] = piece, min(fit(piece, units[previous_row][1]))
        row, column = previous_row, previous_column
    return {"units": [{"char": unit[0], "reading": closest(typed[index], unit[1]), "typed": typed[index], "cost": costs[index]}
                      for index, unit in enumerate(units)],
            "skipped": skipped, "cost": total, "budget": budget(units), "consistent": total <= budget(units),
            "flagged": [index for index in range(count) if costs[index]],
            "doubts": [index for index in range(count) if costs[index] >= 2 or not typed[index]]}


def closest(piece, readings):
    """The reading of a character that its typed letters spell best."""
    return min(readings, key=lambda reading: (distance(piece, reading), readings.index(reading)))


def summary(report):
    """The part of a report worth keeping next to a reply."""
    return {"cost": report["cost"], "budget": report["budget"], "doubts": len(report["doubts"]), "flagged": len(report["flagged"]),
            "skipped": len(report["skipped"])}
