def check_strictly_increasing(array_like) -> bool:
    return all(x < y for x, y in zip(array_like, array_like[1:]))
