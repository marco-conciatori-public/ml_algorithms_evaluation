import global_constants


def check_strictly_increasing(array_like) -> bool:
    return all(x < y for x, y in zip(array_like, array_like[1:]))


def print_formatted_results(results: dict, display_as_percent: bool = True, title: str = 'RESULTS'):
    print(title)
    for key in results:
        if key == global_constants.CONFUSION_MATRIX:
            print(f'\t- {key}:\n{results[key]}')
        else:
            if display_as_percent:
                print(f'\t- {key}: {round(results[key] * 100, max(global_constants.MAX_DECIMAL_PLACES - 2, 0))} %')
            else:
                print(f'\t- {key}: {round(results[key], global_constants.MAX_DECIMAL_PLACES)}')
