from vis import (
    PATTERN_DICT,
    PATTERN_DICT_SKIRTIFIED,
    visualize_pattern
)


DRESS = False


if DRESS:
    pattern_dict = PATTERN_DICT_SKIRTIFIED
else:
    pattern_dict = PATTERN_DICT


if __name__ == "__main__":
    visualize_pattern(DRESS)
