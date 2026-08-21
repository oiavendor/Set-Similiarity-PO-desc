def string_from_enum(enum_value: dict[int, str]) -> str:
    return_string = ""
    for key, value in enum_value.items():
        return_string += value + "\n"
    return return_string