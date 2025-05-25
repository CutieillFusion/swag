from itertools import product

raw_action_mappings = {
    "right": 0b10000000,
    "left": 0b01000000,
    "down": 0b00100000,
    "up": 0b00010000,
    "start": 0b00001000,
    "select": 0b00000100,
    "B": 0b00000010,
    "A": 0b00000001,
    "NOOP": 0b00000000,
}

ACTION_SET = {
    "up_down": ["NOOP", "up", "down"],
    "left_right": ["NOOP", "left", "right"],
    "a": ["NOOP", "A"],
    "b": ["NOOP", "B"],
}

action_permutations = list(
    product(
        ACTION_SET["up_down"],
        ACTION_SET["left_right"],
        ACTION_SET["a"],
        ACTION_SET["b"],
    )
)

action_combinations = sorted([sorted(set(tup)) for tup in action_permutations])

ACTION_SPACE = [
    (
        action
        if "NOOP" not in action or len(action) == 1
        else [a for a in action if a != "NOOP"]
    )
    for action in action_combinations
]

def get_raw_value(action_buttons):
    byte_action = 0
    for button in action_buttons:
        byte_action |= raw_action_mappings[button]
    return byte_action

ACTION_SPACE = sorted(ACTION_SPACE, key=get_raw_value)

action_map = {}
action_meanings = {}

for action_idx, button_list in enumerate(ACTION_SPACE):
    byte_action = get_raw_value(button_list)
    action_map[action_idx] = byte_action
    action_meanings[action_idx] = " ".join(button_list)

mapping_action = {v: k for k, v in action_map.items()}


def convert_int_to_action(input: int) -> str:
    if input in mapping_action.keys():
        return mapping_action[input]

    if input & raw_action_mappings["left"] and input & raw_action_mappings["right"]:
        return convert_int_to_action(input - raw_action_mappings["left"])
    if input & raw_action_mappings["up"] and input & raw_action_mappings["down"]:
        return convert_int_to_action(input - raw_action_mappings["up"])
    if input & raw_action_mappings["select"]:
        return convert_int_to_action(input - raw_action_mappings["select"])
    if input & raw_action_mappings["start"]:
        return convert_int_to_action(input - raw_action_mappings["start"])

    raise ValueError(f"Unresolvable input: {input}")

if __name__ == "__main__":
    _actions = {v: action_meanings[k] for k, v in action_map.items()}
    for k, v in _actions.items():
        print(v, k)
