"""
Places each distinct purchase order line item into the taxonomy in json_files/item_categories.json.

Line matching on the descriptions themselves can only find a split that repeats the same items
across several POs. The other way of splitting a purchase is to divide complementary items between
them, and those descriptions are dissimilar by construction: a processor and the workstation it
goes into share no wording, and an embedding is right to say so. Cosine similarity answers whether
two strings describe the same thing, which is not the same question as whether two items belong to
the same purchase.

A category supplies that second relation. Both a processor and a workstation are end-user
computing, so the pair can be recognised even though nothing about the two strings matches.

Classification is done once per DISTINCT line item and cached to disk, so the cost is the number of
different things the institution buys rather than the number of PO pairs being compared, and so
that the same description always resolves to the same category. An unstable category would make
scores drift between runs.
"""

from typing import Dict, Iterable, List, Optional, Tuple

from utils.GenAI import api_query, generate_system_prompt
from utils.item_categorizer_enums import CONSTRAINTS, CONTEXT, EXAMPLE, ROLE, TASK
from utils.JSONOps import read_json, write_json
from utils.shared_functions import string_from_enum

TAXONOMY_FILE = "item_categories"
CACHE_FILE = "item_category_cache"

# What a line item resolves to when the model could not place it, the answer could not be parsed,
# or the description carries too little information. Two POs that share only this share an absence
# of information rather than an attribute, so category matching drops these entirely.
UNCLASSIFIED = "Items pending classification"

_taxonomy = None


def load_taxonomy() -> Dict[str, List[str]]:
    """
    Loads and caches the taxonomy, as the level 2 categories under each level 1 category.

    Returns:
        Dict[str, List[str]]: The level 2 categories of each level 1 category.

    Raises:
        FileNotFoundError: If the taxonomy file is missing or holds no categories.
    """

    global _taxonomy
    if _taxonomy is None:
        categories = read_json(TAXONOMY_FILE).get("categories")
        if not categories:
            raise FileNotFoundError(
                f"No categories found in the '{TAXONOMY_FILE}' JSON file. Line mode needs it when "
                f"use_categories is on; set use_categories = false in analysis.config to run without."
            )
        _taxonomy = categories
    return _taxonomy


def format_taxonomy() -> str:
    """
    Renders the taxonomy for the prompt, as an indented list of level 2 categories under each
    level 1 category.

    Returns:
        str: The taxonomy as text.
    """

    lines = []
    for level_1, level_2_list in load_taxonomy().items():
        lines.append(level_1)
        lines.extend(f"  - {level_2}" for level_2 in level_2_list)
    return "\n".join(lines)


def parse_answer(answer: str, batch: List[str]) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    Reads the model's answer for one batch back into a category per line item.

    The answer is matched to the batch by the item number it carries rather than by position, so a
    dropped or reordered line misplaces nothing. Anything that does not parse, names a level 1
    category that is not in the taxonomy, or numbers an item that was not asked about, is left out
    and treated as unclassified by the caller. A level 2 that does not belong to its level 1 is
    dropped on its own, keeping the level 1 that did parse, because level 1 alone is still enough
    to rule a pair out.

    Args:
        answer (str): The model's reply.
        batch (List[str]): The line items that were sent, in order.

    Returns:
        Dict[str, Tuple[str, Optional[str]]]: The level 1 and level 2 of each line item that parsed.
    """

    taxonomy = load_taxonomy()
    categories = {}

    for row in (answer or "").splitlines():
        parts = [part.strip() for part in row.split("|")]
        if len(parts) != 3 or not parts[0].isdigit():
            continue

        position = int(parts[0]) - 1
        if not 0 <= position < len(batch):
            continue

        level_1, level_2 = parts[1], parts[2]
        if level_1 not in taxonomy:
            continue
        categories[batch[position]] = (level_1, level_2 if level_2 in taxonomy[level_1] else None)

    return categories


def classify_lines(lines: Iterable[str], batch_size: int = 20,
                   regenerate: bool = False) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    Returns the category of every line item given, asking the language model only about the ones
    not already cached.

    Args:
        lines (Iterable[str]): The normalised line items to classify.
        batch_size (int, optional): How many line items to send per request. Defaults to 20.
        regenerate (bool, optional): Whether to re-ask about line items already cached. Defaults to False.

    Returns:
        Dict[str, Tuple[str, Optional[str]]]: The level 1 and level 2 of each line item. A line item
        the model could not place resolves to the unclassified category, with a level 2 of None.
    """

    wanted = sorted(set(lines))
    if not wanted:
        return {}

    cache = {} if regenerate else {line: tuple(value) for line, value in read_json(CACHE_FILE).items()}
    outstanding = [line for line in wanted if line not in cache]

    if outstanding:
        system_prompt = generate_system_prompt(
            role=ROLE, task=TASK, context=string_from_enum(CONTEXT),
            constraints=string_from_enum(CONSTRAINTS), example=string_from_enum(EXAMPLE))
        system_prompt += f"\n\nThe taxonomy is:\n{format_taxonomy()}\n"

        print(f"Categorising {len(outstanding)} line items not seen before "
              f"({len(wanted) - len(outstanding)} already cached)")

        for start in range(0, len(outstanding), batch_size):
            batch = outstanding[start:start + batch_size]
            print(f"  categorising line items {start + 1} to {start + len(batch)} of {len(outstanding)}")
            user_prompt = "\n".join(f"{position}: {line}" for position, line in enumerate(batch, start=1))
            answered = parse_answer(api_query(system_prompt, user_prompt), batch)

            for line in batch:  # anything the model skipped or garbled stays unclassified
                cache[line] = answered.get(line, ("General & Other", UNCLASSIFIED))

        write_json(CACHE_FILE, {line: list(value) for line, value in cache.items()})

    return {line: cache[line] for line in wanted}


def po_categories(lines: Iterable[str],
                  categories: Dict[str, Tuple[str, Optional[str]]]) -> List[Tuple[str, Optional[str]]]:
    """
    Reduces one PO's line items to the distinct categories it buys from.

    How many line items fall into a category does not matter to the question being asked, only
    which categories appear, so the result is deduplicated. Unclassified line items are left out
    rather than kept with a weight of zero: they carry no information at all, and two POs holding
    only unclassified items are not related by that fact.

    Args:
        lines (Iterable[str]): The normalised line items of one PO.
        categories (Dict[str, Tuple[str, Optional[str]]]): The category of each line item.

    Returns:
        List[Tuple[str, Optional[str]]]: The PO's distinct categories, in first-seen order.
    """

    seen = set()
    result = []
    for line in lines:
        category = categories.get(line)
        if category is None or category[1] == UNCLASSIFIED:
            continue
        if category not in seen:
            seen.add(category)
            result.append(category)
    return result
