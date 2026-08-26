from utils.AnalysisConfig import get_analysis_config

# The column names and fallback verdicts are read once from analysis.config and exposed as the
# constants below, so renaming a column is a config edit rather than a code change. Everything
# else in this file is the prompt wording handed to the language model.

_PO = get_analysis_config("po")
_PAYMENT = get_analysis_config("payment")

PO_SET_COLUMN = _PO["group_column"]
PO_ITEM_DESC_COLUMN = _PO["description_column"]
PO_SPLIT_BOOLEAN_COLUMN = _PO["boolean_column"]
PO_EXPLANATION_COLUMN = _PO["explanation_column"]
PO_DEFAULT_NON_CLUSTERED = _PO["default"]

PAYMENT_SET_COLUMN = _PAYMENT["group_column"]
PAYMENT_ITEM_DESC_COLUMN = _PAYMENT["description_column"]
PAYMENT_DUPP_BOOLEAN_COLUMN = _PAYMENT["boolean_column"]
PAYMENT_EXPLANATION_COLUMN = _PAYMENT["explanation_column"]
PAYMENT_DEFAULT_NON_CLUSTERED = _PAYMENT["default"]

TASK = "Identify if a list of item description are for the same item"

PO_CONTEXT = {
    1: "A group of items have been identified as potentially coming from a split purchase order (PO), where large POs are split up into smaller POs, to circumvent the checks for large POs.",
    2: "A purchase can be split in two ways. It can be split by repeating the same items across several POs, in which case the item descriptions are roughly similar with minor changes to 'mask' the technique.",
    3: "It can instead be split by dividing the complementary parts of one purchase between several POs, for example a workstation on one PO and the processor, memory and monitor that go into it on another. Those item descriptions have nothing in common as text, so descriptions that do not resemble each other do NOT on their own rule out a split PO.",
    4: "The item descriptions given to you are the purchase orders of ONE candidate split, identified as belonging together by an upstream rule that examined the vendor, the dates and the amounts. Judge them as a set. Your task is to say whether their contents support that finding, not to work out which of them belong together."
}

PAYMENT_CONTEXT = {
    1: "A group of items have been identified as potentially coming from a purchase orders for duplicate payment.",
    2: "Duplicate payment have highly similar item descriptions with very minor changes due to human error."
}

CONSTRAINTS = {
    1: "Do not provide any additional explanation beyond 'Yes,{Explanation of similarity}'(no more than 50 words) or 'No,{Explanation of dissimilarity}'(no more than 50 words).",
    2: "Do not add additional keys to the answer, for example: 'similarity explanation:' or 'explanation for dissimilarity:'.",
    3: "For dissimilar items. Begin the explanation with 'Item descriptions are different because...'",
    4: "For similar items. Begin the explanation with 'Item descriptions are similar because...'"
}
