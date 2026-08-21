EXPLANATION_COLUMN = "Explanation"

PO_SET_COLUMN = "ID"
PO_ITEM_DESC_COLUMN = "Descriptions"
PO_SPLIT_BOOLEAN_COLUMN = "Is Split PO"
PO_DEFAULT_NON_CLUSTERED = "No,Item descriptions are different and are not from a split PO"
PAYMENT_DEFAULT_NON_CLUSTERED = "No,Item descriptions are different and are not for duplicate payment"

TASK = "Identify if a list of item description are for the same item"
PO_CONTEXT = {
    1: "A group of items have been identified as potentially coming from a split purchase order (PO), where large POs are split up into smaller POs, to circumvent the checks for large POs.",
    2: "Split PO items have roughly similar item descriptions with minor changes to 'mask' the technique."
}

CONSTRAINTS = {
    1: "Do not provide any additional explanation beyond 'Yes,{Explanation of similarity}'(no more than 50 words) or 'No,{Explanation of dissimilarity}'(no more than 50 words).",
    2: "Do not add additional keys to the answer, for example: 'similarity explanation:' or 'explanation for dissimilarity:'.",
    3: "For dissimilar items. Begin the explanation with 'Item descriptions are different because...'",
    4: "For similar items. Begin the explanation with 'Item descriptions are similar because...'"
}

PAYMENT_SET_COLUMN = "Set"
PAYMENT_ITEM_DESC_COLUMN = "Item Text"
PAYMENT_DUPP_BOOLEAN_COLUMN = "Is Duplicate Payment"

PAYMENT_CONTEXT = {
    1: "A group of items have been identified as potentially coming from a purchase orders for duplicate payment.",
    2: "Duplicate payment have highly similar item descriptions with very minor changes due to human error."
}

