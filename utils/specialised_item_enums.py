CONTEXT = {
    1: "The procurement team is performing a fruad analysis on whether there are any suspicious transactions between departments and vendors.",
    2: "The information provided for the task: Item Description is used to see if the item is highly specialized in nature, such that the department can only procure it from a handful of vendors."
}

CONSTRAINTS = {
    1: "Do not provide any additional explanation beyond 'Yes,{Explanation of why item is specialized}'(no more than 50 words) or 'No,{Explanation of why item is not specialized}'(no more than 50 words).",
    2: "Only respond with Yes if the line item is highly specialized (not commercially available)",
    3: "Do not add additional keys to the answer, for example: 'explanation:', only provide the value.",
    4: "For spcialized items. Begin the explanation with 'Item is specialized because...'",
    5: "For non-specialized items. Begin the explanation with 'Item is not specialized because...'"
}