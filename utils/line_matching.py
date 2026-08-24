"""
Line level matching for purchase order description sets.

Each row this pipeline reads is one purchase order, and its description column holds every line
item of that PO joined by a separator. Embedding that joined string as a single vector averages
the line items together, so a PO's vector is the centroid of its items. Two things go wrong at
that centroid:

    - the line items a pair of POs do NOT share drag the measured similarity down, so genuine
      matches never get a clean vote, and
    - a PO's vector drifts toward generic text as it gets longer, which makes the score partly a
      function of line count rather than content. An 11 line PO compared against a 3 line PO is
      penalised for the size gap alone.

This module reorders the work to avoid that. Instead of aggregate, embed, compare it does embed
each line, compare every line pair, aggregate the scores. Aggregation then happens on similarity
scores, where the formula is a choice, rather than on vectors, where mean pooling is forced. An
unmatched line can no longer damage a matched line's score; it can only fail to be counted.

The same rule of never averaging what you are asking a maximum question about applies three
times: vectors are not pooled, a line's verdict is its best partner rather than its mean score
against all candidates, and the two directions of containment are combined with max rather than
a symmetric ratio.
"""

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy
from scipy.optimize import linear_sum_assignment

from utils.GenAI import embed


def normalize_line(text: str) -> str:
    """
    Normalises one line item description into the form that is embedded and used as the cache key.

    Collapsing whitespace and case folding turns near misses into exact matches, which both
    improves recall and removes them from the set of strings needing an embedding at all. Real
    data carries plenty of these: 'NUS Rabbit Serology  Profile A' with a double space and
    'NUS Rabbit Serology Profile A' are the same item typed twice.

    Punctuation is deliberately left alone. Stripping it would merge descriptions differing only
    by a code or a size in brackets, which for procurement data are usually different items.

    Args:
        text (str): The raw line item description.

    Returns:
        str: The normalised description.
    """

    return " ".join(str(text).split()).casefold()


def parse_lines(text, separator: str = "||") -> List[str]:
    """
    Splits one PO's aggregated description into its normalised line items.

    Empty fragments are dropped, which matters because a trailing separator would otherwise
    contribute an empty string that embeds to noise and inflates the PO's line count. Repeated
    line items within one PO are dropped too: a PO listing the same description twice is no more
    evidence than listing it once, and leaving the duplicate in would let it consume a second
    partner during assignment.

    Args:
        text: The aggregated description of one PO. Non-string values, including NaN, yield [].
        separator (str, optional): The delimiter the line items are joined by. Defaults to '||'.

    Returns:
        List[str]: The PO's unique normalised line items, in the order they first appear.
    """

    if text is None or text != text:  # None, or a NaN, which is not equal to itself
        return []

    seen = set()
    lines = []
    for fragment in str(text).split(separator):
        line = normalize_line(fragment)
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def encode_lines(lines: Iterable[str], provider: str = None,
                 model: str = None) -> Tuple[numpy.ndarray, Dict[str, int]]:
    """
    Embeds a set of line items once each and returns them as a matrix that can be sliced per PO.

    Line items repeat heavily across the POs of a set, so embedding the distinct strings rather
    than every occurrence is where the cost of line level matching is won back. The vectors are
    re-normalised here whatever the provider returned, so that the dot products taken downstream
    are cosine similarities regardless of what llm.config sets for normalize.

    Args:
        lines (Iterable[str]): The normalised line items to embed. Duplicates are ignored.
        provider (str, optional): Overrides the embedding provider configured in llm.config.
        model (str, optional): Overrides the embedding model configured for that provider.

    Returns:
        Tuple[numpy.ndarray, Dict[str, int]]: A unit norm matrix of n rows by however many
        dimensions the model produces, and the row index of each line item within it.

    Raises:
        RuntimeError: If the embedding provider returned nothing.
    """

    unique = sorted(set(lines))
    if not unique:
        return numpy.zeros((0, 0), dtype=numpy.float32), {}

    vectors = embed(unique, provider, model)
    if vectors is None:
        raise RuntimeError("The embedding provider returned no vectors. Check the [embedding] "
                           "section of llm.config and the endpoint it points at.")

    matrix = numpy.asarray(vectors, dtype=numpy.float32)
    norms = numpy.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / numpy.maximum(norms, 1e-12)  # the dot product is now the cosine similarity
    return matrix, {line: index for index, line in enumerate(unique)}


def line_weights(documents: Sequence[Sequence[str]]) -> Dict[str, float]:
    """
    Scores how much each line item can tell you about which POs belong together, as the inverse
    document frequency of that line over the POs it is given.

    Without this, shared boilerplate is the main source of false matches. The sets these analyses
    read come from an upstream rule that over-fires, often on vendor, so every PO in a set tends
    to carry that vendor's fee and freight lines. Matching on those would re-glue exactly the POs
    the analysis is meant to separate. A line every PO carries scores 0 and drops out of the
    result entirely; a line only one PO carries counts for the most.

    Args:
        documents (Sequence[Sequence[str]]): The line items of each PO in the corpus.

    Returns:
        Dict[str, float]: The weight of each line item. Lines absent from the corpus default to
        1.0 when looked up by the caller.
    """

    total = len(documents)
    if total < 2:
        return {}  # nothing to compare against, so every line falls back to a weight of 1.0

    frequency = Counter()
    for document in documents:
        frequency.update(set(document))
    return {line: math.log(total / count) for line, count in frequency.items()}


def po_similarity(lines_a: Sequence[str], lines_b: Sequence[str], vectors: numpy.ndarray,
                  index: Dict[str, int], weights: Dict[str, float],
                  line_threshold: float) -> Tuple[float, float, List[Tuple[str, str, float]]]:
    """
    Scores how much of one PO's line items appear in another's.

    Every line of A is compared against every line of B, and the matrix of scores is resolved into
    a one to one assignment. A plain row or column maximum would let a single line of B stand in
    as the partner of several lines of A, which inflates the score whenever a pair shares
    boilerplate or carries two variants of the same item. Only assigned pairs at or above the line
    threshold count as matches.

    Both scores are weighted by the line weights, so a match on a line every PO in the corpus
    carries contributes nothing. If that leaves either PO with no weight at all, which happens
    when two POs are identical, both fall back to unweighted line counts for that comparison.

    Args:
        lines_a (Sequence[str]): The normalised line items of the first PO.
        lines_b (Sequence[str]): The normalised line items of the second PO.
        vectors (numpy.ndarray): The unit norm embedding matrix from encode_lines.
        index (Dict[str, int]): The row of each line item within that matrix.
        weights (Dict[str, float]): The weight of each line item, from line_weights.
        line_threshold (float): The cosine SIMILARITY at or above which two lines are the same
            item. Higher is stricter, the opposite direction to the distance thresholds blob
            mode clusters at.

    Returns:
        Tuple[float, float, List[Tuple[str, str, float]]]:
            - containment: the larger of the two directions, so a small PO wholly absorbed into a
              larger one still scores high. That is the split PO signature, and a symmetric ratio
              would hide it behind the size gap.
            - jaccard: the symmetric overlap, reported alongside rather than used to cluster.
            - pairs: the matched line items and their similarity, strongest first, as the evidence
              handed to the language model.
    """

    if not lines_a or not lines_b:
        return 0.0, 0.0, []

    a = vectors[[index[line] for line in lines_a]]
    b = vectors[[index[line] for line in lines_b]]
    scores = a @ b.T  # both sides are normalised, so this is the cosine similarity

    rows, columns = linear_sum_assignment(scores, maximize=True)
    matched = [(row, column) for row, column in zip(rows, columns)
               if scores[row, column] >= line_threshold]
    if not matched:
        return 0.0, 0.0, []

    weight_a = [weights.get(line, 1.0) for line in lines_a]
    weight_b = [weights.get(line, 1.0) for line in lines_b]
    total_a, total_b = sum(weight_a), sum(weight_b)

    if total_a <= 0 or total_b <= 0:  # every line is corpus wide boilerplate, or the POs are identical
        weight_a = [1.0] * len(lines_a)
        weight_b = [1.0] * len(lines_b)
        total_a, total_b = float(len(lines_a)), float(len(lines_b))

    matched_a = sum(weight_a[row] for row, _ in matched)
    matched_b = sum(weight_b[column] for _, column in matched)
    intersection = sum(min(weight_a[row], weight_b[column]) for row, column in matched)

    containment = max(matched_a / total_a, matched_b / total_b)
    jaccard = intersection / (total_a + total_b - intersection)

    # A pair that carries no weight on either side contributed nothing to the scores above, so it
    # is not evidence and is left out of what the language model is shown. Without this the
    # boilerplate that was deliberately weighted out of the score would walk back in through the
    # prompt, and a shared freight line would read as a reason to say yes.
    pairs = sorted(((lines_a[row], lines_b[column], float(scores[row, column]))
                    for row, column in matched
                    if max(weight_a[row], weight_b[column]) > 0),
                   key=lambda pair: pair[2], reverse=True)
    return containment, jaccard, pairs
