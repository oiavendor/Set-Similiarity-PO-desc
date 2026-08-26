"""
Checks on the line matching path, run with `python test_line_matching.py`.

Neither the embedding endpoint nor the language model is touched. `embed` is replaced with a
deterministic character trigram encoder, so identical strings score 1.0, related ones score
somewhere in between and unrelated ones score low, which is enough to assert on the behaviour of
the matching itself. That encoder is far blunter than a real embedding model, in particular it
cannot see that two differently worded descriptions of the same service are the same service, so
the scores printed below are a floor rather than what a real run produces.

These checks say the maths does what it is meant to do. They say nothing about whether the
thresholds in analysis.config are the right numbers, which still needs labelled data.
"""

import sys

import numpy
import pandas

import description_matching as dm
import utils.item_categorizer as ic
import utils.line_matching as lm
from utils.line_matching import (category_similarity, encode_lines, line_weights, parse_lines,
                                 po_similarity)

DIMENSIONS = 512
failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """
    Records one assertion and prints its result.

    Args:
        name (str): What is being asserted.
        condition (bool): Whether it holds.
        detail (str, optional): The observed value, printed alongside.
    """

    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def fake_embed(strings, provider=None, model=None, **parameters):
    """
    Stands in for utils.GenAI.embed with a character trigram bag hashed into a fixed width, so the
    checks need no endpoint, no model weights and no network.

    Args:
        strings: The strings to encode.
        provider: Ignored, present to match the signature of the real embed.
        model: Ignored, present to match the signature of the real embed.

    Returns:
        list: One unit norm vector per string.
    """

    vectors = []
    for text in strings:
        vector = numpy.zeros(DIMENSIONS, dtype=numpy.float32)
        padded = f"  {text}  "
        for position in range(len(padded) - 2):
            vector[hash(padded[position:position + 3]) % DIMENSIONS] += 1.0
        norm = numpy.linalg.norm(vector)
        vectors.append((vector / norm if norm else vector).tolist())
    return vectors


lm.embed = fake_embed  # line_matching imported the name directly, so it is patched there
dm.embed = fake_embed

# Nothing here may reach the categorisation endpoint either. Everything is unclassified until
# section 15 replaces this with a stub that places things, so the checks before it exercise the
# line score on its own.
dm.classify_lines = lambda lines, batch_size=20, regenerate=False: {
    line: ("General & Other", ic.UNCLASSIFIED) for line in lines}

# Three real purchase orders, kept as they appear in the source data.
PO1 = ("PCR Rush Charge || Diagnostic Pathology-Med/Lrg Species || NUS Rabbit PCR Profile A || "
       "Diagnostic Rodent Pathology || GPAV MFI Serology Test (s) || NUS Mouse Diet PCR Panel || "
       "NUS Rabbit Serology Profile A || Diagnostic Aquatics Pathology (Non-Fish) || "
       "NUS Guinea Pig PCR Profile A || RHDV-Elisa-Send Out Testing (CRL) || PCR Pooling Fee")
PO2 = ("NUS Guinea Pig PCR Profile C || Diagnostic Pathology Services- medium/large species(non-NHP) || "
       "NUS Guinea Pig Serology Profile A")
PO3 = ("Pathology-Medium service (7-13cm) || Diagnostic Pathology Services- medium/large species(non-NHP) || "
       "NUS Mouse Diet PCR Panel || NUS Rabbit PCR Profile A || RHDV-ELISA || NUS Rabbit Serology  Profile A")


def build(lines_per_po, threshold=0.75, weights=None):
    """
    Encodes a set of POs and returns everything po_similarity needs to score a pair of them.

    Args:
        lines_per_po: The parsed line items of each PO.
        threshold (float, optional): The line matching threshold to use. Defaults to 0.75.
        weights (dict, optional): Overrides the weights derived from these POs.

    Returns:
        tuple: The vectors, the line index, the weights and the threshold.
    """

    vectors, index = encode_lines({line for lines in lines_per_po for line in lines})
    if weights is None:
        weights = line_weights(lines_per_po)
    return vectors, index, weights, threshold


print("\n1. parse_lines")
lines1, lines3 = parse_lines(PO1), parse_lines(PO3)
check("PO1 splits into 11 line items", len(lines1) == 11, f"got {len(lines1)}")
check("PO3 splits into 6 line items", len(lines3) == 6, f"got {len(lines3)}")
check("whitespace and case are normalised",
      "nus rabbit serology profile a" in lines1 and "nus rabbit serology profile a" in lines3)
check("a trailing separator adds no empty line", parse_lines("a || b ||") == ["a", "b"])
check("a repeated line item is kept once", parse_lines("a || A || b") == ["a", "b"])
check("NaN yields no lines", parse_lines(float("nan")) == [])
check("None yields no lines", parse_lines(None) == [])
check("a description with no separator is one line", parse_lines("just one item") == ["just one item"])

print("\n2. po_similarity on the real sample")
vectors, index, weights, threshold = build([lines1, parse_lines(PO2), lines3])
containment, jaccard, pairs = po_similarity(lines1, lines3, vectors, index, weights, threshold)
matched = {(left, right) for left, right, _ in pairs}
check("the exact repeat of the mouse diet panel is matched",
      ("nus mouse diet pcr panel", "nus mouse diet pcr panel") in matched)
check("the exact repeat of the rabbit PCR profile is matched",
      ("nus rabbit pcr profile a", "nus rabbit pcr profile a") in matched)
check("the double-spaced serology profile is matched",
      ("nus rabbit serology profile a", "nus rabbit serology profile a") in matched)
check("containment exceeds jaccard on this uneven pair",
      containment > jaccard, f"containment {containment:.3f} vs jaccard {jaccard:.3f}")
check("containment is a fraction", 0.0 < containment <= 1.0, f"{containment:.3f}")
print(f"        containment {containment:.3f}, jaccard {jaccard:.3f}, {len(pairs)} matched line items")

print("\n3. a plain row maximum would double count, the assignment does not")
a = ["widget type a", "widget type a x", "totally unrelated dredging works"]
b = ["widget type a"]
vectors, index, weights, threshold = build([a, b], threshold=0.6, weights={})
containment, _, pairs = po_similarity(a, b, vectors, index, weights, threshold)
check("one line of B partners at most one line of A", len(pairs) == 1, f"got {len(pairs)} pairs")
check("containment stays within bounds", containment <= 1.0)

print("\n4. corpus-wide boilerplate carries no weight")
boiler = "freight and handling charge"
documents = [[boiler, "steel beam 200mm"], [boiler, "office chair mesh"], [boiler, "laptop docking station"]]
weights = line_weights(documents)
check("a line every PO carries scores 0", weights[boiler] == 0.0, f"got {weights[boiler]:.3f}")
check("a line only one PO carries scores above 0", weights["steel beam 200mm"] > 0)
vectors, index, _, threshold = build(documents, threshold=0.75)
containment, jaccard, pairs = po_similarity(documents[0], documents[1], vectors, index, weights, threshold)
check("two POs sharing only boilerplate score 0", containment == 0.0 and jaccard == 0.0,
      f"containment {containment:.3f}, jaccard {jaccard:.3f}")
check("boilerplate is not offered to the model as evidence either", pairs == [], f"got {pairs}")

print("\n5. identical POs fall back to unweighted counts rather than dividing by zero")
same = ["alpha reagent kit", "beta reagent kit"]
vectors, index, weights, threshold = build([same, list(same)])
containment, jaccard, pairs = po_similarity(same, list(same), vectors, index, weights, threshold)
check("identical POs score 1.0 containment", abs(containment - 1.0) < 1e-9, f"{containment:.3f}")
check("identical POs score 1.0 jaccard", abs(jaccard - 1.0) < 1e-9, f"{jaccard:.3f}")

print("\n6. empty descriptions are safe")
vectors, index, weights, threshold = build([["alpha reagent kit"], []])
check("an empty PO scores 0 with no pairs",
      po_similarity([], ["alpha reagent kit"], vectors, index, weights, threshold) == (0.0, 0.0, []))

print("\n7. end to end, with the language model stubbed")
prompts = []
dm.api_query = lambda system_prompt, user_prompt, **kw: (
    prompts.append(user_prompt),
    "Yes,Item descriptions are similar because they repeat the same tests.")[1]

frame = pandas.DataFrame({
    "Unique_Group_Number": [1, 1, 1, 2, 2, 3],
    "Line_desc": [
        "alpha reagent kit || beta reagent kit || freight and handling charge",
        "alpha reagent kit || beta reagent kit || freight and handling charge",
        "bulldozer hire 20 tonne || site clearance works || freight and handling charge",
        "gamma assay plate || delta assay plate",
        "office chair mesh black || desk riser adjustable",
        "solitary line item with no partner",
    ],
})
result = dm.description_matching(frame, "po", regenerate=True, embedding_model="all-mpnet-base-v2")

check("both verdict columns exist",
      "Is Split PO" in result.columns and "Explanation" in result.columns)
check("no blank verdict is left anywhere", result["Is Split PO"].isnull().sum() == 0)
check("no blank explanation is left anywhere", result["Explanation"].isnull().sum() == 0)
# The group IS the split, so every row of a multi-row group carries the model's answer for that
# group. Nothing is filtered out on the grounds that its wording differs from its neighbours.
check("every row of group 1 carries the group's verdict",
      list(result.loc[[0, 1, 2], "Is Split PO"]) == ["Yes", "Yes", "Yes"],
      str(list(result.loc[[0, 1, 2], "Is Split PO"])))
check("the unrelated pair in group 2 is ruled on rather than pre-filtered",
      list(result.loc[[3, 4], "Is Split PO"]) == ["Yes", "Yes"],
      str(list(result.loc[[3, 4], "Is Split PO"])))
check("the lone PO of group 3 is filled with the default",
      result.at[5, "Is Split PO"] == "No", str(result.at[5, "Is Split PO"]))
check("one prompt per multi-row group, not one per cluster",
      len(prompts) == 2, f"{len(prompts)} prompts for 2 multi-row groups")
check("every PO of a group appears in its prompt",
      all(description in prompts[0] for description in frame.loc[[0, 1, 2], "Line_desc"]))
check("the prompt carries the matched line items as evidence",
      any("matched across the descriptions above" in prompt for prompt in prompts))
check("the prompt still opens with the wording blob mode uses",
      all(prompt.startswith("Item Descriptions: [") for prompt in prompts))

print("\n8. a group is never split up on the strength of its wording")
# A shares a line with B, B shares a different line with C, A and C share nothing. The old
# clustering broke this group apart; the group is the split, so all three are ruled on together.
chain = pandas.DataFrame({
    "Unique_Group_Number": [9, 9, 9],
    "Line_desc": [
        "alpha widget",
        "alpha widget || zeta gadget",
        "zeta gadget",
    ],
})
chain_prompts = []
dm.api_query = lambda s, u, **kw: (
    chain_prompts.append(u),
    "Yes,Item descriptions are similar because they share a reagent.")[1]
flagged = list(dm.description_matching(chain, "po", regenerate=True,
                                       embedding_model="all-mpnet-base-v2")["Is Split PO"])
check("all three POs of the group are ruled on together", flagged == ["Yes"] * 3, str(flagged))
check("the group was put to the model exactly once", len(chain_prompts) == 1,
      f"{len(chain_prompts)} prompts")

print("\n9. fill_empty can be turned off")
sparse = pandas.DataFrame({"Unique_Group_Number": [7],
                           "Line_desc": ["a lone unmatched purchase order line"]})
untouched = dm.description_matching(sparse, "po", regenerate=True, fill_empty=False,
                                    embedding_model="all-mpnet-base-v2")
check("blanks survive when fill_empty is off", untouched["Is Split PO"].isnull().all())

print("\n10. blob mode still runs unchanged")
blob = pandas.DataFrame({"Set": [1, 1],
                         "Item Text": ["invoice for annual licence renewal",
                                       "invoice for annual licence renewal"]})
dm.api_query = lambda s, u, **kw: "Yes,Item descriptions are similar because they are the same invoice."
blobbed = dm.description_matching(blob, "payment", regenerate=True, embedding_model="all-mpnet-base-v2")
check("blob mode flags the duplicate pair",
      list(blobbed["Is Duplicate Payment"]) == ["Yes", "Yes"],
      str(list(blobbed["Is Duplicate Payment"])))

print("\n11. the taxonomy loads and is well formed")
taxonomy = ic.load_taxonomy()
level_2_all = [level_2 for group in taxonomy.values() for level_2 in group]
check("every level 1 category has level 2 categories under it",
      all(len(group) > 0 for group in taxonomy.values()), f"{len(taxonomy)} level 1")
check("no level 2 category is listed twice", len(level_2_all) == len(set(level_2_all)),
      f"{len(level_2_all)} level 2")
check("the unclassified bucket is in the taxonomy", ic.UNCLASSIFIED in level_2_all)
print(f"        {len(taxonomy)} level 1 categories, {len(level_2_all)} level 2 categories")

print("\n12. the categoriser reads the model's answer back correctly")
batch = ["amd ryzen 9 7950x", "lenovo thinkstation p360", "unintelligible ref xyz"]
answer = ("1|IT & Technology|End-user computing devices and peripherals\n"
          "2|IT & Technology|Datacentre compute and storage infrastructure\n"
          "3|Not A Real Level One|Whatever")
parsed = ic.parse_answer(answer, batch)
check("a valid answer parses", parsed[batch[0]] == ("IT & Technology", "End-user computing devices and peripherals"))
check("an answer is matched by its item number, not its position",
      parsed[batch[1]][1] == "Datacentre compute and storage infrastructure")
check("an unknown level 1 is dropped", batch[2] not in parsed)
check("a level 2 that does not belong to its level 1 keeps the level 1 only",
      ic.parse_answer("1|IT & Technology|Hotel and lodging", batch)[batch[0]] == ("IT & Technology", None))
check("a garbled answer yields nothing rather than a wrong category",
      ic.parse_answer("I think item 1 is probably IT related", batch) == {})
check("an out of range item number is ignored", ic.parse_answer("9|IT & Technology|Language services", batch) == {})

print("\n13. category_similarity tiers level 1 against level 2")
IT_USER = ("IT & Technology", "End-user computing devices and peripherals")
IT_DATACENTRE = ("IT & Technology", "Datacentre compute and storage infrastructure")
LAB = ("Laboratory & Research", "Bulk and fine chemicals for research use")
score_exact, _ = category_similarity([IT_USER], [IT_USER], {}, 0.5)
score_partial, _ = category_similarity([IT_USER], [IT_DATACENTRE], {}, 0.5)
score_none, _ = category_similarity([IT_USER], [LAB], {}, 0.5)
check("both levels agreeing scores in full", score_exact == 1.0, f"{score_exact:.2f}")
check("level 1 agreeing alone earns partial credit", score_partial == 0.5, f"{score_partial:.2f}")
check("level 1 disagreeing scores nothing", score_none == 0.0, f"{score_none:.2f}")
check("a category every PO buys from is weighted out",
      category_similarity([IT_USER], [IT_USER], {IT_USER: 0.0}, 0.5)[0] == 0.0)

print("\n14. the AMD Ryzen and Lenovo ThinkStation case")
ryzen, thinkstation = "amd ryzen 9 7950x processor", "lenovo thinkstation p360 workstation"
vectors, index, weights, threshold = build([[ryzen], [thinkstation]])
line_score, _, _ = po_similarity([ryzen], [thinkstation], vectors, index, weights, threshold)
check("the line score does not see the pair, correctly", line_score == 0.0, f"{line_score:.3f}")
for classified_as, expected in ((IT_USER, 1.0), (IT_DATACENTRE, 0.5)):
    score, _ = category_similarity([IT_USER], [classified_as], {}, 0.5)
    check(f"the category score does see it when the processor is filed under {classified_as[1][:28]}",
          score >= 0.5, f"{score:.2f} (expected {expected})")

print("\n15. end to end, complementary split with the categoriser stubbed")
ic.classify_lines = lambda lines, batch_size=20, regenerate=False: {
    line: (("IT & Technology", "End-user computing devices and peripherals") if
           any(word in line for word in ("ryzen", "thinkstation", "monitor", "docking"))
           else ("Laboratory & Research", "Bulk and fine chemicals for research use") if "reagent" in line
           else ("General & Other", ic.UNCLASSIFIED))
    for line in lines}
dm.classify_lines = ic.classify_lines

prompts = []
dm.api_query = lambda s, u, **kw: (prompts.append(u),
                                   "Yes,Item descriptions are similar because they are parts of one computer purchase.")[1]
complementary = pandas.DataFrame({
    "Unique_Group_Number": [1, 1, 2, 2],
    "Line_desc": [
        "amd ryzen 9 7950x processor || 64gb ddr5 memory module",
        "lenovo thinkstation p360 workstation || dell u2723 monitor",
        "sodium chloride analytical reagent 500g",
        "amd ryzen 9 7950x processor || 64gb ddr5 memory module",
    ],
})
ruled = dm.description_matching(complementary, "po", regenerate=True, embedding_model="all-mpnet-base-v2")
check("the complementary split is caught despite no shared wording",
      list(ruled.loc[[0, 1], "Is Split PO"]) == ["Yes", "Yes"],
      str(list(ruled.loc[[0, 1], "Is Split PO"])))
check("the reagent group reaches the model instead of being cleared on wording",
      ruled.at[2, "Is Split PO"] == "Yes", str(ruled.at[2, "Is Split PO"]))
check("the wording-free prompt asks the complementary question",
      any("complementary parts of ONE purchase" in prompt for prompt in prompts))
check("the wording-free prompt says no wording matched",
      any("NO line item matched on wording" in prompt for prompt in prompts))
check("the category evidence is shown",
      any("buy from the same categories" in prompt for prompt in prompts))
if prompts:
    print("        ---- category-only prompt ----")
    for line in prompts[0].splitlines():
        print(f"        {line[:110]}")

print("\n16. categories only supply evidence, they no longer decide anything")
import utils.AnalysisConfig as ac
_real_option = ac.get_analysis_option
dm.get_analysis_option = lambda a, o: "false" if o == "use_categories" else _real_option(a, o)
prompts.clear()
off = dm.description_matching(complementary.drop(columns=["Is Split PO", "Explanation"]), "po",
                              regenerate=True, embedding_model="all-mpnet-base-v2")
# Categories used to be the only thing that could carry a complementary split as far as the model.
# The group carries it now, so turning them off costs evidence in the prompt, not the finding.
check("with categories off the complementary split still reaches the model",
      off.at[0, "Is Split PO"] == "Yes", str(off.at[0, "Is Split PO"]))
check("with categories off no category evidence is shown",
      not any("buy from the same categories" in prompt for prompt in prompts))
check("the complementary question is still asked without categories",
      any("complementary parts of ONE purchase" in prompt for prompt in prompts))
dm.get_analysis_option = _real_option

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURES: {failures}"))
sys.exit(1 if failures else 0)
