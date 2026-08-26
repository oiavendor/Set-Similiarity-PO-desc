"""
Manual test harness for the PO description matching flow.

Run it with `python scratchpad.py`. Set INPUT_FILE and OUTPUT_FILE below first.

It performs two checks in order:
    1. The endpoints configured in llm.config answer a probe request. Only runs when DEBUGMODE is on.
    2. An Excel workbook can be read, matched with description_matching and written back out.

The second check is skipped when the first one fails, since description_matching
would otherwise fill every row with the "Something went wrong" fallback.
"""

from pathlib import Path
import pandas

from description_matching import description_matching
from utils.AnalysisConfig import get_analysis_config, get_cluster_threshold
from utils.GenAI import (api_query, embed, get_embedding_model_name,
                         get_embedding_provider, get_llm_config)

INPUT_FILE = r"C:\Users\airts_2026\Desktop\Github-Repo (Vendor)\Set-Similiarity-PO-desc\data\Analysis - Split PO.xlsx"
OUTPUT_FILE = r"C:\Users\airts_2026\Desktop\Github-Repo (Vendor)\Set-Similiarity-PO-desc\data\Analysis - Split PO_output.xlsx"

ANALYSIS_TYPE = "po"
REGENERATE = True

# None analyses every group. Setting a number caps the run, and the groups it never reaches are
# filled with the analysis default, which reads as 'No' in the output without anything having been
# compared, so only set one for a deliberately partial trial run.
MODIFY_NUMBER = None

DEBUGMODE = True

# Columns description_matching reads from the workbook, and the ones it writes back. Both
# come from the [analysis.<ANALYSIS_TYPE>] section of llm.config, so renaming a column there
# is picked up here too.
ANALYSIS = get_analysis_config(ANALYSIS_TYPE)
GROUP_COLUMN = ANALYSIS["group_column"]
DESCRIPTION_COLUMN = ANALYSIS["description_column"]
BOOLEAN_COLUMN = ANALYSIS["boolean_column"]
REQUIRED_COLUMNS = [GROUP_COLUMN, DESCRIPTION_COLUMN]
GENERATED_COLUMNS = [BOOLEAN_COLUMN, ANALYSIS["explanation_column"]]


def debug(message: str) -> None:
    """
    Prints a message only when DEBUGMODE is on, so the normal run stays readable.

    Args:
        message (str): The message to print.
    """

    if DEBUGMODE:
        print(f"  [debug] {message}")


def describe_section(section: str) -> None:
    """
    Prints the endpoint a section of llm.config points at, with the API key masked.

    Args:
        section (str): The section to describe, either 'llm' or 'embedding'.
    """

    settings = get_llm_config(section)
    api_key = settings.get("api_key") or ""
    print(f"  [{section}] base_url = {settings.get('base_url')}")
    print(f"  [{section}] model    = {settings.get('model')}")
    print(f"  [{section}] api_key  = {'*' * len(api_key)} ({len(api_key)} characters)")
    for option in ("temperature", "max_output_tokens", "timeout", "max_retries", "default_headers"):
        if settings.get(option):
            debug(f"[{section}] {option} = {settings[option]}")


def test_llm_connection() -> bool:
    """
    Sends a probe prompt to the responses endpoint configured in the [llm] section of llm.config.

    Returns:
        bool: True if the endpoint returned any text, False otherwise.
    """

    print("[1/3] Testing the [llm] endpoint")
    describe_section("llm")

    response = api_query("You are a connectivity probe. Answer with a single word.",
                         "Reply with exactly: OK")

    if not response:
        print("  FAILED: no response from the responses endpoint.\n")
        return False

    debug(f"raw response: {response!r}")
    print(f"  Response: {response.strip()!r}")
    print("  OK\n")
    return True


def test_embedding_connection() -> bool:
    """
    Sends a probe string to whichever embedding provider llm.config selects. description_matching
    clusters on these embeddings, so the flow cannot run without them.

    Returns:
        bool: True if the provider returned a vector, False otherwise.
    """

    provider = get_embedding_provider()
    model = get_embedding_model_name()
    print(f"[2/3] Testing the '{provider}' embedding provider")

    if provider == "local":
        print(f"  [embedding.local] model = {model}")
        debug(f"[embedding.local] {get_llm_config('embedding.local')}")
    else:
        describe_section("embedding")

    threshold = get_cluster_threshold(ANALYSIS_TYPE, model)
    print(f"  Clustering threshold for '{model}' / '{ANALYSIS_TYPE}': {threshold}")

    embeddings = embed(["connectivity probe", "connectivity probe"])

    if not embeddings:
        print("  FAILED: no vector from the embedding provider.\n")
        return False

    debug(f"first 8 dimensions: {embeddings[0][:8]}")
    print(f"  Vector of {len(embeddings[0])} dimensions returned")
    print("  OK\n")
    return True


def check_columns(df: pandas.DataFrame) -> bool:
    """
    Verifies the workbook holds the columns description_matching needs, and adds the columns
    it writes to when they are missing so the assignment back onto the dataframe can enlarge them.

    Args:
        df (pandas.DataFrame): The dataframe read from INPUT_FILE. Modified in place.

    Returns:
        bool: True if every required column is present, False otherwise.
    """

    print(f"  Columns found: {list(df.columns)}")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        print(f"  FAILED: missing required column(s): {missing}")
        return False
    print(f"  Required columns present: {REQUIRED_COLUMNS}")

    for column in GENERATED_COLUMNS:
        if column not in df.columns:
            df[column] = pandas.NA
            print(f"  Added empty output column: '{column}'")

    for column in REQUIRED_COLUMNS:
        null_count = df[column].isnull().sum()
        if null_count:
            print(f"  Warning: '{column}' has {null_count} empty value(s); "
                  f"rows with an empty '{GROUP_COLUMN}' are dropped by the grouping.")

    debug(f"dtypes:\n{df.dtypes.to_string()}")

    group_sizes = df.groupby(GROUP_COLUMN).size()
    debug(f"rows per '{GROUP_COLUMN}':\n{group_sizes.to_string()}")
    singletons = int((group_sizes == 1).sum())
    if singletons:
        print(f"  Warning: {singletons} of {len(group_sizes)} '{GROUP_COLUMN}' group(s) hold a single row; "
              f"clustering needs at least two rows per group and will raise on those.")

    return True


def test_description_matching() -> bool:
    """
    Reads INPUT_FILE, runs the PO description matching over it and writes the result to OUTPUT_FILE.

    Returns:
        bool: True if the workbook was matched and written, False otherwise.
    """

    print("[3/3] Testing description_matching on a workbook")

    if not Path(INPUT_FILE).is_file():
        print(f"  FAILED: INPUT_FILE '{INPUT_FILE}' does not exist. Set it at the top of this file.\n")
        return False

    df = pandas.read_excel(INPUT_FILE)
    print(f"  Read {len(df)} row(s) from {INPUT_FILE}")

    if not check_columns(df):
        print("columns check failed")
        return False
    else:
        print("columns check success")


    debug(f"input frame:\n{df.head(10).to_string()}")

    print(f"  Running description_matching(df, '{ANALYSIS_TYPE}', regenerate={REGENERATE}, "
          f"modify_number={MODIFY_NUMBER})")
    df = description_matching(df, ANALYSIS_TYPE, regenerate=REGENERATE, modify_number=MODIFY_NUMBER)

    debug(f"output frame:\n{df.head(10).to_string()}")

    matched = int(df[BOOLEAN_COLUMN].notnull().sum())
    print(f"  {matched} of {len(df)} row(s) hold a '{BOOLEAN_COLUMN}' value")
    if not matched:
        print("  Warning: no row was matched. Either no group held two similar enough descriptions, "
              "or the embeddings came back unusable; the clustering threshold lives in description_matching.")

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"  Wrote {len(df)} row(s) to {OUTPUT_FILE}")
    print("  OK\n")
    return True


def main() -> None:
    """
    Runs the endpoint checks followed by the workbook check, then prints a summary of the results.
    The endpoint checks only run when DEBUGMODE is on.
    """

    results = {}

    if DEBUGMODE:
        results["llm endpoint"] = test_llm_connection()
        results["embedding endpoint"] = test_embedding_connection()

    # An empty result set means the endpoint checks were skipped, so the workbook check still runs.
    if all(results.values()):
        results["description matching"] = test_description_matching()
    else:
        print("[3/3] Skipping description_matching: an endpoint check failed.\n")
        results["description matching"] = False

    print("Summary")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()



""""
PCR Rush Charge || Diagnostic Pathology-Med/Lrg Species || NUS Rabbit PCR Profile A || Diagnostic Rodent Pathology || GPAV MFI Serology Test (s) || NUS Mouse Diet PCR Panel || NUS Rabbit Serology Profile A || Diagnostic Aquatics Pathology (Non-Fish) || NUS Guinea Pig PCR Profile A || RHDV-Elisa-Send Out Testing (CRL) || PCR Pooling Fee
NUS Guinea Pig PCR Profile C || Diagnostic Pathology Services- medium/large species(non-NHP) || NUS Guinea Pig Serology Profile A
Pathology-Medium service (7-13cm) || Diagnostic Pathology Services- medium/large species(non-NHP) || NUS Mouse Diet PCR Panel || NUS Rabbit PCR Profile A || RHDV-ELISA || NUS Rabbit Serology  Profile A

"""