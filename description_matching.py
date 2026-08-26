import numpy
import pandas
from rapidfuzz import fuzz
from utils.GenAI import embed, generate_system_prompt, api_query, get_embedding_model_name
from utils.AnalysisConfig import (get_analysis_option, get_cluster_threshold, get_line_threshold,
                                  get_pair_threshold)
from utils.description_matching_enums import (PO_SET_COLUMN, PO_ITEM_DESC_COLUMN, PO_SPLIT_BOOLEAN_COLUMN,
                                              PO_EXPLANATION_COLUMN, PO_DEFAULT_NON_CLUSTERED,
                                              PAYMENT_SET_COLUMN, PAYMENT_ITEM_DESC_COLUMN,
                                              PAYMENT_DUPP_BOOLEAN_COLUMN, PAYMENT_EXPLANATION_COLUMN,
                                              PAYMENT_DEFAULT_NON_CLUSTERED, PO_CONTEXT, PAYMENT_CONTEXT,
                                              CONSTRAINTS, TASK)
from typing import Literal
from utils.shared_functions import string_from_enum
from utils.line_matching import (parse_lines, encode_lines, line_weights, po_similarity,
                                 category_similarity)
from utils.item_categorizer import UNCLASSIFIED, classify_lines, po_categories
from sklearn.cluster import AgglomerativeClustering

role = "audit associate"
task = "identify if the list of item descriptions are for"

# How many matched line items are shown to the language model per cluster. They are the evidence
# for the verdict, but a large cluster produces hundreds of them and burying the prompt costs more
# than it explains. The strongest matches are kept.
MAX_EVIDENCE_PAIRS = 20


def split_verdict(text: str, default: str) -> tuple:
    """
    Splits a 'Yes,{explanation}' or 'No,{explanation}' answer into its two columns, falling back
    to the analysis default when the answer does not carry a comma to split on.

    Args:
        text (str): The answer to split.
        default (str): The analysis default, itself in the same comma separated form.

    Returns:
        tuple: The boolean verdict and the explanation.
    """

    parts = text.split(",", 1)
    if len(parts) != 2:
        parts = default.split(",", 1)
    return parts[0].strip(), parts[1].strip()


def description_matching(df: pandas.DataFrame, analysis_type: Literal["po", "payment"], regenerate: bool = False,
                         modify_number: int = 500, embedding_provider: str = None,
                         embedding_model: str = None, fill_empty: bool = True) -> pandas.DataFrame:
    """
    Clusters the rows of each group by how alike their item descriptions are, then asks the
    language model to rule on each cluster it finds.

    Two matching modes are available, chosen per analysis by match_mode in analysis.config:

        blob  embeds each row's whole description as a single vector. Simple, but when the
              description is an aggregation of many line items the vector is their centroid, and
              the items a pair of rows do not share drag the score down.
        line  splits the description on line_separator and matches the line items directly,
              aggregating similarity scores rather than vectors. See utils/line_matching.py.

    The po analysis also gets a 'Set Similarity' column: each row's best token_set_ratio (0 to
    100) against the other descriptions in its set. It is a plain string measure written
    alongside the verdicts and feeds nothing downstream. See set_similarity_column.

    When an analysis names local_assessment_column and arbiter_assessment_column in
    analysis.config, line mode also writes per row which stage decided its verdict: what the
    local embedding model concluded, and what the language model ruled where it was consulted.
    Both columns are placed immediately after the description column in the returned frame.

    Args:
        df (pandas.DataFrame): The rows to analyse.
        analysis_type (Literal['po', 'payment']): Which analysis to run, naming a section of analysis.config.
        regenerate (bool, optional): Whether to re-analyse groups that already hold verdicts. Defaults to False.
        modify_number (int, optional): How many groups to analyse before stopping. Defaults to 500.
        embedding_provider (str, optional): Overrides the embedding provider configured in llm.config.
        embedding_model (str, optional): Overrides the embedding model configured for that provider.
        fill_empty (bool, optional): Whether to fill every row left without a verdict with the
            analysis default, so the output carries no blank cells. Defaults to True.

    Returns:
        pandas.DataFrame: The frame, with the boolean and explanation columns written.

    Raises:
        ValueError: If the analysis type or the configured match mode is not one of the known ones.
    """

    if analysis_type == "po":
        group_by = PO_SET_COLUMN
        item_description_column = PO_ITEM_DESC_COLUMN
        context = string_from_enum(PO_CONTEXT)
        boolean_column = PO_SPLIT_BOOLEAN_COLUMN
        explanation_column = PO_EXPLANATION_COLUMN
        default = PO_DEFAULT_NON_CLUSTERED
    elif analysis_type == "payment":
        group_by = PAYMENT_SET_COLUMN
        item_description_column = PAYMENT_ITEM_DESC_COLUMN
        context = string_from_enum(PAYMENT_CONTEXT)
        boolean_column = PAYMENT_DUPP_BOOLEAN_COLUMN
        explanation_column = PAYMENT_EXPLANATION_COLUMN
        default = PAYMENT_DEFAULT_NON_CLUSTERED
    else:
        raise ValueError(f"Unknown analysis_type '{analysis_type}'. Use 'po' or 'payment'.")

    # Created empty when the source data lacks them, so that the skip check below has something to
    # read and the fill at the end has something to fill.
    for column in (boolean_column, explanation_column):
        if column not in df.columns:
            df[column] = pandas.NA

    # Written before the verdict pass and independent of it, so the column is complete even when
    # a run capped by modify_number leaves verdicts for later. Only the po analysis carries it.
    if analysis_type == "po":
        set_similarity_column(df, group_by, item_description_column,
                              get_analysis_option(analysis_type, "line_separator"))

    embedding_model_name = get_embedding_model_name(embedding_provider, embedding_model)
    constraints = string_from_enum(CONSTRAINTS)

    system_prompt = generate_system_prompt(role=role,
                                           task=TASK,
                                           context=context,
                                           constraints=constraints)

    match_mode = get_analysis_option(analysis_type, "match_mode").lower()

    # Only line mode computes anything worth reporting per stage, so the assessment columns are
    # dropped rather than written empty when a blob analysis configures them by mistake.
    local_assessment_column = get_analysis_option(analysis_type, "local_assessment_column")
    arbiter_assessment_column = get_analysis_option(analysis_type, "arbiter_assessment_column")
    if match_mode != "line" and (local_assessment_column or arbiter_assessment_column):
        print(f"Assessment columns are only written in line mode; analysis '{analysis_type}' "
              f"runs in '{match_mode}' mode, so they are skipped.")
        local_assessment_column = arbiter_assessment_column = ""
    for column in (local_assessment_column, arbiter_assessment_column):
        if column and column not in df.columns:
            df[column] = pandas.NA

    arguments = dict(df=df, group_by=group_by, item_description_column=item_description_column,
                     boolean_column=boolean_column, explanation_column=explanation_column,
                     default=default, system_prompt=system_prompt,
                     embedding_model_name=embedding_model_name,
                     embedding_provider=embedding_provider, embedding_model=embedding_model,
                     regenerate=regenerate, modify_number=modify_number)

    if match_mode == "line":
        line_matching(analysis_type=analysis_type, local_assessment_column=local_assessment_column,
                      arbiter_assessment_column=arbiter_assessment_column, **arguments)
    elif match_mode == "blob":
        blob_matching(analysis_type=analysis_type, **arguments)
    else:
        raise ValueError(f"Unknown match_mode '{match_mode}' for analysis '{analysis_type}' in "
                         f"analysis.config. Use 'blob' or 'line'.")

    if fill_empty:
        fill_empty_verdicts(df, boolean_column, explanation_column, default)

    # A row the grouping never saw, because its group value is empty, holds no assessment yet.
    if local_assessment_column:
        empty = df[local_assessment_column].isnull()
        df.loc[empty, local_assessment_column] = (f"Not assessed: the row has no '{group_by}' "
                                                  f"value, so it belongs to no group")
    if arbiter_assessment_column:
        empty = df[arbiter_assessment_column].isnull()
        df.loc[empty, arbiter_assessment_column] = "Not consulted: the group was not analysed"

    assessment_columns = [column for column in (local_assessment_column, arbiter_assessment_column)
                          if column]
    if assessment_columns:
        df = place_columns_after(df, item_description_column, assessment_columns)

    return df


def place_columns_after(df: pandas.DataFrame, anchor: str, columns: list) -> pandas.DataFrame:
    """
    Returns the frame with the given columns moved to sit immediately after the anchor column,
    in the order given. All named columns must exist.

    Args:
        df (pandas.DataFrame): The frame to reorder.
        anchor (str): The column the moved columns should follow.
        columns (list): The columns to move.

    Returns:
        pandas.DataFrame: The reordered frame.
    """

    order = [column for column in df.columns if column not in columns]
    position = order.index(anchor) + 1
    for offset, column in enumerate(columns):
        order.insert(position + offset, column)
    return df[order]


def fill_empty_verdicts(df: pandas.DataFrame, boolean_column: str, explanation_column: str,
                        default: str) -> None:
    """
    Fills every row still without a verdict with the analysis default, in place.

    A row is left empty when its group produced no cluster, when it sat in a group that did
    cluster but did not itself land in one, or when the run stopped at modify_number before
    reaching its group. Those are all 'nothing found here', so the output workbook carries a
    verdict on every row rather than a mix of 'No' and blanks.

    Note that this does not distinguish 'analysed and cleared' from 'never analysed'. If a run is
    capped by modify_number, the groups it never reached are filled as 'No' too.

    Args:
        df (pandas.DataFrame): The frame to fill.
        boolean_column (str): The verdict column.
        explanation_column (str): The explanation column.
        default (str): The analysis default, in 'No,{explanation}' form.
    """

    verdict, explanation = split_verdict(default, default)
    filled = 0

    for column, value in ((boolean_column, verdict), (explanation_column, explanation)):
        empty = df[column].isnull() | (df[column].astype("string").str.strip() == "")
        filled = max(filled, int(empty.sum()))
        df.loc[empty, column] = value

    if filled:
        print(f"Filled {filled} rows that held no verdict with the analysis default '{verdict}'")


def normalize_description(text, separator: str) -> str:
    """
    Normalises one whole aggregated description for token set comparison.

    The separator is replaced with a space rather than stripped, so the words either side of it
    stay separate tokens and the separator itself can never count as wording two rows share.
    Whitespace is collapsed and case folded the same way individual line items are in line mode.

    Args:
        text: The aggregated description of one row. Non-string values, including NaN, yield ''.
        separator (str): The delimiter the line items are joined by.

    Returns:
        str: The normalised description.
    """

    if text is None or text != text:  # None, or a NaN, which is not equal to itself
        return ""
    return " ".join(str(text).replace(separator, " ").split()).casefold()


def set_similarity_column(df: pandas.DataFrame, group_by: str, item_description_column: str,
                          separator: str, column_name: str = "Set Similarity") -> None:
    """
    Writes each row's best token_set_ratio against the other descriptions in its set, in place.

    This is a plain string measure reported alongside the verdicts, not an input to them: nothing
    downstream reads the column. token_set_ratio tokenises both descriptions on whitespace and
    scores the overlap of the two token SETS from 0 to 100, ignoring word order and repetition.
    Because the shared tokens are scored on their own, a small PO whose wording is wholly absorbed
    into a larger one still scores near 100 despite the size gap, which is the same containment
    idea line mode computes, at word level. What it cannot see is meaning: 'laptop' and 'notebook
    computer' share no tokens and score low here even though an embedding places them together.

    It costs no requests, so every group is scored on every run regardless of regenerate and
    modify_number. A row alone in its set has nothing to be compared against and stays blank,
    which keeps 'no partner exists' distinct from 'a partner exists and shares no wording'.

    Args:
        df (pandas.DataFrame): The frame to write the column into.
        group_by (str): The column rows are grouped by; the rows sharing a value are one set.
        item_description_column (str): The column holding the aggregated line item descriptions.
        separator (str): What the line items of one row are joined by.
        column_name (str, optional): The column written. Defaults to 'Set Similarity'.

    Raises:
        ValueError: If the frame's index is not unique.
    """

    if not df.index.is_unique:
        raise ValueError("Set similarity addresses rows by index label, so the frame needs a "
                         "unique index. Call df.reset_index(drop=True) before passing it in.")

    if column_name not in df.columns:
        df[column_name] = pandas.NA

    print(f"Scoring '{column_name}' with token_set_ratio within each {group_by} group")

    for group_ind, group_data in df.groupby(group_by):
        if len(group_data) < 2:
            continue
        texts = [normalize_description(text, separator)
                 for text in group_data[item_description_column]]

        size = len(texts)
        best = [0.0] * size
        for first in range(size):
            for second in range(first + 1, size):
                # Two blank descriptions share an absence of text, not wording, so a blank scores
                # 0 against everything rather than 100 against another blank.
                if texts[first] and texts[second]:
                    ratio = float(fuzz.token_set_ratio(texts[first], texts[second]))
                    best[first] = max(best[first], ratio)
                    best[second] = max(best[second], ratio)

        for row_index, score in zip(group_data.index, best):
            df.at[row_index, column_name] = round(score, 1)


def blob_matching(df: pandas.DataFrame, analysis_type: str, group_by: str, item_description_column: str,
                  boolean_column: str, explanation_column: str, default: str, system_prompt: str,
                  embedding_model_name: str, embedding_provider: str, embedding_model: str,
                  regenerate: bool, modify_number: int) -> None:
    """
    Clusters each group on one vector per row, embedded from the row's whole description, in place.

    This is the original behaviour. It suits a description column holding a single item. Where the
    column aggregates many line items, prefer match_mode = line.

    Args:
        df (pandas.DataFrame): The rows to analyse.
        analysis_type (str): Which analysis is running, used to look up the threshold.
        group_by (str): The column rows are grouped by.
        item_description_column (str): The column holding the description.
        boolean_column (str): The verdict column to write.
        explanation_column (str): The explanation column to write.
        default (str): The verdict for a row in a clustered group that did not land in a cluster.
        system_prompt (str): The instructions handed to the language model.
        embedding_model_name (str): The embedding model in use, used to look up the threshold.
        embedding_provider (str): Overrides the embedding provider configured in llm.config.
        embedding_model (str): Overrides the embedding model configured for that provider.
        regenerate (bool): Whether to re-analyse groups that already hold verdicts.
        modify_number (int): How many groups to analyse before stopping.
    """

    def map_dict(cell, cluster_results, default=default):
        for result, descriptions in cluster_results:
            if cell in descriptions:
                response_list = result.split(",", 1)
                if len(response_list) == 2:
                    return (response_list[0], response_list[1].strip())
        default_list = default.split(",", 1)
        return (default_list[0], default_list[1])

    threshold = get_cluster_threshold(analysis_type, embedding_model_name)
    print(f"Embedding with '{embedding_model_name}', "
          f"clustering at a cosine distance below {threshold}")

    grouped_df = df.groupby(group_by)
    df_size = len(grouped_df)
    count = 1

    for group_ind, group_data in grouped_df:
        print(f"Clustering for {group_by}: {group_ind}/{df_size}")
        if not regenerate and (not group_data[boolean_column].isnull().values.any() or not group_data[explanation_column].isnull().values.any()):
            print(f"Not regenerating for {group_by}: {group_ind}/{df_size} as there are exisiting values for all datapoints.")
            continue
        item_descriptions = group_data[item_description_column].tolist()
        processed_embeddings = embed(item_descriptions, embedding_provider, embedding_model)
        item_clusters = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=threshold
        ).fit(processed_embeddings)
        labels = item_clusters.labels_
        cluster_df = pandas.DataFrame({
            item_description_column: item_descriptions,
            "cluster": labels
        })
        similar_cluster = cluster_df.groupby("cluster").filter(lambda x: len(x) > 1)
        similar_cluster = similar_cluster.groupby("cluster")
        cluster_len = len(similar_cluster)
        if cluster_len > 0:
            cluster_results: list[tuple[str, list[str]]] = []
            print(f"Found {cluster_len} clusters for {group_by}: {group_ind}")
            for cluster_ind, cluster_data in similar_cluster:
                cluster_description_list = cluster_data[item_description_column].tolist()
                print(f"Twin Analysis on cluster: {cluster_ind}")
                user_prompt = f'''Item Descriptions: {cluster_description_list}'''
                result = api_query(system_prompt, user_prompt)
                if not result:
                    result = "Error,Something went wrong with the GenAI analysis."
                cluster_results.append((result, cluster_description_list))
            df.loc[df[group_by] == group_ind, [boolean_column, explanation_column]] = df[df[group_by] == group_ind][item_description_column].apply(lambda desc: pandas.Series(map_dict(desc, cluster_results))).to_numpy()

        if count == modify_number:
            print(f"Modified {count} datapoints")
            break
        count += 1


def write_assessment(df: pandas.DataFrame, column: str, notes: dict, overwrite: bool = True) -> None:
    """
    Writes per row assessment notes into a column, in place. Does nothing when the column is not
    configured, so callers do not need to guard every write.

    Args:
        df (pandas.DataFrame): The frame to write into.
        column (str): The assessment column, or '' when the analysis does not carry one.
        notes (dict): The note for each row, keyed by index label.
        overwrite (bool, optional): Whether to replace a value already present. The skip reasons
            pass False so a verdict kept under regenerate=False keeps its original assessment too.
    """

    if not column:
        return
    for row_index, note in notes.items():
        if overwrite or pandas.isna(df.at[row_index, column]):
            df.at[row_index, column] = note


def line_matching(df: pandas.DataFrame, analysis_type: str, group_by: str, item_description_column: str,
                  boolean_column: str, explanation_column: str, default: str, system_prompt: str,
                  embedding_model_name: str, embedding_provider: str, embedding_model: str,
                  regenerate: bool, modify_number: int, local_assessment_column: str = "",
                  arbiter_assessment_column: str = "") -> None:
    """
    Clusters each group on how much of one row's line items appear in another's, in place.

    Each row's description is split on the configured separator into its line items, every distinct
    line item across the run is embedded once, and rows are compared by matching their line items
    one to one rather than by comparing two averaged vectors. See utils/line_matching.py for why
    that ordering matters.

    Args:
        df (pandas.DataFrame): The rows to analyse.
        analysis_type (str): Which analysis is running, used to look up the thresholds.
        group_by (str): The column rows are grouped by.
        item_description_column (str): The column holding the aggregated line item descriptions.
        boolean_column (str): The verdict column to write.
        explanation_column (str): The explanation column to write.
        default (str): The verdict for a row in a clustered group that did not land in a cluster.
        system_prompt (str): The instructions handed to the language model.
        embedding_model_name (str): The embedding model in use, used to look up the thresholds.
        embedding_provider (str): Overrides the embedding provider configured in llm.config.
        embedding_model (str): Overrides the embedding model configured for that provider.
        regenerate (bool): Whether to re-analyse groups that already hold verdicts.
        modify_number (int): How many groups to analyse before stopping.
        local_assessment_column (str, optional): Where to note what the local model concluded per
            row, or why a row was never assessed. '' leaves the column unwritten.
        arbiter_assessment_column (str, optional): Where to note the language model's raw ruling
            per row, or that it was never consulted. '' leaves the column unwritten.

    Raises:
        ValueError: If the frame's index is not unique, or pair_score is not a known score.
    """

    if not df.index.is_unique:
        raise ValueError("Line matching addresses rows by index label, so the frame needs a unique "
                         "index. Call df.reset_index(drop=True) before passing it in.")

    separator = get_analysis_option(analysis_type, "line_separator")
    pair_score = get_analysis_option(analysis_type, "pair_score").lower()
    if pair_score not in ("containment", "jaccard"):
        raise ValueError(f"Unknown pair_score '{pair_score}' for analysis '{analysis_type}' in "
                         f"analysis.config. Use 'containment' or 'jaccard'.")

    use_categories = get_analysis_option(analysis_type, "use_categories").lower() == "true"
    category_threshold = float(get_analysis_option(analysis_type, "category_threshold"))
    category_partial_credit = float(get_analysis_option(analysis_type, "category_partial_credit"))
    category_batch_size = int(get_analysis_option(analysis_type, "category_batch_size"))

    line_threshold = get_line_threshold(analysis_type, embedding_model_name)
    pair_threshold = get_pair_threshold(analysis_type, embedding_model_name)
    print(f"Embedding with '{embedding_model_name}', matching line items at a cosine similarity of "
          f"at least {line_threshold}, clustering rows at a {pair_score} of at least {pair_threshold}")

    # Every row is parsed, not only the rows being analysed, because the line weights are inverse
    # document frequencies over the whole run: how ordinary a line item is cannot be judged from
    # within one group, which is exactly where the boilerplate is most concentrated.
    row_lines = {index: parse_lines(text, separator)
                 for index, text in df[item_description_column].items()}
    weights = line_weights(list(row_lines.values()))

    grouped_df = df.groupby(group_by)
    df_size = len(grouped_df)

    # Decide what to work on before embedding anything, so the pass below only pays for line items
    # that will actually be compared. Groups that are passed over get the reason written into the
    # assessment column, so a default 'No' in the output can be told apart from an analysed one.
    selected = []
    skipped = {}
    for group_ind, group_data in grouped_df:
        if not regenerate and (not group_data[boolean_column].isnull().values.any() or not group_data[explanation_column].isnull().values.any()):
            print(f"Not regenerating for {group_by}: {group_ind}/{df_size} as there are exisiting values for all datapoints.")
            reason = "Not assessed: the group already held a verdict and regenerate is off"
        elif len(group_data) < 2:
            reason = "Not assessed: the only PO in its group, nothing to compare against"
        elif len(selected) == modify_number:
            reason = f"Not assessed: the run reached its cap of {modify_number} groups before this one"
        else:
            selected.append((group_ind, list(group_data.index)))
            continue
        for row_index in group_data.index:
            skipped[row_index] = reason

    write_assessment(df, local_assessment_column, skipped, overwrite=False)
    write_assessment(df, arbiter_assessment_column,
                     {row_index: "Not consulted: the group was not analysed" for row_index in skipped},
                     overwrite=False)

    if not selected:
        print("No groups to analyse.")
        return

    # One embedding per distinct line item. Line items repeat heavily between the rows of a group,
    # so encoding the distinct strings rather than every occurrence is where line mode wins back
    # the cost of comparing at line level.
    needed = {line for _, row_indices in selected for index in row_indices for line in row_lines[index]}
    print(f"Embedding {len(needed)} distinct line items across {len(selected)} {group_by} groups")
    vectors, index = encode_lines(needed, embedding_provider, embedding_model)

    # What each line item IS, as opposed to how it is worded. Only the rows being analysed are
    # categorised, because unlike parsing this costs a request, so the category weights are inverse
    # document frequencies over those rows rather than over the whole frame.
    row_categories, category_weights = {}, {}
    if use_categories:
        categories = classify_lines(needed, batch_size=category_batch_size, regenerate=regenerate)
        row_categories = {row_index: po_categories(row_lines[row_index], categories)
                          for _, row_indices in selected for row_index in row_indices}
        category_weights = line_weights(list(row_categories.values()))
        placed = sum(1 for value in categories.values() if value[1] != UNCLASSIFIED)
        print(f"Categorised {placed} of {len(categories)} distinct line items, "
              f"clustering rows at a category containment of at least {category_threshold}")

    default_verdict = split_verdict(default, default)

    for count, (group_ind, row_indices) in enumerate(selected, start=1):
        print(f"Clustering for {group_by}: {group_ind}/{df_size}")
        size = len(row_indices)
        distances = numpy.zeros((size, size), dtype=float)
        evidence = {}
        category_evidence = {}
        carried_by_line = set()
        pair_stats = {}

        for first in range(size):
            for second in range(first + 1, size):
                containment, jaccard, pairs = po_similarity(
                    row_lines[row_indices[first]], row_lines[row_indices[second]],
                    vectors, index, weights, line_threshold)
                score = containment if pair_score == "containment" else jaccard
                evidence[(first, second)] = pairs
                best_line = pairs[0][2] if pairs else 0.0
                category_score = 0.0
                linked = score >= pair_threshold
                if linked:
                    carried_by_line.add((first, second))

                # The two scores answer different questions against their own thresholds, so a pair
                # only has to satisfy one of them. The line score asks whether the same items were
                # bought twice; the category score asks whether the two POs buy from the same place
                # in the taxonomy, which is the only way a purchase split into its complementary
                # parts is visible at all.
                if use_categories:
                    category_score, category_pairs = category_similarity(
                        row_categories[row_indices[first]], row_categories[row_indices[second]],
                        category_weights, category_partial_credit)
                    category_evidence[(first, second)] = category_pairs
                    linked = linked or category_score >= category_threshold

                # Recorded as linked or not rather than as a graded distance. Each score has already
                # been read against its own threshold, and the two are not on a comparable scale, so
                # there is nothing left for a distance to say. Complete linkage over this turns the
                # clustering below into exactly 'every pair inside a cluster is linked'.
                distances[first][second] = distances[second][first] = 0.0 if linked else 1.0
                pair_stats[(first, second)] = (score, best_line, category_score, linked)

        # Complete linkage, so a cluster forms only when EVERY pair inside it is linked. Average or
        # single linkage would let A join C on the strength of a shared B, and in sets this loosely
        # built that chaining would quietly rebuild the set the analysis is meant to take apart.
        # The threshold sits between the linked distance of 0 and the unlinked distance of 1.
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='complete',
            distance_threshold=0.5
        ).fit(distances).labels_

        clusters = {}
        for position, label in enumerate(labels):
            clusters.setdefault(label, []).append(position)
        clusters = {label: members for label, members in clusters.items() if len(members) > 1}

        # What the local model concluded for every row of the group, clustered or not, written
        # before the verdict pass so the columns are complete even when no cluster forms. The
        # arbiter note for clustered rows is written below, once the model has actually ruled.
        if local_assessment_column or arbiter_assessment_column:
            member_of = {position: members for members in clusters.values() for position in members}
            local_notes, arbiter_notes = {}, {}
            for position in range(size):
                stats = [pair_stats[(min(position, other), max(position, other))]
                         for other in range(size) if other != position]
                best_score = max(stat[0] for stat in stats)
                best_line = max(stat[1] for stat in stats)
                best_category = max(stat[2] for stat in stats)
                links = sum(1 for stat in stats if stat[3])
                members = member_of.get(position)
                if members:
                    wording = any((min(position, other), max(position, other)) in carried_by_line
                                  for other in members if other != position)
                    if wording:
                        note = (f"Clustered with {len(members) - 1} other PO(s): best line item "
                                f"similarity {best_line:.2f} (threshold {line_threshold}), "
                                f"{pair_score} {best_score:.2f} (threshold {pair_threshold})")
                    else:
                        note = (f"Clustered with {len(members) - 1} other PO(s) on category alone: "
                                f"category containment {best_category:.2f} (threshold "
                                f"{category_threshold}); no line item matched on wording")
                elif links:
                    note = (f"Not clustered: linked to {links} PO(s) but not to every member of a "
                            f"cluster (complete linkage); best {pair_score} {best_score:.2f}")
                else:
                    note = (f"Not clustered: best {pair_score} {best_score:.2f} below "
                            f"{pair_threshold}")
                    if use_categories:
                        note += (f", best category containment {best_category:.2f} below "
                                 f"{category_threshold}")
                local_notes[row_indices[position]] = note
                if not members:
                    arbiter_notes[row_indices[position]] = ("Not consulted: the local model did "
                                                            "not put this PO in any cluster")
            write_assessment(df, local_assessment_column, local_notes)
            write_assessment(df, arbiter_assessment_column, arbiter_notes)

        if not clusters:
            print(f"Found 0 clusters for {group_by}: {group_ind}")
            continue

        print(f"Found {len(clusters)} clusters for {group_by}: {group_ind}")
        verdicts = {row_index: default_verdict for row_index in row_indices}

        for label, members in clusters.items():
            descriptions = [df.at[row_indices[position], item_description_column] for position in members]

            pairs, category_pairs, line_carried = [], [], False
            for first in range(len(members)):
                for second in range(first + 1, len(members)):
                    key = (members[first], members[second])
                    pairs.extend(evidence.get(key, []))
                    category_pairs.extend(category_evidence.get(key, []))
                    line_carried = line_carried or key in carried_by_line
            pairs = sorted(set(pairs), key=lambda pair: pair[2], reverse=True)[:MAX_EVIDENCE_PAIRS]
            category_pairs = sorted(set(category_pairs), key=lambda pair: pair[2],
                                    reverse=True)[:MAX_EVIDENCE_PAIRS]

            # A cluster no pair of which cleared the line threshold was held together entirely by
            # what its rows buy, not by how anything is worded. Asking whether those descriptions
            # look alike would only ever get one answer, so it is asked the other question instead.
            print(f"Twin Analysis on cluster: {label}"
                  f"{'' if line_carried else ' (matched on category only)'}")
            result = api_query(system_prompt,
                               build_user_prompt(descriptions, pairs, category_pairs, line_carried))
            if not result:
                result = "Error,Something went wrong with the GenAI analysis."
            verdict = split_verdict(result, default)
            # The ruling is written raw, so the column shows the model's own words even when a
            # malformed answer makes the verdict columns fall back to the analysis default.
            write_assessment(df, arbiter_assessment_column,
                             {row_indices[position]: result for position in members})
            for position in members:
                verdicts[row_indices[position]] = verdict

        for row_index, (value, explanation) in verdicts.items():
            df.at[row_index, boolean_column] = value
            df.at[row_index, explanation_column] = explanation

        if count == modify_number:
            print(f"Modified {count} datapoints")
            break


def build_user_prompt(descriptions: list, pairs: list, category_pairs: list = None,
                      line_carried: bool = True) -> str:
    """
    Builds the prompt for one cluster, listing the descriptions, the line items that matched
    between them and the categories they share.

    The first line is kept identical to the one blob mode sends, so the wording the system prompt is
    written against does not shift. The evidence is appended below it, which gives the model
    something specific to rule on and to quote in its explanation rather than several long
    aggregated strings.

    When nothing was matched on wording, the cluster exists only because its rows buy from the same
    part of the taxonomy, and the question is put the other way round. Asking whether those
    descriptions resemble each other would get one answer every time, which is exactly how a
    purchase split into its complementary parts goes unnoticed.

    Args:
        descriptions (list): The full descriptions of the rows in the cluster.
        pairs (list): The matched line items, as (first, second, similarity), strongest first.
        category_pairs (list, optional): The matched categories, as (first, second, score).
        line_carried (bool, optional): Whether any pair in the cluster matched on wording. Defaults to True.

    Returns:
        str: The prompt.
    """

    prompt = f'''Item Descriptions: {descriptions}'''

    if pairs:
        matches = "\n".join(f'  {score:.2f}  "{first}"  <->  "{second}"'
                            for first, second, score in pairs)
        prompt += ("\n\nThese individual line items matched across the descriptions above, with "
                   f"their similarity:\n{matches}")

    if category_pairs:
        matches = "\n".join(f'  {score:.2f}  {first[0]} / {first[1]}  <->  {second[0]} / {second[1]}'
                            for first, second, score in category_pairs)
        prompt += ("\n\nThese purchase orders buy from the same categories, scored 1.00 where both "
                   f"levels agree and lower where only the broad category does:\n{matches}")

    if not line_carried:
        prompt += ("\n\nNote that NO line item matched on wording here. These purchase orders were "
                   "put together only because of what they buy. Judge whether they are the "
                   "complementary parts of ONE purchase that has been divided between them, for "
                   "example equipment on one and the components, accessories or installation for "
                   "that same equipment on another. Answer 'No' if they are simply unrelated "
                   "purchases that happen to fall in the same category.")

    return prompt
