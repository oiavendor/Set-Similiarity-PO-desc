import pandas
from rapidfuzz import fuzz
from utils.GenAI import embed, generate_system_prompt, api_query, get_embedding_model_name
from utils.AnalysisConfig import get_analysis_option, get_cluster_threshold, get_line_threshold
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

# How many matched line items are shown to the language model per group. They are the evidence
# for the verdict, but a large group produces hundreds of them and burying the prompt costs more
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
                         modify_number: int = None, embedding_provider: str = None,
                         embedding_model: str = None, fill_empty: bool = True) -> pandas.DataFrame:
    """
    Asks the language model to rule on the item descriptions of each group.

    Two matching modes are available, chosen per analysis by match_mode in analysis.config:

        blob  clusters the rows of a group by how alike their descriptions are, embedding each
              whole description as a single vector, and rules on each cluster it finds. Suits a
              description column holding a single item.
        line  treats the group itself as the unit and rules on it whole, one verdict written to
              every row. The line items are matched only to gather the evidence shown to the
              model. Use this where the grouping already asserts that the rows belong together,
              as the po analysis does. See line_matching and utils/line_matching.py.

    The po analysis also gets a 'Set Similarity' column: each row's best token_set_ratio (0 to
    100) against the other descriptions in its set. It is a plain string measure written
    alongside the verdicts and feeds nothing downstream. See set_similarity_column.

    Args:
        df (pandas.DataFrame): The rows to analyse.
        analysis_type (Literal['po', 'payment']): Which analysis to run, naming a section of analysis.config.
        regenerate (bool, optional): Whether to re-analyse groups that already hold verdicts. Defaults to False.
        modify_number (int, optional): How many groups to analyse before stopping. Defaults to None,
            which analyses every group. A cap leaves the groups it never reached filled with the
            analysis default, which reads as 'No' in the output without anything having been
            compared, so only set one for a deliberately partial trial run.
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
    arguments = dict(df=df, group_by=group_by, item_description_column=item_description_column,
                     boolean_column=boolean_column, explanation_column=explanation_column,
                     default=default, system_prompt=system_prompt,
                     embedding_model_name=embedding_model_name,
                     embedding_provider=embedding_provider, embedding_model=embedding_model,
                     regenerate=regenerate, modify_number=modify_number)

    if match_mode == "line":
        line_matching(analysis_type=analysis_type, **arguments)
    elif match_mode == "blob":
        blob_matching(analysis_type=analysis_type, **arguments)
    else:
        raise ValueError(f"Unknown match_mode '{match_mode}' for analysis '{analysis_type}' in "
                         f"analysis.config. Use 'blob' or 'line'.")

    if fill_empty:
        fill_empty_verdicts(df, boolean_column, explanation_column, default)

    return df


def fill_empty_verdicts(df: pandas.DataFrame, boolean_column: str, explanation_column: str,
                        default: str) -> None:
    """
    Fills every row still without a verdict with the analysis default, in place.

    A row is left empty when it is alone in its group, when its group value is empty, when blob
    mode produced no cluster for it, or when the run stopped at modify_number before reaching its
    group. Those are all 'nothing found here', so the output workbook carries a verdict on every
    row rather than a mix of 'No' and blanks.

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
        modify_number (int): How many groups to analyse before stopping, or None for every group.
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

        if modify_number is not None and count == modify_number:
            print(f"Modified {count} groups, stopping at the modify_number cap. The remaining "
                  f"groups are NOT analysed and are filled with the analysis default.")
            break
        count += 1


def line_matching(df: pandas.DataFrame, analysis_type: str, group_by: str, item_description_column: str,
                  boolean_column: str, explanation_column: str, default: str, system_prompt: str,
                  embedding_model_name: str, embedding_provider: str, embedding_model: str,
                  regenerate: bool, modify_number: int) -> None:
    """
    Rules on each group as a whole, writing one verdict across every row of it, in place.

    The group is the split. An upstream rule has already decided which purchase orders make up a
    candidate split, so this does not re-derive that: it puts the whole group to the language model
    and writes the answer to every row. Two purchase orders in a group are never separated on the
    grounds that their descriptions read differently, because a purchase divided into complementary
    parts, a workstation on one order and its processor on another, has descriptions that share
    nothing by construction. That is the case the analysis most needs to catch.

    The line items are still matched, but only to gather evidence. Each row's description is split
    on the configured separator, every distinct line item is embedded once, and the pairs that match
    across the group are shown to the model alongside the descriptions. When nothing matches on
    wording the prompt asks the complementary question instead. See utils/line_matching.py.

    Args:
        df (pandas.DataFrame): The rows to analyse.
        analysis_type (str): Which analysis is running, used to look up the threshold.
        group_by (str): The column rows are grouped by; one group is one candidate split.
        item_description_column (str): The column holding the aggregated line item descriptions.
        boolean_column (str): The verdict column to write.
        explanation_column (str): The explanation column to write.
        default (str): The verdict used when the model's answer cannot be split into two columns.
        system_prompt (str): The instructions handed to the language model.
        embedding_model_name (str): The embedding model in use, used to look up the threshold.
        embedding_provider (str): Overrides the embedding provider configured in llm.config.
        embedding_model (str): Overrides the embedding model configured for that provider.
        regenerate (bool): Whether to re-analyse groups that already hold verdicts.
        modify_number (int): How many groups to analyse before stopping, or None for every group.

    Raises:
        ValueError: If the frame's index is not unique.
    """

    if not df.index.is_unique:
        raise ValueError("Line matching addresses rows by index label, so the frame needs a unique "
                         "index. Call df.reset_index(drop=True) before passing it in.")

    separator = get_analysis_option(analysis_type, "line_separator")
    use_categories = get_analysis_option(analysis_type, "use_categories").lower() == "true"
    category_partial_credit = float(get_analysis_option(analysis_type, "category_partial_credit"))
    category_batch_size = int(get_analysis_option(analysis_type, "category_batch_size"))

    # Only the line threshold is read. pair_score, pair_threshold and category_threshold decided
    # which rows of a group belonged together, and the group already answers that.
    line_threshold = get_line_threshold(analysis_type, embedding_model_name)
    print(f"Embedding with '{embedding_model_name}', matching line items at a cosine similarity of "
          f"at least {line_threshold}, one verdict per {group_by}")

    # Every row is parsed, not only the rows being analysed, because the line weights are inverse
    # document frequencies over the whole run: how ordinary a line item is cannot be judged from
    # within one group, which is exactly where the boilerplate is most concentrated.
    row_lines = {index: parse_lines(text, separator)
                 for index, text in df[item_description_column].items()}
    weights = line_weights(list(row_lines.values()))

    grouped_df = df.groupby(group_by)
    df_size = len(grouped_df)

    # Decide what to work on before embedding anything, so the pass below only pays for line items
    # that will actually be compared.
    selected = []
    for group_ind, group_data in grouped_df:
        if not regenerate and (not group_data[boolean_column].isnull().values.any() or not group_data[explanation_column].isnull().values.any()):
            print(f"Not regenerating for {group_by}: {group_ind}/{df_size} as there are exisiting values for all datapoints.")
            continue
        if len(group_data) < 2:
            continue  # a group of one row has nothing to be compared against
        selected.append((group_ind, list(group_data.index)))
        if modify_number is not None and len(selected) >= modify_number:
            break

    if not selected:
        print("No groups to analyse.")
        return

    # One embedding per distinct line item. Line items repeat heavily between the rows of a group,
    # so encoding the distinct strings rather than every occurrence is where line mode wins back
    # the cost of comparing at line level.
    needed = {line for _, row_indices in selected for index in row_indices for line in row_lines[index]}
    print(f"Embedding {len(needed)} distinct line items across {len(selected)} {group_by} groups")
    vectors, index = encode_lines(needed, embedding_provider, embedding_model)

    # What each line item IS, as opposed to how it is worded. This no longer decides anything: it
    # names the categories two purchase orders share so the model can be shown them, which is what
    # a complementary split looks like when the wording has nothing in common. Only the rows being
    # analysed are categorised, because unlike parsing this costs a request.
    row_categories, category_weights = {}, {}
    if use_categories:
        categories = classify_lines(needed, batch_size=category_batch_size, regenerate=regenerate)
        row_categories = {row_index: po_categories(row_lines[row_index], categories)
                          for _, row_indices in selected for row_index in row_indices}
        category_weights = line_weights(list(row_categories.values()))
        placed = sum(1 for value in categories.values() if value[1] != UNCLASSIFIED)
        print(f"Categorised {placed} of {len(categories)} distinct line items, "
              f"shown to the model as evidence")

    # One verdict per group. The group IS the candidate split: an upstream rule already decided
    # these purchase orders belong to one another, so the analysis judges that claim rather than
    # rebuilding it. Sub-dividing a group by how alike its descriptions read would throw the claim
    # away, and would answer 'No' for exactly the split being hunted, because a purchase divided
    # into its complementary parts has descriptions that share nothing by construction.
    for count, (group_ind, row_indices) in enumerate(selected, start=1):
        print(f"Assessing {group_by}: {group_ind} ({count} of {len(selected)} selected, "
              f"{df_size} in total, {len(row_indices)} purchase orders)")
        size = len(row_indices)
        pairs, category_pairs = [], []
        line_carried = False

        # Every pair in the group, compared only to gather what the model is shown. Nothing here
        # decides membership; membership was decided upstream.
        for first in range(size):
            for second in range(first + 1, size):
                _, _, matched = po_similarity(
                    row_lines[row_indices[first]], row_lines[row_indices[second]],
                    vectors, index, weights, line_threshold)
                pairs.extend(matched)
                line_carried = line_carried or bool(matched)

                if use_categories:
                    _, matched_categories = category_similarity(
                        row_categories[row_indices[first]], row_categories[row_indices[second]],
                        category_weights, category_partial_credit)
                    category_pairs.extend(matched_categories)

        pairs = sorted(set(pairs), key=lambda pair: pair[2], reverse=True)[:MAX_EVIDENCE_PAIRS]
        category_pairs = sorted(set(category_pairs), key=lambda pair: pair[2],
                                reverse=True)[:MAX_EVIDENCE_PAIRS]

        descriptions = [df.at[row_index, item_description_column] for row_index in row_indices]

        # A group where no line item matched on wording is the complementary case, so the prompt
        # asks the other question. Asking whether those descriptions look alike would only ever
        # get one answer, which is how a divided purchase goes unnoticed.
        print(f"Twin Analysis on {group_by} {group_ind}"
              f"{'' if line_carried else ' (no wording matched between the descriptions)'}")
        result = api_query(system_prompt,
                           build_user_prompt(descriptions, pairs, category_pairs, line_carried))
        if not result:
            result = "Error,Something went wrong with the GenAI analysis."

        value, explanation = split_verdict(result, default)
        for row_index in row_indices:
            df.at[row_index, boolean_column] = value
            df.at[row_index, explanation_column] = explanation


def build_user_prompt(descriptions: list, pairs: list, category_pairs: list = None,
                      line_carried: bool = True) -> str:
    """
    Builds the prompt for one group, listing the descriptions, the line items that matched between
    them and the categories they share.

    The first line is kept identical to the one blob mode sends, so the wording the system prompt is
    written against does not shift. The evidence is appended below it, which gives the model
    something specific to rule on and to quote in its explanation rather than several long
    aggregated strings.

    When nothing matched on wording, the question is put the other way round. Asking whether those
    descriptions resemble each other would get one answer every time, which is exactly how a
    purchase split into its complementary parts goes unnoticed.

    Args:
        descriptions (list): The full descriptions of every row in the group.
        pairs (list): The matched line items, as (first, second, similarity), strongest first.
        category_pairs (list, optional): The matched categories, as (first, second, score).
        line_carried (bool, optional): Whether any pair in the group matched on wording. Defaults to True.

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
        prompt += ("\n\nNote that NO line item matched on wording here, so do not look for repeated "
                   "items. Judge instead whether these are the complementary parts of ONE purchase "
                   "that has been divided between them, for example equipment on one and the "
                   "components, accessories or installation for that same equipment on another. "
                   "Answer 'No' if they are simply unrelated purchases.")

    return prompt
