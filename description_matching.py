import numpy
import pandas
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


def line_matching(df: pandas.DataFrame, analysis_type: str, group_by: str, item_description_column: str,
                  boolean_column: str, explanation_column: str, default: str, system_prompt: str,
                  embedding_model_name: str, embedding_provider: str, embedding_model: str,
                  regenerate: bool, modify_number: int) -> None:
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
    # that will actually be compared.
    selected = []
    for group_ind, group_data in grouped_df:
        if not regenerate and (not group_data[boolean_column].isnull().values.any() or not group_data[explanation_column].isnull().values.any()):
            print(f"Not regenerating for {group_by}: {group_ind}/{df_size} as there are exisiting values for all datapoints.")
            continue
        if len(group_data) < 2:
            continue  # a group of one row has nothing to be compared against
        selected.append((group_ind, list(group_data.index)))
        if len(selected) == modify_number:
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

        for first in range(size):
            for second in range(first + 1, size):
                containment, jaccard, pairs = po_similarity(
                    row_lines[row_indices[first]], row_lines[row_indices[second]],
                    vectors, index, weights, line_threshold)
                score = containment if pair_score == "containment" else jaccard
                evidence[(first, second)] = pairs
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
