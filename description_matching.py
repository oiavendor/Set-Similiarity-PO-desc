import pandas
from utils.GenAI import embed, generate_system_prompt, api_query, get_cluster_threshold, get_embedding_model_name
from utils.description_matching_enums import PO_ITEM_DESC_COLUMN, PO_SET_COLUMN, PAYMENT_ITEM_DESC_COLUMN, PAYMENT_SET_COLUMN, PO_CONTEXT, CONSTRAINTS, TASK, PO_SPLIT_BOOLEAN_COLUMN, EXPLANATION_COLUMN, PO_DEFAULT_NON_CLUSTERED, PAYMENT_CONTEXT, PAYMENT_DUPP_BOOLEAN_COLUMN, PAYMENT_DEFAULT_NON_CLUSTERED
from typing import Literal
from utils.shared_functions import string_from_enum
from sklearn.cluster import AgglomerativeClustering

role = "audit associate"
task = "identify if the list of item descriptions are for"

def description_matching(df: pandas.DataFrame, analysis_type: Literal["po", "payment"], regenerate: bool = False,
                         modify_number: int = 500, embedding_provider: str = None,
                         embedding_model: str = None) -> pandas.DataFrame:

    def map_dict(cell, cluster_results, default=PO_DEFAULT_NON_CLUSTERED if analysis_type == "po" else PAYMENT_DEFAULT_NON_CLUSTERED):
        for result, descriptions in cluster_results:
            if cell in descriptions:
                response_list = result.split(",", 1)
                if len(response_list) == 2:
                    return (response_list[0], response_list[1].strip())
        default_list = default.split(",", 1)
        return (default_list[0], default_list[1])

    if analysis_type == "po":
        group_by = PO_SET_COLUMN
        item_description_column = PO_ITEM_DESC_COLUMN
        context = string_from_enum(PO_CONTEXT)
        boolean_column = PO_SPLIT_BOOLEAN_COLUMN
    elif analysis_type == "payment":
        group_by = PAYMENT_SET_COLUMN
        item_description_column = PAYMENT_ITEM_DESC_COLUMN
        context = string_from_enum(PAYMENT_CONTEXT)
        boolean_column = PAYMENT_DUPP_BOOLEAN_COLUMN
    threshold = get_cluster_threshold(analysis_type, embedding_provider, embedding_model)
    print(f"Embedding with '{get_embedding_model_name(embedding_provider, embedding_model)}', "
          f"clustering at a cosine distance below {threshold}")

    grouped_df = df.groupby(group_by)
    df_size = len(grouped_df)
    count = 1
    explanation_column = EXPLANATION_COLUMN
    constraints = string_from_enum(CONSTRAINTS)

    system_prompt = generate_system_prompt(role=role, 
                                           task=TASK, 
                                           context=context, 
                                           constraints=constraints)

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

    return df

        
            


    
        