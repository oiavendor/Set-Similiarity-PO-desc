import pandas

from utils.GenAI import generate_system_prompt, api_query
from utils.check_item_desc_enums import CONTEXT, CONSTRAINTS, EXAMPLE
from utils.shared_functions import string_from_enum

role = "audit associate"
task = "identify if an item description matches any of the 3 item categories provided for each datapoint"
context = string_from_enum(CONTEXT)

constraints = string_from_enum(CONSTRAINTS)

example = string_from_enum(EXAMPLE)

def check_item_desc(po_details: pandas.DataFrame, regenerate: bool = False, modify_number: int = 500) -> pandas.DataFrame:
    system_prompt = generate_system_prompt(role=role, 
                                           task=task, 
                                           context=context, 
                                           constraints=constraints, 
                                           example=example)
    item_match_column = "Twin Item Match Category"
    match_explanation_column = "Twin Explanation"
    matching_dict: dict[str, str] = {}
    count = 1

    po_details = po_details.copy()
    df_size = len(po_details)
    twin_exist = item_match_column in po_details.columns and match_explanation_column in po_details.columns
    for ind, row in po_details.iterrows():
        if count == modify_number:
            print(f"Modified {count} datapoints")
            break
        po_number = row["PO Number"]
        item_category_1 = row["Level 1 Category"]
        item_category_2 = row["Level 2 Category"]
        item_category_3 = row["Level 3 Category"]
        print(f"Running Item-Cat Match for [{ind + 1}/{df_size}] PO: {po_number}")
        if twin_exist:
            if not regenerate and row[item_match_column] and row[match_explanation_column]:
                continue
        breakout = False
        po_item_description = row["Item Description"]
        po_item_category_list = [po_item_description, item_category_1, item_category_2, item_category_3]
        for matching_dict_key in matching_dict:
            item_category_list = matching_dict_key.split(",")
            if item_category_list == po_item_category_list:
                matching_value = matching_dict[matching_dict_key].split(",")
                po_details.at[ind, item_match_column] = matching_value[0]
                po_details.at[ind, match_explanation_column] = matching_value[1]
                breakout = True
                break
        if breakout:
            continue
        matching_dict_key = ",".join(po_item_category_list)
        user_prompt = f'''Item Description: {po_item_description}\nItem Category 1: {item_category_1}\nItem Category 2: {item_category_2}\nItem Category 3: {item_category_3}'''
        result = api_query(system_prompt, user_prompt)
        if "because" not in result:
            matching_dict[matching_dict_key] = "No,Item description does not provide enough information to match with all of the categories."
        else:
            matching_dict[matching_dict_key]= result
        matching_value = result.split(",", 1)
        if len(matching_value) == 2:
            po_details.loc[ind, item_match_column] = matching_value[0]
            po_details.at[ind, match_explanation_column] = matching_value[1]
        count += 1
    
    return po_details