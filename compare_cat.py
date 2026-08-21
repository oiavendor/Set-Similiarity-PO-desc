import pandas
from collections import Counter

from utils.GenAI import generate_system_prompt, api_query
from utils.JSONOps import read_json, write_json
from utils.enum import MISSCLASSIFICATION_CAT

role = "audit associate"
task = "identify if the 3 categories given are of the same type or makes sense to be grouped together and create an overarching category for them"
context = '''The procurement team is performing a check on whether the item category in the purchase order accurately reflects the item purchased by looking at the item description.  
When performing this comparison, we want to compare if the item description can be categorised in the item category reflected in the purchase order.
However, there are 3 categories given in the purchase order and sometimes the categories do not match or makes sense to be grouped together.
For example: Category 1 is "Laboratory consumables and chemicals", Category 2 is "Laboratory supplies and fixtures", and Category 3 is "General laboratory glassware and plasticware and supplies", all 3 categories are Laboratory related and are of the same type and makes sense to be grouped together
For example: Category 1 is "Tea bags", Category 2 is "Beverages", and Category 3 is "Coffee and tea", all 3 categories are Drinks or Beverages related and are of the same type and makes sense.
For example: Category 1 is "Laboratory equipment maintenance", Category 2 is "Professional engineering services", and Category 3 is "Electrical and electronic engineering", all 3 categories are Laboratory Maintenence related and are of the same type and makes sense.
For example: Category 1 is "Business cards", Category 2 is "Motor vehicles", and Category 3 is "Laboratory supplies and fixtures", the documents do not have a similar overarching caegory and does not makes sense to be grouped together

If the 3 categories makes sense to be grouped together:
- Respond with True, overarching category

If the 3 categories does not makes sense to be grouped together:
- Respond with False, None

If the category is ambiguous like "Vector", use it in a scientific context
For example: Category 1 is "Vectors", Category 2 is "Laboratory and scientific equipment", and Category 3 is "Vectors", all 3 categories are Laboratory Equipment related and are of the same type and makes sense as "Vectors" in this context is referring to the brand/type of the equipment
'''

constraints = '''Do not provide any additional explanation beyond True, overarching category or False, None.
Only respond with False if the categories are grossly mismatched.
Do not return "overarching category:" only return the value
Do not use "," in overarching categories value.
Can use "," to seperate Boolean and overarching category
'''

def compare_cat(po_details: pandas.DataFrame, regenerate: bool = False, modify_number: int = 500) -> pandas.DataFrame:
    system_prompt = generate_system_prompt(role, task, context, constraints)
    matching_cat_column = "Twin Matching Category"
    parent_cat_column = "Twin Parent Category"
    category_dict: dict[str, str] = read_json(MISSCLASSIFICATION_CAT)
    count = 1

    po_details = po_details.copy()
    df_size = len(po_details)
    twin_exist = matching_cat_column in po_details.columns and parent_cat_column in po_details.columns
    for ind, row in po_details.iterrows():
        po_number = row["PO Number"]
        print(f"Running 3 Cat Match for [{ind + 1}/{df_size}] PO: {po_number}")
        if twin_exist:
            if not regenerate and row[matching_cat_column] and row[parent_cat_column]:
                continue
        breakout = False
        item_category_1 = row["Level 1 Category"]
        item_category_2 = row["Level 2 Category"]
        item_category_3 = row["Level 3 Category"]
        po_category_list = [item_category_1, item_category_2, item_category_3]
        for category_dict_key in category_dict:
            dictionary_category_list = category_dict_key.split(",")
            if Counter(po_category_list) == Counter(dictionary_category_list):
                category_value = category_dict[category_dict_key].split(",")
                po_details.at[ind, matching_cat_column] = category_value[0]
                po_details.at[ind, parent_cat_column] = category_value[1]
                breakout = True
                break
        if breakout:
            continue
        category_dict_key = ",".join(po_category_list)
        user_prompt = f'''Category 1: {item_category_1}\nCategory 2: {item_category_2}\nCategory 3: {item_category_3}'''
        result = api_query(system_prompt, user_prompt)
        category_dict[category_dict_key] = result
        category_value = result.split(",")
        if len(category_value) == 2:
            po_details.loc[ind, matching_cat_column] = category_value[0]
            po_details.at[ind, parent_cat_column] = category_value[1] if ":" not in category_value[1] else category_value[1].split(":")[1]
        if count == modify_number:
            print(f"Modified {count} datapoints")
            break
        count += 1
    
    write_json(MISSCLASSIFICATION_CAT, category_dict)
    
    return po_details