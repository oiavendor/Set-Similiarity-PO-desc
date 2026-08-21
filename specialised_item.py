import pandas

from utils.GenAI import generate_system_prompt, api_query
from utils.shared_functions import string_from_enum
from utils.specialised_item_enums import CONTEXT, CONSTRAINTS

role = "procurement associate"
task = "identify if line item provided by a vendor is for a specialized product or service based on item description"

context = string_from_enum(CONTEXT)

constraints = string_from_enum(CONSTRAINTS)

def specialised_item(vendor_details: pandas.DataFrame, regenerate: bool = False, modify_number: int = 500) -> pandas.DataFrame:
    system_prompt = generate_system_prompt(role, task, context, constraints)
    specialised_column = "Is Specialized"
    explanation_column = "Explanation"
    specialised_dict: dict[str, str] = {}
    count = 1

    vendor_details = vendor_details.copy()
    vendor_details = vendor_details.sort_values(by="VendorNo")
    twin_exist = specialised_column in vendor_details.columns and explanation_column in vendor_details.columns
    df_size = len(vendor_details)
    for ind, row in vendor_details.iterrows():
        vendor_number = row["VendorNo"]
        print(f"Running Specialized Item for [{count}/{modify_number}] Vendor Number: {vendor_number}")
        if twin_exist:
            if not regenerate and row[specialised_column] and row[explanation_column]:
                continue
        item_description = row["Text"]
        if item_description == "":
            continue
        if item_description in specialised_dict:
            result_list = specialised_dict[item_description].split(",")
            vendor_details[specialised_column] = result_list[0]
            vendor_details[explanation_column] = result_list[1]
            continue
        user_prompt = f'''Item Description: {item_description}'''
        result = api_query(system_prompt, user_prompt)
        result_list = result.split(",")
        vendor_details.loc[ind, specialised_column] = result_list[0]
        vendor_details.loc[ind, explanation_column] = result_list[1]
        if count == modify_number:
            print(f"Modified {count} datapoints")
            break
        count += 1
    
    return vendor_details