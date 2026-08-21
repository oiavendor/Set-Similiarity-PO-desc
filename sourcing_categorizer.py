import pandas

from utils.JSONOps import read_json, write_json
from utils.GenAI import generate_system_prompt, api_query
from utils.sourcing_categorizer_enums import CONTEXT, CONSTRAINTS, EXAMPLE
from utils.shared_functions import string_from_enum

role = "audit associate"
task = "identify an appropriate broad cateogry of a project given a list of possible categories and a brief project description.\nif there is no approprioate broad category in the list, come up with the best possible broad category to place the project under"
context = string_from_enum(CONTEXT)

constraints = string_from_enum(CONSTRAINTS)

example = string_from_enum(EXAMPLE)

def sourcing_categorizer(sourcing_details: pandas.DataFrame, regenerate: bool = False, modify_number:int = 500) -> pandas.DataFrame:

    system_prompt = generate_system_prompt(role, task, context, constraints)
    project_category_column = "Twin Category"
    try:
        category_list = read_json("sourcing_categories")["category_list"]
    except:
        category_list = []
    count = 1

    sourcing_details = sourcing_details.copy()
    df_size = len(sourcing_details)
    for ind, row in sourcing_details.iterrows():
        project_reference = row["Reference Number"]
        print(f"Running Twin Categorization for [{ind + 1}/{df_size}] Project: {project_reference}")
        if project_category_column in row:
            if not regenerate and row[project_category_column]:
                continue
        proj_desc = row["Proj Desc"]
        user_prompt = f'''Project Description: {proj_desc}\nCategory List: {category_list}'''
        result = api_query(system_prompt, user_prompt)
        if result not in category_list:
            category_list.append(result)
        sourcing_details.loc[ind, project_category_column] = result
        if count == modify_number:
            print(f"Modified {count} datapoints")
            break
        count += 1
    write_json("sourcing_categories", {"category_list": category_list})
    return sourcing_details