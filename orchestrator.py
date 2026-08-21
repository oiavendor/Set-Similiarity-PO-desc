from utils.DBOps import ReadSQL, InsertTable
from check_item_desc import check_item_desc
from utils.enum import DQIC_TARGET_COLUMNS
from sourcing_categorizer import sourcing_categorizer
from description_matching import description_matching
from specialised_item import specialised_item


def data_quality_item_cat():
    df = ReadSQL("STGOUT_MASTER_PODETAILS")
    df = check_item_desc(df, regenerate=True, modify_number=100)
    InsertTable(df[DQIC_TARGET_COLUMNS], "STGOUT_MASTER_PODETAILS")

def item_description_match_PO():
    df = ReadSQL("STGOUT_11_SPLITPO_MATCH")
    df = description_matching(df, "po", regenerate=True, modify_number=50)
    InsertTable(df, "STGOUT_11_SPLITPO_MATCH")

def item_description_match_payment():
    df = ReadSQL("STGOUT_15_DUPPMT_MATCHED")
    df = description_matching(df, "payment", regenerate=True, modify_number=50)
    InsertTable(df, "STGOUT_15_DUPPMT_MATCHED")

def item_description_specialised():
    df = ReadSQL("STGOUT_14_VENDORTRNSw1DEPT_DETAILS_SPECIALISED")
    df = specialised_item(df, regenerate=True, modify_number=200)
    InsertTable(df, "STGOUT_14_VENDORTRNSw1DEPT_DETAILS_SPECIALISED")


def sourcing_fill_null():
    df = ReadSQL("STGOUT_1_to_10_SOURCING_W_CAT")
    category_list = df["Project Category"].drop_duplicates().to_list()
    df = sourcing_categorizer(df, category_list=category_list)
    InsertTable(df, "SOURCINGSTGOUT_1_to_10_SOURCING_W_CAT_W_CAT")


def sourcing_cat(command_check: bool = True):
    """
    RUNNING THIS WILL OVERWRITE SOURCING_W_CAT
    """
    if command_check:
        if input("RUNNING THIS WILL OVERWRITE SOURCING_W_CAT\nTo run enter 1: ") != "1":
            return
    df = ReadSQL("STGOUT_1_to_10_SOURCING")
    df = sourcing_categorizer(df)
    InsertTable(df, "STGOUT_1_to_10_SOURCING_W_CAT")



