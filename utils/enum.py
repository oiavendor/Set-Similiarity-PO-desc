DQIC_TARGET_COLUMNS = ["PO Number", "Line Number", "Item Description", "Level 1 Category", "Level 2 Category", "Level 3 Category", "Twin Item Match Category", "Twin Explanation"]

MISSCLASSIFICATION_CAT = "missclassification_categories"

# Resolved against the project root rather than the working directory, so the JSON files are found
# whichever folder the pipeline is launched from. read_json swallows a miss and returns {}, so a
# relative path here fails as a silent cache miss followed by a write that raises.
from pathlib import Path

JSON_DIR = str(Path(__file__).resolve().parent.parent / "json_files") + "/"
JSON_FILE_EXT = ".json"

