from typing import Union
from pathlib import Path
import configparser


ANALYSIS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "analysis.config"

# Every analysis section has to set all of these.
ANALYSIS_OPTIONS = ("group_column", "description_column", "boolean_column", "explanation_column", "default")

# Settings an analysis section MAY set, with the value used when it does not. These are optional
# so that a section written before line matching existed keeps working unchanged: leaving them out
# selects the original blob behaviour.
ANALYSIS_DEFAULTS = {
    "match_mode": "blob",       # 'blob' embeds the whole joined description, 'line' each line item
    "line_separator": "||",     # what the line items of one PO are joined by
    "pair_score": "containment",  # which line mode score decides that two POs belong together
}

_analysis_config = None


def load_analysis_config(config_path: Union[str, Path] = ANALYSIS_CONFIG_PATH,
                         reload: bool = False) -> configparser.ConfigParser:
    """
    Loads and caches the analysis.config file describing the columns each analysis reads and writes.

    Args:
        config_path (Union[str, Path], optional): Path to the configuration file. Defaults to analysis.config in the project root.
        reload (bool, optional): Whether to re-read the file instead of using the cached configuration. Defaults to False.

    Returns:
        configparser.ConfigParser: The parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """

    global _analysis_config
    if _analysis_config is None or reload:
        parser = configparser.ConfigParser(interpolation=None)
        if not parser.read(config_path, encoding="utf-8"):
            raise FileNotFoundError(f"Analysis configuration file not found at '{config_path}'.")
        _analysis_config = parser
    return _analysis_config


def get_analysis_config(analysis_type: str) -> dict:
    """
    Returns the column names and fallback verdict for an analysis type.

    Args:
        analysis_type (str): The analysis being run, matching a section name in analysis.config.

    Returns:
        dict: The settings of that section.

    Raises:
        KeyError: If the section is missing, or does not set every option the analysis needs.
    """

    config = load_analysis_config()
    if not config.has_section(analysis_type):
        raise KeyError(f"Section '[{analysis_type}]' is missing from '{ANALYSIS_CONFIG_PATH}'.")

    settings = {option: value.strip() for option, value in config[analysis_type].items()}

    missing = [option for option in ANALYSIS_OPTIONS if not settings.get(option)]
    if missing:
        raise KeyError(f"Section '[{analysis_type}]' of '{ANALYSIS_CONFIG_PATH}' does not set: {', '.join(missing)}.")
    return settings


def get_analysis_option(analysis_type: str, option: str) -> str:
    """
    Returns one of the optional settings of an analysis section, or its default when the section
    does not set it.

    Args:
        analysis_type (str): The analysis being run, matching a section name in analysis.config.
        option (str): The setting to read, one of the keys of ANALYSIS_DEFAULTS.

    Returns:
        str: The configured value, or the default for that option.

    Raises:
        KeyError: If the option is not one this function knows a default for.
    """

    if option not in ANALYSIS_DEFAULTS:
        raise KeyError(f"'{option}' is not an optional analysis setting. Known settings are: "
                       f"{', '.join(sorted(ANALYSIS_DEFAULTS))}.")

    config = load_analysis_config()
    if not config.has_section(analysis_type):
        raise KeyError(f"Section '[{analysis_type}]' is missing from '{ANALYSIS_CONFIG_PATH}'.")

    value = config[analysis_type].get(option)
    return value.strip() if value and value.strip() else ANALYSIS_DEFAULTS[option]


def _get_threshold(kind: str, analysis_type: str, model: str, description: str) -> float:
    """
    Reads a per model threshold out of a [<kind>.<analysis_type>] section.

    Every model has its own distribution of scores, so a threshold calibrated for one model does
    not carry over to another, and the lookup fails loudly rather than falling back to a default.

    The model name is passed in rather than looked up, so that reading the analysis settings does
    not depend on how the embedding provider is configured.

    Args:
        kind (str): The threshold family, e.g. 'threshold' or 'line_threshold'.
        analysis_type (str): The analysis being run.
        model (str): The embedding model in use, e.g. 'all-MiniLM-L6-v2'.
        description (str): What the number means, used in the error message.

    Returns:
        float: The configured threshold.

    Raises:
        KeyError: If no threshold is configured for that analysis type and model.
    """

    config = load_analysis_config()
    section = f"{kind}.{analysis_type}"

    if not config.has_section(section):
        raise KeyError(f"Section '[{section}]' is missing from '{ANALYSIS_CONFIG_PATH}'.")

    value = config[section].get(model.strip().lower())
    if value is None:
        raise KeyError(
            f"No {description} for embedding model '{model}' in section '[{section}]' of "
            f"'{ANALYSIS_CONFIG_PATH}'. Add one and calibrate it before running with this model."
        )
    return float(value)


def get_cluster_threshold(analysis_type: str, model: str) -> float:
    """
    Returns the cosine DISTANCE below which two whole joined descriptions join the same cluster
    in blob mode. Lower is stricter.

    Args:
        analysis_type (str): The analysis being run, matching a [threshold.<analysis_type>] section.
        model (str): The embedding model in use, e.g. 'all-MiniLM-L6-v2'.

    Returns:
        float: The distance below which two descriptions join the same cluster.

    Raises:
        KeyError: If no threshold is configured for that analysis type and model.
    """

    return _get_threshold("threshold", analysis_type, model, "clustering threshold")


def get_line_threshold(analysis_type: str, model: str) -> float:
    """
    Returns the cosine SIMILARITY at or above which two individual line items are treated as the
    same item in line mode. Higher is stricter, the opposite direction to get_cluster_threshold,
    because the line mode maths works in similarity rather than distance.

    Args:
        analysis_type (str): The analysis being run, matching a [line_threshold.<analysis_type>] section.
        model (str): The embedding model in use, e.g. 'all-MiniLM-L6-v2'.

    Returns:
        float: The similarity at or above which two line items are the same item.

    Raises:
        KeyError: If no threshold is configured for that analysis type and model.
    """

    return _get_threshold("line_threshold", analysis_type, model, "line matching threshold")


def get_pair_threshold(analysis_type: str, model: str) -> float:
    """
    Returns how much of one PO's line items must be found in another's before the two are put in
    the same cluster in line mode. Higher is stricter.

    Args:
        analysis_type (str): The analysis being run, matching a [pair_threshold.<analysis_type>] section.
        model (str): The embedding model in use, e.g. 'all-MiniLM-L6-v2'.

    Returns:
        float: The containment or jaccard score at or above which two POs cluster together.

    Raises:
        KeyError: If no threshold is configured for that analysis type and model.
    """

    return _get_threshold("pair_threshold", analysis_type, model, "PO pair threshold")
