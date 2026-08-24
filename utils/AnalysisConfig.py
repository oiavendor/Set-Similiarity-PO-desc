from typing import Union
from pathlib import Path
import configparser


ANALYSIS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "analysis.config"

# Every analysis section has to set all of these.
ANALYSIS_OPTIONS = ("group_column", "description_column", "boolean_column", "explanation_column", "default")

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


def get_cluster_threshold(analysis_type: str, model: str) -> float:
    """
    Returns the cosine distance threshold an analysis clusters at, for a given embedding model.
    Every model has its own distance distribution, so a threshold calibrated for one model does
    not carry over to another.

    The model name is passed in rather than looked up, so that reading the analysis settings does
    not depend on how the embedding provider is configured.

    Args:
        analysis_type (str): The analysis being run, matching a [threshold.<analysis_type>] section.
        model (str): The embedding model in use, e.g. 'all-MiniLM-L6-v2'.

    Returns:
        float: The distance below which two descriptions join the same cluster.

    Raises:
        KeyError: If no threshold is configured for that analysis type and model.
    """

    config = load_analysis_config()
    section = f"threshold.{analysis_type}"

    if not config.has_section(section):
        raise KeyError(f"Section '[{section}]' is missing from '{ANALYSIS_CONFIG_PATH}'.")

    value = config[section].get(model.strip().lower())
    if value is None:
        raise KeyError(
            f"No clustering threshold for embedding model '{model}' in section '[{section}]' of "
            f"'{ANALYSIS_CONFIG_PATH}'. Add one and calibrate it before running with this model."
        )
    return float(value)
