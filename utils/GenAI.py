from typing import Union, Tuple, List
from pathlib import Path
from openai import OpenAI, OpenAIError
import configparser
import json
import os
import requests
import numpy as np
import pandas as pd
# from pypdf import PdfReader
# from docx import Document
# from docx.table import Table
# from docx.text.paragraph import Paragraph
# from googlesearch import search


LLM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "llm.config"

# Environment variables that take precedence over the values in llm.config.
LLM_CONFIG_ENV_OVERRIDES = {
    "llm": {
        "base_url": "OPENAI_BASE_URL",
        "api_key": "OPENAI_API_KEY",
        "model": "OPENAI_MODEL",
    },
    "embedding": {
        "base_url": "OPENAI_EMBEDDING_BASE_URL",
        "api_key": "OPENAI_API_KEY",
        "model": "OPENAI_EMBEDDING_MODEL",
    },
}

_llm_config = None
_llm_clients = {}
_embedding_models = {}


def load_llm_config(config_path: Union[str, Path] = LLM_CONFIG_PATH, reload: bool = False) -> configparser.ConfigParser:
    """
    Loads and caches the llm.config file holding the settings of the OpenAI-compatible endpoints.

    Args:
        config_path (Union[str, Path], optional): Path to the configuration file. Defaults to llm.config in the project root.
        reload (bool, optional): Whether to re-read the file instead of using the cached configuration. Defaults to False.

    Returns:
        configparser.ConfigParser: The parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """

    global _llm_config
    if _llm_config is None or reload:
        parser = configparser.ConfigParser(interpolation=None)
        if not parser.read(config_path, encoding="utf-8"):
            raise FileNotFoundError(f"LLM configuration file not found at '{config_path}'.")
        _llm_config = parser
        _llm_clients.clear()  # rebuild the clients from the reloaded settings
        _embedding_models.clear()
    return _llm_config


def get_llm_config(section: str = "llm") -> dict:
    """
    Returns the settings of a section of llm.config, with the environment variable overrides applied.

    Args:
        section (str, optional): The section to read, either 'llm' or 'embedding'. Defaults to 'llm'.

    Returns:
        settings (dict): The settings of the section.

    Raises:
        KeyError: If the section is missing from the configuration file.
    """

    config = load_llm_config()
    if not config.has_section(section):
        raise KeyError(f"Section '[{section}]' is missing from '{LLM_CONFIG_PATH}'.")

    settings = {option: value.strip() for option, value in config[section].items()}
    for option, env_var in LLM_CONFIG_ENV_OVERRIDES.get(section, {}).items():
        env_value = os.environ.get(env_var)
        if env_value:
            settings[option] = env_value
    return settings


def get_llm_client(section: str = "llm") -> OpenAI:
    """
    Returns a cached OpenAI client built from the settings of a section of llm.config.

    Args:
        section (str, optional): The section to build the client from, either 'llm' or 'embedding'. Defaults to 'llm'.

    Returns:
        client (OpenAI): A client pointed at the configured endpoint.
    """

    if section not in _llm_clients:
        settings = get_llm_config(section)
        client_arguments = {"api_key": settings.get("api_key") or "EMPTY"}  # self-hosted endpoints accept any key
        for option, option_type in (("base_url", str), ("default_headers", json.loads), ("timeout", float), ("max_retries", int)):
            if settings.get(option):
                client_arguments[option] = option_type(settings[option])
        _llm_clients[section] = OpenAI(**client_arguments)
    return _llm_clients[section]


def generate_system_prompt(role: str, task: str, context: str, constraints: str, definitions: str = None,
                           example: str = None) -> str:
    """
    Generates a system prompt for a given role and task with specific context and constraints.

    Args:
        role (str): The role the user is supposed to assume in the task.
        task (str): The specific task that needs to be completed.
        context (str): Relevant background information or context for the task.
        constraints (str): Any limitations or constraints that should be considered for the task.
        definitions (str, optional): Definitions of specific terms used in the task or context. Defaults to None.
        example (str, optional): An example or sample to illustrate how the task should be approached. Defaults to None.

    Returns:
        prompt (str): A formatted system prompt that provides a clear set of instructions based on the role, task, context, and constraints provided.
    """

    prompt = f'''You are a {role}
    Your task is to {task}
    Some context to the task are as follows: {context}
    Some constraints are as follows: {constraints}
    '''

    if definitions is not None:
        prompt += f"Some terms that are used in the documents are: {definitions}\n"

    if example is not None:
        prompt += f"Here is an example: {example}\n"

    prompt += "Focus on being precise, framing your response directly to the context of the task."
    return prompt


def compare_similarity(query_embedding: np.ndarray, page_embeddings: np.ndarray,
                       valid_indices: np.ndarray = None) -> np.ndarray:
    """
    Compares the similarity between a query embedding and a list of chunk embeddings, only considering valid indices.

    Args:
        query_embedding (np.ndarray): A 2D numpy array representing the embedding of the query (1 x N).
        page_embeddings (np.ndarray): A 2D numpy array representing the embeddings of the pages (N x M).
        valid_indices (np.ndarray): A 1D boolean numpy array representing which page embeddings are valid.

    Returns:
        similarities (np.ndarray): An array of cosine similarity values for each embedding. Invalid embeddings have a similarity value of 0.
    """

    page_embeddings = page_embeddings.T
    p1 = query_embedding @ page_embeddings  # dot product, preferred over np.dot for 2d
    p1 = p1[0]

    query_norm = np.linalg.norm(query_embedding)
    page_norms = np.sqrt(
        np.einsum('ij,ij->j', page_embeddings, page_embeddings))  # einops to optimize norm calculations
    p2 = query_norm * page_norms

    if valid_indices is None:
        similarities = np.divide(p1, p2)

    else:
        similarities = np.divide(p1, p2, where=valid_indices)
        similarities[~valid_indices] = 0  # set invalid indices to have 0 similarity

    return similarities


def api_pdf_chunk(base_url, file_path, params):
    """
    Sends a request to the API to extract chunks from a PDF file.

    Args:
        base_url (str): The base URL of the API endpoint.
        file_path (str): The path to the PDF file that needs to be processed.
        params (dict): Additional parameters to be included in the request.

    Returns:
        dict: A JSON response containing the extracted chunks from the PDF file.
        Returns None if the request fails, and prints the error message.
    """

    with open(file_path, "rb") as pdf_file:
        files = {
            "file": ("file.pdf", pdf_file, "application/pdf")
        }
        response = requests.post(base_url + 'get_chunks_pdf', files=files, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None


def api_url_chunk(base_url, params):
    """
    Sends a request to the API to extract chunks from a URL.

    Args:
        base_url (str): The base URL of the API endpoint.
        web_url (str): The URL of the webpage that needs to be processed.
        params (dict): Additional parameters to be included in the request.

    Returns:
        dict: A JSON response containing the extracted chunks from the webpage.
        Returns None if the request fails, and prints the error message.
    """

    response = requests.post(base_url + 'get_chunks_url', params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def api_chunk(base_url, strings, params):
    """
    Sends a request to the API to generate chunks for a list of strings.

    Args:
        base_url (str): The base URL of the API endpoint.
        strings (dict): A dictionary containing a list of strings to generate chunks for.
        params (dict): Additional parameters to be included in the request.

    Returns:
        dict: A JSON response containing the chunks for each string.
        Returns None if the request fails, and prints the error message.
    """

    response = requests.post(base_url + 'get_chunks', json=strings, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def api_embed(strings: List[str], model: str = None, **parameters) -> List[List[float]]:
    """
    Generates embeddings for a list of strings with the embeddings endpoint of the OpenAI client.

    Args:
        strings (List[str]): The strings to generate embeddings for.
        model (str, optional): Overrides the model configured in the [embedding] section of llm.config.
        **parameters: Additional embedding parameters (e.g. dimensions, encoding_format).

    Returns:
        List[List[float]]: The embedding vector of each string, in the same order as the strings provided.
        Returns None if the request fails, and prints the error message.
    """

    settings = get_llm_config("embedding")

    try:
        response = get_llm_client("embedding").embeddings.create(model=model or settings["model"], input=strings,
                                                                 **parameters)
    except OpenAIError as e:
        print(f"Error: {e}")
        return None

    embeddings = sorted(response.data, key=lambda embedding: embedding.index)
    return [embedding.embedding for embedding in embeddings]


def get_embedding_provider(provider: str = None) -> str:
    """
    Returns the embedding provider in force, either the one passed in or the one set in llm.config.

    Args:
        provider (str, optional): Overrides the provider configured in the [embedding] section.

    Returns:
        str: Either 'api' or 'local'.

    Raises:
        ValueError: If the resolved provider is not one of those two.
    """

    provider = (provider or get_llm_config("embedding").get("provider") or "api").strip().lower()
    if provider not in ("api", "local"):
        raise ValueError(f"Unknown embedding provider '{provider}' in '{LLM_CONFIG_PATH}'. Use 'api' or 'local'.")
    return provider


def get_embedding_model_name(provider: str = None, model: str = None) -> str:
    """
    Returns the name of the embedding model the given provider will use. Needed to look up the
    clustering threshold, which is calibrated per model.

    Args:
        provider (str, optional): Overrides the provider configured in the [embedding] section.
        model (str, optional): Overrides the model configured for that provider.

    Returns:
        str: The model name.
    """

    if model:
        return model
    section = "embedding.local" if get_embedding_provider(provider) == "local" else "embedding"
    return get_llm_config(section)["model"]


def get_embedding_model(model: str = None):
    """
    Returns a cached sentence-transformers model built from the [embedding.local] section of llm.config.
    The import is deferred so that running with provider = api needs no sentence-transformers install.

    Args:
        model (str, optional): Overrides the model configured in the [embedding.local] section.

    Returns:
        SentenceTransformer: The loaded model, cached so it loads once per process.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """

    settings = get_llm_config("embedding.local")
    name = model or settings["model"]

    if name not in _embedding_models:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "provider = local needs sentence-transformers. Install it with "
                "'pip install sentence-transformers', or set provider = api in llm.config."
            ) from e

        arguments = {}
        if settings.get("device"):
            arguments["device"] = settings["device"]
        for option in ("trust_remote_code", "local_files_only"):
            if settings.get(option):
                arguments[option] = settings[option].strip().lower() == "true"

        print(f"Loading embedding model '{name}' (first run downloads the weights)")
        _embedding_models[name] = SentenceTransformer(name, **arguments)
    return _embedding_models[name]


def local_embed(strings: List[str], model: str = None, **parameters) -> List[List[float]]:
    """
    Generates embeddings for a list of strings with a sentence-transformers model on this machine.

    Args:
        strings (List[str]): The strings to generate embeddings for.
        model (str, optional): Overrides the model configured in the [embedding.local] section of llm.config.
        **parameters: Additional encode parameters that override llm.config (e.g. batch_size, device).

    Returns:
        List[List[float]]: The embedding vector of each string, in the same order as the strings provided.
    """

    settings = get_llm_config("embedding.local")
    encoder = get_embedding_model(model)

    arguments = {"normalize_embeddings": settings.get("normalize", "true").strip().lower() == "true"}
    if settings.get("batch_size"):
        arguments["batch_size"] = int(settings["batch_size"])
    arguments.update(parameters)

    return encoder.encode(list(strings), **arguments).tolist()


def embed(strings: List[str], provider: str = None, model: str = None, **parameters) -> List[List[float]]:
    """
    Generates embeddings with whichever provider llm.config selects, so callers do not care which.

    Args:
        strings (List[str]): The strings to generate embeddings for.
        provider (str, optional): Overrides the provider configured in the [embedding] section, 'api' or 'local'.
        model (str, optional): Overrides the model configured for that provider.
        **parameters: Additional parameters passed on to the chosen provider.

    Returns:
        List[List[float]]: The embedding vector of each string, in the same order as the strings provided.
        The api provider returns None if the request fails.
    """

    if get_embedding_provider(provider) == "local":
        return local_embed(strings, model=model, **parameters)
    return api_embed(strings, model=model, **parameters)


def get_cluster_threshold(analysis_type: str, provider: str = None, model: str = None) -> float:
    """
    Returns the cosine distance threshold to cluster at, for the embedding model in force. Every model
    has its own distance distribution, so a threshold calibrated for one model does not carry to another.

    Args:
        analysis_type (str): The analysis being run, either 'po' or 'payment'.
        provider (str, optional): Overrides the provider configured in the [embedding] section.
        model (str, optional): Overrides the model configured for that provider.

    Returns:
        float: The distance threshold below which two descriptions join the same cluster.

    Raises:
        KeyError: If no threshold is configured for that analysis type and model.
    """

    name = get_embedding_model_name(provider, model)
    section = f"threshold.{analysis_type}"
    value = get_llm_config(section).get(name.strip().lower())

    if value is None:
        raise KeyError(
            f"No clustering threshold for embedding model '{name}' in section '[{section}]' of "
            f"'{LLM_CONFIG_PATH}'. Add one and calibrate it before running with this model."
        )
    return float(value)


def api_query(system_prompt: str, user_prompt: str, model: str = None, **parameters) -> str:
    """
    Sends system and user prompts to the responses endpoint of the OpenAI client and retrieves the generated response.

    Args:
        system_prompt (str): The system-level prompt providing context or instructions for the language model.
        user_prompt (str): The user-level prompt containing the main input or query for the language model.
        model (str, optional): Overrides the model configured in the [llm] section of llm.config.
        **parameters: Additional response parameters (e.g. temperature, max_output_tokens) that override llm.config.

    Returns:
        str: The output text generated by the language model if the request is successful.
             Returns None if the request fails, and prints the error message.
    """

    settings = get_llm_config("llm")
    arguments = {
        "model": model or settings["model"],
        "instructions": system_prompt,
        "input": user_prompt
    }

    if "max_tokens" in settings:  # the responses endpoint names the chat completions max_tokens max_output_tokens
        settings.setdefault("max_output_tokens", settings.pop("max_tokens"))

    for option, option_type in (("temperature", float), ("top_p", float), ("max_output_tokens", int)):
        value = parameters.pop(option, settings.get(option))
        if value is not None and value != "":
            arguments[option] = option_type(value)
    arguments.update(parameters)  # any other OpenAI parameter passed by the caller

    try:
        response = get_llm_client("llm").responses.create(**arguments)
    except OpenAIError as e:
        print(f"Error: {e}")
        return None

    return response.output_text


def api_token_count(base_url, query):
    """
    Sends a query to the token count API endpoint and retrieves the number of tokens in the query string.

    Args:
        base_url (str): The base URL of the API endpoint.
        query (str): The text input for which the token count will be calculated.

    Returns:
        dict: A JSON response containing the number of tokens if the request is successful.
              Returns None if the request fails, and prints the error message.
    """

    params = {'query': query}
    response = requests.post(base_url + 'get_token_count', params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def api_ocr(base_url, image_path, language="en", return_annotated_images=False):
    """
    Sends an image file and parameters to an OCR API and retrieves the extracted text (and optionally annotated images).

    Args:
        base_url (str): The base URL of the API endpoint.
        image_path (str): The path to the image file to be processed by OCR.
        language (str): The language to be used for OCR.
        return_annotated_images (bool): Whether to return the annotated image along with OCR results.

    Returns:
        dict: A JSON response containing the OCR results, including extracted text and optionally annotated images,
              if the request is successful. Returns None if the request fails, and prints the error message.
    """

    params = {
        "language": language,
        "return_annotated_images": return_annotated_images
    }

    with open(image_path, 'rb') as img:
        files = {"file": ("image.jpg", img, "image/jpeg")}
        response = requests.post(f"{base_url}/perform_ocr", params=params, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def refine_prompt(user_prompt: str, desired_adjustment: str) -> str:
    """
    Sends a prompt to a language model API to refine it for a desired outcome.

    Args:
        user_prompt (str): The original prompt that needs to be refined.
        desired_adjustment (str): The specific adjustment or outcome desired for the prompt refinement.

    Returns:
        str: The refined prompt generated by the language model if the request is successful.
             If the request fails, returns the original prompt and prints the error message.
    """

    role = "language assistant"
    task = "refine a system prompt to achieve a desired outcome"
    context = (
        f"The user will provide a prompt and a desired adjustment. "
        f"Adjust the prompt to achieve the user's desired adjustment. "
        f"Suggest each of the following: role, task, context, constraints, definitions (optional), example (optional)."
    )
    constraints = (
        f"The refined prompt must achieve the following outcome: {desired_adjustment}. "
        f"Reply strictly in a RFC8259-compliant JSON format containing at least the four mandatory keys: "
        f"role, task, context, constraints, example, definitions, where example and definitions are optional."
    )
    example = {
        "role": "news summarizer",
        "task": "to help a user summarize a news article",
        "context": "The article provided will be long and contain unnecessary information. Keep your summary concise.",
        "constraints": "Respond only with the generated summary."
    }

    refinement_prompt = generate_system_prompt(
        role=role,
        task=task,
        context=context,
        constraints=constraints,
        example=json.dumps(example)
    )

    response = api_query(refinement_prompt, user_prompt)

    if response is not None:
        try:
            result = json.loads(response)

            role = result.get('role', role)
            task = result.get('task', task)
            context = result.get('context', context)
            constraints = result.get('constraints', constraints)
            definitions = result.get('definitions', None)
            example = result.get('example', None)

            refined_prompt = generate_system_prompt(
                role=role,
                task=task,
                context=context,
                constraints=constraints,
                definitions=definitions,
                example=json.dumps(example) if example else None
            )
            return refined_prompt

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing API response: {e}")
            return user_prompt

    else:
        print("Error: No response received from the API.")
        return user_prompt

#
# def load_pdf(file_path: str) -> str:
#     """
#     Load and process a PDF file, returning the text with a new page character separating pages.
#
#     Args:
#         file_path (str): Path to the PDF file.
#
#     Returns:
#         str: Full text of the PDF with a new page character separating each page.
#     """
#     try:
#         if not file_path.endswith('.pdf'):
#             raise ValueError("Invalid file type. Please provide a PDF file.")
#
#         reader = PdfReader(file_path)
#         new_page_char = "\f"
#         text = new_page_char.join(page.extract_text() or "" for page in reader.pages)
#
#         return text
#
#     except Exception as e:
#         raise RuntimeError(f"Error reading the PDF file: {e}")
#
#
# def load_docx(file_path: str, return_df: bool = False) -> Union[str, Tuple[str, List[pd.DataFrame]]]:
#     """
#     Load and process a DOCX file, optionally returning tables as dataframes.
#
#     Args:
#         file_path (str): Path to the DOCX file.
#         return_df (bool): If True, return the text with table indices and a list of dataframes.
#
#     Returns:
#         str: Full text of the DOCX with a newline character separating sections (when return_df=False).
#         Tuple[str, List[pd.DataFrame]]: Text with table placeholders and a list of DataFrames (when return_df=True).
#     """
#     try:
#         if file_path.endswith('.docx'):
#             doc = Document(file_path)
#             content = []
#
#             if return_df:
#                 df_list = []
#                 df_counter = 0
#                 for element in doc.iter_inner_content():
#                     if isinstance(element, Paragraph):
#                         content.append(element.text)
#                     elif isinstance(element, Table):
#                         table_rows = []
#                         for row_idx, row in enumerate(element.rows):
#                             cells = [c.text.replace('\n', ' ').strip() for c in row.cells]
#                             if row_idx == 0:
#                                 columns = cells
#                             else:
#                                 table_rows.append(cells)
#                         df = pd.DataFrame(data=table_rows, columns=columns)
#                         df_list.append(df)
#                         content.append(f"{{df{df_counter}_index}}")
#                         df_counter += 1
#
#                 return '\n'.join(content), df_list
#
#             else:
#                 for element in doc.iter_inner_content():
#                     if isinstance(element, Paragraph):
#                         content.append(element.text)
#                     else:
#                         for r in element.rows:
#                             cells = [c.text for c in r.cells]
#                             content.append('\t'.join(cells))
#
#                 return '\n'.join(content)
#
#         else:
#             raise ValueError("Invalid file type. Please provide a DOCX file.")
#
#     except Exception as e:
#         raise RuntimeError(f"Error reading the file: {e}")


def load_txt(file_path: str, encoding: str = "utf-8") -> str:
    """
    Load and return the content of a TXT file.

    Args:
        file_path (str): Path to the TXT file.
        encoding (str): Encoding used to read the file (default is 'utf-8').

    Returns:
        str: Full text of the TXT file.
    """
    try:
        with open(file_path, encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at '{file_path}' was not found.")
    except IOError as e:
        raise IOError(f"Error reading the file at '{file_path}': {e}")


def load_excel(file_path: str) -> List[pd.DataFrame] | pd.DataFrame:
    """
    Load and return the dataframe(s) of all sheets in an Excel file.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        List[pd.DataFrame] | pd.DataFrame: A list of dataframes if there are multiple sheets,
            or a single dataframe if only one sheet is present.
    """
    try:
        file = pd.ExcelFile(file_path)
        try:
            sheets = file.sheet_names
            if len(sheets) == 1:
                return pd.read_excel(file_path)
            else:
                dfs = [pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets]
                return dfs
        except IOError as e:
            raise IOError(f"Error reading the file at '{file_path}': {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at '{file_path}' was not found.")

#
# def websearch(search_query: str, num_results: int = 2, lang: str = "en", region: str = None) -> List[str]:
#     """
#     Perform a web search using the provided search function and return a list of URLs.
#
#     Args:
#         search_query (str): The query string to search for.
#         num_results (int): Number of results to return. Default is 2.
#         region (str): Google region to target. Default is None.
#         lang (str): Language of the search results. Default is "en".
#
#     Returns:
#         List[str]: A list of URLs from the search results.
#     """
#     try:
#         results = []
#         for result in search(term=search_query, num_results=num_results, lang=lang, region=region):
#             results.append(result)
#         return results
#     except Exception as e:
#         print(f"An error occurred while performing the search: {e}")
#         return []
