"""
Utilities for working with the local cache and the HuggingFace hub.
Functions are adapted from the HuggingFace transformers library at
https://github.com/huggingface/transformers/.
"""
import os
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm

from huggingface_hub import hf_hub_url, cached_download

WEIGHTS_NAME = 'pytorch_model.pt'
WEIGHTS_NAME_SCALE = 'pytorch_model_{scale}x.pt'
CONFIG_NAME = 'config.json'


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def get_model_url(
    model_id: str, filename: str, revision: Optional[str] = None
) -> str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface hub url.
    """
    if revision is None:
        revision = "main"
    return hf_hub_url(model_id, revision=revision, filename=filename)


def get_model_path(
    url_or_filename,
    cache_dir=None,
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path
    Args:
        url_or_filename: the url or filename.
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletely received file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        use_auth_token: Optional string or boolean to use as Bearer token for remote files. If True,
            will get token from ~/.huggingface.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and override the folder where it was extracted.
    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.
    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = cached_download(url_or_filename, cache_dir=cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    return output_path
