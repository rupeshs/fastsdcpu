""" Configuration base class and utilities."""

import os
import copy
import json
import logging
from typing import Any, Dict, Tuple, Union

from .file_utils import (
    CONFIG_NAME,
    is_remote_url,
    get_model_url,
    get_model_path
)

logger = logging.getLogger(__name__)


class PretrainedConfig:
    model_type: str = ""

    def __init__(self, **kwargs):
        self._name_or_path = str(kwargs.pop('name_or_path', ''))
        self.scale = None

    @property
    def name_or_path(self) -> str:
        return self._name_or_path

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save a configuration object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~super_image.PretrainedConfig.from_pretrained` class method.
        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)
        logger.info(f"Configuration saved in {output_config_file}")

    @classmethod
    def get_config_dict(
            cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        :class:`~super_image.PretrainedConfig` using ``from_dict``.
        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.
        """
        scale = kwargs.pop("scale", None)
        cache_dir = kwargs.pop("cache_dir", None)
        revision = kwargs.pop("revision", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            config_file = get_model_url(
                pretrained_model_name_or_path, filename=CONFIG_NAME, revision=revision
            )

        try:
            # Load from URL or cache if already cached
            resolved_config_file = get_model_path(
                config_file,
                cache_dir=cache_dir
            )
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
            if scale is not None:
                config_dict['scale'] = scale

        except EnvironmentError as err:
            logger.error(err)
            msg = (
                f"Can't load config for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier \n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {CONFIG_NAME} file\n\n"
            )
            raise EnvironmentError(msg)

        except json.JSONDecodeError:
            msg = (
                f"Couldn't reach server at '{config_file}' to download configuration file or "
                "configuration file is not a valid JSON file. "
                f"Please check network or file content here: {resolved_config_file}."
            )
            raise EnvironmentError(msg)

        if resolved_config_file == config_file:
            logger.info(f"loading configuration file {config_file}")
        else:
            logger.info(f"loading configuration file {config_file} from cache at {resolved_config_file}")

        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> Tuple["PretrainedConfig", Dict[str, Any]]:
        """
        Instantiates a :class:`~super_image.PretrainedConfig` from a Python dictionary of parameters.
        Args:
            config_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~super_image.PretrainedConfig.get_config_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.
        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from those parameters.
        """
        config = cls(**config_dict)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Model config {config}")
        return config, kwargs

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PretrainedConfig":
        """
        Instantiates a :class:`~super_image.PretrainedConfig` from the path to a JSON file of parameters.
        Args:
            json_file (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file containing the parameters.
        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from that JSON file.
        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        return output

    @property
    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.
        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.
        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string)
