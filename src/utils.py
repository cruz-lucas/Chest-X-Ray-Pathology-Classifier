import yaml


def get_config() -> dict:
    """Reads configuration file.

    Returns:
        dict: configuration dictionary.
    """
    with open('config.YAML') as f:
        return yaml.safe_load(f)


def extract_config(config: dict, scope: str, config_name: str):
    """Returns config value with scope.

    Args:
        config (dict): config dict
        scope (str): first level of the config dict
        config_name (str): name for the variable

    Returns:
        _type_: value of configuration variable
    """
    return config.get(scope).get(config_name)
