import importlib

import yaml


def load_domain_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def load_test_questions(config: dict) -> list[str]:
    module_path, attr = config["test_questions_module"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)
