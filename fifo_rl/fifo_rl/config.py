"""Training config: configs/default.yaml + command-line overrides -> one plain dict.

    parser = argparse.ArgumentParser()
    add_config_args(parser)              # adds --config and one --<key> per config key
    args = parser.parse_args()
    cfg = config_from_args(args)

Override values are read as YAML, so "--double false" gives False, "--lr 3e-4" a float and
"--hidden [64,64]" a list. Unknown keys in a YAML file are an error (a typo would otherwise be
silently ignored and the default used).
"""

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path=DEFAULT_CONFIG, overrides=None):
    """Input:  path, a YAML file; overrides, an optional dict of keys to replace.
    Output: the config dict.
    Logic:  the defaults file is always read first, then the given file (if it is a different
            one) on top, then the overrides; every key must exist in the defaults.
    """
    cfg = _read(DEFAULT_CONFIG)
    known = set(cfg)
    if Path(path).resolve() != DEFAULT_CONFIG:
        cfg.update(_checked(_read(path), known, f"in {path}"))
    cfg.update(_checked(overrides or {}, known, "in the overrides"))
    return cfg


def add_config_args(parser):
    """Input: an argparse parser. Output: nothing; adds --config and --<key> for every default key."""
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config file")
    group = parser.add_argument_group("config overrides (value read as YAML)")
    for key, value in _read(DEFAULT_CONFIG).items():
        group.add_argument(f"--{key.replace('_', '-')}", dest=f"cfg_{key}", type=parse_value,
                           default=None, metavar="V", help=f"default: {value}")


def parse_value(text):
    """Input: a command-line string. Output: its YAML value; "3e-4" (a string in YAML 1.1) -> float."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def config_from_args(args):
    """Input: parsed args from a parser set up by add_config_args. Output: the config dict."""
    overrides = {name[len("cfg_"):]: value for name, value in vars(args).items()
                 if name.startswith("cfg_") and value is not None}
    return load_config(args.config, overrides)


def save_config(cfg, path):
    """Input: config dict, file path. Output: nothing; writes it as YAML (keeps the key order)."""
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


def _read(path):
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a mapping of config keys")
    return data


def _checked(values, known, where):
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys {where}: {unknown}")
    return values
