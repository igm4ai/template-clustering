import os
from functools import wraps, partial
from textwrap import dedent
from typing import Tuple, List, Optional, TypeVar

import numpy as np
import pandas as pd
from InquirerPy.base import BaseSimplePrompt
from prompt_toolkit.application import run_in_terminal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def normpath(path: str):
    return os.path.normcase(os.path.expanduser(os.path.normpath(os.path.abspath(path))))


def is_csv_file(path: str):
    return os.path.exists(path) and os.path.isfile(path) and os.access(path, os.R_OK) and \
           os.path.splitext(os.path.normcase(path))[1] == '.csv'


def recommend_pca_dims(df: pd.DataFrame, n_features: int, thresholds: Tuple[float, ...]) -> Tuple[int, ...]:
    standardize = StandardScaler()
    standardize.fit(df)
    std_data = standardize.transform(df)

    pca_all = PCA(n_features)
    pca_all.fit(std_data)
    evr = np.cumsum(pca_all.explained_variance_ratio_)
    return tuple(np.searchsorted(evr, t) + 1 for t in thresholds)


def _operate_in_inquirer(func):
    @wraps(func)
    def _new_func(*args, **kwargs):
        run_in_terminal(partial(func, *args, **kwargs))

    return _new_func


_DEFAULT_HELP_HOTKEY = ['c-w']

PromptType = TypeVar('PromptType', bound=BaseSimplePrompt)


def prompt_with_help(p: PromptType, help_text: str, hotkey: Optional[List[str]] = None) -> PromptType:
    hotkey = hotkey or _DEFAULT_HELP_HOTKEY
    help_text = dedent(help_text).lstrip()
    is_help_shown: bool = False

    @_operate_in_inquirer
    def _show_help(_):
        nonlocal is_help_shown
        if not is_help_shown:
            print(help_text)
            is_help_shown = True

    p.kb_maps = {'show_help': [{"key": key} for key in hotkey]}
    p.kb_func_lookup = {'show_help': [{"func": _show_help}]}

    return p
