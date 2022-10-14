from typing import Optional, List

import pandas as pd
from InquirerPy import inquirer
from InquirerPy.base import Choice
from igm.conf import InquireRestart

from utils import normpath, is_csv_file, recommend_pca_dims, prompt_with_help

title = 'my-clustering'
file = ''
features: Optional[List[str]] = None
need_pca: bool = True
pca_dims: Optional[int] = None


def inquire_func():
    retval = {}

    global title
    title = inquirer.text(
        message='Title of your project:',
        default=title,
        validate=lambda x: len(x) > 0,
        invalid_message='Title should not be empty.'
    ).execute()
    retval['title'] = title

    global file
    last_file = file
    file = inquirer.filepath(
        message='Select the data source:',
        default=file,
        validate=is_csv_file,
        invalid_message='Data source should be a readable csv file.',
    ).execute()
    retval['data_source'] = normpath(file)
    is_file_changed = normpath(last_file) != normpath(file)

    global features
    df: pd.DataFrame = pd.read_csv(file)
    columns = list(filter(lambda name: not name.startswith('Unnamed:'), df))
    if is_file_changed or features is None:
        features = columns
    features = inquirer.checkbox(
        message='Select the feature columns:',
        choices=[Choice(name, enabled=name in features) for name in columns],
        validate=lambda x: len(x) >= 2,
        invalid_message='No less than 2 features should be selected.'
    ).execute()
    retval['features'] = features

    global need_pca, pca_dims
    need_pca = inquirer.confirm(
        message='Need PCA to reduce the dims?',
        default=need_pca,
    ).execute()
    if need_pca:
        _min_s_dims, _default_dims, _max_s_dims = recommend_pca_dims(df[features], len(features), (2 / 3, 0.85, 0.95))
        pca_dims = inquirer.number(
            message=f'Target dims for PCA (recommendation: {_min_s_dims} - {_max_s_dims})',
            default=int(_default_dims if pca_dims is None else pca_dims),
            filter=int,
            float_allowed=False,
            min_allowed=1,
            max_allowed=len(features),
            invalid_message=f'Dims should be an integer within [{1}, {len(features)}].',
        ).execute()
        retval['need_pca'], retval['pca_dims'] = True, pca_dims
    else:
        retval['need_pca'], retval['pca_dims'] = False, None

    if inquirer.confirm(
            message='Do you known exactly how many clusters are?',
    ).execute():  # kmeans
        retval['algorithm'] = 'kmeans'
        print('KMeans algorithm will be used.')

        clusters = prompt_with_help(inquirer.number(
            message='How many?',
            filter=int,
            float_allowed=False,
            min_allowed=2,
        ), help_text="""
        This is the help information of clusters.
        """).execute()
        retval['clusters'] = clusters
        algo_title = f'KMeans(n_clusters={clusters!r})'
        print(f'{algo_title} algorithm will be applied.')

    else:  # dbscan or optics
        if inquirer.confirm(
                message='Is the distribution between different clusters uniform?',
        ).execute():  # dbscan
            algorithm = 'dbscan'
            print('DBSCAN algorithm will be used.')
        else:  # optics
            algorithm = 'optics'
            print('OPTICS algorithm will be used.')
        retval['algorithm'] = algorithm

        eps = inquirer.number(
            message='Eps value (after standardization):',
            default=0.2,
            filter=float,
            float_allowed=True,
            min_allowed=0.0,
        ).execute()
        min_samples = inquirer.number(
            message='Min samples of each cluster:',
            default=5,
            filter=int,
            float_allowed=False,
            min_allowed=2,
            invalid_message='Min samples should be no less than 2.'
        ).execute()
        retval['eps'], retval['min_samples'] = eps, min_samples
        algo_title = f'{algorithm.upper()}(eps={eps!r}, min_samples={min_samples!r})'
        print(f'{algo_title} algorithm will be applied.')

    if len(features) > 3:
        tsne_dims: Optional[int] = int(inquirer.select(
            message='Visual dims for high dimension data:',
            choices=[2, 3],
            default=2,
            multiselect=False,
        ).execute())
        retval['need_tsne'], retval['tsne_dims'] = True, tsne_dims
    else:
        retval['need_tsne'], retval['tsne_dims'] = False, None

    if inquirer.confirm(
            message=f'Algorithm {algo_title} will be used, confirm?',
    ).execute():
        return retval
    else:
        raise InquireRestart('Not confirmed.')
