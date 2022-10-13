import pandas as pd
from ditk import logging
from hbutils.color import rnd_colors
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from config import DATA_SOURCE_FILE, FEATURES

if __name__ == '__main__':
    logging.try_init_root(logging.INFO)

    logging.info(f'Loading data from {DATA_SOURCE_FILE!r} ...')
    source = pd.read_csv(DATA_SOURCE_FILE)
    raw_data = source[FEATURES]

    logging.info('Standardizing ...')
    standardize = StandardScaler()
    standardize.fit(raw_data)
    std_data = standardize.transform(raw_data)

    # {% if user.need_pca %}
    logging.info('Reducing dimensions from {{len(user.features) | potc}} to {{user.pca_dims | potc}} with PCA ...')
    pca = PCA(n_components={{user.pca_dims | potc}})
    pca.fit(std_data)
    pca_data = pca.transform(std_data)

    logging.info('Standardizing ...')
    standardize = StandardScaler()
    standardize.fit(pca_data)
    std_data = standardize.transform(pca_data)
    # {% endif %}

    # {% if user.algorithm == 'kmeans' %}
    logging.info('Clustering with KMeans algorithm to {{user.clusters | potc}} parts ...')
    algo = KMeans(n_clusters={{user.clusters | potc}})
    # {% elif user.algorithm == 'dbscan' %}
    logging.info('Clustering with DBSCAN algorithm ...')
    algo = DBSCAN(eps={{user.eps | potc}}, min_samples={{user.min_samples | potc}})
    # {% elif user.algorithm == 'optics' %}
    logging.info('Clustering with OPTICS algorithm ...')
    algo = OPTICS(eps={{user.eps | potc}}, min_samples={{user.min_samples | potc}})
    # {% endif %}
    algo.fit(std_data)
    pred = algo.fit_predict(std_data)

    dst = source.copy(deep=False)
    dst['pred'] = pred
    # {% if user.need_tsne %}
    logging.info('Reducing dimensions from {{len(user.features) | potc}} '
                 'to {{user.tsne_dims | potc}} with TSNE for visualization ...')
    tsne = TSNE({{user.tsne_dims | potc}}, learning_rate='auto')
    visual_data = tsne.fit_transform(raw_data)
    # {% if user.tsne_dims == 2 %}
    dst['_v_x'] = visual_data[:, 0]
    dst['_v_y'] = visual_data[:, 1]
    # {% elif user.tsne_dims == 3 %}
    dst['_v_x'] = visual_data[:, 0]
    dst['_v_y'] = visual_data[:, 1]
    dst['_v_z'] = visual_data[:, 2]
    # {% endif %}
    # {% endif %}

    print('Visualizing ...')
    fig = plt.figure()
    # {% if (user.need_tsne and user.tsne_dims == 3) or (not user.need_tsne and len(user.features) == 3) %}
    ax = fig.add_subplot(projection='3d')
    # {% else %}
    ax = fig.add_subplot()
    # {% endif %}

    # noise points
    noises = dst[dst['pred'] < 0]
    if not noises.empty:
        # {% if user.need_tsne %}
        # {% if user.tsne_dims == 2 %}
        ax.scatter(noises['_v_x'], noises['_v_y'], c='black', label='Noise')
        # {% else %}
        ax.scatter(noises['_v_x'], noises['_v_y'], noises['_v_z'], c='black', label='Noise')
        # {% endif %}
        # {% else %}
        # {% if len(user.features) == 2 %}
        ax.scatter(noises[{{user.features[0] | potc}}], noises[{{user.features[1] | potc}}], c='black', label='Noise')
        # {% else %}
        ax.scatter(noises[{{user.features[0] | potc}}], noises[{{user.features[1] | potc}}],
                   noises[{{user.features[2] | potc}}], c='black', label='Noise')
        # {% endif %}
        # {% endif %}

    n_clus = dst['pred'].max() + 1  # clusters
    for i, color in zip(range(n_clus), rnd_colors(n_clus)):
        cluster = dst[dst['pred'] == i]
        # {% if user.need_tsne %}
        # {% if user.tsne_dims == 2 %}
        ax.scatter(cluster['_v_x'], cluster['_v_y'], c=str(color), label=f'clu-{i}')
        # {% else %}
        ax.scatter(cluster['_v_x'], cluster['_v_y'], cluster['_v_z'], c=str(color), label=f'clu-{i}')
        # {% endif %}
        # {% else %}
        # {% if len(user.features) == 2 %}
        ax.scatter(cluster[{{user.features[0] | potc}}], cluster[{{user.features[1] | potc}}],
                   c=str(color), label=f'clu-{i}')
        # {% else %}
        ax.scatter(cluster[{{user.features[0] | potc}}], cluster[{{user.features[1] | potc}}],
                   cluster[{{user.features[2] | potc}}], c=str(color), label=f'clu-{i}')
        # {% endif %}
        # {% endif %}

    ax.set_title({{('Visualization of ' + user.title) | potc}})
    # {% if not user.need_tsne %}
    # {% if len(user.features) == 2 %}
    ax.set_xlabel({{user.features[0] | potc}})
    ax.set_ylabel({{user.features[1] | potc}})
    # {% else %}
    ax.set_xlabel({{user.features[0] | potc}})
    ax.set_ylabel({{user.features[1] | potc}})
    ax.set_zlabel({{user.features[2] | potc}})
    # {% endif %}
    # {% endif %}
    ax.legend()
    plt.show()

    logging.info('Complete!')
