import os
import zipfile
from numbers import Number
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wget
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

PLOTTING_COLORS = ("b", "g", "r", "c", "m", "y", "k")


def plot_vectors(vector_list: List[Tuple[Number, Number]]):
    """Plot a list of vectors

        ```
        v1 = [1,2]
        v2 = [0.5, 3]
        v3 = [-0.2, -4]
        v4 = [-10, 0.25]
        plot_vectors(vector_list = [v1, v2, v3, v4])
        ```

    Will create matplotlib quiver plot, of the listed vectors supplied

    Args:
        vector_list (List[Tuple[Number, Number]]): _description_
    """
    n_vectors = len(vector_list)
    V = np.array(vector_list)
    origin = np.zeros(shape=(2, n_vectors))
    ax = plt.gca()
    ax.quiver(*origin, V[:, 0], V[:, 1], color=PLOTTING_COLORS[:n_vectors], scale=4 * np.abs(V).max())
    plt.show()


def compare_two_vectors(vector1: Tuple[int, int], vector2: Tuple[int, int]):
    """Compare two vectors, and print out the cosine_sim + euclid_dist

        ```
        v1 = [1, 2]
        v2 = [2, 5]
        compare_two_vectors(v1, v2)
        0.996546: cosine_sim
        3.162278: euclid_dist
        ```
    Args:
        v1 (_type_): _description_
        v2 (_type_): _description_
    """
    distances_between_two_vectors(vector1, vector2)
    plot_vectors(vector_list=[vector1, vector2])


def distances_between_two_vectors(vector1: Tuple[int, int], vector2: Tuple[int, int]):
    vector_stack = np.array([vector1, vector2])

    cosine_sim = cosine_similarity(X=vector_stack)[1, 0]
    euclid_dist = euclidean_distances(X=vector_stack)[1, 0]

    print("{:02f}: cosine_sim".format(cosine_sim))
    print("{:02f}: euclid_dist".format(euclid_dist))
    return cosine_sim, euclid_dist


def download_word_vecs():
    """downloads word vec from the FB website"""
    wget.download("https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip")
    with zipfile.ZipFile("./wiki-news-300d-1M.vec.zip") as f:
        f.extractall()


def import_word_vecs(word_vec_text_file: str = "wiki-news-300d-1M.vec") -> Dict[str, List[float]]:
    """Designed to read in the text file and the vectors into mem

        Sample format
        ```
        man 0.125 0.253 0.6236 0.6235 ....
        cat  0.523 0.262 0.234 0.623 ...
        kitten 0.236 0.5745 0.378 0.765 ...
        ```

    Args:
        word_vec_text_file (str): the FB word file

    Returns:
        Dict[str, List[float]]: returns a "word" -> float vector
    """
    word_vecs = dict()
    with open(word_vec_text_file, "r") as fin:
        n, d = map(int, fin.readline().split())
        counter = 0
        for line in fin:
            tokens = line.rstrip().split(" ")
            word_vecs[tokens[0]] = np.array(tokens[1:]).astype(float)
            counter += 1
            if counter % 10_000 == 0:
                print("{:,}{:12,}".format(n, counter))
    return word_vecs


def load_and_cache_word_vecs(
    word_vec_text_file: str = "wiki-news-300d-1M.vec", cache_dir: str = "./"
) -> pd.DataFrame:
    """Load / parse / cache word vectors

        1. download word vectors in the text file format
        2. parse float values
        3. save in parquet format for later

    Args:
        word_vec_text_file (str, optional): _description_. Defaults to "wiki-news-300d-1M.vec".
        cache_dir (str, optional): _description_. Defaults to "./".

    Returns
        pd.DataFrame
    """
    cache_filename = Path(cache_dir) / "word_vecs.parquet"
    if os.path.exists(str(cache_filename)):
        return pd.read_parquet(str(cache_filename))

    word_vecs = import_word_vecs(word_vec_text_file)
    word_df = pd.DataFrame(word_vecs).transpose()
    word_df.columns = [f"v{j:03}" for j in range(300)]

    word_df.to_parquet(cache_filename, engine="fastparquet")
    return word_df
