from typing import List

import numpy as np
import pandas as pd


def get_genre2movie(netflix_movies: pd.DataFrame) -> pd.DataFrame:
    """convert netflix shows into genre2movie exploded format

    Args:
        netflix_movies (pd.DataFrame): a dataframe loaded from csv

    Returns:
        pd.DataFrame: exploded dataframe, with columns cleaned + dropped
        output columns are:
            - show_id_int
            - title
            - genre_norm
    """
    movies = netflix_movies.copy()
    movies["show_id_int"] = movies.index
    movies["listed_in"] = movies["listed_in"].map(lambda x: x.split(","))
    exploded_genres = movies.explode("listed_in")
    exploded_genres["genre_norm"] = (
        exploded_genres["listed_in"]
        .str.strip()
        .str.lower()
    )
    return exploded_genres[["show_id_int", "title", "genre_norm"]]


def generate_shows_watched_per_year(n: int = 1) -> List[int]:
    """ Will generate the number of shows watched per year,

    The current distribtution is designed around the following:
    50 shows in 1 year is roughly the median
    but the longtail max is around 150

    Args:
        n: the number of samples to generate

    Returns
        List[int]: a list of watched show_id_ints
    """
    # returns whole numbers
    shows_watched = np.random.poisson(lam=4, size=n) * 20 + 20

    # smoothes out the distribution with some noise
    shows_watched += np.random.randint(1, 5, size=n)

    if n == 1:
        return shows_watched[0]
    return shows_watched
