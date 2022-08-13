import json
from collections import Counter, OrderedDict
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np

from datasets.util.logging import get_logger
from datasets.generate.name_generator import generate_names

PARALLELISM = 4
logger = get_logger()

# ======================================================================
# MAIN Runtime
# ======================================================================


def generate_synthetic_dataset(
    csv_path: str, n_users: int = 10, n_genres: int = 5, show_report: bool = False
):
    netflix_movies = pd.read_csv(csv_path)

    genre2movie = get_genre2movie(netflix_movies)
    genres_unique = genre2movie["genre_norm"].unique()

    profile_collection = generate_user_profiles(
        genres_unique, n_users=n_users, dim=n_genres
    )

    user2watches = generate_synthetic_watch_records(
        profile_collection, genre2movie
    )

    movie_info = (
        genre2movie
        .groupby("show_id_int")
        .agg(
            title=pd.NamedAgg("title", "first"),
            genres=pd.NamedAgg("genre_norm", tuple),
        )
        .to_dict("index")
    )

    if show_report:
        info_dicts = report_user_info(profile_collection, user2watches, genre2movie)
        return user2watches, movie_info, info_dicts

    return user2watches, movie_info

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================


def report_user_info(profile_collection: List[tuple], user2watches: Dict[str, list], genre2movie: pd.DataFrame):
    info_dicts = dict()
    for user, profile in profile_collection:
        movie_ids = user2watches[user]
        synthetic_counts = (
            genre2movie.loc[np.array(movie_ids), "genre_norm"]
            .value_counts()
            .to_dict()
        )

        info_dicts[user] = dict(
            genre_profile_src=profile,
            genre_counts_synthetic=synthetic_counts,
        )
    logger.info(json.dumps(info_dicts, indent=2))
    return info_dicts


def generate_synthetic_watch_records(
    profile_collection: List[tuple],
    genre2movie: pd.DataFrame
) -> Dict[str, List[int]]:
    """Generates synthetic watch records

    Args:
        profile_collection (List[tuple]): a list of tuples representing
        different users + their genre preferences
            ("george user", {"action" 0.5, "mystery" 0.3, ...})
        genre2movie (pd.DataFrame): a simple dataframe that has
            show_id_int 111
            title       "sherlock holmes"
            genre_norm  "mystery"

    Returns:
        Dict[str, List[int]]: a dictionary of user names + the show_id_ints
        that they have watched (over a year)
    """
    single_func = partial(
        sample_watch_records_for_one_user,
        genre2movie=genre2movie
    )
    pool = ThreadPool(PARALLELISM)
    user2watched_show_ids = dict()
    for result in pool.imap_unordered(single_func, profile_collection):
        username, watched_show_ids = result
        user2watched_show_ids[username] = watched_show_ids
    return user2watched_show_ids


def sample_watch_records_for_one_user(
    user_tuple: tuple, genre2movie: pd.DataFrame
) -> Tuple[str, List[int]]:
    """Generates watch records for a single user

    Args:
        user_tuple (tuple): has the (username, genre_preferences)
        genre2movie (pd.DataFrame): has the exploded genre + movie listing

    Returns:
        Tuple[str, List[int]]: returns the username, and a list of show_ids
        that they have watched
    """
    username, profile = user_tuple
    try:
        watches = generate_shows_watched_per_year(n=1)
        logger.info(f"sampling records [{watches}] for: {username}")
        watches_by_genre = split_watch_ct_by_genre_probs(watches, profile)
        genre2show_id_probs = calculate_show_probability_based_on_user_profile(
            genre2movie, profile
        )

        watched_show_ids = []
        for genre in watches_by_genre:
            genre_watch_ct = watches_by_genre[genre]
            show_id_probs = genre2show_id_probs[genre]

            probs = show_id_probs["show_prob"]
            show_ids = np.random.choice(
                show_id_probs["show_id_int"],
                p=probs,
                replace=True,
                size=genre_watch_ct
            )
            watched_show_ids.extend(show_ids)

        return username, watched_show_ids
    except Exception as e:
        logger.error(f"{username} {e}")
        raise e


def split_watch_ct_by_genre_probs(watches, profile):
    genres, probs = zip(*profile.items())
    probs = list(probs)
    probs[0] = 1 - sum(probs[1:])
    genre_per_watch = np.random.choice(genres, p=probs, size=watches)
    ctr = Counter(genre_per_watch)
    return dict(ctr)


def calculate_show_probability_based_on_user_profile(
    genre2movie: pd.DataFrame, profile: dict
) -> pd.DataFrame:
    user_copy = genre2movie.copy()

    # keep only the genres that the user is interested in
    mask = user_copy["genre_norm"].isin(profile.keys())
    user_copy = user_copy[mask].copy()

    # apply the genre scores to each movie instance, some may have 2
    user_copy["genre_probs"] = user_copy["genre_norm"].map(profile)
    show_id_probs = user_copy.groupby("show_id_int")["genre_probs"].sum()

    # normalize within genre
    user_copy["show_score"] = user_copy["show_id_int"].map(show_id_probs)
    user_copy["show_prob"] = user_copy["show_score"] / user_copy.groupby("genre_norm")["show_score"].transform(sum)

    genre2show_id_probs = (
        user_copy.groupby("genre_norm")[["show_id_int", "show_prob"]].agg(list).to_dict("index")
    )

    return genre2show_id_probs


def generate_weighted_profile(n: int = 2, dim: int = 6) -> np.ndarray:
    """ Generates n_profiles, with probability distribution

    Example of a 2 x 5 = 2 users, 5 genre's each
        [[0.5, 0.3, 0.15, 0.03, 0.02]]
        [[0.45, 0.25, 0.15, 0.08, 0.07]]

    Args:
        n (int): number of users
        dim (int): dimension (how many genres they will have probs for)

    Returns:
        np.ndarray: a [n x dim] array of probabilities
    """
    floats = np.random.rand(n, dim)

    # negatives are to sort in reverse order
    arr = - np.sort(
        - np.round((floats * 1000)), axis=1,
    )
    arr = np.power(arr, 1.5)
    arr_norm = np.divide(arr, arr.sum(axis=1).reshape(-1, 1))
    weights = np.round(arr_norm, 4)

    # ensure rounded probabilities add to 1
    weights[:, 0] = 1 - weights[:, 1:].sum(axis=1)
    return weights


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


def _generate_up(
    tup: tuple, dim: int, profiles: np.ndarray, genres_unique: List[str]
) -> Tuple[str, dict]:
    """generate the user profile, executes a single record

    Intended to be used with a multiprocessing setup

    Args:
        tup (tuple): a tuple of (j: id, and n: string name)
        dim (int): how many genres to sample
        profiles (np.ndarray): global list of profiles [user x genre]
        genres_unique (List[str]): global list of possible genres

    Returns:
        Tuple[str, dict]: returns a tuple of
            (name, {genre: prob, ...})
    """
    j, n = tup
    genres = np.random.choice(genres_unique, size=dim)
    weights = list(np.round(profiles[j], 4))
    weights[0] = 1 - sum(weights[1:])
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"weights not totaling 1: {weights}, {sum(weights)}")

    profile = OrderedDict([(g, w) for g, w in zip(genres, weights)])
    logger.info(f"generating preferences for : {j:3} {n} \n {profile}")
    return tuple([
        n,
        profile
    ])


def generate_user_profiles(
    genres_unique: np.ndarray, n_users: int = 3, dim: int = 5
) -> List[Tuple[str, dict]]:
    """Generates (n_user) x profiles sized (dim)

    Args:
        genres_unique (np.ndarray): unique listing of genre (str)
        n_users (int, optional): number of users to make. Defaults to 3.
        dim (int, optional): how many genres per user to make. Defaults to 5.

    Returns:
        List[Tuple[str, dict]]: Returns a collection of
            (user, profile<dict>)
    """
    names = generate_names(n_users)
    profiles = generate_weighted_profile(n=n_users, dim=dim)

    items = list(enumerate(names))
    single_func = partial(
        _generate_up,
        dim=dim,
        profiles=profiles,
        genres_unique=genres_unique
    )

    profile_collection = []
    pool = ThreadPool(PARALLELISM)
    for result in pool.imap_unordered(single_func, items):
        profile_collection.append(result)
    pool.close()
    pool.join()

    return profile_collection


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
