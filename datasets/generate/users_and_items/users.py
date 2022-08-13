from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from datasets.generate.name_generator import generate_names
from datasets.generate.constants import PARALLELISM
from datasets.util.logging import get_logger

logger = get_logger()


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
    arr = np.power(arr, 2)
    arr_norm = np.divide(arr, arr.sum(axis=1).reshape(-1, 1))
    weights = np.round(arr_norm, 4)

    # ensure rounded probabilities add to 1
    weights[:, 0] = 1 - weights[:, 1:].sum(axis=1)
    return weights


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
    pool = Pool(PARALLELISM)
    for result in pool.imap_unordered(single_func, items):
        profile_collection.append(result)
    pool.close()
    pool.join()

    return profile_collection


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
