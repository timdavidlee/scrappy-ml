import json
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from datasets.generate.users_and_items.shows import get_genre2movie, generate_shows_watched_per_year
from datasets.generate.users_and_items.users import generate_user_profiles
from datasets.generate.constants import PARALLELISM
from datasets.util.logging import get_logger


logger = get_logger()

# ======================================================================
# MAIN Runtime
# ======================================================================


class NetflixUserShowDataset:
    def __init__(self, csv_path: str, n_users: int = 10, n_genres: int = 5):
        self.csv_path = csv_path
        self.n_users = n_users
        self.n_genres = n_genres
        self.user2watches = None
        self.movie_info = None
        self.info_dicts = None

        self.user2int = None
        self.int2user = None

        self.show2int = None
        self.int2show = None

        self.watches_df = None

    def _load(self):
        x, y, z = generate_synthetic_dataset(
            self.csv_path, self.n_users, self.n_genres, show_report=True
        )
        self.user2watches = x
        self.movie_info = y
        self.info_dicts = z
        self.int2user = {j: user for j, user in enumerate(sorted(x.keys()))}
        self.user2int = {user: j for j, user in enumerate(sorted(x.keys()))}
        self.show2int = {v["title"]: k for k, v in self.movie_info.items()}
        self.int2show = {k: v["title"] for k, v in self.movie_info.items()}

    def make(self):
        self._load()

    @property
    def _as_df(self):
        if self.watches_df is not None:
            return self.watches_df

        dfs = []
        for user in self.user2watches:
            df = pd.DataFrame(dict(
                user_id_int=self.user2int[user],
                show_id_int=self.user2watches[user],
            ))
            dfs.append(df)
        concat_df = pd.concat(dfs, axis=0, ignore_index=True)
        agg = concat_df.groupby(["user_id_int", "show_id_int"], as_index=False).size()
        agg = agg.rename(columns={"size": "watches"})
        self.watches_df = agg
        return agg

    @property
    def as_coo_matrix(self):
        agg = self._as_df
        return coo_matrix(
            (
                agg["watches"].values,
                (agg["user_id_int"].values, agg["show_id_int"].values),
            )
        )


def generate_synthetic_dataset(
    csv_path: str, n_users: int = 10, n_genres: int = 5, show_report: bool = False
) -> tuple:
    """Create a recommender dataset

        user2watches, movie_info = generate_synthetic_dataset(csv_path, n_users=100)

    Args:
        csv_path (str): path the netflix movie/genre dataset
        n_users (int, optional): Number of fake users to generate. Defaults to 10.
        n_genres (int, optional): Number of genre preferences per user. Defaults to 5.
        show_report (bool, optional): Return an info dict about the users. Defaults to False.

    Returns:
        users2watches: username : show_id_ints
            {'kip feist': [5396, 7954, 5570, ...}
        movie_info:
            {
                0: {'title': 'Dick Johnson Is Dead', 'genres': ('documentaries',)},
                1: {'title': 'Blood & Water',
                'genres': ('international tv shows', 'tv dramas', 'tv mysteries')},
                2: {'title': 'Ganglands',
                'genres': ('crime tv shows',
                'international tv shows',
                'tv action & adventure')},
                3: {'title': 'Jailbirds New Orleans', 'genres': ('docuseries', 'reality tv')},
                ...
        info_dicts:
            {
                "sheldon mudpuppy": {
                    "genre_profile_src": {
                        "international tv shows": 0.5241,
                        "anime features": 0.4334,
                        "stand-up comedy": 0.023,
                        "dramas": 0.0162,
                        "cult movies": 0.0033
                    },
                    "genre_counts_synthetic": {
                        "international tv shows": 45,
                        "anime features": 37,
                        "international movies": 29,
                        "action & adventure": 27,
                        "tv dramas": 19,
                        "crime tv shows": 12,
    """
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
    pool = Pool(PARALLELISM)
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
