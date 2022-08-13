"""python -m lessons.B01_user_item_recommenders.example"""
import json
import implicit

import numpy as np

from datasets.generate.users_and_items.make import NetflixUserShowDataset


def main():
    csv_path = "/Users/timlee/Downloads/netflix_titles.csv.zip"
    dataset = NetflixUserShowDataset(csv_path, n_users=500)
    dataset.make()

    csr_data = dataset.as_coo_matrix.tocsr()
    # initialize a model
    model = implicit.als.AlternatingLeastSquares(factors=50)

    # train the model on a sparse matrix of user/item/confidence weights
    model.fit(dataset.as_coo_matrix)

    random_username = np.random.choice(list(dataset.info_dicts.keys()))
    profile = dataset.info_dicts[random_username]
    random_user_id = dataset.user2int[random_username]

    print(random_username)
    print(random_user_id)
    print(json.dumps(profile, indent=2))

    # recommend items for a user
    recommendations = model.recommend(random_user_id, csr_data[random_user_id])

    for item_id, score in zip(*recommendations):
        print(f"{score}\t{item_id}\t{dataset.int2show[item_id]}\t{dataset.movie_info[item_id]['genres']}")


if __name__ == "__main__":
    main()
