import main as M

from sklearn.metrics import mean_squared_error
from src import config as C
from src import utils as U


def main():
    mm = M.MatchMaker(C.FILE_NAME, C.MINILM_L6_V2)
    mm.make_pipeline(preprocess=True, embed_sentence=True)

    rank_true = U.get_all_ranks(mm.embeddings)

    # paramter tuning
    n_neighbors_list = [5, 10, 15, 20, 53]
    min_dist_list = [0, 0.1, 0.2, 0.5, 0.99]
    random_state = C.RANDOM_STATE

    parameters = [(n_neighbors, min_dist) for n_neighbors in n_neighbors_list for min_dist in min_dist_list]

    least_mse = float("inf")
    tuned_params = []
    for n_neighbors, min_dist in parameters:
        umap_dict = mm.dimension_reduction(n_neighbors, min_dist, random_state)
        rank_pred = U.get_all_ranks(umap_dict)
        mse = mean_squared_error(rank_true, rank_pred)
        if least_mse > mse:
            least_mse = mse
            tuned_params = [n_neighbors, min_dist]

        print(mse, n_neighbors, min_dist)

    print("tuned params ", tuned_params)


if __name__ == "__main__":
    main()
