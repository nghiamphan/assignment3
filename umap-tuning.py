import optuna
import main as M

from sklearn.metrics import mean_squared_error
from src import config as C
from src import utils as U


class Objective:
    def __init__(self, mm: M.MatchMaker, random_state: int):
        self.mm = mm
        self.mm.make_pipeline(preprocess=True, embed_sentence=True)
        self.rank_true = U.get_all_ranks(mm.embeddings)
        self.random_state = random_state

    def __call__(self, trial):
        n_neighbors = trial.suggest_int("n_neighbors", 2, 53)
        min_dist = trial.suggest_float("min_dist", 0, 0.99)
        umap_dict = self.mm.dimension_reduction(n_neighbors, min_dist, self.random_state)
        rank_pred = U.get_all_ranks(umap_dict)
        mse = mean_squared_error(self.rank_true, rank_pred)

        return mse


def umap_tuning_with_optuna(mm: M.MatchMaker, random_state: int) -> dict:
    study = optuna.create_study()
    study.optimize(Objective(mm, random_state), n_trials=20)
    print(f"\nBest parameters: {study.best_params}. Least MSE: {study.best_value}")
    return study.best_params


def umap_tuning_without_optuna(mm: M.MatchMaker, random_state: int):
    mm.make_pipeline(preprocess=True, embed_sentence=True)

    rank_true = U.get_all_ranks(mm.embeddings)

    # paramter tuning
    n_neighbors_list = [5, 10, 15, 20, 53]
    min_dist_list = [0, 0.1, 0.2, 0.5, 0.99]

    parameters = [(n_neighbors, min_dist) for n_neighbors in n_neighbors_list for min_dist in min_dist_list]

    least_mse = float("inf")
    best_params = {"n_neighbors": n_neighbors_list[0], "min_dist": min_dist_list[0]}
    for n_neighbors, min_dist in parameters:
        umap_dict = mm.dimension_reduction(n_neighbors, min_dist, random_state)
        rank_pred = U.get_all_ranks(umap_dict)
        mse = mean_squared_error(rank_true, rank_pred)
        if least_mse > mse:
            least_mse = mse
            best_params["n_neighbors"] = n_neighbors
            best_params["min_dist"] = min_dist

        print(
            f"Parameters: {{'n_neighbors': {n_neighbors}, 'min_dist': {min_dist}}}. MSE: {mse}. Current least MSE: {least_mse}"
        )

    print(f"\nBest parameters: {best_params}. Least MSE: {least_mse}")
    return best_params


def main():
    mm = M.MatchMaker(C.FILE_NAME, C.MINILM_L6_V2)

    # Best parameters: {'n_neighbors': 20, 'min_dist': 0}. Least MSE: 293.27721335268507
    # params_dict = umap_tuning_without_optuna(mm, C.RANDOM_STATE)

    params_dict = umap_tuning_with_optuna(mm, C.RANDOM_STATE)

    # draw graph with tuned params
    mm.dimension_reduction(
        n_neighbors=params_dict["n_neighbors"], min_dist=params_dict["min_dist"], random_state=C.RANDOM_STATE
    )
    image_file_name = (
        f"n_neighbors_{params_dict['n_neighbors']}_min_dist_{params_dict['min_dist']}_random_{C.RANDOM_STATE}.png"
    )
    mm.visualization(image_file_name)

    # draw graph with tuned params and a different seed
    mm.dimension_reduction(
        n_neighbors=params_dict["n_neighbors"], min_dist=params_dict["min_dist"], random_state=C.RANDOM_STATE_2
    )
    image_file_name = (
        f"n_neighbors_{params_dict['n_neighbors']}_min_dist_{params_dict['min_dist']}_random_{C.RANDOM_STATE}.png"
    )
    mm.visualization(image_file_name)


if __name__ == "__main__":
    main()
