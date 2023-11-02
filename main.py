import re
import warnings

from sentence_transformers import SentenceTransformer
from src import config as C
from src import utils as U


class MatchMaker:
    def __init__(self, file_name: str, model_name: str):
        """
        Parameters
        ----------
        file_name: str
            data file name
        model_name: str
            model name of SentenceTransformer package. E.g.: entence-transformers/all-MiniLM-L6-v2

        Instance Variables
        ------------------
        self.data: dict{str: str}
            dictionary {name: description}
        self.model: SentenceTransformer
            SentenceTransformer object
        self.embeddings: {name: list[float]}
            dictionary {name: description_embedding}
        self.dim_reduced_embeddings: {name: list[float]}
            dictionary {name: dim_reduced_embeddings}
        """
        self.data = U.read_file(file_name)
        self.model = SentenceTransformer(model_name)

    def set_model(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def preprocessing(self) -> dict:
        """
        Preprocess the data if necessary
        """
        # remove punctuation
        for name in self.data.keys():
            self.data[name] = re.sub(r"[^\w\s]", "", self.data[name])

        return self.data

    def sentence_embedding(self, model_name: str = None) -> dict:
        if model_name != None:
            self.model = SentenceTransformer(model_name)

        self.embeddings = U.sentence_embedding(self.model, self.data)
        return self.embeddings

    def dimension_reduction(
        self, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = C.RANDOM_STATE
    ) -> list:
        """
        Dimension reduction using tools such as UMAP.
        """
        warnings.simplefilter("ignore")
        self.dim_reduced_embeddings = U.dimension_reduction(self.embeddings, n_neighbors, min_dist, random_state)
        return self.dim_reduced_embeddings

    def visualization(self):
        U.umap_visualization(self.dim_reduced_embeddings)

    def make_pipeline(
        self,
        preprocess: bool = False,
        embed_sentence: bool = False,
        reduce_dimens: bool = False,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = C.RANDOM_STATE,
    ):
        """
        Combine methods preprocessing(), sentence_embedding(), dimension_reduction().
        """
        if preprocess:
            self.preprocessing()
        if embed_sentence:
            self.sentence_embedding()
        if reduce_dimens:
            self.dimension_reduction(n_neighbors, min_dist, random_state)


def main():
    mm = MatchMaker(C.FILE_NAME, C.MINILM_L6_V2)
    mm.make_pipeline(preprocess=True, embed_sentence=True, reduce_dimens=True)
    mm.visualization()


if __name__ == "__main__":
    main()
