import main as M
import matplotlib.pyplot as plt

from src import config as C
from src import utils as U


def compare_models(file_name: str, model_name_1: str, model_name_2: str, target_person: str) -> None:
    """
    Compare two models from SentenceTransformer.
    """
    # calculate embeddings for model 1
    mm = M.MatchMaker(file_name, model_name_1)
    mm.make_pipeline(preprocess=True, embed_sentence=True)
    top_matches_1 = U.similarity_ranking(mm.embeddings, target_person)

    # calculate embeddings for model 2
    mm.sentence_embedding(model_name_2)
    top_matches_2 = U.similarity_ranking(mm.embeddings, target_person)

    x = [i for i in range(len(top_matches_1))]

    # calculate the difference in cosine similarity and ranking of everyone vs target person between the two models
    cosine_diff = [top_matches_2[i][1] - top_matches_1[i][1] for i in range(len(top_matches_1))]
    rank_diff = [top_matches_1[i][2] - top_matches_2[i][2] for i in range(len(top_matches_1))]

    # print result
    print(f"{target_person}'s description: {mm.data[target_person]}")
    top_5 = sorted(top_matches_1, key=lambda x: x[2])[:5]
    print_top_5(mm, top_5, target_person, model_name_1)
    top_5 = sorted(top_matches_2, key=lambda x: x[2])[:5]
    print_top_5(mm, top_5, target_person, model_name_2)

    # draw graph
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 1, 1)
    plt.scatter(x, cosine_diff, s=8)
    plt.axhline(y=0, color="red", linestyle="-")
    plt.tick_params(axis="x", bottom=False, labelbottom=False)
    plt.ylabel("Cosine Difference")
    for i in range(len(top_matches_1)):
        plt.annotate(top_matches_1[i][0][:6], (x[i], cosine_diff[i]), fontsize="8")

    plt.subplot(2, 1, 2)
    plt.scatter(x, rank_diff, s=8)
    plt.axhline(y=0, color="red")
    plt.tick_params(axis="x", bottom=False, labelbottom=False)
    plt.ylabel("Ranks gained")
    for i in range(len(top_matches_1)):
        plt.annotate(top_matches_1[i][0][:6], (x[i], rank_diff[i]), fontsize="8")

    plt.suptitle(
        f"Difference in cosine similarity and ranking of everyone vs {target_person} between {model_name_1} and {model_name_2}",
        size=16,
    )
    plt.show()


def print_top_5(mm: M.MatchMaker, top_5: list[list], target_person: str, model_name: str):
    print(f"\nTop five closest people to {target_person} by embeddings of {model_name}")
    print(f"\n{'Rank':<5} {'Name':<30} {'Cosine Similarity':<20} {'Description'}")
    for name, cosine_similarity, rank in top_5:
        print(f"{rank:<5} {name:<30} {cosine_similarity:<20.4f} {mm.data[name]}")


def main():
    compare_models(C.FILE_NAME, C.MINILM_L6_V2, C.ALL_MPNET_BASE_V2, "Greg Kirczenow")


if __name__ == "__main__":
    main()
