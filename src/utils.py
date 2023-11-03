import csv
import matplotlib.pyplot as plt
import umap

from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


def read_file(file_name: str) -> dict:
    """
    Read a csv file and return a dictionary.

    Parameters
    ----------
    file_name: str
        The csv file has columns: name and description.

    Return
    ------
    people: dict{str: str}
        dictionary {name: description}
    """
    people = {}
    with open(file_name, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            name, description = row
            people[name] = description
    return people


def sentence_embedding(model: SentenceTransformer, people: dict) -> dict:
    """
    Embedding descriptions of people.

    Parameters
    ----------
    model: SentenceTransformer
        SentenceTransformer object
    people: dict{str: str}
        dictionary {name: description}

    Return
    ------
    person_embeddings: {str: list[float]}
        dictionary {name: description_embedding}
    """
    person_embeddings = {}
    for name in people.keys():
        person_embeddings[name] = model.encode(people[name])
    return person_embeddings


def dimension_reduction(person_embeddings: dict, n_neighbors: int, min_dist: float, random_state: int) -> dict:
    """
    Return UMAP vectors

    Parameters
    ----------
    person_embeddings: {str: list[float]}
        dictionary {name: description_embedding}

    Return
    ------
    umap_dict: dict{str: list[float]}
        dictionary {name: umap_vector}
    """
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    umap_vectors = umap_model.fit_transform(list(person_embeddings.values()))

    umap_dict = {}
    for i, name in enumerate(person_embeddings.keys()):
        umap_dict[name] = umap_vectors[i]

    return umap_dict


def umap_visualization(umap_dict: dict, image_file_name: str = None) -> None:
    """
    Visualize people's embeddings using UMAP.

    Parameters
    ----------
    umap_dict: dict{str: list[float]}
        dictionary {name: umap_vector}
    image_file_name: str
        file name to save the image
    """
    plt.figure(figsize=(20, 10))

    umap_vectors = list(umap_dict.values())
    x = [row[0] for row in umap_vectors]
    y = [row[1] for row in umap_vectors]

    plt.scatter(x, y)
    for i, name in enumerate(umap_dict):
        plt.annotate(name, (x[i], y[i]), fontsize=9)

    if image_file_name:
        plt.savefig(image_file_name, dpi=800)
    plt.show()


def similarity_ranking(person_embeddings: dict, target_person: str) -> list[list]:
    """
    Given the embeddings of people's description and a target person,
    return a list of cosine similarities and cosine similarity rankings
    of other people (sorted alphabetically by name) to the target person.

    Parameters
    ----------
    person_embeddings: {str: list[float]}
        dictionary {name: description_embedding}
    target_person: str
        name of the person from whom to calculate cosine similarities with other people

    Return
    ------
    top_matches: list[list[str, float, int]]
        [[name, cosine_similarity, rank]] ranked alphabetically by name
    """
    # calculate cosine similarity
    top_matches = []
    for name in person_embeddings.keys():
        top_matches.append([name, 1 - cosine(person_embeddings[target_person], person_embeddings[name])])

    # add similarity ranking
    top_matches.sort(key=lambda x: -x[1])  # sort by cosine similarity
    for i, match in enumerate(top_matches):
        match.append(i)

    # sort by name
    top_matches = top_matches[1:]
    top_matches.sort(key=lambda x: x[0])
    return top_matches


def get_all_ranks(person_embeddings: dict) -> list:
    """
    Given the embeddings of people's description, for each person, get a list of
    cosine similarity rankings of other people (sorted alphabetically) by name to that person.
    Return the concanation of all those lists.

    Parameters
    ----------
    person_embeddings: {str: list[float]}
        dictionary {name: embedding}

    Return
    ------
    rank: list[int]
    """
    rank = []

    for target_person in person_embeddings.keys():
        top_matches = similarity_ranking(person_embeddings, target_person)
        rank += [top_matches[i][2] for i in range(len(top_matches))]

    return rank
