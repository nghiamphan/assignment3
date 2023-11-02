from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from src import config as C


def compare_embedding(str1: str, str2: str) -> float:
    """
    Given two strings, compare their embeddings' cosine similarity.
    """
    model = SentenceTransformer(C.MINILM_L6_V2)
    embedding_1 = model.encode(str1)
    embedding_2 = model.encode(str2)

    cosine_similarity = 1 - cosine(embedding_1, embedding_2)
    return cosine_similarity


def main():
    original = "I like hiking, swimming, traveling, enjoying nice weather and sun (and h.a.t.e winters!)"

    # Change word order
    # hiking, swimming, traveling -> swimming, traveling, hiking
    mod_1 = "I like swimming, traveling, hiking, enjoying nice weather and sun (and h.a.t.e winters!)"
    # move "swimming, traveling, hiking" to the end
    mod_2 = "I like enjoying nice weather and sun (and h.a.t.e winters!), swimming, traveling, hiking"

    # Substitute with synonyms
    # h.a.t.e -> hate
    mod_3 = "I like hiking, swimming, traveling, enjoying nice weather and sun (and hate winters!)"
    # like -> love
    mod_4 = "I love hiking, swimming, traveling, enjoying nice weather and sun (and h.a.t.e winters!)"
    # like -> luv
    mod_5 = "I luv hiking, swimming, traveling, enjoying nice weather and sun (and h.a.t.e winters!)"

    # Use antonyms
    # like -> don't like
    mod_6 = "I don't like hiking, swimming, traveling, enjoying nice weather and sun (and h.a.t.e winters!)"
    # like -> don't like; h.a.t.e -> love
    mod_7 = "I don't like hiking, swimming, traveling, enjoying nice weather and sun (and love winters!)"
    # like -> hate
    mod_8 = "I hate hiking, swimming, traveling, enjoying nice weather and sun (and h.a.t.e winters!)"
    # like -> hate; h.a.t.e -> love
    mod_9 = "I hate hiking, swimming, traveling, enjoying nice weather and sun (and love winters!)"

    # Shorten sentence
    mod_10 = "I like hiking, swimming, traveling, enjoying nice weather and sun"
    mod_11 = "I like hiking, swimming, traveling and h.a.t.e winters"

    # Rewrite with opposite meaning
    mod_12 = "I like sitting at home, enjoy cold weather and winters"
    mod_13 = "I like sitting at home"

    print("Cosine similarity of the original sentence vs modified sentences")

    print("\nChange word order")
    print(1, compare_embedding(original, mod_1))
    print(2, compare_embedding(original, mod_2))

    print("\nSubstitute with synonyms")
    print(3, compare_embedding(original, mod_3))
    print(4, compare_embedding(original, mod_4))
    print(5, compare_embedding(original, mod_5))

    print("\nUse antonyms")
    print(6, compare_embedding(original, mod_6))
    print(7, compare_embedding(original, mod_7))
    print(8, compare_embedding(original, mod_8))
    print(9, compare_embedding(original, mod_9))

    print("\nShorten sentence")
    print(10, compare_embedding(original, mod_10))
    print(11, compare_embedding(original, mod_11))

    print("\nRewrite with opposite meaning")
    print(12, compare_embedding(original, mod_12))
    print(13, compare_embedding(original, mod_13))


if __name__ == "__main__":
    main()
