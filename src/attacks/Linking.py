from sklearn.metrics.pairwise import cosine_distances

from attacks import Attack

class Linking(Attack):
    """
    1-1 linkability
    """

    def picker(self, entry1, entry2):
        super().picker()
        vec1 = list(self.vecframe[str(entry1)])
        vec2 = list(self.vecframe[str(entry2)])



    def attack(self, vec1, vec2, threshold=0.01):
        super().attack()

        """
        Temporal linkability attack.
        Checks whether cosine similarity of two entries is below `threshold` and if so return a match.
        :param df: StepFrame of all the people's steps
        :param entry1: first entry to compare
        :param entry2: second entry to compare
        :param threshold: threshold below each two users are considered to be the same
        :return: bool, whether two entries come from the same user
        """

        return cosine_distances([vec1], [vec2])[0][0] < threshold
