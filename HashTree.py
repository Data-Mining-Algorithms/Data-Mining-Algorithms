#######################################################################################################################
#								   					HASH TREE														  #
#																													  #
#  A persistent data structure that can be used to implement sets and maps, intended to replace hash tables in purely #
#  functional programming. In its basic form, a hash tree stores the hashes of its keys, regarded as strings of bits, #
#  in a trie, with the actual keys and (optional) values stored at the trie's "final" nodes.						  #
#  - Hash tree, Wikipedia.																					          #
#																													  #
#######################################################################################################################


class Node:
    """
    Class that implements a leaf node in the hash tree.
    """

    def __init__(self, k, max_leaf_size, depth):
        """
        Initialize the leaf node.

        Parameters
        ----------
        k : int, optional
            denominator of the hash function.
        max_leaf_size : int, optional
            maximum number of itemsets that can be present in a leaf node.
        depth : int, optional
            the depth at which the leaf node is present in the hash tree.

        """
        self.max_leaf_size = max_leaf_size
        self.depth = depth
        self.children = {}
        self.k = 0
        self.isTree = False

    def add(self, candidate):
        """
        Initialize the leaf node.

        Parameters
        ----------
        candidate : list
            the candidate that needs to be inserted.

        """
        self.children[tuple(candidate)] = 0


class HashTree:

    def __init__(self, candidate_list, k=3, max_leaf_size=3, depth=0):
        """
        Initializes a new hash tree and inserts the items present in the c_list into the hash tree.

        Parameters
        ----------
        (list) c_list: list of itemsets to be inserted into the hash tree.
        (int) k: optional, denominator of the hash function.
        (int) max_leaf_size : optional, maximum number of itemsets that can be present in a leaf node.
        (int) depth : optional, the current depth at which the tree is present.

        Usage
        -----
        # >>> t=Tree(c_list=[[1,2], [2,3], [3,4]], k=3, max_leaf_size=3, depth=0)
        The tree has been created and the itemsets [1,2], [2,3] and [3,4] have been innserted into the tree.

        """
        self.k = k
        self.max_leaf_size = max_leaf_size
        self.depth = depth

        self.children = {}
        self.is_tree = True
        self.candidate_len = len(candidate_list[0])
        self.build_tree(candidate_list)

    def update_tree(self):
        """
        Function which splits the leaf node of the tree if it contains more elements than self.max_leaf_size.

        """
        for child in self.children:
            if len(self.children[child].children) > self.max_leaf_size:
                if self.depth + 1 < self.candidate_len: 	# Make sure that only fewer than k divisions are performed
                    child = HashTree(list(self.children[child].children.keys()), k=self.k, max_leaf_size=self.max_leaf_size, depth=self.depth + 1)

    def build_tree(self, c_list):
        """
        Function that builds the tree and inserts the itemsets into the tree.

        Parameters
        ----------
        (list) c_list: list of itemsets to be inserted into the hash tree.

        """
        for candidate in c_list:
            if candidate[self.depth] % self.k not in self.children:
                self.children[candidate[self.depth] % self.k] = Node(k=self.k, max_leaf_size=self.max_leaf_size, depth=self.depth)
            self.children[candidate[self.depth] % self.k].add(candidate)
        self.update_tree()

    def check_candidate_exists(self, candidate, update=False):
        """
        Function to check if candidate is present in the hash tree and to update support counts of tree elements.

        Parameters
        ----------
        (list) candidate: the candidate that needs to be checked.
        (bool) update: optional, If true, the count of candidate is incremented in the tree.

        Returns
        ----------
        (int) support: Support count of the candidate.

        """
        support = 0
        if candidate[self.depth] % self.k in self.children:
            child = self.children[candidate[self.depth] % self.k]
            if child.isTree:
                support = child.check_candidate_exists(candidate)
            else:
                if tuple(candidate) in list(child.children.keys()):
                    if update:
                        child.children[tuple(candidate)] += 1
                    return child.children[tuple(candidate)]
                else:
                    return 0
        return support


def generate_subsets(transaction, k):
    """
    Function to recursively generate k subsets of the transaction.

    Parameters
    ----------
    (list) transaction : the transaction whose subsets have to be generated.
    (int) k : number of items in each subset.

    Returns
    ----------
    (list) res: List containing subsets of the transaction.
    """
    res = []
    n = len(transaction)
    transaction.sort()

    def recurse(transaction, k, i=0, curr=[]):
        """
        Recursion function used in subset generation
        """
        if k == 1:
            for j in range(i, n):
                res.append(curr + [transaction[j]])
            return None
        for j in range(i, n-k+1):
            temp = curr + [transaction[j]]
            recurse(transaction, k-1, j+1, temp[:])
    recurse(transaction, k)

    return res


if __name__ == '__main__':
    temp_list = [[1, 2, 3], [2, 3, 4], [3, 5, 6], [4, 5, 6], [5, 7, 9], [7, 8, 9], [4, 7, 9]]
    t = HashTree(temp_list, k=3, max_leaf_size=3, depth=0)
