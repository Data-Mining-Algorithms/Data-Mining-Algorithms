#######################################################################################################################
#								    ASSOCIATION RULES WITH THE APRIORI ALGORITHM									  #
#																													  #
#  Apriori is an algorithm for frequent item set mining and association rule learning over relational databases. 	  #
#  It proceeds by identifying the frequent individual items in the database and extending them to larger and larger   #
#  item sets as long as those item sets appear sufficiently often in the database. The frequent item sets determined  #
#  by Apriori can be used to determine association rules which highlight general trends in the database: 			  #
#  this has applications in domains such as market basket analysis.													  #
#  - Apriori Algorithm, Wikipedia.																					  #
#																													  #
#  Resources: 																									      #
#	- https://medium.com/analytics-vidhya/association-analysis-in-python-2b955d0180c								  #
# 	- https://towardsdatascience.com/association-rules-2-aa9a77241654												  #
#																													  #
#######################################################################################################################

import os
import csv
import copy
import pickle
from functools import reduce
from operator import add
import re

from HashTree import HashTree, generate_subsets
from Utilities import time_function


# TODO: Check all possible rules for a subset and which one has strong confidence.
# TODO: Create graph visualisation

# region DATA PRE-PROCESSING
@time_function
def load_data(path, prep_required=False):
    """
    Reads the dataset row by row.

    Parameters
    ----------
    (string) path: path to file containing transactions.

    Returns
    -------
    (list) dataset: 2D list that contains the full dataset as row data.
    (list) u_element_list: list containing all the unique items.
    """
    with open(path, 'r') as file_object:
        reader = csv.reader(file_object)
        dataset = list(reader)

    if prep_required:
        dataset = clean_data(copy.deepcopy(dataset[1:]))

    # Reduces the dimension of the dataset from 2D to 1D. This is done by applying the add function from
    # the operator module to all rows in the dataset, equivalent to -> for row in dataset: temp_list += row.
    temp_list = reduce(add, dataset)

    # Fetches all the unique elements from the list and sorts them, this allows to easier manage the data.
    u_element_list = sorted(set(temp_list))

    print("<< ", len(dataset), " >> rows loaded from the given dataset.")
    print("<< ", len(u_element_list), " >> Unique Elements Found ---> ", u_element_list, end="\n\n")

    return dataset, u_element_list


def create_map(items):
    """
    Maps each unique item to an integer to reduce computation time and storage. A reverse mapping was created
    so that the item names could be written in the final output.

    Parameters
    ----------
    (list) items: list of unique items.

    Returns
    -------
    (dict) map: Items --> integers mapping.
    (dict) reverse_map: Integers --> items mapping.
    """
    map = {x: i for i, x in enumerate(items)}
    reverse_map = {i: x for i, x in enumerate(items)}
    return map, reverse_map


def applymap(transaction, map_):
    """
    Applies mapping to items, i.e. each item in the current subset will be replaced by its ID in accordance with
    the map_ dictionary.

    Parameters
    ----------
    (list) transaction: single transaction.
    (dict) map_: mapping.

    Returns
    -------
    (dict) element_map: mapped transaction.
    """
    return [map_[item] for item in transaction]


@time_function
def missing_data_scan(num_columns, dataset):
    """
    Scans the dataset for missing data and displays the insights to the user.

    Parameters
    ----------
    (int) num_columns:
    (int) num_rows:
    (2D list) dataset:

    Returns
    -------
    (dict) missing_data_pos: dictionary that contains a list of the rows that have missing data, for each column.
    """
    # "missing_data_pos" contains a list of rowIds for for each column.
    # Format => {columnID0: [rowID0, rowID1, ...], columnID1: [rowID0, rowID1, ...], ...}
    # Generate the compromised_rows and missing_data_pos dictionaries.
    missing_data_pos = {count: [] for count in range(num_columns)}

    num_rows = len(dataset)

    # Look through the dataset cell by cell and check if the data is missing.
    for row_count in range(num_rows):
        row = dataset[row_count]
        print("Scanning row..")  # SLOW PROCESSING TODO: Make more efficient.

        for column_count in range(len(row)):
            column = row[column_count]

            # First check for missing data, if it is missing no point applying a complete regex validation on it.
            # A full match request of this regex --> "\s+" means there are only spaces in the column string.
            if column == "" or re.fullmatch("\s+", column):
                missing_data_pos[column_count].append(row_count)

    # region Display the Missing Data Information
    # To improve performance removed any column IDs that do not contain any missing data.
    cols_to_remove = []

    print("\n\tMissing Value Count:")
    for column_id in missing_data_pos:
        print("\tColumnID:", column_id, ",  Num Rows:", len(missing_data_pos[column_id]))

        if len(missing_data_pos[column_id]) == 0:
            cols_to_remove.append(column_id)

    print()

    for column_id in cols_to_remove:
        del missing_data_pos[column_id]
    # endregion

    return missing_data_pos


def regex_validation(dataset):
    """
    Regular expressions are a great way to scan the dataset for any abnormalities, it can also be used for
    more complex searching.

    Found this website which allows extensive testing of the regexs, the link provided is the one used to test
    the AllNumbers regex: https://regex101.com/r/I8k3th/2

    1) AllNumbers-S "^\d[\d]*$" - all characters in the column string are numbers, can be in any order
       but no spaces allowed.
    2) AnyCharOrNum+S - 2 or more of any character and digit including spaces, but not just spaces.
    3) UnitPrice - "^£{0,1}([0-9]+\.[0-9]*[1-9]+)|([1-9][0-9]*)" - Must contain any number of digits + . + 2 or more digits
       Format that passes this regex: £3.44, 78.95, 213548.897
    4) Date - "^([0]\d|[1][0-2])\/([0-2]\d|[3][0-1])\/([2][01]|[1][6-9])\d{2}(\s([0-1]\d|[2][0-3])(\:[0-5]\d){1,2})?$"
       This is a regular expression to validate a date string in the formats:
       DD/MM/YYYY; DD/MM/YYYY HH:MM; DD/MM/YYYY HH:MM:SS
    5) TitleCaseWords - "[A-Z][a-z]+" - any word that starts with capital letter but the rest are all lowercase letters.
    """
    # This list contains some common pre-constructed regex to allow easy reuse.
    prep_regex = {"EmptyOrJustSpaces": "\s", "AllNumbers-S": "^\d[\d]*$", "AnyCharOrNum+S": ".+",
                  "UnitPrice": "^£{0,1}([0-9]+\.[0-9]*[1-9]+)|([1-9][0-9]*)", "TitleCaseWords": "[A-Z][a-z]+",
                  "Date": "(^([0-2]\d|[3][0-1])\/([0]\d|[1][0-2])\/([2][01]|[1][6-9])\d{2}(\s{1,4}([0-1]\d|[2][0-3])(\:[0-5]\d){1,2})?$)"
                  }

    # The index of this list represents the column index in the dataset.
    prep_str = [["\d{6}"], ["\d{5}"], [prep_regex["AnyCharOrNum+S"]], [prep_regex["AllNumbers-S"]],
                [prep_regex["Date"]], [prep_regex["UnitPrice"]], ["\d{5}"], [prep_regex["TitleCaseWords"]]]

    num_columns = len(dataset[0])
    # "compromised_rows" contains a list of compromised entries for each column.
    # Format => {columnID0: {rowID0: data0}, columnID1: {rowID1: data1}, ...}
    # Example Structure: {0: {50:"0"}, 1: {8:"85099C"}, 2:{}}
    compromised_rows = {count: {} for count in range(num_columns)}

    # Look through the dataset cell by cell and apply the regex validation to each cell/value.
    for row_count in range(len(dataset)):
        row = dataset[row_count]
        print("Scanning row..")

        for column_count in range(len(row)):
            column = row[column_count]

            # Fetch the all the validation strings for the column
            curr_regex = None
            for regex in prep_str[column_count]:
                curr_regex = regex

            # Extract the data that does not match the set format by applying regex validation on it.
            if not (re.fullmatch(curr_regex, column)):
                compromised_rows[column_count].update({row_count: column})

    # region Display the information extracted from the pre-processing.
    print("\n\tPotentially Erroneous Data:")
    for column_id in compromised_rows:
        print("\t- ColumnID:", column_id, ",  Num Rows:", len(compromised_rows[column_id]))
    # print("\n Frequency & Recency Calculations for the Period (01/12/2010 to 09/12/2011): ")

    inspect = input("\nWould you like to inspect the abnormalities in the data? (Y/N)\n")

    while inspect.upper() == "Y":
        selected_column = input("Please enter the ColumnID of the column you are interested in here: ")

        # Check that the entered input is valid ColumnID
        if selected_column.isdigit() and (0 <= int(selected_column) < num_columns):
            for key, value in compromised_rows[int(selected_column)].items():
                print("Value -->  ", value, " found at row -->  ", key)

            print("\n\tPotentially Erroneous Data:")
            for column_id in compromised_rows:
                print("ColumnID:", column_id, ",  Num Rows:", len(compromised_rows[column_id]))

            inspect = input("\nWould you like to inspect another column? (Y/N)\n")
        else:
            print("Invalid Input, the ColumnID ranges from 0 to", num_columns - 1, "!")
    # endregion
    """
    # This code is not essential for the overall program so it has been commented out to improve performance. 
    cust_invo_match = {}

    for i in range(len(dataset)):
        row = dataset[i]
        invoice_id = row[0]
        customer_id = row[6]

        if invoice_id not in cust_invo_match:
            cust_invo_match.update({invoice_id: [customer_id]})
        else:
            if customer_id not in cust_invo_match[invoice_id]:
                cust_invo_match[invoice_id].append(customer_id)

    print("Num Invoices (Unique Invoice IDS): ", len(cust_invo_match.keys()))

    single_custID = 0
    for value in cust_invo_match.values():
        if len(value) == 1:
            single_custID += 1

    print("Num Invoices with single CustomerID: ", single_custID)"""

    return dataset


@time_function
def remove_missing_data(dataset, missing_data_pos):

    for row_index in missing_data_pos:
        print("Removing Missing Data..")
        del dataset[row_index]

    return dataset


"""@time_function
def remove_missing_data(dataset, missing_data_pos):
    cleaner_dataset = []

    for row_id in range(len(dataset)):
        print("Removing Missing Data..")
        row = dataset[row_id]

        if (row_id not in missing_data_pos) and (row not in cleaner_dataset):
            cleaner_dataset.append(dataset[row_id])

        elif row_id in missing_data_pos:
            del missing_data_pos[0]  # The index here is always 0 because the list is sorted.

    return cleaner_dataset"""


def clean_data(dataset):
    """
        Cleans the loaded data according to the pre-processing strings - prep_str, which are constructed with
        regular expressions (regex).

        (string list) prep_str: 2D List of encoded strings that describes the type of pre-processing needed.
        Each column may have 0 or more pre-processing strings to be executed.
        The prep_str should be in this format for dataset with 3 columns -->  [[], [], []]
        each inner list contains the (regexs) for a column.

        Parameters
        ----------
        (list) dataset: contains the loaded row data from the dataset.
        (bool) insight_required: this variable turn on and off the insight generation about the missing and erroneous data found.

        Returns
        -------
        (list) dataset: 2D list that contains the now clean dataset.
    """
    num_columns = len(dataset[0])
    missing_data_pos = missing_data_scan(num_columns, dataset)

    # Combine all the missing data lists so that it is easier to validate a row.
    temp = []
    for row_id_list in missing_data_pos.values():
        temp += row_id_list

    missing_data_pos = list(set(temp))
    missing_data_pos.sort(reverse=True)

    if input("\nWould you like to remove all the missing data? Y/N\n").upper() == "Y":
        dataset = remove_missing_data(dataset, missing_data_pos)

    return dataset
# endregion


class AssociationRules:
    def __init__(self, data_path):
        # TODO: add option to set these variables through the constructor so that multiple association rules algorithms
        #       can be run at the same time if needed.
        self.MIN_RELATIVE_SUP = 0.024  # 0.024    0.5, 0.5, 10
        self.MIN_CONF = 0.5
        self.HASH_DENOMINATOR = 10

        # APPLY DATA PRE-PROCESSING
        # Load the dataset into the transactions variable & construct an alphabetically sorted list of unique items.
        self.transactions, self.unique_items = load_data(data_path)

        # Generate maps of integer to item and item to integer representation.
        self.item_id_map, self.id_item_map = create_map(self.unique_items)
        # Serialize the unique item to ID map.
        pickle.dump(self.id_item_map, open('reverse_map.pkl', 'wb+'))

    # TODO: Check if more efficient and if so make into static function.
    def generate_combinations(self, subsets_list):
        """
        Function to generate c(k+1) from l(k).

        Parameters
        ----------
        (list) subsets_list: l(k)

        Returns
        -------
        (list) next_unfiltered_k_lvl: c(k+1).
        """
        next_unfiltered_k_lvl = []
        rejected = []

        if len(subsets_list) != 0:
            new_k = len(list(subsets_list[0])) + 1
        else:
            pass    # Possible opportunity for performance improvements.
            # print(":/")

        # Generate all possible combinations of the survived itemsets for current_k+1.
        for i in range(len(subsets_list)):
            for j in range(i + 1, len(subsets_list)):
                temp_a = subsets_list[i]
                temp_b = subsets_list[j]

                # If all but the last element are the same, merge them.
                if temp_a[:-1] == temp_b[:-1]:
                    temp_c = []
                    temp_c.extend(temp_a)
                    temp_c.append(temp_b[-1])
                    temp_c = sorted(temp_c)
                    next_unfiltered_k_lvl.append(temp_c)

                else:
                    temp_c = []
                    temp_c.extend(temp_a)
                    temp_c.append(temp_b[-1])
                    temp_c = sorted(temp_c)
                    rejected.append([temp_a, temp_b, temp_c])

        return next_unfiltered_k_lvl

    @time_function
    def count_k_itemsets(self, candidate_list, transactions):
        """
        Finds the frequency of all the candidates and updates the hash tree.

        Parameters
        ----------
        (dict) candidate_counts: contains a key-value pair for each subset, with the subset being the key and the value
        being the frequency/support of this subset.
        """
        candidate_counts = {}
        tree = HashTree(candidate_list, k=self.HASH_DENOMINATOR, max_leaf_size=100)

        for transaction in transactions:
            subsets = generate_subsets(transaction, len(candidate_list[0]))
            for sub in subsets:  # If the set exists in the tree update its frequency, otherwise add it to the tree.
                tree.check_candidate_exists(sub, update=True)

        for candidate in candidate_list:
            candidate_counts[tuple(candidate)] = tree.check_candidate_exists(candidate, update=False)

        return candidate_counts

    def generate_frequent_itemsets(self):
        """
        Reads data at the given path and generates frequent itemsets list through the Apriori algorithm.

        Parameters
        ----------
        (string) data_path: path to file containing transactions.

        Returns
        -------
        (list) L_final: list of dictionaries containing the final L set.
        """
        # Generate a 1-itemset list in the correct format for the applymap function.
        one_itemset = [[itemset] for itemset in self.unique_items]
        # Generate an ID list for all unique items and for all transaction.
        mapped_u_items = [applymap(itemset, self.item_id_map) for itemset in one_itemset]
        mapped_transactions = [applymap(transaction, self.item_id_map) for transaction in self.transactions]

        # Find the frequency of each 1-itemset in the dataset.
        curr_level_subsets = self.count_k_itemsets(mapped_u_items, mapped_transactions)
        curr_survival_subsets = {}  # Format: {itemset0: frequency, itemset1: frequency, ... itemsetN: frequency}

        # Find the highest frequency to use to convert from min relative support to min support.
        max_freq = max(curr_level_subsets.values())  # 2513
        min_support = round(self.MIN_RELATIVE_SUP * max_freq)  # 60

        # Remove the subsets that do not survive
        for itemset in curr_level_subsets.keys():
            if curr_level_subsets[itemset] >= min_support:
                curr_survival_subsets[itemset] = curr_level_subsets[itemset]

        # The list contains a dictionary per k-level (1-itemsets, 2-itemsets, etc)
        complete_survival_subsets_list = [curr_survival_subsets]
        #complete_survival_subsets_list.append(curr_survival_subsets)

        # While there are more itemset & frequency pairs to look through
        while len(curr_survival_subsets):

            # Use Apriori to construct a list of sets out of the given k-itemsets.
            # Example: if input is 1-itemset list, output is 2-itemset list.
            next_unfiltered_k_level = self.generate_combinations(list(curr_survival_subsets.keys()))

            if len(next_unfiltered_k_level):
                next_level_subsets = self.count_k_itemsets(next_unfiltered_k_level, mapped_transactions)
                curr_survival_subsets = {}

                # Possible efficiency improvement by removing the .keys() call
                # Remove the subsets that do not survive
                for c in next_level_subsets.keys():
                    if next_level_subsets[c] >= min_support:
                        curr_survival_subsets[tuple(sorted(c))] = next_level_subsets[c]

                if len(curr_survival_subsets):
                    complete_survival_subsets_list.append(curr_survival_subsets)
            else:
                break

        pickle.dump(complete_survival_subsets_list, open('l_final.pkl', 'wb+'))
        return complete_survival_subsets_list

    def generate_rules(self, frequent_items):
        """
        Function to generate rules from frequent itemsets.

        Parameters
        ----------
        (list) frequent_items: list containing all frequent itemsets.

        Returns
        -------
        (list) rules: list of generated rules.
        "rules" is stored in the following format: [(X, Y), (X,Y)]
        """
        rules = []

        for k_itemset in frequent_items:
            k = len(list(k_itemset.keys())[0])  # Find out what is the current k.
            if k == 1:  # Association rules cannot be generated from 1-itemsets.
                continue

            for itemset, support in k_itemset.items():
                curr_items = [[item] for item in itemset]
                to_remove = []
                for item in curr_items:
                    X = tuple(sorted(set(itemset) - set(item)))
                    Y = tuple(sorted(item))
                    confidence = support / (frequent_items[k - 2][X])

                    if confidence > self.MIN_CONF:
                        rule = []
                        rule.append(X)
                        rule.append(Y)
                        rules.append({tuple(rule): confidence})
                    else:
                        to_remove.append(item)

                curr_items = [item for item in curr_items if item not in to_remove]

                if len(curr_items) > 1:
                    for m in range(1, k - 1):
                        if k > m + 1:
                            next = self.generate_combinations(curr_items)
                            to_remove = []
                            for item in next:
                                X = tuple(sorted(set(itemset) - set(item)))
                                Y = tuple(sorted(item))
                                confidence = support / (frequent_items[k - m - 2][X])
                                if confidence > self.MIN_CONF:
                                    rule = []
                                    rule.append(X)
                                    rule.append(Y)
                                    rules.append({tuple(rule): confidence})
                                else:
                                    to_remove.append(item)
                            next = [x for x in next if x not in to_remove]
                            curr_items = next
                        else:
                            break
        return rules

    # TODO: Check if more efficient and if so make into static function.
    def display_rules(self, dataset_name, rules, frequent_items, write=False):
        """
        Function to display and write rules to file in the prescribed format.

        Prescribed Format
        -----------------
        Precedent (itemset (support count)) ---> Antecedent (itemset (support count)) - confidence value
        Frequent itemset (support count)

        Parameters
        ----------
        (list) rules: list containing all rules generated by generate_rules function.
        (list) frequent_items: list containing all frequent itemsets.
        (bool) write: write to file if true. Two files are created- association_rules.txt and frequent_itemsets.txt
        """

        reverse_map = pickle.load(open('reverse_map.pkl', 'rb'))
        bad_chars = "[]''"
        with open(os.getcwd()+'\\outputs\\association_rules.txt', 'w+') as fo:
            for rule in rules:
                X, Y = list(rule.keys())[0]
                x_support_count, y_support_count = (frequent_items[len(X) - 1][X], frequent_items[len(Y) - 1][Y])
                confidence = list(rule.values())[0]

                rule_string = '{' + str([reverse_map[x] for x in X]).strip(bad_chars).replace("'", '') + '}, sup(' \
                    + str(x_support_count) + '), rel sup(' + str(round(x_support_count/len(self.transactions)*100)) \
                    + '%) ---> {' + str([reverse_map[y] for y in Y]).strip(bad_chars).replace("'", '') + '}, sup('\
                    + str(y_support_count) + '), rel sup(' + str(round(y_support_count/len(self.transactions)*100)) \
                    + '%)' + ' - conf(' + str(round(confidence*100)) + '%)'

                print(rule_string)
                fo.write(rule_string + '\n')

        with open(os.getcwd()+'\\outputs\\frequent_itemsets.txt', 'w+') as fo:
            fo.write("Dataset: " + dataset_name + "\n")
            for k_itemset in frequent_items:
                for itemset, support in k_itemset.items():
                    fo.write(str([reverse_map[x] for x in itemset]).strip(bad_chars).replace("'", '') + ' sup(' + str(
                        support) + ')' + '\n')
            fo.write("\n\n")


if __name__ == '__main__':
    path = os.getcwd()+'\\Data Repository\\Online Retail.csv'     # simple_dataset, groceries, Online Retail
    rules = AssociationRules(path)  # TODO: Convert to Singleton.

    # All the itemsets that have survived Apriori
    frequent_items = rules.generate_frequent_itemsets()

    print("\nRULE GENERATION..")
    print("FILTERING RULES WITH LOW CONFIDENCE...\n\nSURVIVED ASSOCIATION RULES: ")
    associations = rules.generate_rules(frequent_items)
    rules.display_rules('groceries.csv', associations, frequent_items, write=True)
    num_itemsets = 0

    print("")
    for k_itemsets_lvl in frequent_items:
        num_itemsets += len(k_itemsets_lvl)

    print("After Filtering there are: << ", len(associations), " >> number of rules and "
                                     "<< ", num_itemsets, " >> number of different itemsets.")
