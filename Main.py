# TODO:
#######################################################################################################################
#                                             Data Mining Algorithms                                                  #
#                                                                                                                     #
#           This project was developed for the Data Mining module at Teesside University with the aim to              #
#           demonstrate and evaluate the use of popular computational techniques for data mining. The                 #
#           project contains implementations of popular Data Mining techniques such as Dimension Reduction,           #
#           Association Rules Mining, Collaborative Filtering and a variety of data sets to test on.                  #
#                                                                                                                     #
#                                                                                                                     #
#   Data Repository Contains:                                                                                         #
#       >                                                                                                             #
#       >                                                                                                             #
#                                                                                                                     #
#                                                  Developed by                                                       #
#                               Aleksandra Petkova, Victor Essien, Nour Al Mubarak                                    #
#                                                                                                                     #
#######################################################################################################################

import os
import AssociationRulesCore as ARMC


# Runs association rules mining core
def run_ARMC():
    print("\n\nRUNNING APRIORI ASSOCIATION RULES CORE PYTHON VERSION..")

    path = os.getcwd() + '\\Data Repository\\groceries.csv'  # simple_dataset, groceries, Online Retail
    rules = ARMC.AssociationRules(path, minimum_relative_sup=0.025, minimum_confidence=0.5)

    # All the itemsets that have survived Apriori
    frequent_items = rules.generate_frequent_itemsets()

    print("\nRULE GENERATION..")
    print("REMOVING RULES WITH LOW CONFIDENCE...\n\nSURVIVED ASSOCIATION RULES: ")
    associations = rules.generate_rules(frequent_items)
    rules.display_rules('groceries.csv', associations, frequent_items, write=True)
    num_itemsets = 0

    print("")
    for k_itemsets_lvl in frequent_items:
        num_itemsets += len(k_itemsets_lvl)

    print("After Filtering there are: << ", len(associations), " >> number of rules and "
                                                               "<< ", num_itemsets, " >> number of different itemsets.")


# Runs association rules mining with library
def run_ARML():
    print("TODO: ARML CODE HERE")


def run_CFL():
    print("TODO: CFL CODE HERE")


def main():
    alg_run_dict = {'1': run_ARMC, '2': run_ARML, '3': run_CFL}

    print("\nPlease select the algorithm you wish to run! (1, 2 or 3)"
          "\n\t1) Association Rules Mining (Core Python)"
          "\n\t2) Association Rules Mining (Library)"
          "\n\t3) Collaborative filtering (Library)")

    alg_option = input("\nEnter chosen option here: ")

    while not (alg_option.isdigit() and 0 < int(alg_option) < 4):
        print("\nInvalid input, must be a number between 0 and 4!")
        alg_option = input("Re-enter option here: ")
    else:
        alg_run_dict[alg_option]()


main()
