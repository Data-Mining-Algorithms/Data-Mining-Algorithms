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
    pass


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

    alg_option = input("\n\tEnter chosen option here: ")

    while not (alg_option.isdigit() and 0 < int(alg_option) < 4):
        print("\nInvalid input, must be a number between 0 and 4!")
        alg_option = input("Re-enter option here: ")
    else:
        alg_run_dict[alg_option]()


main()
