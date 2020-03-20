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


# Runs association rules mining core
def run_ARMC():
    print("TODO: ARMC CODE HERE") # This print never runs, the dictionary ignores them. TODO: Find out why.
    a = 10
    a += 1
    return a


# Runs association rules mining with library
def run_ARML():
    print("TODO: ARML CODE HERE")  # This print never runs, the dictionary ignores them. TODO: Find out why.


def run_CFL():
    print("TODO: CFL CODE HERE")  # This print never runs, the dictionary ignores them. TODO: Find out why.


def main():

    alg_run_dict = {'1': run_ARMC(), '2': run_ARML(), '3': run_CFL()}

    print("Please select the algorithm you wish to explore! (Example: 1) \n"
          "1) Association Rules Mining (Core Python)"
          "2) Association Rules Mining (Library)"
          "3) Collaborative filtering (Library)")

    alg_option = input("Enter option number here: ")

    print(alg_run_dict[alg_option])


main()
