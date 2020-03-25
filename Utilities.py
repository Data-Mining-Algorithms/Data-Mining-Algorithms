import time
import os
from csv import reader
import xml.etree.ElementTree as xml_tree


# region LOAD SECTION
def load_xml(path, options=""):
	"""
	Load the xml file into a tree-like structure.

	Parameters
	----------
	(string) path: path to the file containing the data.
	(string) options: Currently not used. Would have to pass options to the xml file as well because it is in the same
	 				  dictionary as the csv load. Can use this options variable later to customise how to parse the file.
	Returns
    -------
	(Element Object) root: The root node of the XML Tree.
	"""
	return xml_tree.parse(path).getroot()


# Returns a file object
def load_txt(path, options="r"):
	return open(path, options)


# This function will return the reader object
def load_csv(path, options="r"):
	return reader(open(path, options))


# region EXTRA TASKS FOR FUNCTION
# TODO: Add more file formats: XAML, JSON, etc.
#
# TODO: Add configuration options to the function.
# 		1) Choosing between different formats for the final dataset structure:
# 			1.1) Grouping by rows, columns, attributes.
# 		2) Specifying which part of the file to read.
# 			2.1) Where to start and where to end.
# 			2.2) From a specific character index to another character index.
#
# TODO: Look at using the * operator to pass any number of options as one construct.
# endregion
def load_file(file_type, file_path, options="r"):
	# TODO: assert that file_type is a string

	# Implemented exception handling for loading files.
	try:
		loaded_files = {"XML": load_xml, "TXT": load_txt, "CSV": load_csv}
		return loaded_files[file_type.upper()](file_path, options)

	except IOError:    # The exception thrown is IOError, check if my try catch will stop it
		print("\n"+117*"-"+"\n"\
			  "\tNO SUCH FILE OR DIRECTORY: Current Working Directory  - ", os.getcwd(), " \n"\
			  "\tPlease make sure your data set is stored within the 'data' directory "\
			  "in your current working directory!"\
			  "\n"+117*"-"+"\n")

		return None
# endregion


# region Performance Tracking Section
# Metrics to profile: Speed(Time), Calls(Frequency)
def time_function(fun):
	def wrapper(*pos_args, **kw_args):
		start = time.time()
		returned = fun(*pos_args, **kw_args)
		print("TIMED FUNCTION: ", round(time.time() - start, 2), "seconds spent in ", fun.__name__)
		return returned

	return wrapper
# endregion
