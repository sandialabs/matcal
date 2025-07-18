"""
Basic Python Overview
=====================

This chapter serves as a basic primer into using Python and working with object-oriented code. If you are familiar with these concepts, feel free to skip this chapter. 

Python Introduction  
-------------------

Python is a very flexible coding language that has been around since the early 1990's. Python has a wealth of pre-written code that makes it very easy to 
get up and running quicky with almost any coding project. MatCal is one of these libraries of pre-written code. Using MatCal does not require any 
advanced knowledge of Python but understanding some of Python's fundamental operations is critical for using MatCal. 

This primer covers the topics that are relevant for using MatCal. The selected topics are by no means an exhaustive list of the topics in Python
but should serve as a decent starting place for working with MatCal. 

How to Run Python
-----------------

To run Python code, your computer needs a Python interpreter. Because Python is so widely used, most computers comes with a pre-existing Python interpreter. 
Using a terminal, you can run a Python file by executing the following.::

  python my_python_file.py

This will call up the Python interpreter and run the commands written in the file 'my_python_file.py'. Another way of executing Python code it to use the Python 
interpreter interactively. To do this simply execute the Python command without any filenames.:: 

  python

This will cause the terminal to become an interactive Python session. Now any thing you type will be interpreted as a Python command, and typing ``exit()``, will leave
the interactive session. An interactive session is useful for doing simple calculations, and confirming syntax. However, recording commands in a 
Python file and executing Python on the file is the generally the best way to work on larger tasks with Python. 

A Python file is simply a plain text file that has Python commands written in it. To create a new Python file, open a text editor, like "gedit", and write a few 
Python commands and save it. The file can now be run by a Python interpreter. By convention, Python files are saved with a ".py" extension, however this is not necessary. 

MatCal is written in Python3, and any interpreter used needs to be running a Python version 3 or newer. If MatCal is run on an older Python version, then 
it will not work, and it will report errors about valid sections of code.

Variables in Python
-------------------

In Python variables are declared using ``=``, as the assignment operator, where the variable on the left is assigned the value of whatever is on the right. 
"""
x = 3
y = 4.5
z = 11/3
#%%
# In the above code the variables ``x``, ``y``, and ``z`` are assigned the values of 3, 4.5, and 3.66666666667. Any operations to the right of the ``=`` will be performed before the 
# variable value is assigned. Text can be stored in a Python variable through the use of quotation marks. 

what_program = 'MatCal'

#%%
# Underscores can be used in a variable name, and numbers can be used as well, however they cannot be the first character in variable name. 


#%%
# Lists,  Dictionaries, and Objects
# ---------------------------------
#
# Python can store more complicated pieces of information in variables as well. Two of these types of variables are lists and dictionaries. 
# Lists store an ordered collection of variables, that are typically accessed one at a time in the order they are positioned in the list. 
# In a MatCal workflow it is often useful to assemble several items into a list, and then pass it off to one of MatCal's tools.
# An empty list in Python can be created by 

my_list = []

#%%
# Here the brackets tell Python to make a list of the items in between the brackets. In this case, since there is nothing in between the brackets, an empty list is created.
# We can create a populated list by creating a comma separated list of items between the brackets.

my_list_with_fruit = ['apple', 'orange']

#%%
# Entries can be added to the list by appending them to the list. 

my_list.append('apple')
my_list.append('orange')

#%%
# In the above code, we added the strings 'apple' and 'orange' to the list ``my_list``, using the ``append`` method. Now ``my_list`` and ``my_list_with_fruit`` contain the same information. 
# A "method" is a feature specific to object-oriented programming, we can use methods in Python because all variables 
# in Python are concepts known as objects.
# 
# Objects are useful tools because they allow us to store data and the ways of manipulating said data in a compact form.
# In the previous snippet, ``my_list`` is the object we are interacting with. The data the object stores are the entries of the list,
# and one of the actions we can tell the list to do is to add more items to it. This is done using the ``append`` method. Methods are functions that cause the object to do something.
# These methods act on the object they are called and can use input from users through the form of arguments as would any function. 
# Lists have other methods as well such as ``pop``, which allows you to remove items from a list. To find out the exhaustive list of methods available 
# to each object it is best to look up its official documentation. 
#
# We retrieve entries from a list using brackets. 

my_list[0] #'apple'

#%%
# Here we are retrieving the first entry from my_list, and any action taken on the above snippet will treat it like it is the string 'apple'. 
# Indexing in Python is zero based, thus to index the first entry we use 0, and to index the last entry we use N-1, where N is the number of 
# entries in the list. To index the second entry we use 

my_list[1] #'orange'

#%%
# Another container object (an object that stores other objects), that is useful in the MatCal workflow is a dictionary. A dictionary allows one to store 
# some object along with a key label, and then retrieve the object using the key. In MatCal this is useful when defining Python functions to simulate material responses. 
# For example, we can make a simple dictionary containing some information about MatCal.

my_dict = {'name':'MatCal', 'python_version':3}

#%%
# The line above builds a dictionary with the keys 'name' and 'python_version'. If we use the get-item brackets, we can then retrieve the corresponding 
# information from the dictionary as follows:

my_dict['name'] # 'MatCal'
my_dict['python_version'] # 3

#%% 
# and new items can be added to the dictionary through direct assignment of a key. 

my_dict['fruit_list'] = my_list

#%%
# Here we added the list we made earlier to the dictionary. Dictionaries and lists can store more complicated objects other than just numerical values and strings. 


#%%
# For Loops
# ---------
# 
# Often when interacting with data, we are performing the same action repeatedly on different pieces of data. Rather than do this explicitly for each piece of data, 
# this type of operation can be written as a loop. Loops in MatCal are often useful for assembling lists of experimental data to pass to calibration studies. 
# For a simple example of how to use a for loop, we demonstrate how to sum values in a dictionary. We make a list with the dictionary keys, and a dictionary 
# containing a value for each piece of fruit (the dictionary keys).

my_fruit = ['apple', 'apple', 'orange']
fruit_price = {'apple':2, 'orange':1.25, 'grape':.02}

#%% We then create a loop which iterates through the entries of ``my_fruit``, and then looks up the value associated with the key in the 
# dictionary and adds that to a running total. 

total_spent = 0
for fruit_index in range(len(my_fruit)):
  what_fruit = my_fruit[fruit_index]
  total_spent = total_spent + fruit_price[what_fruit]
print(total_spent)

#%%
# Here we created a for loop to loop over all the indices present in ``my_fruit``. Which then allowed us to systematically total up the values in the dictionary. 
# In this example there are a few important things to notice. The first is how we defined what we are looping over. A common Python loop format is ``for X in Y``. Where X will 
# iterate over all entries defined by Y. In our case ``fruit_index`` will loop through 0, 1, and 2. We get these numbers using the built-in Python functions ``range`` (which defines an 
# appropriate index list based on the total number of entries in an object with length) and ``len`` (which returns the length of an object). 
#
# The second thing to notice is how we defined the start and end of the code to be executed in our loop. In the loop declaration it ends with a ``:``. Colons in Python
# define what is known as a context block. A context block represents a subset of code that another line (or lines) of code manages. All code inside a context block 
# must be indented, this creates a nice readable demarcation of where context blocks are. The indentations in a given block must be the same number of spaces. 

#%%
# Functions
# ---------
#
# Storing and assigning values is useful but performing more complicated actions on data is often required for computing tasks. 
# We can do this by defining functions in our code. Functions are useful in MatCal becuase they can be used for fast approximations of material responses, 
# and they can used to streamline other data operations. A simple example of a function is below.

def add_a_and_b(a, b):
  c = a + b
  return c

#%% 
# A function is defined by a line naming the function and then a context block detailing what the function does. The leading 'def' 
# tells Python to expect a function name and a definition. The next entry ``add_a_and_b`` is the name of the function, and will be what you can use later
# to invoke the function. The inputs to the function are defined in parentheses. Here you list variable names the function should expect. You can have 
# as many input variables as desired, or none at all.
#
# Inside the context block of the function is where we describe the operations performed with the variables passed to it. The return command informs Python that
# whatever is to the right of it should be returned to where the function was invoked. 
#
# One important note, in Python variables are very exposed. As such we could write 
g = 10
def messy_function(a):
  return a + g

#%% 
# and, in most cases, the function would operate as expected. This is because the function can see the value for the global variable ``g``. 
# However, this is bad form and can 
# lead to errors or strange behavior in your code. It is recommended that you write a function so that all of the variables it
# uses are passed to it or defined within the function. In programming, this is referred to as keeping variables in 
# different namespaces where a namespace defines a scope where variable names are valid. 
#
# Importing Libraries
# -------------------
# 
# As we mentioned earlier, Python has a large collection of pre-written code. To get access to the objects and functions others have written
# you need to import those libraries into your code. 

import numpy as np

#%%
# The above command illustrates how to import the NumPy library :cite:p:`harris2020array` in to your code. Any lines after the import command have access to the NumPy 
# library. The NumPy library is a commonly imported library that helps
# you manage data,  perform linear algebra calculations and access other mathematical and numerical tools. Here the ``import`` command tells Python that the next term is something that it 
# should import into our code, which in this case is NumPy. Then the 'as np' gives our imported library the alias 'np' and puts all tools from NumPy into 
# a protected namespace accessed through ``np``. We can now use the alias
# to reference the items we want to use in the imported library. For example

np.power(3, 2) # = 9

#%%
# is invoking NumPy to perform the power operation to calculate 3^2. If we don't want to invoke code from our imported libraries using 
# an alias, we can instead import the tools we want directly as follows. 

from numpy import power
power(4, 3) # = 64

#%%
# Here we just imported the power function from the NumPy library and can use it without accessing it via the library alias. We can import 
# all items in a library by using

from matcal import *
my_parameter = Parameter('m', -1, 1)

#%% 
# which uses the wildcard character (*) to import all objects, variables, and functions that the developers intended to be 
# imported into the global namespace. The wildcard 
# import should be used carefully. When importing tools from multiple libraries, this can result in conflicts if these libraries 
# have the same name for different objects.  
#
# Conclusion
# ----------
# This concludes the MatCal Python primer. There is a wealth of Python information on the internet, so you can 
# refer to external websites for more detailed information. Understanding the brief introduction here, should be 
# enough to get you using MatCal. 





