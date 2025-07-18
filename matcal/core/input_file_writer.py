from abc import ABC, abstractmethod
from collections import OrderedDict
from numbers import Number, Integral
from io import TextIOWrapper

from matcal.core.utilities import (check_item_is_correct_type, 
                                   check_value_is_nonempty_str, 
                                   check_value_is_positive_integer,
                                   check_value_is_bool)


_default_indent = "    "


class InputFileLine():
    """
    Create an input file line to be added to a file or input file block.

    :param leading_statement: first word in the line that is always printed.
    :type leading_statement: str

    :param args: list of values to be added to the line after the leading 
        statement.
    :type args: list(str, float, int)

    :param name: optional name for the line. It is used only for
        accessing the line as part of a block or input file and is not
        printed. Default is the leading statement if one is not passed.
    :type name: str
    """

    default_symbol = "="
    default_symbol_index = 1
    def __init__(self, leading_statement, *args, name=None):
        self._symbol_location = self.default_symbol_index
        self._use_symbol = True
        self._symbol = self.default_symbol
        check_value_is_nonempty_str(leading_statement, "leading_statement", 
                                    "InputFileLine")
        if name is None:
            name = leading_statement
        else:
            check_value_is_nonempty_str(name, "name", 
                                        "InputFileLine", )
        self.name = name

        self._my_values = []
        self._my_values.append(leading_statement)
        for arg in args:
            check_item_is_correct_type(arg, (str, Number), "InputFileLine", 
                                      "arg")
            self._my_values.append(arg)

    def write(self, f, indent=0):
        """
        Write the line to a file handle object.

        :param f: the file object that line will be written to.
        :type f: TextIOWrapper

        :param indent: The number of indents to be applied to the line. 
            The default indent is four spaces.
        :type indent: int
        """
        check_item_is_correct_type(indent, int, "write", "ident")
        check_item_is_correct_type(f, TextIOWrapper, "write", "f")
        lines = self.get_string(indent)
        f.write(lines)

    def get_string(self, indent=0):
        """
        Return the line string with a specified indent.

        :param indent: the number of indents
            to be applied to the line. 
            The default indent is four spaces.
        :type indent: int
        """
        check_item_is_correct_type(indent, int, "get_string", "ident")
        line = _default_indent * indent
        for idx, item in enumerate(self._my_values):
            if item is None:
                continue
            if idx != 0:
                line += " "
            if self._use_symbol and idx == self._symbol_location:
                line += self._symbol + " "
            if isinstance(item, float):
                line += "{:.12g}".format(item)
            else:
                line += str(item)
        line += "\n"
        return line

    def suppress_symbol(self):
        """
        Do not output a symbol between the line leading 
        statement and the following values.
        """
        self._use_symbol = False

    def set_symbol(self, symbol):
        f"""
        Change the symbol between the leading statement and the following 
        values. Default symbol is \"{self.default_symbol}\".

        :param symbol: the updated symbol to be used
        :type symbol: str  
        """
        check_item_is_correct_type(symbol, str, "set_symbol", "symbol")
        self._symbol = symbol
        self._use_symbol = True

    def set_symbol_location(self, index):
        f"""
        Set the position for the symbol location. 
        The index is the index of the value list field before which 
        the symbol will appear.

        :param index: the index of the line value before which the symbol 
            will appear. The default values is {self.default_symbol_index}.
        :type index: int
        """
        check_value_is_positive_integer(index, "set_symbol_location", "index")
        self._symbol_location = index

    def set_at_end(self, value):
        """
        Set the line's last value. 

        :param value: the last value the line will contain.
        :type value: float or str
        """
        idx = len(self._my_values)
        self.set(value, idx)
       

    def set(self, value, index=1):
        """
        Set the line's value. By default it sets the lines value 
        after the leading statement (default index of 1). If an index  
        greater than the current length is chosen, empty entries are added 
        until the line reaches the correct number of entries

        warning :: a non-string or float can be added. It must be able to be 
            converted to a string using the str function to be printed correctly. 

        :param value: the value to be added to the line.
        :type value: float or str

        :param index: the location of the value place in the line list.
        :type index: int
        """
        check_value_is_positive_integer(index, "index", "set")
        if len(self._my_values) <= index:
            for i in range(index - len(self._my_values) + 1):
                self._my_values.append(None)
        self._my_values[index] = value

    def get_values(self):
        """
        Returns a list of all values in the line including the leading statement.

        :rtype: list(str)
        """
        return self._my_values


class InputFileBlock:
    """
    Create an input file block to be added to a file or input file block
    as a subblock.

    :param title: The block title. This is printed by default and all lines 
        in the subblock are printed as indented lines.
    :type title: str

    :param name: An optional name. Set to title by default. The name is how 
        to access the subblock from any parent container object.
    :type args: list(str, float, int)

    :param name: optional name for the line. It is used for
        accessing the subblock as part of a block or input file and is not
        printed by default. The name can be printed in place of the title if 
        one is used.
    :type name: str

    :param begin_end: Prepend "Begin " to the block title and close the block 
        with an "End {title}" line.
    :type being_end: bool
    """
    def __init__(self, title, name=None, begin_end=False):
        check_value_is_nonempty_str(title, "title", 
                                    "InputFileBlock")
        self._title = title
        self._name = None
        check_item_is_correct_type(begin_end, bool, "InputFileBlock", 
                                   "begin_end")
        self._begin_end = begin_end
        if name is None:
            self.set_name(title)
        else:
            check_value_is_nonempty_str(name, "name", 
                                    "InputFileBlock")
            self.set_name(name)

        self._lines = OrderedDict()
        self._subblocks = OrderedDict()
        self._tables = OrderedDict()
        self._indent = 0
        self._print_title = True
        self._print_name = False

    @property
    def title(self):
        """
        Returns the subblock title.

        :rtype: str
        """
        return self._title
    
    @property
    def name(self):
        """
        Returns the subblock name.

        :rtype: str
        """
        return self._name

    @property
    def lines(self):
        """
        Returns a list of all block lines.

        :rtype: list(:class:`~matcal.core.input_file_writer.InputFileLine`)
        """
        return self._lines

    @property
    def subblocks(self):
        """
        Returns a list of all block subblocks.

        :rtype: list(:class:`~matcal.core.input_file_writer.InputFileBlock`)
        """
        return self._subblocks

    @property
    def tables(self):
        """
        Returns a list of all block tables.

        :rtype: list(:class:`~matcal.core.input_file_writer.InputFileTable`)
        """
        return self._tables

    def print_title(self):
        """
        This method returns if the block will print the title.
        """
        return self._print_title

    def add_line(self, line, replace=False):
        """
        Add a line to the input file block.

        :param line: the line to be added.
        :type line: :class:`~matcal.core.input_file_writer.InputFileLine`

        :param replace: replace existing value if the line is in
            the block lines when sen to True
        :type replace: bool
        """

        check_item_is_correct_type(line, InputFileLine, "add_line", "line")
        if line.name in self._lines and not replace:
            raise KeyError(f"A line with the name '{line.name}' is already included in block. " 
                           "Use the 'replace' keyword argument if you want to replace an "
                           "existing line.")
        self._lines[line.name] = line

    def add_lines(self, *lines, replace=False):
        """
        Add a set or list of lines to the input file block.

        :param lines: the lines to be added.
        :type lines: list(:class:`~matcal.core.input_file_writer.InputFileLine`)

        :param replace: replace existing value if the line is in
            the block lines when sen to True
        :type replace: bool
        """
        for line in lines:
            self.add_line(line, replace=replace)

    def add_lines_from_dictionary(self, dictionary, replace=False):
        """
        Use a dictionary to add several keyword, value pairs 
        to the block as lines. The keywords will end up being the lines'
        leading statements and the values will be the values. 

        :param dictionary: the dictionary containing the line information to 
            be added to the block.
        :type dictionary: dict(str or float or tuple(float, str) or list(float, str))

        :param replace: replace lines if already existing in the subblock
        :type replace: bool
        """
        check_item_is_correct_type(dictionary, dict, "add_lines_from_dictionary", 
                                   "dictionary")
        for key, val in dictionary.items():
            if isinstance(val, bool): 
                if val:
                    self.add_line(InputFileLine(key), replace)
            elif isinstance(val, (tuple, list)):
                self.add_line(InputFileLine(key, *val), replace)
            else:
                self.add_line(InputFileLine(key, val), replace)
            
    def add_subblock(self, subblock, replace=False, 
                     set_begin_end_to_parent=False):
        """
        Add a subblock to the input file block or input file. 

        :param subblock: the subblock to be added. Can be unpopulated.
        :type subblock: :class:`matcal.core.input_file_writer.InputFileBlock`

        :param replace: the subblock will replace an existing one if found.
        :type replace: bool
        """
        self._verify_valid_subblock_to_add(subblock)
        self._handle_preexisting(subblock, replace)

        subblock._indent = self._indent + 1
        self._adjust_subblock_indent(subblock)
        self._subblocks[subblock._name] = subblock
        if set_begin_end_to_parent:
            self._subblocks[subblock._name]._begin_end = self._begin_end
            self._adjust_subblock_begin_end(self._subblocks[subblock._name], self._begin_end)

    def _verify_valid_subblock_to_add(self, subblock):
        if not isinstance(subblock, InputFileBlock):
            raise TypeError()
        if self._name == subblock._name:
            raise ValueError(f"A subblock with name \"{self._name}\" ", 
                             "cannot be added to this subblock "
                             "because it has the same name.")

    def _handle_preexisting(self, subblock, replace):
        if subblock._name in self.subblocks and replace:
            self.remove_subblock(subblock._name)
        elif subblock._name in self.subblocks:
            raise ValueError(f"A subblock with name \"{subblock._name}\" ", 
                            "already exists in this block.")

    def remove_subblock(self, subblock):
        """
        Remove and return the passed subblock or subblock name.

        :param subblock: the subblock to be removed from the input 
            file block.
        :type subblock: str or :class:`matcal.core.input_file_writer.InputFileBlock`
        """
        try:
            if isinstance(subblock, str):
                self._subblocks.pop(subblock)
            elif isinstance(subblock, InputFileBlock):
                self._subblocks.pop(subblock._name)
        except KeyError:
            raise KeyError(f"The passed subblock \"{subblock}\" "
                           "is not in the current input file block.")

    def add_table(self, table):
        """
        Add an InputFileTable to the subblock.
        """
        check_item_is_correct_type(table, InputFileTable, 
                                   "add_table", "table")
        self._tables[table.name] = table

    def get_line(self, line_name):
        """
        Returns a line with the passed line name. 

        :param line_name: the name of the line desired.
        :type line_name: str

        :rtype: :class:`~matcal.core.input_file_writer.InputFileLine`
        """
        check_value_is_nonempty_str(line_name, "line_name", "get_line")
        if line_name in self._lines.keys():
             return self._lines[line_name]
        else:
            raise KeyError(
                f"A line with name \"{line_name}\" is not a line of \"{self._name}\". "
                f"Possible options are: {self._lines.keys()}")

    def get_line_value(self, key, index = 1):
        """
        Return the value from the line values at a specified index.
        By default it returns the second value, which is the value after the 
        line keyword. 

        :param key: the name of the line. Usually the keyword.
        :type key: str

        :param index: an optional parameter to set a 
            different index for the returned value from the line
        :type index: int
        """
        check_item_is_correct_type(key, str, "get_line_value", "key", TypeError)
        check_item_is_correct_type(index, Integral, "get_line_value", "index",
                                   TypeError )
        line = self.get_line(key)
        values = line.get_values()
        if index > len(values)-1:
            raise ValueError(f"The line \"{key}\" in block \"{self._name}\" "
                             "does not have a value at index {index}")
        return values[index]

    def get_subblock(self, key):
        """
        Get a subblock by name from the block. 

        :param key: the name of the desired subblock.
        :type key: str
        """
        if key in self._subblocks.keys():
             return self._subblocks[key]
        else:
            raise KeyError(
                f"The key \"{key}\" is not a subblock of \"{self._name}\". "
                f"Possible options are: {self._subblocks.keys()}")
        
    def get_table(self, name):
        """
        Returns a table for a given table name if it is in the input file/block. 

        :param name: name of the desired table.
        :type name: str
        """
        if name in self._tables.keys():
            return self._tables[name]
        else:
            raise KeyError(
                f"The name \"{name}\" is not a table of \"{self._name}\". "
                f"Possible options are: {self._tables.keys()}")
                                                               
    def _adjust_subblock_indent(self, block):
        if len(block._subblocks) < 1:
            return ()
        else:
            for sb in block._subblocks.values():
                sb._indent = block._indent + 1
                sb._adjust_subblock_indent(sb)

    def _adjust_subblock_begin_end(self, block, BE):
        if len(block._subblocks) < 1:
            return ()
        else:
            for sb in block._subblocks.values():
                sb._begin_end = BE
                sb._adjust_subblock_begin_end(sb, BE)

    def write(self, f):
        """
        Write the subblock to a file handle object.

        :param f: the file object that line will be written to.
        :type f: TextIOWrapper
        """
        str = self.get_string()
        f.write(str)
        
    def get_string(self):
        """
        Return the subblock string.
        """
        input_string = ""
        space = _default_indent * self._indent
        input_string += self._get_block_start(space)
        
        for line in self._lines.values():
            input_string += line.get_string(self._indent + 1)

        for table in self._tables.values():
            input_string += table.get_string(self._indent + 1)

        for sb in self._subblocks.values():
            input_string += "\n"
            input_string += sb.get_string()
            input_string += "\n"

        if self._begin_end:
            input_string += self._get_block_end(space)

        return input_string 

    def _get_block_start(self, indent):
        return self._get_block_bound_str(indent, "Begin")
    
    def _get_block_end(self, indent):
        return self._get_block_bound_str(indent, "End")

    def _get_block_bound_str(self, indent, bound_key):
        block_title = self._get_block_title_string()
        bound_string = indent
        if self._begin_end:
            bound_string += bound_key + " " + block_title + "\n"
        else:
            bound_string += block_title + "\n"
        if bound_string.isspace():
            bound_string = ""
        return bound_string

    def _get_block_title_string(self):
        block_title = ""
        if self._print_name and self._print_title:
            block_title = self._title+" "+self._name
        elif self._print_name:
            block_title = self._name
        elif self._print_title:
            block_title = self._title
        return block_title

    def set_print_name(self, print_name=True, print_title=False):
        """
        Controls whether to print the name of the subblock. By default, 
        it prints the name and not the title. 

        :param print_name: print the name if True or do not if False.
        :type print_name: bool
        """
        check_value_is_bool(print_name, "print_name", 
                            "InputFileBlock.print_name")
        self._print_name = print_name

    def set_print_title(self, print_title=True):
        """
        Controls how to print the title of the subblock. By default, 
        it prints the name and not the title. If the title is printed,
        it is printed before the name.
        
        :param print_title: print the title with the name or as the name
        :type print_title: bool
        """
        check_value_is_bool(print_title, "print_title", 
                            "InputFileBlock.print_name")
        self._print_title = print_title

    def set_name(self, name):
        """
        Set the block name.
        
        :param name: the new block name.
        :type name: str
        """
        check_value_is_nonempty_str(name, "name", 
                                    "set_name")
        self._name = name

    def suppress_symbols_for_lines(self):
        """Suppress the symbol for all lines in the block. 
        Does not apply to subblocks."""
        for line in self._lines:
            self._lines[line].suppress_symbol()

    def set_symbol_for_lines(self, symbol):
        """Set the symbol for all lines in the block. Does not apply to subblocks.
        
        :param symbol: symbol to be used for all block lines. If none, 
            the symbol will be suppressed.

        :type symbol: None or str
        """
        
        if symbol is None:
            self.suppress_symbols_for_lines()
        else:
            check_value_is_nonempty_str(symbol, "symbol", 
                                        "InputFileBlock.set_symbol_for_lines")
            for line in self._lines:
                self._lines[line].set_symbol(symbol)

    def get_input_string(self):
        """Returns a string of the entire 
        input file/block."""
        return self.get_string()
    
    def write_input_to_file(self, filename):
        """Writes the input file/block to the given 
        filename.
        
        :param filename: the filename to write the input file to.
        :type filename: str
        """
        check_value_is_nonempty_str(filename, "filename", 
                                    "InputFileBlock.write")
        with open(filename, 'w') as f:
            self.write(f)

    def reset_lines(self):
        """
        Clears out the lines in the input block.
        """
        self._lines = OrderedDict()

class _BaseTypedInputFileBlock(InputFileBlock, ABC):

    @property
    @abstractmethod
    def type(self):
        """Returns the type of subblock for the input deck. 
        This is not the type of object in a code sense. However, 
        an input deck block type that can be reused in an 
        analysis input deck. Such as a dirichlet boundary condition block."""
    
    @property
    @abstractmethod
    def required_keys(self):
        """These are input block lines that must be specified in 
        order to write the input block string."""

    @property
    @abstractmethod
    def default_values(self):
        """The default values for certain input block keyword value pairs 
        for the input block type."""

    def __init__(self, *args, **kwargs):
        super().__init__(self.type, *args, **kwargs)
        self.add_lines_from_dictionary(self.default_values)

    def _prewrite_check(self):
        for key in self.required_keys:
            if key not in self._lines.keys():
                raise ValueError(f"The keyword \"{key}\" is"
                               f" required for the \"{self.type}\" "
                               " input block. Check input")
    def get_string(self):
        """Verifies all required lines and input are added and then 
        generates and returns the input block/file string."""
        self._prewrite_check()
        return super().get_string()

    def get_subblock_by_type(self, block_type):
        """This will return the subblock of a given type. It will return the 
        first one found if there are multiple subblocks of the same type.
        If the block_type is not found as a subblock, the function returns None.
        
        :param block_type: the type of the subblock to be returned. This is the 
            input file type, not a Python type.
        :type block_type: str

        :rtype: :class:`~matcal.core.input_file_writer.InputFileBlock`
        """
        desired_subblock = None
        for subblock in self.subblocks.values():
            if subblock.type == block_type:
                desired_subblock = subblock
                break
        return desired_subblock

    def get_subblocks_by_type(self, block_type):
        """This will return all subblocks of a given type. 
           If the block_type is not found as a subblock, the function returns None.
        
        :param block_type: the type of the subblock to be returned. This is the 
            input file type, not a Python type.
        :type block_type: str
        :rtype: list(:class:`~matcal.core.input_file_writer.InputFileBlock`)
       """
        desired_subblocks = []
        for subblock in self.subblocks.values():
            if subblock.type == block_type:
                desired_subblocks.append(subblock)
        return desired_subblocks

    def remove_subblocks_by_type(self, block_type):
        """This will remove all subblocks of a given type.
        If the block_type is not found as a subblock, none are removed.
        
        :param block_type: the type of the subblocks to be removed. This is the 
            input file type, not a Python type.
        :type block_type: str
        """
        block = self.get_subblock_by_type(block_type)
        while block is not None:
            self.remove_subblock(block)
            block = self.get_subblock_by_type(block_type)


class InputFileTable:

    class NonstringLabelError(RuntimeError):
        pass

    class InvalidColumnNumberError(RuntimeError):
        pass

    class InvalidNameError(RuntimeError):
        pass

    class ColumnNumberMismatchError(RuntimeError):
        pass

    class ColumnLengthMismatchError(RuntimeError):
        pass

    def __init__(self, label, n_col, name=None, begin_end_values=True):
        """
        Stores a table of values for input blocks/files.

        :param label: Similar to the title for an input block. 
            This is a label that can be printed in the input string and used 
            for identification if a name is not provided.
        :type label: str

        :param n_col: the number of columns in the table.
        :type n_col: int

        :param name: an optional table name. If provided, this can be printed 
            in the input string and is used as the table identifier when 
            interacting with the input block/file.
        :type name: str

        :param begin_end_values: add "Begin Values" and "End Values" as the first and 
            last line to the printed table string.
        :param begin_end_values: bool
        """
        if not isinstance(label, str) or len(label) < 1:
            raise self.NonstringLabelError()
        if not isinstance(n_col, int) or n_col < 1:
            raise self.InvalidColumnNumberError()
        self._label = label
        self._print_label = False
        check_item_is_correct_type(begin_end_values, bool,
                                   "InputFileTable", begin_end_values)
        self._begin_end_values = begin_end_values
        self._n_col = n_col
        self._values = []
        for i in range(self._n_col):
            self._values.append([])

        if name is not None:
            if not isinstance(name, str):
                raise self.InvalidNameError()
            self.name = name
        else:
            self.name = self._label
        
    def get_num_col(self):
        """Returns the number of columns in the table.
        :rtype: int"""
        return (self._n_col)

    def set_values(self, *cols):
        """
        Sets the column values. This will overwrite
        preexisting data. The number of arguments passed 
        must match the number of columns specified in the table initialize.
        
        :param cols: the values to be added to the table. These must be 
            iterable items of the same length.
        :type cols: list(list(str,float))
        """
        if len(cols) != self._n_col:
            raise self.ColumnNumberMismatchError()
        for i, col in enumerate(cols):
            if len(cols[0]) != len(col):
                raise self.ColumnLengthMismatchError()
            self._values[i] = col

    def append_values(self, *vals):
        """
        Add values to the end of the columns. The number of values 
        passed must match the number of columns.

        :param vals: the values to be added to the table. 
        :type vals: list(str,float)
        """
        if len(vals) != self._n_col:
            raise self.ColumnNumberMismatchError()

        for i in range(self._n_col):
            self._values[i].append(vals[i])

    def append_lists(self, *lists):
        if len(lists) != self._n_col:
            raise self.ColumnNumberMismatchError()

        for i in range(self._n_col):
            if len(lists[0]) != len(lists[i]):
                raise self.ColumnLengthMismatchError()
            self._values[i] += lists[i]

    def set_print_label(self, print_label=True):
        """Set if the label is to be printed in the table string.
        
        :param print_label: optional flag to turn off or on label printing.
            Default is set to turn on label printing.
        :type print_label: bool
        """
        check_item_is_correct_type(print_label, bool, 
                                   "set_print_label", "print_label")
        self._print_label = print_label

    def get_string(self, indents):
        """
        Get the input table string with a given 
        number of indents.

        :param indents: number of indents to apply
        :type indents: int
        """
        n_row = len(self._values[0])
        spaces = indents * _default_indent
        sub_indent = indents
        block_string = ""
        if self._print_label:
            block_string += spaces + self._label + "\n"
            sub_indent += 1
        if self._begin_end_values:
            block_string += sub_indent*_default_indent + "Begin Values\n"
            sub_indent += 1

        for i in range(n_row):
            line = sub_indent*_default_indent
            for j in range(self._n_col):
                line += str(self._values[j][i])
                if j + 1 != self._n_col:
                    line += " "

            block_string += line + "\n"
        if self._begin_end_values:
            sub_indent -= 1
            block_string += sub_indent*_default_indent + "End Values\n"

        return block_string

    def write(self, f, indents):
        """Write the input table to a file with a 
        given number of indents. 

        :param f: the file object to write to.
        :type f: TextIO

        :param indents: the number of indents to apply to the table.
        :type indents: int
        """
        lines = self.get_string(indents)
        f.write(lines)
   