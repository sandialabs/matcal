
import sys

from matcal.core.object_factory import BasicIdentifier
from matcal.core.utilities import (check_value_is_nonempty_str)


def raise_error_if_no_pyprepro_path_added():
        err_str = (f"The pyprepro path fetching function has not been registered in " +
        f"the \"MatCalPypreproPathIdentifier\". "+
        "Import the identifier from "+
        "\"matcal.dakota.file_modifications\" and set  " +
        f"the appropriate function that returns the pyprepro python path when called. "+
        "Set the default using \"MatCalPypreproPathIdentifier.set_default(function_name)\".")
        raise RuntimeError(err_str)


MatCalPypreproPathIdentifier = BasicIdentifier(raise_error_if_no_pyprepro_path_added)


def add_pyprepro_to_path():
    pyprepro_path = MatCalPypreproPathIdentifier.identify()
    if pyprepro_path not in sys.path:
        sys.path.append(pyprepro_path)


class pyprepro_delimiters:
    inline = "{ }"
    code_block = "{% %}"
    code = "%"
    

def _check_pyprepro_delimiter_inputs(inline, code_block, code):
    check_value_is_nonempty_str(inline, "inline", 
                                "set_pyprepro_delimiters")
    check_value_is_nonempty_str(code_block, "code_block", 
                              "set_pyprepro_delimiters")
    check_value_is_nonempty_str(code, "code", 
                                "set_pyprepro_delimiters")
    if inline==code_block or code in inline.split(" ") or code in code_block.split(" "):
        raise ValueError("The delimiters for pyprepro must be unique for each type.")
    if inline.count(" ") > 1 or code_block.count(" ") > 1:
        raise ValueError("The inline and code_block delimiters for pyprepro "
                         "must not contain more than one space.")         


def set_pyprepro_delimiters(inline="{ }", code_block="{% %}", 
                         code="%"):
    """
    Set the delimiters for pyprepro templates.

    :param inline: The delimiters for inline code.
        It should be a single string 
        of the form where the start and end delimiter strings are separated
        by a space.
    :type inline: str
    :param code_block: The delimiters for code blocks. It should be a single string 
        of the form where the start and end delimiter strings are separated
        by a space.
    :type code_block: str
    :param code: The delimiter for code.
    :type code: str
    :raises ValueError: If the delimiters are not unique or if inline or 
        code_block contain more than one space.
    """
    _check_pyprepro_delimiter_inputs(inline, code_block, code)
    pyprepro_delimiters.inline = inline
    pyprepro_delimiters.code_block = code_block
    pyprepro_delimiters.code = code


def pyprepro_processor(contents, replacements):
    add_pyprepro_to_path()
    from pyprepro import pyprepro
    return pyprepro(contents, env=replacements, 
                    code = pyprepro_delimiters.code,
                    code_block = pyprepro_delimiters.code_block,
                    inline=pyprepro_delimiters.inline)