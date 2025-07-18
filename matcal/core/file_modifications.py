from matcal.core.object_factory import IdentifierByTestFunction
from matcal.core.utilities import is_text_file, check_value_is_nonempty_str


class jinja2_delimiters:
    variable = "{{ }}"
    block = "{% %}"
    comment = "{# #}"
    line_statement_prefix = None
    line_comment_prefix = None


def _check_jinja_delimiters_input(variable, block, comment, line_statement_prefix, 
                                  line_comment_prefix):
    check_value_is_nonempty_str(variable, "variable", 
                                "set_jinja_delimiters")
    check_value_is_nonempty_str(block, "block", 
                              "set_jinja_delimiters")
    check_value_is_nonempty_str(comment, "comment", 
                                "set_jinja_delimiters")
    if line_statement_prefix is not None:
        check_value_is_nonempty_str(line_statement_prefix, "line_statement_prefix", 
                                "set_jinja_delimiters")
    if line_comment_prefix is not None:
        check_value_is_nonempty_str(line_comment_prefix, "line_comment_prefix", 
                                "set_jinja_delimiters")


def set_jinja_delimiters(variable="{{ }}", block="{% %}", 
                         comment="{# #}", line_statement_prefix=None, 
                         line_comment_prefix=None):
    """
    Set the delimiters for Jinja2 templates. 

    :param variable: The delimiter for variables. It should be a single string 
        of the form where the start and end delimiter strings are separated
        by a space.
    :type variable: str
    :param block: The delimiter for blocks.It should be a single string 
        of the form where the start and end delimiter strings are separated
        by a space.
    :type block: str
    :param comment: The delimiter for comments. It should be a single string 
        of the form where the start and end delimiter strings are separated
        by a space.
    :type comment: str
    :param line_statement_prefix: The prefix for line statements.
    :type line_statement_prefix: str, optional
    :param line_comment_prefix: The prefix for line comments.
    :type line_comment_prefix: str, optional
    """
    _check_jinja_delimiters_input(variable, block, comment, line_statement_prefix, 
                                  line_comment_prefix)
    jinja2_delimiters.variable = variable
    jinja2_delimiters.block = block
    jinja2_delimiters.comment = comment
    jinja2_delimiters.line_statement_prefix = line_statement_prefix
    jinja2_delimiters.line_comment_prefix = line_comment_prefix


def _get_start_delimiter(delim):
    return delim.split(" ")[0]


def _get_end_delimiter(delim):
    return delim.split(" ")[1]


def jinja2_processor(contents, replacements):
    from jinja2 import Template, Environment
    env = Environment(
    block_start_string=_get_start_delimiter(jinja2_delimiters.block),
    block_end_string=_get_end_delimiter(jinja2_delimiters.block),
    variable_start_string=_get_start_delimiter(jinja2_delimiters.variable), 
    variable_end_string=_get_end_delimiter(jinja2_delimiters.variable),
    comment_start_string=_get_start_delimiter(jinja2_delimiters.comment),
    comment_end_string=_get_end_delimiter(jinja2_delimiters.comment),
    line_comment_prefix=jinja2_delimiters.line_comment_prefix,
    line_statement_prefix=jinja2_delimiters.line_statement_prefix,   
    )
    template = env.from_string(contents)
    modified_content = template.render(replacements)
    return modified_content


MatCalTemplateFileProcessorIdentifier = IdentifierByTestFunction(jinja2_processor)


def use_jinja_preprocessor():
    """
    Forces the preprocessor for templated files to return to the 
    default processor, jinja2.
    """
    MatCalTemplateFileProcessorIdentifier._registry = {}


def process_template_file(file_path, replacements):
    if is_text_file(file_path):
        with open(file_path, 'r') as file:
            contents = file.read()
    
        template_processor = MatCalTemplateFileProcessorIdentifier.identify()
        modified_contents = template_processor(contents, replacements)
        with open(file_path, 'w') as file:
            file.write(modified_contents)


