# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

copyright = ('2025 National Technology & Engineering Solutions of Sandia, '+
    'LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government '+
    'retains certain rights in this software')


import sys
import os
import inspect
import shutil
from glob import glob
from sphinx_gallery.sorting import FileNameSortKey


sys.path.insert(0, os.path.abspath("../../"))
from matcal.version import __version__

site_docs_path = os.path.join("..", "..", "..", "site_matcal", "documentation")
rsts_to_include = []
if os.path.exists(site_docs_path):
    for filename in glob(os.path.join(site_docs_path, "*.rst")):
        shutil.copyfile(filename, os.path.join(os.getcwd(), os.path.basename(filename)))
        rsts_to_include.append(os.path.basename(filename))
with open("site_includes.rst", 'w') as f:
    for rst in rsts_to_include:
        f.write(f'\t{rst}\n')
    

# -- Project information -----------------------------------------------------

project = 'MatCal Users Guide'
author = 'Kyle Karlson, Matthew Kury, Reese Jones'

# The full version, including alpha/beta/rc tags
release = __version__
nitpicky = True
nitpick_ignore = [('py:class', "matcal.core.pruner.DirectoryPrunerBase"),
                ('py:class', "matcal.core.objective.MetricFunctionBase"),
                ('py:class', "TextIOWrapper"), 
                ("py:class", "matcal.core.input_file_writer.InputFileTable"),
                ("py:meth", "os.path.split")]
nitpick_ignore_regex = [('py:exc', ".*Err.*"), 
                        ('py:class', '.*HWD.*SurfaceExtractor'), 
                        ('py:class', 'numpy.*typ.*')]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_gallery.gen_gallery',
    'sphinx_copybutton',
    'sphinxcontrib.bibtex',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.imgmath',
    'sphinx.ext.autosectionlabel']

autosummary_generate = True
numfig = True
# Document both class doc strings and the __init__ function doc strings
autoclass_content = 'both'

# list of members to document.
autodoc_default_options = {
    'members': True,
    'noindex': True,
    'member-order': 'bysource',
    'exclude-members': 'Exception',
    'inherited-members': True,
    'private-members': False
}

sphinx_gallery_conf = {
    'examples_dirs': ['../introduction_examples',
                      '../advanced_examples',
                      '../matcal_model_v_and_v', 
                      '../full_field_verification_examples',
                      '../full_field_study_verification_examples',  
                      ],
    # '../solid_mechanics_steel'],   # path to your example scripts

    'gallery_dirs': ['introduction_examples',
                     'advanced_examples',
                     'matcal_model_v_and_v', 
                     'full_field_verification_examples',
                     'full_field_study_verification_examples', 
                     ],
    # 'solid_mechanics_steel'],  # path to where to save gallery generated out
    'within_subsection_order': FileNameSortKey,
    'nested_sections': False,
#    'filename_pattern': '/plot_6061',
}


# way to exclude exceptions
def autodoc_skip_filter(app, what, name, obj, skip, options):
    exclude = skip
    filters = []
    filters += [autodoc_skip_exceptions, autodoc_skip_private_functions_classes,
                 autodoc_skip_nonmatcal_objects, 
                 autodoc_skip_explicit_class_patterns]
    for current_filter in filters:
        exclude = exclude or current_filter(app, what, name, obj, skip, options)
        if exclude:
            break
    autodoc_probe(app, what, name, obj, skip, options, exclude)
    return exclude


def autodoc_skip_exceptions(app, what, name, obj, skip, options):
    exclusions = ('exception')

    exclude = what in exclusions
    try:
        exclude = exclude or issubclass(obj, Exception)
        if exclude:
            print(f"Skipping: {name} --- Exception")
    except:
        pass

    return skip or exclude


def autodoc_skip_private_functions_classes(app, what, name, obj, skip, options):
    exclude = False
    if (what == "function" or what == "class") and name[0] == "_":
        exclude = True
        print(f"Skipping: {name} --- Private Function")
    return exclude or skip


def _is_function_like(what):
    return what not in ["module", "exception"]


def autodoc_skip_nonmatcal_objects(app, what, name, obj, skip, options):
    exclude = False
    if _is_function_like(what):
        try:
            source_file = inspect.getfile(obj)
        except TypeError:
            source_file = ""
        source_file = source_file.split("/")
        # property methods in classes are wrapped as a seperate class known as a property
        # This means that when using a file path to see if the object has a matcal path
        # we need to exclude the properties
        if ('matcal' not in source_file) and not isinstance(obj, property):
            exclude = True
            print(f"Skipping: {name} --- Not MatCal")
    return skip or exclude


def _is_class(what):
    return what == "class"


def autodoc_skip_explicit_class_patterns(app, what, name, obj, skip, options):
    skip_list = ["CollectionBase", "Mock", "mock", "Base"]
    exclude = False
    if _is_class(what):
        for skip_pattern in skip_list:
            if (skip_pattern in name and "StudyBase" not in name and
                "DataConditionerBase" not in name):
                exclude = True
                #print(f"Skipping Class: {name} --- Fit pattern: {skip_pattern}")
                break
    return exclude or skip


def autodoc_probe(app, what, name, obj, skip, options, exclude):
    is_on = True
    if is_on:
        if _is_function_like(what):
            pass
            #print(what)
            #print(name)
            #print(obj)
            #print(f"skip: {skip}")
            #print(f"exclude: {exclude}")
            #print(options)
            #print("------")
    return skip


def setup(app):
    # it appears you can only add one
    app.connect('autodoc-skip-member', autodoc_skip_filter)


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', ]

# The master toctree document.
#master_doc = 'matcal_user_documentation'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'full_field_verification_examples/index.rst', 
                    'matcal_model_v_and_v/index.rst', 
                    'full_field_study_verification_examples/index.rst',
                    'advanced_examples/user_model_studies/index.rst',
                    'introduction_examples/sierra_material_point_examples/index.rst']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a filenamed "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']  # allows any width to text instead of default limit


latex_elements = {
    'preamble': "\\usepackage{amsmath}\n"
}
imgmath_image_format = 'svg'

try:
    imgmath_latex_preamble  # check whether this is already defined
except NameError:
    imgmath_latex_preamble = ""

with open("latex_macros.sty", "r") as f:
    for macro in f.readlines():
        # used when building latex and pdf versions
        latex_elements['preamble'] += macro
        # used when building html version
        imgmath_latex_preamble += macro

# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.


intersphinx_mapping = {"numpy":('https://numpy.org/doc/stable/', None), 
        "matplotlib":('https://matplotlib.org/stable/', None)}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


##Bibtext files for references###
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'plain'

extra_foot_txt = "<img src=\"images/snl.jpg\" alt=\"Sandia National Laboratories\" style=\"height:40px\">"



html_theme_options = {
    'display_version': True,
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    "extra_footer": extra_foot_txt,
}
