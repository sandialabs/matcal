"""
This module contains the Material object meant to define 
a SIERRA material file for use with a MatCal generated SIERRA model.
"""

import os

class Material(object):
    """
    MatCal object for creating a material to be calibrated. It requires the name of the material
    model, the material model input file, and the material model type.
    """

    class InvalidNameError(RuntimeError):
        pass

    def __init__(self, name, material_filename, material_model_type):
        """
        :param name: The name of the material model. This is the name that will be used in MatCal generated input decks.
        :type name: str

        :param material_filename: The filename that has the parameterized input deck syntax for the material model. The
            file should have both state and design parameters if need.
        :type material_filename: str

        :param material_model_type: The type of the material model. The type is specific to the code being used to simulate
            the model. For example, the "Johnson Cook" material model type is "johnson_cook" in SierraSM.
        :type material_model_type: str
        """
        if not name or not isinstance(name, str):
            raise self.InvalidNameError()

        if not os.path.isfile(material_filename):
            raise FileNotFoundError("The material file {} does not exist".format(material_filename))

        self._name = name
        self._filename = os.path.abspath(material_filename)
        self._model = material_model_type

    @property
    def name(self):
        """
        :return: the material name
        :rtype: str
        """
        return self._name

    @property
    def filename(self):
        """
        :return: the material model input filename
        :rtype: str
        """
        return self._filename

    @property
    def model(self):
        """
        :return: the material model name
        :rtype: str
        """
        return self._model
