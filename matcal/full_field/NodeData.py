import numpy as np
from collections import OrderedDict

from matcal.core.data import convert_dictionary_to_data

class NodeData:

    class MissingFieldError(RuntimeError):
        def __init__(self):
            super().__init__("Method 'get_node_data' requires at least one field name. 'get_node_data' was passed no "
                             "field names")

    def __init__(self):
        self._data = OrderedDict()

    def add_node_data(self, fieldname, data_array):
        self._data[fieldname] = data_array

    def get_node_data(self, *fieldnames):
        self._check_fieldnames(*fieldnames)
        number_of_fields, number_of_nodes = self._get_node_array_size(fieldnames)
        data_array = self._create_node_data_array(fieldnames, number_of_fields, number_of_nodes)
        return data_array

    def get_full_data(self):
        return convert_dictionary_to_data(self._data)

    def _check_fieldnames(self, *fieldnames):
        if len(fieldnames) == 0:
            raise self.MissingFieldError()

    def _get_node_array_size(self, fieldnames):
        number_of_nodes = len(self._data[fieldnames[0]])
        number_of_fields = len(fieldnames)
        return number_of_fields, number_of_nodes

    def _create_node_data_array(self, fieldnames, number_of_fields, number_of_nodes):
        data_array = np.zeros([number_of_fields, number_of_nodes])
        for index, name in enumerate(fieldnames):
            data_array[index, :] = self._data[name]
        return data_array
