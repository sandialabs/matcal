from matcal.core.data import convert_data_to_dictionary
from matcal.core.serializer_wrapper import json_serializer, _format_serial 
from matcal.core.object_factory import BasicIdentifier

from matcal.full_field.data import FieldData


def export_full_field_data_to_json(target_filename:str, data_to_export:FieldData, 
                                   *args, **kwargs):
    serial_data_fields = serialize_full_field_data(data_to_export)
    with open(target_filename, 'w') as f:
        json_serializer.dump(serial_data_fields, f)


def serialize_full_field_data(data_to_export):
    serial_mesh_skeleton = data_to_export.skeleton.serialize()
    serial_data_fields = convert_data_to_dictionary(data_to_export)
    for key, value in serial_data_fields.items():
        serial_data_fields[key] = _format_serial(value)
    serial_data_fields.update(serial_mesh_skeleton)
    return serial_data_fields


class FieldDataExporterSelector(BasicIdentifier):
    pass


MatCalFieldDataExporterIdentifier = FieldDataExporterSelector()
MatCalFieldDataExporterIdentifier.register('json', 
                                          export_full_field_data_to_json)