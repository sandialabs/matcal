from matcal.core.constants import TIME_KEY
from matcal.full_field.data_exporter import matcal_field_data_exporter_identifier

from matcal.exodus.mesh_modifications import copy_mesh_and_store_data


def exodus_field_data_exporter_function(target_filename, data_to_export, 
                 fields, reference_source_mesh, independent_field, *args, **kwargs):

    if TIME_KEY not in data_to_export.field_names:
        data_to_export = data_to_export.add_field(TIME_KEY, 
                                                  data_to_export[independent_field])
    elif independent_field != TIME_KEY:
        data_to_export[TIME_KEY] = data_to_export[independent_field]

    return copy_mesh_and_store_data(reference_source_mesh, target_filename, 
                                    data_to_export, fields)


matcal_field_data_exporter_identifier.register("e", exodus_field_data_exporter_function)
matcal_field_data_exporter_identifier.register("exo", exodus_field_data_exporter_function)
matcal_field_data_exporter_identifier.register("g", exodus_field_data_exporter_function)