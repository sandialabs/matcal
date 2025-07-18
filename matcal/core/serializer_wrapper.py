from builtins import ImportError
import numpy as np

try:
    import pickle as pickle_serializer
except ImportError as e:
    raise e("MatCal requires pickle")

try:
    import json as json_serializer
except ImportError as e:
    raise e("MatCal requires json")

try:
    import joblib as joblib_serializer
except ImportError as e:
    raise e("MatCal requires joblib")

serial_file_ext = '.serialized'

def _get_extension(filename):
    extension = filename.split('.')[-1]
    return extension

def _dump_json(filename, object_to_be_saved):
    with open(filename, 'w') as f:
        json_serializer.dump(object_to_be_saved, f)

def _load_json(filename):
    info = None
    with open(filename, 'r') as f:
        info = json_serializer.load(f)
    return info

def _dump_joblib(filename, object_to_be_saved):
    joblib_serializer.dump(object_to_be_saved, filename)

def _load_joblib(filename):
    return joblib_serializer.load(filename)

def _dump_pickle(filename, object_to_be_saved):
    with open(filename, 'wb') as f:
        pickle_serializer.dump(object_to_be_saved, f)

def _load_pickle(filename):
    info = None
    with open(filename, 'rb') as f:
        info = pickle_serializer.load(f)
    return info


serializer_lookup = {'serialized':(_dump_json, _load_json), 'json':(_dump_json, _load_json), 'joblib':(_dump_joblib, _load_joblib),
                     'pcl':(_dump_pickle, _load_pickle), 'pickle':(_dump_pickle, _load_pickle)}

def matcal_save(filename, object_to_dump):
    """
    Store an object generated in MatCal to a file. It is recommended that filenames of .joblib or .json are used.
    Note that using different filename extensions will call different savers which allow for storing different levels of complex objects. 

    :param filename: Name of file to store object. Must have an extension of (.serialized, .json, .joblib, .pcl, .pickle)
    :type filename: str

    :param object_to_dump: The object to be store. 
    """
    extension = _get_extension(filename)
    dumper = serializer_lookup[extension][0]
    dumper(filename, object_to_dump)

def matcal_load(filename):
    """
    Load a previously stored object. 

    :param filename: Name of file to load object from. Must have an extension of (.serialized, .json, .joblib, .pcl, .pickle)
    :type filename: str
    """
    extension = _get_extension(filename)
    loader = serializer_lookup[extension][1]
    return loader(filename)

def _format_serial(potential_array):
    if not isinstance(potential_array, np.ndarray):
        return potential_array
    else:
        return potential_array.tolist()
