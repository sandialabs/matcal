from matcal.core.object_factory import IdentifierByTestFunction


def throw_no_exo_error():
    raise ImportError("Setup your exodus python paths for MatCal. "+
                      "See documentation for MatCal exodus module.")


def import_exodus_helper():
    try:
        import exodus_helper as exo
        return exo
    except:
        throw_no_exo_error
        

MatcalExodusImporterIdentifier = IdentifierByTestFunction(import_exodus_helper)


def create_exodus_class_instance(*args, **kwargs):
    exo_importer = MatcalExodusImporterIdentifier.identify()
    exo = exo_importer()
    try:
        return exo.exodus(*args, **kwargs)
    except AttributeError:
        return exo.Exodus(*args, **kwargs)
