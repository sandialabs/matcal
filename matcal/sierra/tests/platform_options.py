from matcal.core.computing_platforms import HPCComputingPlatform
from matcal.core.object_factory import IdentifierByTestFunction
from matcal.sierra.models import (UniaxialLoadingMaterialPointModel, UserDefinedSierraModel, 
VFMUniaxialTensionConnectedHexModel, VFMUniaxialTensionHexModel)


def raise_error_if_no_platform_test_identifier_registered(*args, **kwargs):
    err_str = ("No \'MatCalTestPlatformOptionsFunctionIdentifier\' has been set. "+
        "Some platform specific testing maybe missed. Add  "+
        "\'MatCalTestPlatformOptionsFunctionIdentifier.set_default(set_platform_options)\' "+
        "to matcal.sierra.tests.platform_options or to an __init__.py in you site directory. "+
        "See the example function in matcal.sierra.tests.platform_options to see "
        "what 'set_platform_options' can/should do. "+
        "This function is intended to set different options on the model by platform.")
    raise RuntimeError(err_str)


MatCalTestPlatformOptionsFunctionIdentifier = IdentifierByTestFunction(
    raise_error_if_no_platform_test_identifier_registered)


def get_cluster_platform():
    """"""
    ####    
    # get_cluster_platform will need to be written specific to your hardware and software
    # environment. It should return a matcal.core.computing.HPCComputingPlatform object
    ####

VALID_QUEUE_ID = "QUEUE_ID"

def set_platform_options(model, remote=True):
    is_cluster = isinstance(model.computer, HPCComputingPlatform)
    if is_cluster and remote:
        model.run_in_queue(VALID_QUEUE_ID, 30.0/60)
    elif is_cluster:

        cluster = get_cluster_platform()
        if not isinstance(model, (UniaxialLoadingMaterialPointModel, UserDefinedSierraModel, 
                                  VFMUniaxialTensionConnectedHexModel, VFMUniaxialTensionHexModel)):
            model.set_number_of_cores(int(cluster.get_processors_per_node()/2))
    else:
        model.set_number_of_cores(16)

    return model