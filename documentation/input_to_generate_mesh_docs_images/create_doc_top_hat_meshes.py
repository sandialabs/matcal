from matcal.sierra.models.geometry import TopHatShearGeometry
from matcal.test.sandia.sierra_sm_models_for_tests import TopHatShearModelForTests
import os
import shutil


geo_params = TopHatShearModelForTests.geo_params

def create_folder(name):
    if os.path.exists(name):
        shutil.rmtree(name)
    os.mkdir(name)
    

def create_meshes(name, params):

    create_folder(name)
    model = TopHatShearGeometry(f"{name}.g", TopHatShearGeometry.Parameters(**params))
    model.create_mesh(name)

geo_params["element_size"] = 0.0254*0.015
geo_params["numsplits"] = 0
create_meshes("top_hat_no_numsplits",geo_params)

geo_params["element_size"] = 0.0254*0.005
geo_params["numsplits"] = 1
create_meshes("top_hat_one_numsplit", geo_params)

geo_params["element_size"] = 0.0254*0.003
geo_params["numsplits"] = 2
create_meshes("top_hat_two_numsplits", geo_params)

