from matcal.sierra.models.geometry import SolidBarTorsionGeometry
import os
import shutil

gauge_radius=0.125
round_params =    {"extensometer_length": ((0.277-0.277*0.5)/2+0.277*0.5),
               "gauge_length": 0.227,
               "gauge_radius": 0.125,
               "grip_radius": 0.25,
               "total_length": 1.25,
               "fillet_radius": 0.350,
               "taper": 0.0001,
               "necking_region":0.5,
               "element_size": 0.125/6,
               "mesh_method": 1,
               "grip_contact_length":0.125}


def create_folder(name):
    if os.path.exists(name):
        shutil.rmtree(name)
    os.mkdir(name)
    

def create_meshes(name, round_params):
    round_folder = f"round_{name}"

    create_folder(round_folder)
 
    round_model = SolidBarTorsionGeometry(f"round_{name}.g", SolidBarTorsionGeometry.Parameters(**round_params))

    round_model.create_mesh(round_folder)

def create_meshes_for_each_method(name, round_params):

    round_params["mesh_method"] = 1
    round_params["element_size"] = gauge_radius/8
    create_meshes(f"{name}_method_1", round_params)
    
create_meshes_for_each_method("standard_torsion", round_params)


