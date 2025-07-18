from matcal.sierra.models.geometry import RoundUniaxialTensionGeometry, RectangularUniaxialTensionGeometry
import os
import shutil


thickness = 0.125 
rect_params = {"gauge_width": 0.5, 
"grip_width":2.0, 
"extensometer_length":1.0,
"gauge_length":1.25, 
"total_length":3.5, 
"fillet_radius":0.5, 
"taper":0.001, 
"necking_region":0.375, 
"mesh_method":1, 
"grip_contact_length":0.5, 
"element_size":thickness/2,
"thickness":thickness}

gauge_radius=0.5/2
round_params = {"gauge_radius": gauge_radius,
"grip_radius":2.0/2, 
"extensometer_length":1.0,
"gauge_length":1.25,  
"total_length":3.5,  
"fillet_radius":0.5, 
"taper":0.001,   
"necking_region":0.375,  
"mesh_method":1,
"grip_contact_length":0.5, 
"element_size":gauge_radius/4}

def create_folder(name):
    if os.path.exists(name):
        shutil.rmtree(name)
    os.mkdir(name)
    

def create_meshes(name, round_params, rect_params):
    round_folder = f"round_{name}"
    rect_folder = f"rect_{name}"

    create_folder(round_folder)
    create_folder(rect_folder)
 
    round_model = RoundUniaxialTensionGeometry(f"round_{name}.g", RoundUniaxialTensionGeometry.Parameters(**round_params))
    rect_model = RectangularUniaxialTensionGeometry(f"rect_{name}.g", RectangularUniaxialTensionGeometry.Parameters(**rect_params))

    round_model.create_mesh(round_folder)
    rect_model.create_mesh(rect_folder)

def create_meshes_for_each_method(name, round_params, rect_params):

    round_params["mesh_method"] = 1
    round_params["element_size"] = gauge_radius/4
    rect_params["mesh_method"] = 1
    rect_params["element_size"] = thickness/2
    create_meshes(f"{name}_method_1", round_params, rect_params)
    
    round_params["mesh_method"] = 2
    round_params["element_size"] = gauge_radius/8
    rect_params["mesh_method"] = 2
    rect_params["element_size"] = thickness/4
    create_meshes(f"{name}_method_2", round_params, rect_params)
    
    round_params["mesh_method"] = 3
    round_params["element_size"] = gauge_radius/16
    rect_params["mesh_method"] = 3
    rect_params["element_size"] = thickness/6
    create_meshes(f"{name}_method_3", round_params, rect_params)
    
    round_params["mesh_method"] = 4
    round_params["element_size"] = gauge_radius/24
    rect_params["mesh_method"] = 4
    rect_params["element_size"] = thickness/12
    create_meshes(f"{name}_method_4", round_params, rect_params)
    
    
    round_params["mesh_method"] = 5
    round_params["element_size"] = gauge_radius/30
    rect_params["mesh_method"] = 5
    rect_params["element_size"] = thickness/24
    create_meshes(f"{name}_method_5", round_params, rect_params)

create_meshes_for_each_method("wide", round_params, rect_params)

round_params["grip_radius"]=0.5
rect_params["grip_width"]=1.0

create_meshes_for_each_method("standard", round_params, rect_params)


