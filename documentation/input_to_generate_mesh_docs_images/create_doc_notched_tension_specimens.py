from matcal.sierra.models.geometry import RoundNotchedTensionGeometry
import os
import shutil
import glob

gauge_radius = 0.25
notch_gauge_radius = 0.125
round_params = {"gauge_radius": gauge_radius,
"grip_radius":1.0, 
"extensometer_length":1.0,
"gauge_length":1.25,  
"total_length":3.5,  
"fillet_radius":0.5, 
"necking_region":0.375,  
"mesh_method":1,
"grip_contact_length":0.5,
"notch_radius":0.5,
"notch_gauge_radius":notch_gauge_radius, 
"element_size":gauge_radius/4}

def create_folder(name):
    if os.path.exists(name):
        shutil.rmtree(name)
    os.mkdir(name)
    
def create_meshes(name, round_params):
    round_folder = f"round_notched_{name}"
    create_folder(round_folder)
 
    round_model = RoundNotchedTensionGeometry(f"round_notched_{name}.g", RoundNotchedTensionGeometry.Parameters(**round_params))
    round_model.create_mesh(round_folder)

def create_meshes_for_each_method(name, round_params):

    round_params["mesh_method"] = 1
    round_params["element_size"] = notch_gauge_radius/4
    create_meshes(f"{name}_method_1_ngr_over_4", round_params)
    
    round_params["mesh_method"] = 2
    round_params["element_size"] = notch_gauge_radius/4
    create_meshes(f"{name}_method_2_ngr_over_4", round_params)
    
    round_params["mesh_method"] = 3
    round_params["element_size"] = notch_gauge_radius/4
    create_meshes(f"{name}_method_3_ngr_over_4", round_params)

    round_params["mesh_method"] = 1
    round_params["element_size"] = notch_gauge_radius/8
    create_meshes(f"{name}_method_1_ngr_over_8", round_params)
    
    round_params["mesh_method"] = 2
    round_params["element_size"] = notch_gauge_radius/8
    create_meshes(f"{name}_method_2_ngr_over_8", round_params)
    
    round_params["mesh_method"] = 3
    round_params["element_size"] = notch_gauge_radius/8
    create_meshes(f"{name}_method_3_ngr_over_8", round_params)
    
    round_params["mesh_method"] = 1
    round_params["element_size"] = notch_gauge_radius/16
    create_meshes(f"{name}_method_1_ngr_over_16", round_params)
    
    round_params["mesh_method"] = 2
    round_params["element_size"] = notch_gauge_radius/16
    create_meshes(f"{name}_method_2_ngr_over_16", round_params)
    
    round_params["mesh_method"] = 3
    round_params["element_size"] = notch_gauge_radius/16
    create_meshes(f"{name}_method_3_ngr_over_16", round_params)

    round_params["mesh_method"] = 4
    round_params["element_size"] = notch_gauge_radius/16
    create_meshes(f"{name}_method_4_ngr_over_16", round_params)

    round_params["mesh_method"] = 1
    round_params["element_size"] = notch_gauge_radius/32
    create_meshes(f"{name}_method_1_ngr_over_32", round_params)
    
    round_params["mesh_method"] = 2
    round_params["element_size"] = notch_gauge_radius/32
    create_meshes(f"{name}_method_2_ngr_over_32", round_params)
    
    round_params["mesh_method"] = 3
    round_params["element_size"] = notch_gauge_radius/32
    create_meshes(f"{name}_method_3_ngr_over_32", round_params)

    round_params["mesh_method"] = 4
    round_params["element_size"] = notch_gauge_radius/32
    create_meshes(f"{name}_method_4_ngr_over_32", round_params)
    
    round_params["mesh_method"] = 5
    round_params["element_size"] = notch_gauge_radius/32
    create_meshes(f"{name}_method_5_ngr_over_32", round_params)

round_params["grip_radius"]=0.5
create_meshes_for_each_method("standard", round_params)

rotate_command_1 = "rotate Volume all angle 150 about Y include_merged\n"
rotate_command_2 = "rotate Volume all angle 30 about X include_merged\n"
init_dir = os.getcwd()
for filename in sorted(glob.glob("*_ngr*/*.jou")):

    print(f"Making image for {filename}")
    with open(filename, "r") as f:
        lines = f.readlines()

    mesh_method_string = filename.split("/")[0].split("standard_")[-1]
    screen_shot_cmd = f'hardcopy "isomesh_view_{mesh_method_string}.png" png\n'
    lines = lines[:-2]
    lines.append("export mesh 'mesh.g' overwrite\n")
    lines.append(rotate_command_1)
    lines.append(rotate_command_2)
    lines.append(screen_shot_cmd)
    lines.append("exit\n")

    with open(filename+".tmp", "w") as f:
       for line in lines:
          f.write(line)

    shutil.move(filename+".tmp", filename)

    os.chdir(os.path.dirname(filename))
    os.system(f"cubit -nojournal {filename.split('/')[-1]}")
    os.chdir(init_dir)



#round_params["grip_radius"]=1.0
#create_meshes_for_each_method("wide_grip", round_params)

#round_params["grip_radius"]=0.5
#round_params["notch_radius"] = 0.078
#create_meshes_for_each_method("small_notch", round_params)


#round_params["fillet_radius"] = 0.1
#round_params["grip_radius"]=1.0
#create_meshes_for_each_method("small_notch_wide_grip_small_fillet_radius", round_params)

