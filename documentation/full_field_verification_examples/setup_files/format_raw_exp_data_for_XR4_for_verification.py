
exp_file_data = FileData("XR4-Data_0254_3.51900e+00.csv")
exp_file_data = exp_file_data.remove_field("Z0")
exp_file_data = exp_file_data.remove_field("sigma")
exp_file_data = exp_file_data.remove_field("U")
exp_file_data = exp_file_data.remove_field("W")
exp_file_data = exp_file_data.remove_field("IR_temperature_0")
exp_file_data = exp_file_data.remove_field("IR_temperature_1")
exp_file_data["X0"] = exp_file_data["X0"]/1000
exp_file_data["Y0"] = exp_file_data["Y0"]/1000

x_specimen_mesh = extract_exodus_mesh("x_specimen_mesh.g")
x= x_specimen_mesh.spatial_coords[:,0]
y= x_specimen_mesh.spatial_coords[:,1]

exp_file_data = exp_file_data[((exp_file_data["X0"] > np.min(x))& (exp_file_data["X0"] < np.max(x))&
                               (exp_file_data["Y0"] > np.min(y)) & (exp_file_data["Y0"] < np.max(y)))] 
print(exp_file_data.field_names)
np.savetxt("x_specimen_XR4_peak_load_data.csv", exp_file_data, header="X0, Y0, V", comments="", delimiter=",")
