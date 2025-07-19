import numpy as np
import matplotlib.pyplot as plt
from matcal import *
from site_matcal.sandia.computing_platforms import is_sandia_cluster

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)
figsize = (4,3)


fig = plt.figure(figsize=figsize)


tension_data = BatchDataImporter("ductile_failure_ASTME8_304L_data/*.dat",
                                    file_type="csv",
                                    fixed_states={"displacement_rate":2e-4,
                                                  "temperature":530}).batch

tension_data = scale_data_collection(tension_data, "engineering_stress", 1000)

tension_data.plot("engineering_strain", "engineering_stress",
                    figure=fig, color="#bdbdbd", labels="experiments", show=False)

def get_common_var_results(filenames_reg_exp):
    common_time = np.linspace(0, 3132, 300)
    data_combined_load = np.array([])
    data_combined_disp = np.array([])

    filenames = glob(filenames_reg_exp)
    max_times = []
    for filename, count in zip(filenames, range(len(filenames))):
        A = FileData(filename)
        max_times.append(np.max(A["time"]))
        data_interp_load = np.interp(common_time, A["time"], A["engineering_strain"])
        data_interp_disp = np.interp(common_time, A["time"], A["engineering_stress"])

        if count == 0:
            data_combined_load = data_interp_load
            data_combined_disp = data_interp_disp
        else:
            data_combined_load = np.vstack((data_combined_load, data_interp_load))
            data_combined_disp = np.vstack((data_combined_disp, data_interp_disp))

    print(np.max(max_times))
    return common_time, data_combined_load, data_combined_disp



from glob import glob

filenames = glob("UQ_sampling_study/matcal_workdir.*/ASTME8_tension_model/batch_fixed_state/results.csv")
label="simulation with uncertainty"

com_tim, com_strain, com_stress = get_common_var_results("UQ_sampling_study/matcal_workdir.*/ASTME8_tension_model/batch_fixed_state/results.csv")

plt.plot(np.percentile(com_strain, 5, axis=0), np.percentile(com_stress, 5, axis=0), color='tab:blue', linestyle='--', linewidth=2, label=r"calibrated sim $5^{th}$ percentile")
plt.plot(np.percentile(com_strain, 50, axis=0), np.percentile(com_stress, 50, axis=0), color='tab:blue',linestyle='-', linewidth=2, label=r"calibrated sim $50^{th}$ percentile")
plt.plot(np.percentile(com_strain, 95, axis=0), np.percentile(com_stress, 95, axis=0), color='tab:blue',linestyle='-.', linewidth=2, label=r"calibrated sim $95^{th}$ percentile")
plt.xlabel("engineering strain")
plt.ylabel("engineering stress (psi)")

plt.legend()
plt.show()


