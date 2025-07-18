from matcal import *

from matcal.sierra.tests.sierra_sm_models_for_tests import UniaxialLoadingMaterialPointModelForTests

model_setup_class = UniaxialLoadingMaterialPointModelForTests()
model = model_setup_class.init_model(plasticity=True, coupled=False)
data_collections = model_setup_class.boundary_condition_data_sets
pc = model_setup_class.get_material_parameter_collection()

neg_bc_dc = scale_data_collection(data_collections[0], "engineering_stress", -1)
neg_bc_dc = scale_data_collection(neg_bc_dc, "engineering_strain", -1)
model.add_boundary_condition_data(neg_bc_dc)

for state in data_collections[0].states.values():
    model.run(state, pc)
