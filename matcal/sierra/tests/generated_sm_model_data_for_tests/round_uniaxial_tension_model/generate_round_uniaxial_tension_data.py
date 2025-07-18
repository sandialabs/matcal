from matcal import *

from matcal.sierra.tests.sierra_sm_models_for_tests import RoundUniaxialTensionModelForTests

model_setup_class = RoundUniaxialTensionModelForTests()
model = model_setup_class.init_model(plasticity=True, coupled=True)
data_collections = model_setup_class.boundary_condition_data_sets
mat_props = model_setup_class.get_material_properties()
pc = model_setup_class.get_material_parameter_collection()
model.add_boundary_condition_data(data_collections[0])
model.add_constants(**mat_props, coupling="coupled")
model.use_under_integrated_element()
model.set_number_of_time_steps(100)

for state in data_collections[0].states.values():
    model.run(state, pc)
