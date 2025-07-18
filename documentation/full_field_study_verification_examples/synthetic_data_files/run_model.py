from matcal import *

mat_file_string = """begin material test_material
  density = 1

  begin parameters for model j2_plasticity
    youngs modulus  = {elastic_modulus*1e9}
    poissons ratio  = {poissons}
    yield_stress    = {yield_stress*1e6}

    hardening model = voce
    hardening modulus = {A*1e6}
    exponential coefficient = {n}
  end
  begin parameters for model hill_plasticity
    youngs modulus  = {elastic_modulus*1e9}
    poissons ratio  = {poissons}
    yield_stress    = {yield_stress*1e6}

    hardening model = voce
    hardening modulus = {A*1e6}
    exponential coefficient = {n}

    R11 = {R11}
    R22 = {R22}
    R33 = {R33}
    R12 = {R12}
    R23 = {R23}
    R31 = {R31}

    coordinate system = rectangular_coordinate_system
    direction for rotation = 1
    alpha = 0
    second direction for rotation = 3
    second alpha = {angle}
  end
end
"""

with open("modular_plasticity.inc", 'w') as fn:
    fn.write(mat_file_string)

model = UserDefinedSierraModel("adagio", "test_model_input.i", "test_mesh.g", "modular_plasticity.inc")
model.set_number_of_cores(112)
#{elastic_modulus = 200e9}
#{poissons = 0.27}
#{yield_stress   = 200.0}
#{A = 1500}
#{n = 2}
#{R11 = 0.95}
#{R22 = 1.00}
#{R33 = 0.9}
#{R12 = 0.85}
#{R23 = 1.00}
#{R31 = 1.00}

E = Parameter("elastic_modulus", 50, 300, 200.0)
nu = Parameter("poissons", 0.2, 0.4, 0.27)
Y = Parameter("yield_stress", 100, 1000.0, 200.0)
A = Parameter("A", 100, 4000, 1500)
n = Parameter("n", 1, 30, 2)

R11 = Parameter("R11", 0.8, 1.1, 0.95)
R22 = Parameter("R22", 0.8, 1.1, 1.0)
R33 = Parameter("R33", 0.8, 1.1, 0.9)
R12 = Parameter("R12", 0.8, 1.1, 0.85)
R23 = Parameter("R23", 0.8, 1.1, 1.0)
R31 = Parameter("R31", 0.8, 1.1, 1.0)

param_c = ParameterCollection("params", E, nu, Y, A, n, R11, R22, R33, R12, R23, R31)
model.add_constants(material_model="hill_plasticity")
model.read_full_field_data("surf_results.e")
model.run_in_queue("add_queue_id", 1)
model.continue_when_simulation_fails()
model.set_name("anisotropic")
model.run(State("45_degree", angle=45), param_c)
model.run(State("0_degree", angle=0), param_c)
model.run(State("90_degree", angle=90), param_c)

