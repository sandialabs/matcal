$ Algebraic Preprocessor (Aprepro) version 5.14 (2019/11/20)
# pmdi_bulk_density = 20 lb/ft^3
$ Algebraic Preprocessor (Aprepro) version 5.14 (2019/11/20)
# foam_name = PMDI_20lb_foam_1T
# foam_defintion = PMDI.material.1T.i
# foam_specheat = PMDI.properties.specheat.i
# reaction_parameters = PMDI.Reactions.PC.i
# chemistry_parameters = PMDI.ChemEq.params.i
# chemistry_definition = PMDI.ChemEq.i
# solver_parameters = PMDI.ChemEq.PC.solver.i; 1

BEGIN ARIA MATERIAL PMDI_20lb_foam_1T

    Emissivity = constant e = 0.8

    Density = constant rho = 320.3692  # Convert lb/ft^3 to kg/m^3


    Species Names = FOAMA FOAMB FOAMC CHAR CO2 LMWO HMWO

    Mass_Fraction of FOAMA = From_ChemEq
    Mass_Fraction of FOAMB = From_ChemEq
    Mass_Fraction of FOAMC = From_ChemEq
    Mass_Fraction of CHAR  = From_ChemEq

    # We only need the gasses here for post-processing
    Mass_Fraction of CO2            = From_ChemEq
    Mass_Fraction of HMWO           = From_ChemEq
    Mass_Fraction of LMWO           = Fracbal

    Solid_Density of FOAMA = Constant  rho_solid = 1500
    Solid_Density of FOAMB = Constant  rho_solid = 1500
    Solid_Density of FOAMC = Constant  rho_solid = 1500
    Solid_Density of CHAR  = Constant  rho_solid = 1500

    # This requires expressions for mass fractions and bulk densities for all solid species
    Volume_Fraction_Gas = from_chemeq_mass_fractions         # phi = 1 - rho * sum_solidi(Y_i/rho_b_i)

    Species = ChemEq_Gas

    Bulk Conductivity = Linear_Temperature_And_Density  C_0 = -0.00758  C_T = 0.0001  C_Rho = 0.000081
    Radiative Conductivity = ChemEq_Foam                Kc0 =  0.00268  Kc1 = 0.0
    Thermal_Conductivity = Summed  Contributions="bulk_conductivity radiative_conductivity"

    Heat Conduction = basic

    Specific Heat = user_function  name = f_cp_PMDI_20lb_foam_1T  X = temperature

    pressure = Pressurization_Model

    BEGIN PARAMETERS FOR CHEMEQ MODEL PMDI_20lb_foam_1T_reaction_model


      species names  are             FOAMA     FOAMB     FOAMC     CHAR      CO2 LMWO HMWO
      species phases are             Condensed Condensed Condensed Condensed Gas Gas  Gas
      species molecular weights are  1         1         1         1         44  80   120


      # Formulate the mechanism on a per-mass basis

      Concentration units = mass fractions

      Begin Reaction R_A
        Reaction is FOAMA -> 0.56 CO2 + 0.44 LMWO
        Rate Function = Arrhenius  A = 8.076663742e+12  Ea = 179441062  R = 8314  beta = 0
        #Temperature Phase = Solid_Phase
        Heat of Reaction = 0
        Concentration Function = Standard  mu = Automatic
      End Reaction R_A

      Begin Reaction R_B
        Reaction is FOAMB -> HMWO
        Rate Function = Arrhenius  A = 1.788833939e+11  Ea = 179441062  R = 8314  beta = 0
        #Temperature Phase = Solid_Phase
        Heat of Reaction = 0
        Concentration Function = Standard  mu = Automatic
      End Reaction R_B

      Begin Reaction R_C
        Reaction is FOAMC -> 0.5 CHAR + 0.5 HMWO
        Rate Function = Arrhenius  A = 8906079764  Ea = 179441062  R = 8314  beta = 0
        #Temperature Phase = Solid_Phase
        Heat of Reaction = 0
        Concentration Function = Standard  mu = Automatic
      End Reaction R_C

    END PARAMETERS FOR CHEMEQ MODEL PMDI_20lb_foam_1T_reaction_model
    
END ARIA MATERIAL PMDI_20lb_foam_1T
 
  Begin Definition for Function f_cp_PMDI_20lb_foam_1T  $specific heat capacity
    type = piecewise linear
    Begin Values

      $Temperature       Specific heat (J/kg/K)-- data from tprl data- from gill memo via ken
        50.0             1269
        296.0            1269
        323.0            1356
        373.0            1497
        423.0            1843
        473.0            1900
        523.0            2203
        2000.0           2203        $ Extrapolated value

    End Values
  End Definition for Function f_cp_PMDI_20lb_foam_1T
#
# -------- PRESSURIZATION MODEL DEFINITION --------
#    Requires un-commenting Pressurization Model
#     definition in the relevant aria region
# -------------------------------------------------
#
#      Begin Pressurization Model PMDI_20lb_foam_1T_pressure_zone
#        Initial Pressure = 101325
#        Pressure Unit = Pa
#        Pressurization Source Blocks = LIST BLOCKS WITH RELEVANT DECOMPOSING FOAM
#        Pressurized Blocks           = LIST BLOCKS INTO WHICH GASES EXPAND HERE
#        Venting Model = Closed
#        Equation of State     = Ideal_Gas
#        Temperature Averaging = Ideal_Gas
#        Excess Volume             = ANY UNMESHED VOLUME INTO WHICH GASES CAN EXPAND
#        Excess Volume Temperature = INITIAL TEMPERATURE OF THE UNMESHED VOLUME
#      End Pressurization Model PMDI_20lb_foam_1T_pressure_zone
#
# -------- CHEMEQ SOLVER DEFINITION --------
#    Requires un-commenting ChemEQ Solver
#   definition in the relevant aria region
# ------------------------------------------
#
#      Begin ChemEQ Solver for PMDI_20lb_foam_1T_reaction_model
#
#         Relative Tolerance = 1e-3
#         Absolute Tolerance = 1e-6
#
#         species FOAMA  = 0.45
#         species FOAMB  = 0.15
#         species FOAMC  = 0.40
#         species CHAR   = 0.0
#         species CO2    = 0.0
#         species HMWO   = 0.0
#         species LMWO   = 0.0
#
#         ODE solver = CVODE
#
#      End ChemEQ Solver for PMDI_20lb_foam_1T_reaction_model
#
