# Problem Setup:

# mesh = {mesh = "test_block.g"}
# useChem = {useChem = 1}
# usePressurization = {usePressurization = 1}
# useAir = {useAir = 1}

# resultsDir = {resultsDir = "results"}
# {Thick_foam = .015}
# H_ref = {H_ref = 5}
# T_air_ref = {T_air_ref = 500}
# T_inf_ref = {T_inf_Ref = 1000}



# T_0 = {T_0 = 300}
# P_0 = {P_0 = 101325}
# EndTime = {EndTime = 60 * 60 * 2}
# output_rate = {output_rate = 10} # float of 'all'


Begin Sierra myJob

  ################################################
  ############ Finite Element Model ##############
  ################################################


  Begin Finite Element Model all
    Database Name = {mesh} 
    Coordinate System Is cartesian
    Use Material 304ss For block_1 block_3  
    Use Material PMDI_20lb_foam_1T For block_2
    decomposition method = rib
  End Finite Element Model all


  ################################################
  ############ Solvers ###########################
  ################################################
{include(include/SolverModule.inc)}


  ################################################
  ############ Material Data & Params ############
  ################################################

  Begin Global Constants
    Stefan Boltzmann Constant = 5.670373e-8
    ideal gas constant = 8314.

  end Global Constants

  ###########################
  ### Inorganic Matierals ###
  ###########################
{include(include/InorganicMaterials.inc)}
  
  
  #########################
  ### Organic Matierals ###
  #########################
{include(include/materials_foams_PMDI_20lb_foam_1T.i)}


  ################################################
  ############ Output Scheduler ##################
  ################################################
  
  Begin Output Scheduler forResults
{if(tostring(output_rate)=="all")}
    At Step 0, Increment = 1 
{else}
    At Time 0, Increment = {output_rate}
{endif}
  end Output Scheduler forResults


  ################################################
  ############ Procedure Block ###################
  ################################################

  Begin Procedure myAriaProcedure

    Begin Solution Control Description
      Use System Main

      Begin System Main
        Begin Transient The_Transient_Block
          Advance all
        end Transient The_Transient_Block

        Simulation Start Time = 0.0
        Simulation Termination Time = {EndTime}
      end System Main

      Begin Parameters For Transient The_Transient_Block
        Begin Parameters For Aria Region all
          Initial Time Step Size = 1.e-4
          maximum time step size = 10.0
          minimum time step size = 1.e-9
          Time Step Variation = adaptive
          Time Integration Method = BDF2
          predictor order = 1
          predictor-corrector tolerance = 1.0e-3

          Maximum Time Step Size Ratio = 1.2

        end Parameters For Aria Region all
      end Parameters For Transient The_Transient_Block


    End solution control description
    
    ################################################
    ############ Aria Region #######################
    ################################################
    
    BEGIN ARIA REGION all



      use finite element model all
      use data block Heater_Control

    ################################################
    ############ Mesh Groups #######################
    ################################################

      Mesh Group organic          = block_2
      Mesh Group inorganic_exterior = surface_1
      Mesh Group inorganic        = block_1 block_3
      
      
    ################################################
    ############ Equation System ###################
    ################################################

    ##### Pressurization from decomposing organics ##### 

{if(usePressurization)}

{include(include/pmdi_pressure_block.inc)}

{endif}

    ##### Chemisty solver for decomposing organics ##### 
{include(include/pmdi_chemeq_solver.inc)}

      Begin Equation System full_system
        Nonlinear Relaxation Factor    = 1.0
        Nonlinear Solution Strategy    = newton
        Use Linear Solver gmres_ddilu_stiff
        Maximum Nonlinear Iterations = 7
       Nonlinear Residual Tolerance = 2.e-4

 
        EQ energy FOR temperature ON inorganic USING Q1 WITH mass diff

        Eq energy for temperature on organic using Q1 with mass diff
        source for energy on organic = ChemEq_heating


        #######################
        ## Boundary/Interface
       
        BC flux for energy on inorganic_exterior = generalized_nat_conv H={H} T_ref = {T_air}
        BC flux for energy on inorganic_exterior = generalized_rad T_ref = {T_inf}

        #######################
        ## Postprocess
        
        Postprocess density ON organic
        Postprocess Heat_Flux ON inorganic
        postprocess value of expression temperature at point 0 {Thick_foam /2}  as TC_top
        postprocess value of expression temperature at point 0 -{Thick_foam / 2}  as TC_bottom

      end Equation System full_system
      
      ################################################
      ############ Initial Conditions ################
      ################################################
     
      #######################
      ## Inorganic 
      IC const FOR inorganic Temperature = {T_0}
     
     ####################### 
      # Organic
      IC const FOR organic Temperature = {T_0}
      

      ################################################
      ############ Predictor Fields ##################
      ################################################
      Predictor Fields = Not SPECIES
      

      ##################################################
      ##############  Outputs ##########################
      ##################################################

      Begin Heartbeat csv_output
        stream name= {resultsDir//"/results.csv"}
        use output scheduler forResults
        precision =7
        timestamp format ''
        variable = global time
        variable = global TC_top
        variable = global TC_bottom
        labels = off
        legend = on
      end Heartbeat csv_output



      ####################### 
      ## Exodus Output
      
      BEGIN RESULTS OUTPUT LABEL output
        database Name = {resultsDir//"/results.e"}
        Use Output Scheduler forResults

        Nodal Variables = solution->TEMPERATURE as Temp
        Nodal Variables = pp->heat_flux as q
        element FOAMA as Y_FoamA
        element FOAMB as Y_FoamB
        element FOAMC as Y_FoamC
        element CHAR as Y_Char

      END RESULTS OUTPUT LABEL output


    End aria region all
  End Procedure myAriaProcedure
END SIERRA myJob
