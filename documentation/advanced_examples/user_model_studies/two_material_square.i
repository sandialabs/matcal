
# mesh_name = {mesh_name = "two_material_square.g"}
# top_block = {top_block = "block_1"}
# bottom_block = {bottom_block = "block_2"}

# top_surface = {top_surface = "surface_1"}
# middle_surface = {middle_surface = "surface_3"}
# bottom_surface = {bottom_surface = "surface_2"}

# T_0 = {T_0 = 273}

# time_end = {time_end = 20}
# n_data = {n_data = 20}
# out_dt = {out_dt = time_end / n_data}

# results_basename = {results_basename = "two_material_results"}
# csv_results_name = {csv_results_name = results_basename//".csv"}
# exo_results_name = {exo_results_name = results_basename//".e"}
# useExo = {useExo = 0}
# runRef = {runRef = 0}

{if(runRef)}
{include(ref_cond.inc)}
{endif}

Begin Sierra myJob

  ################################################
  ############ Finite Element Model ##############
  ################################################


  Begin Finite Element Model all
    Database Name = two_material_square.g 
    Use Material example_ceramic_foam For {top_block}
    Use Material example_steel For {bottom_block}
    decomposition method = rib
  End Finite Element Model all


  ################################################
  ############ Solvers ###########################
  ################################################



  BEGIN Tpetra EQUATION SOLVER scalar
    Begin GMRES Solver
      Begin SGS Preconditioner
        number of sweeps = 1
      End
      MAXIMUM ITERATIONS = 100
      CONVERGENCE TOLERANCE = 1.e-6
      Residual Scaling = R0
    END

    bc enforcement = solver_no_column_mod
  END Tpetra EQUATION SOLVER scalar
Begin TPETRA EQUATION SOLVER GMRES_DDILU
  BEGIN GMRES SOLVER
    BEGIN DD-ILU PRECONDITIONER
      SUBDOMAIN OVERLAP LEVEL = 1
    END
    MAXIMUM ITERATIONS = 500
    RESTART ITERATIONS = 500
    CONVERGENCE TOLERANCE = 1e-06
    RESIDUAL SCALING = R0
  END
  MATRIX SCALING = ONE_NORM
END

Begin TPETRA EQUATION SOLVER GMRES_DDILU_STIFF
  BEGIN GMRES SOLVER
    BEGIN DD-ILU PRECONDITIONER
    END
    MAXIMUM ITERATIONS = 2000
    RESTART ITERATIONS = 2000
    CONVERGENCE TOLERANCE = 1e-06
    RESIDUAL SCALING = R0
  END
  MATRIX SCALING = ONE_NORM
END

Begin TPETRA EQUATION SOLVER GMRES_DDILUT
  BEGIN GMRES SOLVER
    BEGIN DD-ILUT PRECONDITIONER
      FILL FRACTION = 10.0
      SUBDOMAIN OVERLAP LEVEL = 0
    END
    MAXIMUM ITERATIONS = 500
    RESTART ITERATIONS = 500
    CONVERGENCE TOLERANCE = 1e-06
    RESIDUAL SCALING = R0
  END
  MATRIX SCALING = ONE_NORM
END

BEGIN TPETRA EQUATION SOLVER multiphysics
  BEGIN PRESET SOLVER
    SOLVER TYPE = MULTIPHYSICS
  END
END
  ################################################
  ############ Material Data & Params ############
  ################################################

  Begin Global Constants
    Stefan Boltzmann Constant = 5.670373e-8
    ideal gas constant = 8314.

  end Global Constants

  Begin Aria Material example_ceramic_foam
        Density              = constant  rho = .38e3  # [kg/m^3]
        Thermal Conductivity = constant  value = {K_foam} # goal: .15
# specific_heat_slope =  { dcdt = .01}
        Specific Heat        = polynomial variable = temperature order=1 c0 = {1 + dcdt * 273} c1 = {dcdt}
        Heat Conduction = basic                                                 
  end Aria Material 

  Begin Aria Material example_steel
        Density              = constant  rho = 8e3  # [kg/m^3]
        Thermal Conductivity = constant  value = {K_steel} # goal: 45
        Specific Heat        = constant value = 1.
        Heat Conduction = basic                                                 
  end Aria Material 

  ################################################
  ############ Output Scheduler ##################
  ################################################
  
  Begin Output Scheduler forResults
    #At Step 0, Increment = 1  # Rapid output for testing
    At Time 0, Increment = {out_dt}
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
        Simulation Termination Time = {time_end}
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

      
      Begin Equation System full_system
        Nonlinear Relaxation Factor    = 1.0
        Nonlinear Solution Strategy    = newton
        Use Linear Solver gmres_ddilu_stiff
	      Maximum Nonlinear Iterations = 7
	      Nonlinear Residual Tolerance = 2.e-4

 
        EQ energy FOR temperature ON {top_block} USING Q1 WITH mass diff
        EQ energy FOR temperature ON {bottom_block} USING Q1 WITH mass diff

      BC flux for energy on {top_surface} = Constant value = {-exp_flux}
      BC flux for energy on {bottom_surface} = scalar_string_function f="10 * (temperature - 273)"


        
      end Equation System full_system
      
      postprocess average of expression temperature on {bottom_surface} as T_bottom
      postprocess average of expression temperature on {middle_surface} as T_middle
      
      ################################################
      ############ Initial Conditions ################
      ################################################
     
      #######################
      ## Canister 
      IC const FOR all_blocks Temperature = {T_0}
  
     


      ##################################################
      ##############  Outputs ##########################
      ##################################################


      BEGIN HEARTBEAT out_1
        STREAM NAME = {csv_results_name}
        Use Output Scheduler forResults
        Precision = 7
        timestamp format ''
        Variable = Global time as time
        variable = global count as count
        variable = global T_bottom
        variable = global T_middle
        labels = off
        legend = on
      END HEARTBEAT out_1

{if(useExo)}
      BEGIN RESULTS OUTPUT LABEL output
        database Name = {exo_results_name}
        Use Output Scheduler forResults
        Nodal Variables = solution->TEMPERATURE as T
      END RESULTS OUTPUT LABEL output
{endif}



    End aria region all
  End Procedure myAriaProcedure
END SIERRA myJob
