#
# input file for Aria test problem:
#  simple 3D, diffusion problem
#

Begin Sierra The_Job

   Begin ARIA Material mat
      density = constant value =10
      specific heat = constant cp = 10
      thermal Conductivity = Constant value = {K}
      heat conduction = basic
   End
 
    BEGIN TPETRA EQUATION SOLVER SOLVER
      BEGIN UMFPACK SOLVER
      END
    END TPETRA EQUATION SOLVER

   Begin Finite Element Model rect
      Database Name = square.g
      Begin Parameters For Block block_1
        Material mat
      End
   End

   Begin Procedure The_Procedure
      
      Begin Solution Control Description
	 Use System Main
	 Begin System Main
	    Begin transient MySolveBlock
	       Advance myRegion
	    End
            simulation start time = 0
            simulation termination time = 100
	 End
         begin parameters for transient mysolveblock
          begin parameters for aria region myRegion
            Initial Time Step Size = 1.e-4
            maximum time step size = 10.0
            minimum time step size = 1.e-9
            Time Step Variation = adaptive
            Time Integration Method = first_order
            predictor order = 1
            predictor-corrector tolerance = 1.0e-3
          end
         end
      End

      Begin Aria Region myRegion

	 Use Finite Element Model rect

	 Use Linear Solver SOLVER

	 Nonlinear Solution Strategy  = Newton
	 Maximum Nonlinear Iterations = 10
	 Nonlinear Residual Tolerance = 1.0e-12
	 Nonlinear Relaxation Factor  = 1.0

	 EQ energy for temperature on all_blocks using Q1 with Mass Diff

	 BC Dirichlet for temperature on bottom = constant value = 500

        IC for temperature on all_blocks = constant value = 300

        postprocess value of expression temperature at point 0 0.5 as TC1 with tolerance 0.001

        begin heartbeat out1
          stream name = results.csv
          precision = 7
          at time 0, increment =1
          timestamp format ""
          variable = global time as time
          variable = global TC1 as temperature
          labels = off
          legend = on
        end heartbeat out1

	 
      End

   End

End


