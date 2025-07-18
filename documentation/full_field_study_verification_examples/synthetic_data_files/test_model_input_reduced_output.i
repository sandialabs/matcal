

{include(state_values.inc)}
{include(design_parameter_values.inc)}
#end_time = {end_time = 10.0}
#start_time = {start_time = 0.0}
#time_step = {time_step = (end_time-start_time)/450}
#element_type = {element_type = "total_lagrange"}



begin sierra vfm_uniaxial_model

## COORDINATE SYSTEM AND DIRECTIONS #############################################

  define direction X with vector 1.0 0.0 0.0
  define direction Y with vector 0.0 1.0 0.0
  define direction Z with vector 0.0 0.0 1.0

  define point o with coordinates 0.0 0.0 0.0
  define point x with coordinates 1.0 0.0 0.0
  define point z with coordinates 0.0 0.0 1.0

  define coordinate system rectangular_coordinate_system rectangular with point o point z point x


  begin definition for function displacement_function
    type is piecewise linear
    begin values
      0, 0
      {end_time}, 0.045
    end
  end

# MATERIAL PROPERTIES ==================================================


{include("modular_plasticity.inc")}


# ADAGIO MESH ==========================================================
  begin total lagrange section total_lagrange
    volume average j = on
  end


# FINITE ELEMENT MODEL =================================================
  begin finite element model tension_solid_mechanics
    Database name = test_mesh.g
    Database type = exodusII

    Begin Parameters for Block block_main
      Material = test_material
      Model = hill_plasticity
      Section = {element_type}
    End

  end

# PROCEDURE DEFINITIONS ================================================

  begin adagio procedure adagio_proc
        begin time control 
            # a small step to get an initial elastic stress state
            begin time stepping block init
                start time = {start_time}
                begin parameters for adagio region solid_mechanics_region
                    time increment = {time_step/1000}
                end 
            end 

            # quasi-statics
            begin time stepping block tb0
                start time = {start_time+time_step/1000}
                begin parameters for adagio region solid_mechanics_region
                    time increment = {time_step}   # big for quasi-statics
                end
            end
            termination time = {end_time}
        end


# Solid Mechanics Region ----------------------------------
    begin adagio region solid_mechanics_region
      use finite element model tension_solid_mechanics

      begin adaptive time stepping
        target iterations = 65
        iteration window = 10
       cutback factor = 0.5
        growth factor = 1.5
        maximum failure cutbacks = 40
        maximum multiplier = 1
        minimum multiplier = 1e-10
      end

        Begin prescribed displacement top
          node_set = top_nodes
          function = displacement_function
          component = y
          scale factor = 0.5
        End 

        Begin prescribed displacement bottom
          node_set = bottom_nodes
          function = displacement_function
          component = y
          scale factor = -0.5
        end

        Begin fixed displacement
          node_set = back_node_set
          component = z
        end
        Begin fixed displacement
          node_set = fixed_x_node_set
          component = x 
        end


      begin initial temperature 
        include all blocks
        magnitude = 298
      end

        Begin user output
          node set = top_nodes
          compute global partial_displacement as average of nodal displacement(2)
          compute at every step
        end user output

        Begin user output
          node set = top_nodes
          compute global partial_load as sum of nodal reaction(2)
          compute at every step
        end user output

        Begin user output
          compute global displacement from expression "partial_displacement*2;"
          compute global  load from expression "partial_load*2;"
          compute at every step
        end user output

      Begin user variable max_load
        type = global real length = 1
        global operator = max
        initial value = 0
      end

      Begin user output
        compute global max_load from expression "\#
        (load >= max_load) ? load : max_load;"
        compute at every step
      end user output

      Begin user output
        compute global terminate_solution from expression "\#
        (time > {end_time/10})?(load < max_load*(0.5))?0:1:1;"
        compute at every step
      end user output

      begin solution termination terminator
        terminate global terminate_solution < 1
        terminate type = entire_run
      end


      # REQUESTED OUTPUT
      begin results output solid_mechanics_output
        database name = ./surf_results.e
        database type = exodusII
        include = front_DIC_surf 
        exclude = block_main
        at step 0, increment = 1
        global load
        global displacement
        nodal displacement
        output mesh = exposed surface
      end
      # SOLVER
      begin solver
        begin cg
          reference = belytschko
          target residual = 1e-08
          target relative residual = 1e-10
      
          maximum iterations = 80
          minimum iterations = 1
       
          acceptable residual = 1e-7
          acceptable relative residual = 1e-08
      
          begin full tangent preconditioner
            small number of iterations = 20
            #minimum smoothing iterations = 15
          end
        end
      end

    end
  end
end
