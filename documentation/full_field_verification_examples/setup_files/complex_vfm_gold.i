#end_time = {end_time = 10.0}
#start_time = {start_time = 0.0}
#{end_displacement = 0.007}
#time_step = {time_step = (end_time-start_time)/300}
begin sierra vfm_complex_model

## COORDINATE SYSTEM AND DIRECTIONS #############################################

  define direction X with vector 1.0 0.0 0.0
  define direction Y with vector 0.0 1.0 0.0
  define direction Z with vector 0.0 0.0 1.0

  define point o with coordinates 0.0 0.0 0.0
  define point x with coordinates 1.0 0.0 0.0
  define point z with coordinates 0.0 0.0 1.0

  define coordinate system CSYS rectangular with point o point z point x

  begin definition for function displacement_function
    type is piecewise linear
    begin values
      0, 0
      {end_time}, {end_displacement}
    end
  end

# MATERIAL PROPERTIES ==================================================

{include("j2_plasticity_material.inc")}

# ADAGIO MESH ==========================================================
  begin total lagrange section total_lagrange
    volume average j = on
  end

  Begin Solid Section default
    strain incrementation = strongly_objective
  End

  Begin Shell Section shell
      thickness = {thickness}
    Formulation = BT_shell
  End

  Begin Membrane Section membrane
    thickness = {thickness}
    Formulation = selective_deviatoric
    deviatoric parameter = 1.0
  End

# FINITE ELEMENT MODEL =================================================
  begin finite element model tension_solid_mechanics
    Database name = {mesh_name}
    Database type = exodusII

    Begin Parameters for Block block_main
      Material = matcal_test
      Model = j2_plasticity
      Section = shell
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
                    time increment = {time_step}
                end
            end
            termination time = {end_time}
        end


# Solid Mechanics Region ----------------------------------
    begin adagio region solid_mechanics_region
      use finite element model tension_solid_mechanics

      begin adaptive time stepping
        target iterations = 70
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
          node_set = fixed_z_node_set
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
        end user output

        Begin user output
          node set = top_nodes
          compute global load as sum of nodal reaction(2)
        end user output

        Begin user output
          compute global displacement from expression "partial_displacement*2;"
          compute at every step
        end user output

      # REQUESTED OUTPUT
      begin results output solid_mechanics_output
        database name = ./complex_plastic_results.e
        database type = exodusII
        at step 0, increment = 1
        element temperature
        nodal temperature
        nodal displacement
        global time
        global displacement
        global load
      end

      # SOLVER
      begin solver
        begin cg
          reference = belytschko
          target residual = 1e-08
          target relative residual = 1e-10

          maximum iterations = 80
          minimum iterations = 1

          acceptable residual = 1e-4
          acceptable relative residual = 1e-05

          begin full tangent preconditioner
            small number of iterations = 20
          end
        end
      end

    end
  end
end
