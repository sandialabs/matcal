
  begin definition for function al6061T6_yield_temp_dependence
    type is piecewise linear
    begin values
         533.07, 1
         671.67,  {Y_scale_factor_672}
         851.67,  {Y_scale_factor_852}
        1031.67, {Y_scale_factor_1032}
        1391.67, 0.01
    end
  end

  begin definition for function al6061T6_hardening_mod_temp_dependence
    type is piecewise linear
    begin values
         533.07, 1
         671.67,  {A_scale_factor_672}
         851.67,  {A_scale_factor_852}
        1031.67, {A_scale_factor_1032}
        1391.67, 0.01
    end
  end

  begin definition for function al6061T6_hardening_exp_coeff_temp_dependence
    type is piecewise linear
    begin values
         533.07, 1
         671.67,  {b_scale_factor_672}
         851.67,  {b_scale_factor_852}
        1031.67, {b_scale_factor_1032}
        1391.67, 0.01
    end
  end

  begin material ductile_failure_6061T6
    density = 0.00025253231636626974
    begin parameters for model hill_plasticity
      poissons ratio                = 0.33
      youngs modulus                = 10300000.0

      yield stress                  = {yield_stress*1e3}
      yield stress function = al6061T6_yield_temp_dependence

      r11                           =   1
      r22                           =   {R22}
      r33                           =   {R33}
      r12                           =   {R12}
      r23                           =   {R23}
      r31                           =   {R31}
      coordinate system             =   rectangular_coordinate_system
      
      {if(direction=="R11")}
      direction for rotation        = 3
      alpha                         = 90.0
      {elseif((direction=="R33") || (direction=="R31"))}
      direction for rotation        = 1
      alpha                         = -90.0
      {elseif(direction=="R23")}
      direction for rotation        = 2
      alpha                         = 90.0
      {endif}

      hardening model = flow_stress_parameter
      isotropic hardening model = voce_parameter

      hardening modulus = {hardening*1e3}
      hardening modulus function = al6061T6_hardening_mod_temp_dependence

      exponential coefficient = {b}
      exponential coefficient function = al6061T6_hardening_exp_coeff_temp_dependence

      rate multiplier = rate_independent
    end
  end
