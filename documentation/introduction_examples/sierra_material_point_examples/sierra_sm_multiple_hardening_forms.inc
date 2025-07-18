  
 begin function double_voce
    type is analytic
    evaluate expression = "{Y*1e3}+{A1*1e3}*(1-exp(-{b1}*x))+{A2*1e3}*(1-exp(-{b2}*x))"
    differentiate expression = "{A1*b1*1e3}*(exp(-{b1}*x))+{A2*b2*1e3}*(exp(-{b2}*x))"
  end

  begin material j2_double_voce
    density = 0.000254
    begin parameters for model j2_plasticity
      youngs modulus                = 9.9e6
      poissons ratio                =   0.33
      yield stress                  = {Y*1e3}

      hardening model = user_defined
      hardening function = double_voce
    end
  end

  begin material j2_power_law
    density = 0.000254
    begin parameters for model j2_plasticity
      youngs modulus                = 9.9e6
      poissons ratio                =   0.33
      yield stress                  = {Y*1e3}

      hardening model = power_law
      hardening constant = {A*1e3}
      hardening exponent = {n}
      luders strain = {epsilon_l}
    end
  end
