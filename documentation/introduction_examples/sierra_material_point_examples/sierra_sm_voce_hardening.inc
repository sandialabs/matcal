  
  begin material j2_voce
    density = 0.000254
    begin parameters for model j2_plasticity
      youngs modulus                = 9.9e6
      poissons ratio                =   0.33
      yield stress                  = {Y*1e3}

      hardening model = voce
      hardening modulus = {A*1e3}
      exponential coefficient = {b} 
    end
  end
