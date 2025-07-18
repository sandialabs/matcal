 
    begin definition for function 304L_yield_temp_dependence
        # loose linear estimate of data from MMPDS10 Figure 6.2.1.1.4a
        type is piecewise linear
        begin values
        5.20e+02,1.0
        1.0e+03,6.0e-01
        end
    end

    begin definition for function 304_elastic_mod_temp_dependence
        #Stender et. al.
        type is piecewise linear
        begin values
            294.11,     1
            1673,      0.4
        end
    end 

    begin definition for function 304L_thermal_strain_temp_dependence
        #Stender et. al.
        type is piecewise linear
        begin values
            294.11, 0.0
            1725.0, 0.02
        end
    end

    begin material 304L_viscoplastic
        #density and elastic parameters from Granta's MMPDS10 304L database Table 2.7.1.0(b3). 
        #Design Mechanical and Physical Properties of AISI 304 Stainless Steels

        density = {density}
        thermal engineering strain function = 304L_thermal_strain_temp_dependence
    
        begin parameters for model j2_plasticity
            youngs modulus                = 29e6
            poissons ratio                =   0.27
            yield stress                  = {Y_0*1e3}

            youngs modulus function = 304_elastic_mod_temp_dependence

            hardening model = decoupled_flow_stress

            isotropic hardening model = voce
            hardening modulus = {A*1e3}
             exponential coefficient = {b}

            yield rate multiplier = johnson_cook
            yield rate constant = {10^C}
            yield reference rate = {ref_strain_rate}


            yield temperature multiplier = user_defined
            yield temperature multiplier function = 304L_yield_temp_dependence 

            hardening rate multiplier = rate_independent
            hardening temperature multiplier = temperature_independent

            thermal softening model      = {coupling}
            beta_tq                      = 0.95
            specific heat = {specific_heat}
        end
    end
