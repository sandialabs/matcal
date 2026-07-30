  Begin Aria Material 304SS  # 304 Stainless Steel, Isotropic 
        Density              = constant  rho = 7901.  # [kg/m^3]
        Thermal Conductivity = user_function  name=f_steelcond X=Temperature  # [W/m-K]
        Specific Heat        = user_function  name=f_steelspheat X=Temperature  # [J/kg-K]
        Emissivity           = constant  e=0.8
        Heat Conduction = basic                                                 
        gap_conductance_coefficient = user_function name= pmdi20lb_PMVLE_contact_conductance X=time
  end Aria Material 304SS  # 304 Stainless Steel, Isotropic 

  Begin Aria Material 321SS  # Copy properties from 304SS
        Density              = constant  rho = 7901.  # [kg/m^3]
        Thermal Conductivity = user_function  name=f_steelcond X=Temperature  # [W/m-K]
        Specific Heat        = user_function  name=f_steelspheat X=Temperature  # [J/kg-K]
        Emissivity           = constant  e=0.8
        Heat Conduction = basic
        gap_conductance_coefficient = user_function name= pmdi20lb_PMVLE_contact_conductance X=time
  end Aria Material 321SS  # Copy properties from 304SS

  Begin Definition for Function f_steelcond  # AISI-304 stainless steel conductivity
    Type is PIECEWISE LINEAR
    Begin Values
      #Temperature    Conductivity (W/mK) from W.Gill
      50.0            14.5
      300.0           14.5     
      330.0           15      
      344.5           15.33
      372.8           15.77
      415.5           16.68
      470.1           17.56
      511.7           18.26
      562.2           19.17
      606.5           19.97
      743.1           22.2
      783.2           22.91
      840.6           23.91
      880.1           24.6
      2000.0          24.6     # Extrapolated value
   End Values
  End Definition for Function f_steelcond
  
  Begin Definition for Function f_steelspheat  # AISI-304 stainless steel specific heat 
    Type is PIECEWISE LINEAR
    Begin Values
      #Temperature    Specific Heat (J/kgK) from W.Gill
      50.0            460.31 
      273.5           460.31   
      293.3           472.86
      300.3           477.05
      350.3           497.97
      400.3           514.71
      450.3           527.26
      500.3           539.82
      550.3           548.19
      600.3           556.56
      650.3           564.92
      700.3           569.11
      750.3           577.48
      800.3           581.66
      850.3           585.85
      900.3           594.22
      950.3           602.59
      1000.3          610.96
      1100.3          623.51
      1200.3          640.25
      1300.3          652.8
      1400.3          665.36
      1500.3          682.09
    End Values
  End Definition for Function f_steelspheat
  
  
  ################################
  ## Can Thermal boundary Properties  
  # conv_coeff   = { conv_coeff   = 10.}
  # emis_can     = { emis_can    = 0.8}
