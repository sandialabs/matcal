
      Begin Pressurization Model PMDI_20lb_foam_1T_pressure_zone
        Initial Pressure = {P_0}
        Pressure Unit = Pa
        Pressurization Source Blocks = organic # LIST BLOCKS WITH RELEVANT DECOMPOSING FOAM
        Pressurized Blocks           = organic # LIST BLOCKS INTO WHICH GASES EXPAND HERE
        Venting Model = Closed
        Equation of State     = Ideal_Gas
        Temperature Averaging = Ideal_Gas
        #Excess Volume             = 0 #ANY UNMESHED VOLUME INTO WHICH GASES CAN EXPAND
        #Excess Volume Temperature = 273 #INITIAL TEMPERATURE OF THE UNMESHED VOLUME
      End Pressurization Model PMDI_20lb_foam_1T_pressure_zone
