    

.. note:: 
    The thickness direction is the Z direction and 
    the loading direction must be the Y direction.
    User provided meshes and data should conform 
    to these directions.

.. warning::
    SIERRA/SM is inherently 3D and, as implemented, the VFM 
    models may produce non-negligible through the thickness stresses depending on 
    the boundary value problem being simulated. In the virtual
    internal work calculation, all through thickness stresses are currently ignored. 
    This leads to errors in calibration, however, if the boundary value problem conforms
    well to the plane stress assumption
    these errors should only be on the order of 1% or less for the parameter values.

.. warning:: 
    The experimental full-field data must be properly aligned with the mesh.
    We currently do not assist in aligning the mesh coordinate system with the 
    experimental data coordinate system. Work with the experimentalist to do so or 
    perform alignment as a preprocessing step. 