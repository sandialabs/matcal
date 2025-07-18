import matplotlib.pyplot as plt
    

def determine_slope_and_intercept(data, first_index, independent_field_name, 
                                  dependent_field_name):
    """
    This function will determine the intercept and slope
    of a line through two consecutive points in a data set. 

    :param data: the data for which the slope and intercept will 
        be determined.
    :type data: :class:`~matcal.core.data.Data`

    :param first_index: the index of the first point of the consecutive 
       points of interest
    :type first_index: int

    :param x_field: the independent variable field name
    :type x_field: str

    :param y_field: the independent variable field name
    :type y_field: str

    :return: slope and intercept
    :rtype: tuple(float, float)
    """
    
    x_pt1 = data[independent_field_name][first_index]
    y_pt1 = data[dependent_field_name][first_index]
    
    x_pt2 = data[independent_field_name][first_index+1]
    y_pt2 = data[dependent_field_name][first_index+1]
    
    m = (y_pt2-y_pt1)/(x_pt2-x_pt1)
    b = y_pt1 - m*x_pt1
    return m,b


def determine_line_intersection(m1, b1, m2, b2):
    """
    This function determines
    the intersection point location
    for two two-dimensional lines given their slopes 
    and intercepts. 

    :parameter m1: the slope of the first line
    :type m1: float

    :parameter b1: the intercept of the first line
    :type m1: float
    
    :parameter m2: the slope of the second line
    :type m2: float

    :parameter b2: the intercept of the second line
    :type m1: float
    
    :return: intersection point independent and dependent field values
    :rtype: tuple(float, float)
    """
    intersection_x = (b1 - b2)/(m2-m1)
    intersection_y = m1*intersection_x+b1
    return intersection_x, intersection_y
    

def determine_pt2_offset_yield(stress_strain_data, elastic_mod,  
                               strain_field="engineering_strain", 
                               stress_field="engineering_stress", 
                               plot=False, show_plot=True, 
                               blocking_plot=True):
    """
    This function calculates the 0.2% offset 
    yield stress values for a given data set.

    :param stress_strain_data: the data set that the function 
        will operate on and use 
        to calculate the 0.2% offset stress.
    :type stress_strain_data: :class:`~matcal.core.data.Data`

    :param elastic_mod: the elastic modulus for the material from which the data
        was collected. The units must match the units of the provided
        experimental stress strain data.

    :param strain_field: the name of the field for the strain 
        values in the data set
    :type strain_field: str

    :param stress_field: the name of the field for the stress 
        values in the data set
    :type stress_field: str

    :param plot: optionally plot the provided stress strain data, the 0.2% 
        offset elastic data, and the 0.2% yield stress for the data set.
    :type plot: bool

    :param show_plot: boolean to show or not show the plot when the function 
        is called.
    :type plot: bool
    
    :param blocking_plot: If True, this stops the code until 
        the figure is closed. If False, the code continues after the figure is 
        made
    :type blocking_plot: bool

    :return: the 0.2% offset strain (first index) and stress (second index)
    :rtype: tuple(float, float)
    """
    elastic_stress = elastic_mod*stress_strain_data[strain_field]-elastic_mod*0.002
    error = stress_strain_data[stress_field] - elastic_stress
    error = error[error > 0]
    m,b=determine_slope_and_intercept(stress_strain_data,len(error)-1, 
                                      strain_field, stress_field)
    yield_point = determine_line_intersection(m, b, elastic_mod,                                                
                                                -elastic_mod*0.002)
    if plot:
        _plot_pt2_off_set(stress_strain_data, strain_field, 
                          stress_field,
                          elastic_stress, yield_point, show_plot, 
                          blocking_plot)
    return yield_point

def _plot_pt2_off_set(stress_strain_data, strain_field, 
                      stress_field, elastic_stress, yield_pt, show_plot, 
                      blocking_plot):
    strain = stress_strain_data[strain_field]
    stress = stress_strain_data[stress_field]
    plt.plot(strain, stress, '.')
    pts_to_plot = elastic_stress < stress
    plt.plot(strain[pts_to_plot], elastic_stress[pts_to_plot], 'r-', 
             label="offset elastic data")
    plt.plot(yield_pt[0], yield_pt[1], 'o', label="0.2% offset yield")
    plt.xlabel(strain_field)
    plt.ylabel(stress_field)
    if show_plot:
        plt.show(block=blocking_plot)