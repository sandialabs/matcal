def linear_python_model(**variables):
    time_max = 10
    num_time_steps = 100

    import numpy as np
    time = np.linspace(0, time_max, num_time_steps)
    values = variables['slope'] * time + variables['intercept']
    return {'time': time, "Y": values}
