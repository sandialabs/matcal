import os

import matcal as mc
import numpy as np
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest


class spring_drag:
    
    def __init__(self, **mean_parameters):
        self._mean_eval = self._raw_calc(**mean_parameters)
    
    def predict(self, **parameters):
        raw_results = self._raw_calc(**parameters)
        for key in self._mean_eval:
            if key == 'time':
                continue
            raw_results[key] -= self._mean_eval[key]
        return raw_results
        
    def _raw_calc(self, **parameters):
        import numpy as np
        from scipy.integrate import odeint
        n_eval_points = 25 #100
        time_end = 10
        times = np.linspace(0, time_end, n_eval_points)
        qoi_names = ['qoi_1', 'qoi_2', 'qoi_dot_1', 'qoi_dot_2']
        qoi_0 = np.array([1, -1, 0, 0])

        spring_const = parameters['spring_const']
        drag = parameters['drag']

        def qoi_dot_dot(q, t, spring_connst, drag):
            delta = q[0] - q[1]
            f = spring_connst * delta
            q_dot = np.zeros_like(q)
            q_dot[2] = - f - drag * q[2] 
            q_dot[3] = f - drag * q[3] 
            q_dot[0] = q[2]
            q_dot[1] = q[3]
            return q_dot

        qois = odeint(qoi_dot_dot, qoi_0, times, args=(spring_const, drag,))
        results = {'time': times}
        for name, value in zip(qoi_names, qois.T):
            results[name] = value
        return results
  
def spring_drag_fun(**parameters):
    import numpy as np
    from scipy.integrate import odeint
    n_eval_points = 100
    time_end = 10
    times = np.linspace(0, time_end, n_eval_points)
    qoi_names = ['qoi_1', 'qoi_2', 'qoi_dot_1', 'qoi_dot_2']
    qoi_0 = np.array([1, -1, 0, 0])

    spring_const = parameters['spring_const']
    drag = parameters['drag']

    def qoi_dot_dot(q, t, spring_connst, drag):
        delta = q[0] - q[1]
        f = spring_connst * delta
        q_dot = np.zeros_like(q)
        q_dot[2] = - f - drag * q[2] 
        q_dot[3] = f - drag * q[3] 
        q_dot[0] = q[2]
        q_dot[1] = q[3]
        return q_dot

    qois = odeint(qoi_dot_dot, qoi_0, times, args=(spring_const, drag,))
    results = {'time': times}
    for name, value in zip(qoi_names, qois.T):
        results[name] = value
    return results
  
  
def _generate_lhs_parameters(goal_mean, goal_sd):
    param_instances = []
    n_sd = 3
    for name in goal_mean:
        mean = goal_mean[name]
        sd = goal_sd[name]
        low = mean - n_sd * sd
        high = mean + n_sd * sd
        param_instances.append(mc.Parameter(name, low, high))
    return param_instances
    
def _generate_bayes_parameters(goal_mean, goal_sd):
    param_instances = []
    n_sd = 4
    for name in goal_mean:
        mean = goal_mean[name]
        sd = goal_sd[name]
        low = mean - n_sd * sd
        high = mean + n_sd * sd
        param_instances.append(mc.Parameter(name, low, high, mean,
                                            distribution='uniform_uncertain'))
    return param_instances

def _generate_data(goal_mean, goal_sd, hifi_fun, n_noisy_instances):
    data_instances = []
    noiseless_data = hifi_fun(**goal_mean)
    noiseless_data = mc.convert_dictionary_to_data(noiseless_data)
    data_instances.append(noiseless_data)
    
    for i_data in range(n_noisy_instances):
        cur_vals = {}
        for name in goal_mean:
            mean = goal_mean[name]
            sd = goal_sd[name]
            val = np.random.normal(mean, sd)    
            cur_vals[name] = val
        cur_data = mc.convert_dictionary_to_data(hifi_fun(**cur_vals))
        data_instances.append(cur_data)
    return data_instances
    


            
def one_parameter_one_signal(**parameters):
    A = parameters['A']
    n_time = 2
    end_time = 1
    time = np.linspace(0, end_time, n_time)
    qoi = np.ones(n_time) * A - 1
    return {'time': time, 'qoi':qoi}        
    

    

def two_parameters_one_signal(**parameters):
    A = parameters['A']
    B = parameters['B']
    n_time = 10
    B_start_idx = n_time // 2
    end_time = 1
    time = np.linspace(0, end_time, n_time)
    qoi = np.ones(n_time) * A / 1
    qoi[B_start_idx:] = B / 2

    qoi = qoi - 1
    return {'time': time, 'qoi':qoi}

def two_parameters_two_signals(**parameters):
    A = parameters['A']
    B = parameters['B']
    n_time = 2
    end_time = 1
    time = np.linspace(0, end_time, n_time)
    qoi_a = np.ones(n_time) * A / 10 - 1
    qoi_b = np.ones(n_time) * B / 20 - 1

    return {'time': time, 'qoi_a':qoi_a, 'qoi_b':qoi_b}



class BayesCalibration(MatcalUnitTest):
    """
    """
    def setUp(self):
        super().setUp(__file__)
        
    def test_one_parameters_one_signal_dram(self):
        goal_mean = {'A':1}
        goal_sd = {'A':.05}
        hifi_fun = one_parameter_one_signal
        n_exp_repeats = 10000
        n_burnin = int(2e2)
        n_cal_samples = int(1e3)
        mean_err_tol = 2e-2
        sd_err_tol = 25e-2
        
        data_instances = _generate_data(goal_mean, goal_sd, hifi_fun, n_exp_repeats)
        my_noisy_data = mc.DataCollection("noisy", *data_instances)
        my_noiseless_data = _generate_data(goal_mean, goal_sd, hifi_fun, 0)[0]
        my_data_stats = my_noisy_data.report_statistics('time')

        weight_lookup = {'qoi': np.mean(my_data_stats['matcal_default_state']['qoi']['std dev'])}
        


        my_objective = mc.CurveBasedInterpolatedObjective("time", "qoi")
        my_objective.set_field_weights(mc.NoiseWeightingConstant(**weight_lookup))



        my_hifi_model = mc.PythonModel(hifi_fun)        
        
        bayes_param_instances = _generate_bayes_parameters(goal_mean, goal_sd)
        my_bayes_parameters = mc.ParameterCollection('bayes', *bayes_param_instances)
            
        cal = mc.DramBayesianCalibrationStudy(my_bayes_parameters, library='queso')
        cal.add_evaluation_set(my_hifi_model, my_objective, my_noiseless_data)

        cal.set_random_seed(831)
        proposal_covar = np.ones(1)
        cal.set_proposal_covariance(*proposal_covar)
        cal.set_number_of_burnin_samples(n_burnin)
        cal.set_number_of_samples(n_cal_samples)

        cal.run_in_serial()
        cal.set_use_threads(True)
        
        results = cal.launch()

        

        
        for param_name in goal_mean:
            self.assertAlmostEqual(results.outcome[f"mean:{param_name}"], goal_mean[param_name], delta=goal_mean[param_name] * mean_err_tol)
            self.assertAlmostEqual(results.outcome[f"stddev:{param_name}"], goal_sd[param_name], delta=goal_sd[param_name] * sd_err_tol)        
        

    def test_two_indep_parameters_two_signals_dram(self):
        goal_mean = {'A':10, 'B': 20}
        goal_sd = {'A':1, 'B': 2}
        hifi_fun = two_parameters_two_signals
        n_exp_repeats = 10000
        n_burnin = int(2e2)
        n_cal_samples = int(1e3)
        mean_err_tol = 2e-2
        sd_err_tol = 25e-2
        
        data_instances = _generate_data(goal_mean, goal_sd, hifi_fun, n_exp_repeats)
        my_noisy_data = mc.DataCollection("noisy", *data_instances)
        my_noiseless_data = _generate_data(goal_mean, goal_sd, hifi_fun, 0)[0]
        my_data_stats = my_noisy_data.report_statistics('time')

        weight_lookup = {'qoi_a': np.mean(my_data_stats['matcal_default_state']['qoi_a']['std dev']), 
                         'qoi_b': np.mean(my_data_stats['matcal_default_state']['qoi_b']['std dev'])}
        
        # my_objective = mc.DirectCurveBasedInterpolatedObjective("time", "qoi_a", "qoi_b")
        # my_objective.use_log_residual()

        my_objective = mc.CurveBasedInterpolatedObjective("time", "qoi_a", "qoi_b")
        my_objective.set_field_weights(mc.NoiseWeightingConstant(**weight_lookup))


        my_hifi_model = mc.PythonModel(hifi_fun)        
        
        bayes_param_instances = _generate_bayes_parameters(goal_mean, goal_sd)
        my_bayes_parameters = mc.ParameterCollection('bayes', *bayes_param_instances)
            
        cal = mc.DramBayesianCalibrationStudy(my_bayes_parameters, library='queso')
        cal.add_evaluation_set(my_hifi_model, my_objective, my_noiseless_data)

        cal.set_random_seed(831)
        proposal_covar = np.ones(2)
        cal.set_proposal_covariance(*proposal_covar)
        cal.set_number_of_burnin_samples(n_burnin)
        cal.set_number_of_samples(n_cal_samples)
        cal.set_results_storage_options(False, False, False)
        cal.run_in_serial()
        cal.set_use_threads(True)
        
        results = cal.launch()

        
        for param_name in goal_mean:
            self.assertAlmostEqual(results.outcome[f"mean:{param_name}"], goal_mean[param_name], delta=goal_mean[param_name] * mean_err_tol)
            self.assertAlmostEqual(results.outcome[f"stddev:{param_name}"], goal_sd[param_name], delta=goal_sd[param_name] * sd_err_tol)   

        
 