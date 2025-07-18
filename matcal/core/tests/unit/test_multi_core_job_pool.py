from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from matcal.core.parameter_batch_evaluator import BatchRestartCSV
import numpy as np
import os
import time

from matcal.core.computing_platforms import local_computer
from matcal.core.data import DataCollection, convert_dictionary_to_data
from matcal.core.evaluation_set import StudyEvaluationSet
from matcal.core.multi_core_job_pool import (Dispatcher, Job, TooManyJobCoresError, 
    _create_job_dispatcher, dispatch_jobs, prepare_parameter_evaluation_jobs, run_jobs_serial, 
    _retrieve_restart_results)
from matcal.core.objective import (CurveBasedInterpolatedObjective, ObjectiveCollection, 
                                   ObjectiveSet)
from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.models import PythonModel
from matcal.core.state import SolitaryState, State, StateCollection
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest
from matcal.core.tests.utilities_for_tests import MockExecutableModel


class Wait():
    
    class WaitResult():
        def __init__(self, cores, wait):
            self.cores = cores
            self.wait = wait
            self.stdout = "cores {}, wait {}".format(self.cores, self.wait)
            self.stderr = ""
            self.source_filename = "wait.fake.csv"

    def __init__(self, cores, wait, computer=local_computer):
        self._cores = cores
        self._wait = wait
        self._computer = computer
        
    def run(self, parameters={}, working_dir=None):
        time.sleep(self._wait)

        return self._my_results()

    def _my_results(self):
        return self.WaitResult(self._cores, self._wait)

    @property
    def results_filename(self):
        return "wait.fake.csv"

    @property
    def computer(self):
        return self._computer

    def __eq__(self, other):
        return self._cores == other._cores and self._wait == self._wait


class WaitJob(Job):
    def __init__(self, cores, wait, submit_delay=None):
        self._my_job = Wait(cores, wait)
        fake_param = {}
        super().__init__(self._my_job, cores, fake_param, submit_delay=submit_delay)
    
    def stub_get_results(self):
        return self._my_job._my_results()


class TestJob(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)
        self.job = WaitJob(2, 0.1)

    def test_cores(self):
        self.assertEqual(self.job.cores, 2)

    def test_run(self):
        self.assertEqual(self.job.run().stdout, "cores 2, wait 0.1")

    def test_get_job(self):
        self.assertEqual(self.job.task, Wait(2, 0.1))

    def test_job_delay(self):
        job = WaitJob(1, 0.1)
        self.assertEqual(job.submit_delay, 0)
        job = WaitJob(1, 0.1, 10)
        self.assertEqual(job.submit_delay, 10)


def model_func(x,y):
    return {"x":x, "y":y}

class TestRetrieveRestartResults(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)

    def test_retrieve_restart_results_no_file(self):
        model = PythonModel(model_func)
        sim = model.build_simulator(SolitaryState())
        with self.assertRaises(RuntimeError):
            _retrieve_restart_results(("workdir.1", "model1", "state1"), "nofile.json", sim)

    def test_retrieve_restart_results_python_model(self):
        from matcal.core.serializer_wrapper import matcal_save
        model = PythonModel(model_func)
        sim = model.build_simulator(SolitaryState())
        goal = convert_dictionary_to_data({"x":1, "y":2})
        matcal_save("test.joblib", goal)
        res = _retrieve_restart_results(("workdir.1", "model1", "state1"), "test.joblib", sim)
        self.assert_close_dicts_or_data(res.results_data, goal)


class TestJobDispatch:
    def __init__():
        pass
    
    class CommonTests(MatcalUnitTest):
        _use_threads = None

        def setUp(self) -> None:
            super().setUp(__file__)
            self._sim_1 = WaitJob(2, 0.1)
            self._sim_2 = WaitJob(5, 0.1)
            self._sim_3 = WaitJob(20, 0.1)
            self._sim_4 = WaitJob(4, 0.1)
                        
        def _make_job_key(self, wrk_dir_idx, model_idx, state_id):
            wrkdir = f"work_dir.{wrk_dir_idx}"
            os.mkdir(wrkdir)
            return (wrkdir, f"model_{model_idx}", f"state_{state_id}")

        def test_multi_core_process_pool_get_results(self):
            jobs = []
            goal = {}
            for idx, sim in enumerate([self._sim_1, self._sim_3]):
                key = self._make_job_key(idx, idx, idx)
                jobs.append([key, sim])
                goal[key] = sim.stub_get_results()

            max_cores = 20
            job_results = dispatch_jobs(jobs, max_cores, BatchRestartCSV(True))
            for key, val in goal.items():
                result = job_results[key]
                self.assertEqual(result.stdout, val.stdout)

        def test_serial_run(self):
            jobs = []
            goal = {}
            for idx, sim in enumerate([self._sim_1, self._sim_3]):
                key = self._make_job_key(idx, idx, idx)
                jobs.append([key, sim])
                goal[key] = sim.stub_get_results()

            max_cores = 20
            br = BatchRestartCSV(True)
            job_results = run_jobs_serial(jobs, br)
            for key, val in goal.items():
                result = job_results[key]
                self.assertEqual(result.stdout, val.stdout)


        def test_multi_core_process_too_many_cores(self):
            jobs = self._make_all_jobs()
            with self.assertRaises(TooManyJobCoresError):
                results = dispatch_jobs(jobs, 10,  BatchRestartCSV(True))

        def _make_all_jobs(self):
            jobs = []
            for idx, sim in enumerate([self._sim_1, self._sim_2, self._sim_3, self._sim_4]):
                key = self._make_job_key(idx, idx, idx)
                jobs.append([key, sim])
            return jobs

        def test_correct_dispatcher(self):
            max_cores = 10
            br = BatchRestartCSV(True)
            dispathcer = _create_job_dispatcher(max_cores, self._use_threads, br)
            self.assertEqual(isinstance(dispathcer.pool, ThreadPoolExecutor), self._use_threads)


        class DispatcherSpy(Dispatcher):

            class StubFuture:
                class StubFutureResult:
                    def __init__(self):
                        self.stdout=""
                        self.stderr=""
                        self.results = np.ones(5)
                        self.source_filename = 'stub.csv'
                        
                def __init__(self):
                    self._done = False

                def done(self):
                    return self._done

                def result(self):
                    return self.StubFutureResult()

                def stub_make_done(self):
                    self._done = True

            def _dispatch(self, job):
                return self.StubFuture()           

            def confirm_core_use(self, goal):
                return np.isclose(goal, self.current_core_use)

            def end_job(self, job_index):
                self._running_jobs[job_index][1].stub_make_done()
                self._wait()

        def _get_pool(self):
            if self._use_threads:
                return ThreadPoolExecutor
            else:
                return ProcessPoolExecutor


        def test_correct_increment_when_dispatched(self):
            jobs, dispatcher = self._setup_dispatcher()
            key, job = jobs[0]
            dispatcher.dispatch_job_when_available(key, job)
            self.assertTrue(dispatcher.confirm_core_use(2))
            key, job = jobs[2]
            dispatcher.dispatch_job_when_available(key, job)
            self.assertTrue(dispatcher.confirm_core_use(22))
            dispatcher.end_job(0)
            self.assertTrue(dispatcher.confirm_core_use(20))
            dispatcher.end_job(0)
            self.assertTrue(dispatcher.confirm_core_use(0))

        def _setup_dispatcher(self):
            max_cores = 100
            jobs = self._make_all_jobs()
            br = BatchRestartCSV(True)
            dispatcher = self.DispatcherSpy(max_cores, self._get_pool(), br)
            return jobs,dispatcher

        def test_can_detect_existing_jobs(self):
            jobs, disp = self._setup_dispatcher()
            self.assertFalse(disp._has_running_jobs())
            disp.dispatch_job_when_available(jobs[0][0], jobs[0][1])
            self.assertTrue(disp._has_running_jobs())
            disp.end_job(0)
            self.assertFalse(disp._has_running_jobs())

        def test_raise_error_when_reusing_dispatcher(self):
            jobs, dispatcher = self._setup_dispatcher()
            results = dispatcher.get_results_when_finished()
            with self.assertRaises(Dispatcher.ClosedDispatchError):
                dispatcher.dispatch_job_when_available(jobs[1][0], jobs[1][1])


class TestJobDispatchThreads(TestJobDispatch.CommonTests):
    _use_threads = True


class TestJobDispatchProcesses(TestJobDispatch.CommonTests):
    _use_threads = False

        
class JobPreparationTests(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)
        self._run_vars = None
        self.param = Parameter("a", 0, 10)
        self.parameter_collection = ParameterCollection("Test", self.param)
        self.state = SolitaryState()
        self.state_collection = StateCollection("Test", self.state)
        data_dict = {"X":[0,1,2,3,4], "Y":[0,1,4,9,16]}
        self.data = convert_dictionary_to_data(data_dict)
        self.data.set_state(self.state)
        self.data_collection = DataCollection("Test", self.data)
        self.obj = CurveBasedInterpolatedObjective("X", "Y")
        self.objective_collection = ObjectiveCollection("test", self.obj)
        self.results_file = os.path.join(self.get_current_files_path(__file__), 
                                         "test_reference", "multi_core_job_pool", 
                                         "results.csv")
        self.model = MockExecutableModel(self.results_file)
        self.objective_set = ObjectiveSet(self.objective_collection, self.data_collection, 
                                          self.state_collection)
        self.eval_set = StudyEvaluationSet(self.model, self.objective_set)

    def test_get_one_item_bill(self):
        self.eval_set.prepare_model_and_simulators('.')
        eval_sets = {self.model:self.eval_set}
        wrk_dir =  'wrk_dir_name'
        params = {}
        job_bill = prepare_parameter_evaluation_jobs(eval_sets, params, wrk_dir)
        self.assertEqual(len(job_bill), 1)
        key = job_bill[0][0]
        self.assertEqual(wrk_dir, key[0])
        self.assertEqual(self.model.name, key[1])
        self.assertEqual(self.state.name, key[2])
        self.assertIsInstance(job_bill[0][1], Job)

    def test_get_three_item_bill_from_two_eval_sets(self):
        s1 = State("ONE")
        s2 = State("TWO")
        model_1 = MockExecutableModel(self.results_file)
        model_2 = MockExecutableModel(self.results_file)

        data1 = deepcopy(self.data)
        data1.set_state(s1)

        data2 = deepcopy(self.data)
        data2.set_state(s2)

        data_collection = DataCollection("my data", data1, data2)

        obj_set_1 = ObjectiveSet(self.objective_collection, data_collection, 
                                 StateCollection('state', s1))
        eval_set_1 = StudyEvaluationSet(model_1, obj_set_1)

        obj_set_2 = ObjectiveSet(self.objective_collection, data_collection, 
                                 StateCollection('state', s1, s2))
        eval_set_2 = StudyEvaluationSet(model_2, obj_set_2)

        eval_set_1.prepare_model_and_simulators()
        eval_set_2.prepare_model_and_simulators()


class UseThreadsTests(MatcalUnitTest):

    def setUp(self):
        super().setUp(__file__)

    def test_do_not_use_threads(self):
        from matcal.core.multi_core_job_pool import _use_threads
        jobs = []
        jobs.append([("wordkir.1", "mod1", "state1"), WaitJob(1, 1, 0)])
        jobs.append([("wordkir.1", "mod2", "state1"), WaitJob(1, 1, 0)])
        jobs.append([("wordkir.1", "mod2", "state1"), WaitJob(10, 1, 0)])
        self.assertFalse(_use_threads(jobs, max_cores_available=10))

        jobs = [[("wordkir.1", "mod2", "state1"), WaitJob(5, 1, 0)]]
        jobs.append([("wordkir.1", "mod2", "state1"), WaitJob(6, 1, 0)])
        self.assertFalse(_use_threads(jobs, max_cores_available=10))
    
    def test_use_threads(self):
        from matcal.core.multi_core_job_pool import _use_threads
        jobs = []
        jobs.append([("wordkir.1", "mod1", "state1"), WaitJob(1, 1, 0)])
        jobs.append([("wordkir.1", "mod2", "state1"), WaitJob(1, 1, 0)])
        self.assertTrue(_use_threads(jobs, max_cores_available=10))
        jobs.append([("wordkir.1", "mod2", "state1"), WaitJob(9, 1, 0)])
        self.assertTrue(_use_threads(jobs, max_cores_available=10))
        jobs.append([("wordkir.1", "mod2", "state1"), WaitJob(9, 1, 0)])
        self.assertTrue(_use_threads(jobs, max_cores_available=1, always_use_threads=True))
        