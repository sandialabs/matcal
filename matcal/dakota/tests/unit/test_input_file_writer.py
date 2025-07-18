import os
from abc import ABC, abstractmethod

from matcal.core.parameters import Parameter, ParameterCollection
from matcal.core.tests.MatcalUnitTest import MatcalUnitTest

from matcal.dakota.input_file_writer import (DakotaEnvironment, 
                                             PythonInterfaceBlock, 
                                             DakotaModelBlock, 
                                             DEFAULT_FILENAME, 
                                             ContinuousDesignBlock, 
                                             UniformUncertainBlock, 
                                             GeneralGradientMethodType, 
                                             GradientResponseBlock, 
                                             LeastSquaresResponseBlock, 
                                             DakGradientKeys, 
                                             DakEnvKeys, DakVarKeys, 
                                             NumericalGradientBlock, 
                                             NongradientResponseBlock, 
                                             NongradientResidualsResponseBlock, 
                                             DakModelKeys,
                                             DakInterfaceKeys, 
                                             DakMethodKeys, 
                                             GeneralNongradientMethodType, 
                                             DakotaFileBase, 
                                             InputFileBlock, 
                                             ResponseBlock)


class EnvironmentTest(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)

    def test_default_environment_string(self):
        e = DakotaEnvironment()
        self.assertTrue(DakEnvKeys.tabular_data in e.lines)
        self.assertTrue(DakEnvKeys.output_file in e.lines)
        self.assertTrue(DakEnvKeys.error_file in e.lines)
        self.assertEqual(e.get_line_value(DakEnvKeys.output_file), 
                         DakotaEnvironment.default_values[DakEnvKeys.output_file])
        self.assertEqual(e.get_line_value(DakEnvKeys.error_file), 
                         DakotaEnvironment.default_values[DakEnvKeys.error_file])

    def test_restart_env_string(self):
        e = DakotaEnvironment()
        self.assertFalse(DakEnvKeys.read_restart in e.lines)
        self.assertFalse(DakEnvKeys.write_restart in e.lines)

        e.set_read_restart_filename("restart.rst")
        e.set_write_restart_filename("dakota.rst")

        self.assertTrue(DakEnvKeys.read_restart in e.lines)
        self.assertTrue(DakEnvKeys.write_restart in e.lines)
        self.assertEqual(e.get_line_value(DakEnvKeys.read_restart).strip('"'), 
                         "restart.rst")
        self.assertEqual(e.get_line_value(DakEnvKeys.write_restart).strip('"'), 
                         "dakota.rst")

        e.set_read_restart_filename("updated_restart.rst")
        e.set_write_restart_filename("updated_dakota.rst")
        self.assertEqual(e.get_line_value(DakEnvKeys.read_restart).strip('"'), 
                         "updated_restart.rst")
        self.assertEqual(e.get_line_value(DakEnvKeys.write_restart).strip('"'), 
                         "updated_dakota.rst")


class DakotaVariableBlockTest(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)
        
    def test_write_continuous_design_vars(self):
        
        var_block = ContinuousDesignBlock()
        
        param1 = Parameter("a", 0, 1, 0.1)
        param2 = Parameter("b", 0, 2, 0.2)
        pc = ParameterCollection("test", param1, param2)
        var_block.set_parameters(pc)
        self.assertEqual(var_block._title, f"{ContinuousDesignBlock.type} = 2")
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.scale_types).strip('"'), "auto")
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.descriptor).strip('"'), "a")
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.descriptor, index=2).strip('"'), "b")
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.initial_point), 0.1)
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.initial_point, index=2), 0.2)
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.lower_bounds), 0)
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.lower_bounds, index=2), 0)
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.upper_bounds), 1)
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.upper_bounds, index=2), 2)
         
    def test_write_uniform_uncertain_vars(self):
        var_block = UniformUncertainBlock()
        param1 = Parameter("a", 0, 1, 0.1, distribution="uniform_uncertain")
        param2 = Parameter("b", 0, 2, 0.2, distribution="uniform_uncertain")
        pc = ParameterCollection("test", param1, param2)
        var_block.set_parameters(pc)

        self.assertEqual(var_block._title, f"{UniformUncertainBlock.type} = 2")
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.descriptor).strip('"'), "a")
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.descriptor, index=2).strip('"'), "b")
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.initial_point), 0.1)
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.initial_point, index=2), 0.2)
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.lower_bounds), 0)
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.lower_bounds, index=2), 0)
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.upper_bounds), 1)
        self.assertEqual(var_block.get_line_value(
            DakVarKeys.upper_bounds, index=2), 2)


class TestResponseBlock(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)

    def test_gradient_response_block_no_objective(self):
        rb = GradientResponseBlock()
        with self.assertRaises(ValueError):
            rb.get_string()
   
    def test_gradient_response_block(self):
        rb = GradientResponseBlock()
        self.assertTrue(DakGradientKeys.no_hessians in rb.lines)
        num_grads_b = rb.get_subblock(NumericalGradientBlock.type)
        self.assertEqual(num_grads_b.get_line_value(DakGradientKeys.method_source), 
                         "dakota")
        self.assertEqual(num_grads_b.get_line_value(DakGradientKeys.interval_type), 
                         "forward")
        self.assertEqual(num_grads_b.get_line_value(DakGradientKeys.fd_step_size), 
                         5e-5)
        
        rb.set_number_of_expected_responses(5)
        self.assertEqual(rb.get_number_of_expected_responses(), 
                         5)
        
        self.assertTrue(isinstance(rb.get_string(), str))

    def test_least_squares_response_block(self):
        rb = LeastSquaresResponseBlock()
        self.assertTrue(DakGradientKeys.no_hessians in rb.lines)
        num_grads_b = rb.get_subblock(NumericalGradientBlock.type)
        self.assertEqual(num_grads_b.get_line_value(DakGradientKeys.method_source), 
                         "dakota")
        self.assertEqual(num_grads_b.get_line_value(DakGradientKeys.interval_type), 
                         "forward")
        self.assertEqual(num_grads_b.get_line_value(DakGradientKeys.fd_step_size), 
                         5e-5)
        
        rb.set_number_of_expected_responses(5)
        self.assertEqual(rb.get_number_of_expected_responses(), 
                         5)
        self.assertTrue(isinstance(rb.get_string(), str))

    def test_nongradient_response_block(self):
        rb = NongradientResponseBlock()
        self.assertTrue(DakGradientKeys.no_hessians in rb.lines)
        self.assertTrue(DakGradientKeys.no_gradients in rb.lines)
       
        rb.set_number_of_expected_responses(5)
        self.assertEqual(rb.get_number_of_expected_responses(), 
                         5)
        self.assertTrue(isinstance(rb.get_string(), str))

    def test_nongradient_residual_response_block(self):
        rb = NongradientResidualsResponseBlock()
        self.assertTrue(DakGradientKeys.no_hessians in rb.lines)
        self.assertTrue(DakGradientKeys.no_gradients in rb.lines)
       
        rb.set_number_of_expected_responses(5)
        self.assertEqual(rb.get_number_of_expected_responses(), 
                         5)
        self.assertTrue(isinstance(rb.get_string(), str))


class DakotaModelTest(MatcalUnitTest):

    def setUp(self) -> None:
        super().setUp(__file__)

    def test_write_dakota_model(self):
        dm = DakotaModelBlock()
        self.assertTrue(DakModelKeys.single in dm.subblocks)
        self.assertEqual(dm._title, DakotaModelBlock.type)
        self.assertTrue(isinstance(dm.get_string(), str))


class DakotaInterfaceBlockTest(MatcalUnitTest):
  
    def setUp(self) -> None:
        super().setUp(__file__)

    def test_interface_str(self):    
        ib = PythonInterfaceBlock()
        self.assertEqual(ib._title, PythonInterfaceBlock.type)
        self.assertTrue(DakInterfaceKeys.batch in ib.lines)
        driver_subblock = ib.get_subblock(DakInterfaceKeys.analysis_driver)
        self.assertTrue(DakInterfaceKeys.python in driver_subblock.lines)
        self.assertTrue(isinstance(ib.get_string(), str))

    def test_add_do_no_save_evaluation_cache(self):
        ib = PythonInterfaceBlock()
        self.assertFalse(DakInterfaceKeys.deactivate in ib.lines)
        ib.do_not_save_evaluation_cache()
        deactive_cache_line = ib.get_line(DakInterfaceKeys.deactivate_cache_name)
        values = deactive_cache_line.get_values()
        self.assertEqual(values[0], DakInterfaceKeys.deactivate)
        self.assertEqual(values[1], DakInterfaceKeys.eval_cache)
        self.assertTrue("=" not in deactive_cache_line.get_string())        
        self.assertTrue(isinstance(ib.get_string(), str))


class GeneralGradientMethodTest(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)
        
    def test_write_nl2sol_gradient_method(self):
        method = GeneralGradientMethodType()
        method.set_method_name('nl2sol')
        self.assertEqual(method._title, GeneralGradientMethodType.type)
        self.assertEqual(method.name, "nl2sol")
        self.assertTrue(DakMethodKeys.scaling in method.lines)
        self.assertTrue(DakMethodKeys.speculative in method.lines)
        self.assertEqual(method.get_line_value(DakMethodKeys.convergence_tol), 
                         1e-3)
        self.assertEqual(method.get_line_value(DakMethodKeys.max_func_evals), 
                         1e3)
        self.assertEqual(method.get_line_value(DakMethodKeys.max_iterations), 
                         1e2)
        
        method_str = method.get_string()
        self.assertTrue(isinstance(method_str, str))

class GeneralNongradientMethodTest(MatcalUnitTest):
    def setUp(self) -> None:
        super().setUp(__file__)
        
    def test_write_nl2sol_gradient_method(self):
        method = GeneralNongradientMethodType()
        method.set_method_name('optpp_pds')
        self.assertEqual(method.name, "optpp_pds")
        self.assertEqual(method._title, GeneralNongradientMethodType.type)
        
        self.assertTrue(DakMethodKeys.scaling in method.lines)
        self.assertEqual(method.get_line_value(DakMethodKeys.max_func_evals), 
                         1e4)
        self.assertEqual(method.get_line_value(DakMethodKeys.max_iterations), 
                         1e3)
        
        method_str = method.get_string()
        self.assertTrue(isinstance(method_str, str))


def get_param_collection(dist=Parameter.distributions.continuous_design):
    param_x = Parameter("x", 0, 10, 3, dist)
    param_y = Parameter("y", -5, 5, 0, dist)
    param_collection_cd_xy = ParameterCollection("xy", param_x, param_y)
    return param_collection_cd_xy


class TestDakotaFileBase:
    def __init__():
        pass
    class CommonTests(MatcalUnitTest, ABC):

        def setUp(self, file=__file__) -> None:
            super().setUp(file)
            self.df = self._dakota_file_class()
            self.pc = get_param_collection()

        @property
        @abstractmethod
        def _dakota_file_class(self):
            """"""

        def test_init(self):
            self.assertTrue(DakotaEnvironment.type in self.df.subblocks)
            self.assertTrue(DakotaModelBlock.type in self.df.subblocks)
            self.assertTrue(PythonInterfaceBlock.type in self.df.subblocks)
            self.assertTrue(DakotaModelBlock.type in self.df.subblocks)
            method_block = self.df.get_method_block()
            self.assertTrue(self.df._method_class.type in method_block.subblocks)

        def test_populate_variables(self):
            self.assertFalse(DakotaFileBase.Keywords.variables in self.df.subblocks)
            self.df.populate_variables(self.pc)
            self.assertTrue(DakotaFileBase.Keywords.variables in self.df.subblocks)
            vars_block = self.df.get_variables_block()
            self.assertTrue(ContinuousDesignBlock.type in vars_block.subblocks
                            or UniformUncertainBlock.type in vars_block.subblocks)

        def test_get_environment_block(self):  
            env_block = self.df.get_environment_block()
            self.assertIsInstance(env_block, DakotaEnvironment)

        def test_get_variables_block(self):
            with self.assertRaises(KeyError):  
                var_block = self.df.get_variables_block()
            self.df.populate_variables(self.pc)
            var_block = self.df.get_variables_block()

            self.assertIsInstance(var_block, InputFileBlock)

        def test_get_response_block(self):
            resp_block = self.df.get_response_block()
            self.assertIsInstance(resp_block, ResponseBlock)
        
        def test_get_interface_block(self):  
            interface_block = self.df.get_interface_block()
            self.assertIsInstance(interface_block, PythonInterfaceBlock)

        def test_number_of_expected_responses(self):
            self.df.set_number_of_expected_responses(10,10)
            resp_block = self.df.get_response_block()
            self.assertEqual(resp_block.get_number_of_expected_responses(), 10)

        def test_set_read_restart_filename(self):
            fn = "test.rst"
            with open(fn, "w") as f:
                f.write("\n")
            self.df.set_read_restart_filename(fn)
            env_block = self.df.get_environment_block()
            self.assertTrue(DakEnvKeys.read_restart in env_block.lines)
            self.assertEqual(self.df.get_read_restart_filename(), 
                             fn)

        def test_set_restart_filename(self):
            fn = "test.rst"
            with open(fn, "w") as f:
                f.write("\n")
            self.df.set_restart_filename(fn)
            env_block = self.df.get_environment_block()
            self.assertTrue(DakEnvKeys.write_restart in env_block.lines)
            self.assertEqual(self.df.get_write_restart_filename(), 
                             fn)
        
        def test_do_not_save_evaluation_cache(self):
            interface_block = self.df.get_interface_block()
            self.assertFalse(DakInterfaceKeys.deactivate_cache_name in 
                            interface_block.lines)
            self.df.do_not_save_evaluation_cache()
            self.assertTrue(DakInterfaceKeys.deactivate_cache_name in 
                            interface_block.lines)
        
        def test_set_output_verbosity(self):
            with self.assertRaises(TypeError):
                self.df.set_output_verbosity(1)
            with self.assertRaises(ValueError):
                self.df.set_output_verbosity("not_valid_verbosity")
            self.df.set_output_verbosity("normal")
            meth_block = self.df.get_method_block()
            self.assertEqual(meth_block.get_line_value(DakMethodKeys.output), 
                             "normal")
            
        def test_set_method_type_block_line(self):
            self.df.set_method_type_block_line("test", 1, 2, 3)
            meth_type_b = self.df.get_method_type_block()

            self.assertTrue("test" in meth_type_b.lines)
            self.assertEqual(meth_type_b.get_line_value("test"), 1)            
            self.assertEqual(meth_type_b.get_line_value("test", index=2), 2)            
            self.assertEqual(meth_type_b.get_line_value("test", index=3), 3)            
            self.df.set_method_type_block_line("test", 4)
            self.assertEqual(meth_type_b.get_line_value("test"), 4)
            
        def test_get_input_string(self):
            param_c = get_param_collection()
            self.df.populate_variables(param_c)
            self.df.set_number_of_expected_responses(10)
            
            input_str = self.df.get_input_string()
            self.assertIsInstance(input_str, str)

        def test_write_input_file(self):
            param_c = get_param_collection()
            self.df.populate_variables(param_c)
            self.df.set_number_of_expected_responses(10)
            self.df.write_input_file()

            self.assert_file_exists(DEFAULT_FILENAME)
