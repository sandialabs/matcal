from matcal.core.simulators import ExecutableSimulator
from matcal.core.computing_platforms import HPCComputingPlatform


class SierraSimulator(ExecutableSimulator):
    def __init__(self,
                 model_name,
                 compute_information,
                 results_information,
                 state,
                 full_input_file_path,
                 template_dir='.', epu_results=False, custom_commands=None,
                 check_input=False, check_syntax=False, model_constants=None):
        self._input_filename = full_input_file_path.split("/")[-1]
        self._epu_results = epu_results
        self._custom_commands = custom_commands
        self._check_input = check_input
        self._check_syntax = check_syntax
        self._compute_information = compute_information
        commands_list = self._get_commands()

        super().__init__(model_name,
                        compute_information,
                        results_information,
                        state, 
                        template_dir=template_dir, 
                        commands=commands_list, 
                        model_constants=model_constants)
        self._set_modules()

    def _pass_parameters_to_simulators(self, workdir_full_path, parameters):
        """"""

    def run(self,  parameters, **kwargs):
        if self._check_input:
            results = self.run_check_input(parameters, **kwargs)
        elif self._check_syntax:
            results = self.run_check_syntax(parameters, **kwargs)
        else:
            results = super().run(parameters, **kwargs)
        return results

    def run_check_syntax(self, parameters, **kwargs):
        return self._run_with_appended_command("--check-syntax", parameters, 
                                               get_results=False, **kwargs)   

    def run_check_input(self,  parameters, **kwargs):
        return self._run_with_appended_command( "--check-input",  parameters, 
                                               get_results=False, **kwargs)      

    def _run_with_appended_command(self, custom_command, *args, **kwargs):
        self._add_run_and_pre_sierra_options()
        self._commands.append(custom_command)
        results = super().run(*args, **kwargs)
        self._remove_run_and_pre_sierra_options()
        self._commands.remove(custom_command)
        return results
        
    def _remove_run_and_pre_sierra_options(self):
        self._commands.remove("--run")
        if self._compute_information.number_of_cores > 1:
            self._commands.remove("--pre")

    def _add_run_and_pre_sierra_options(self):
        self._commands.remove("--run")
        self._commands.insert(1, "--run")
        if self._compute_information.number_of_cores > 1:
            self._commands.insert(1, "--pre")

    def _set_modules(self):
        if self._compute_information.modules_to_load is None:
            self._modules_to_load = ['sierra']
        else:
            self._modules_to_load = self._compute_information.modules_to_load

    def _get_commands(self):
        commands = []
        commands.append('sierra')
        commands.append('--run')
        if self._compute_information.number_of_cores > 1 and self._epu_results:
            commands.append('--post')
        commands.append('-n')
        commands.append(str(self._compute_information.number_of_cores))

        if self._compute_information.computer.queues is not None:
            queue_commands = self._add_sierra_queue_commands()
            self._append_commands(commands, queue_commands)
        make_methods = [self._make_executable_commands, self._make_timeout_commands, 
            self._make_custom_commands]
        for command_maker in make_methods:
            new_commands = command_maker()
            self._append_commands(commands, new_commands)
        return commands
    
    def _make_timeout_commands(self):
        if self._compute_information.executable == 'aria':
            time_sec = self._compute_information.time_limit_seconds
            if time_sec is not None:
                return ["--graceful_timeout", str(int(time_sec))]
        else:
            return None

    def _make_custom_commands(self):
        return self._custom_commands

    def _make_executable_commands(self):
        exe_commands = []
        exe_commands.append('-a')
        exe_commands.append(self._compute_information.executable)
        exe_commands.append('-i')
        exe_commands.append(self._input_filename)
        return exe_commands

    def _append_commands(self, commands, new_commands):
        if new_commands is None:
            return commands
        else:
            commands.extend(new_commands)
            return commands 

    def _add_sierra_queue_commands(self):
        queue_commands = []
        computer = self._compute_information.computer
        queue_names = computer.get_usable_queue_names(self._compute_information.time_limit_seconds,
                                                      self._compute_information.number_of_cores)
        queue_commands.append('--queue-name')
        queue_commands.append(','.join(queue_names))
        queue_commands.append('-T')
        queue_commands.append(str(int(self._compute_information.time_limit_seconds)))
        queue_commands.append('--account')
        queue_commands.append(str(self._compute_information.queue_id))
        if isinstance(self._compute_information.computer, HPCComputingPlatform):
            processors_per_node = self._compute_information.computer.get_processors_per_node()
            if processors_per_node is not None:
                queue_commands.append('--ppn')
                queue_commands.append(f'{processors_per_node}')

        return queue_commands
