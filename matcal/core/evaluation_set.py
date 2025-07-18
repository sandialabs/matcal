from collections import OrderedDict
import os

from matcal.core.state import StateCollection


class StudyEvaluationSet:
    class ZeroObjectiveError(Exception):
        def __init__(self, *args):
            super().__init__(*args)

    class InputError(RuntimeError):
        def __init__(self, *args):
            super().__init__(*args)

    def __init__(self,  model, objective_set):
        self._model = model
        self._residual_vector_length = None
        self._number_of_objectives = None
        self._simulators = OrderedDict()
        self._template_directory = os.getcwd()
        self._objective_sets = []
        self._objective_names = []
        self._objective_auto_named_count = 0
        self._state_collection = StateCollection("eval set active states")

        self.add_objective_set(objective_set)
        self._determine_active_states()
        self._set_residual_and_objective_sizes()

    def _determine_active_states(self):
        for objective_set in self._objective_sets:
            objective_states = objective_set.states
            for state in objective_states.values():
                if state not in self._state_collection.values():
                    self._state_collection.add(state)

    def add_objective_set(self, new_objective_set):
        self._check_for_repeated_objectives(new_objective_set)
        self._objective_sets.append(new_objective_set)
        self._determine_active_states()
        self._set_residual_and_objective_sizes()

    def _check_for_repeated_objectives(self, objective_set):
        for new_objective in objective_set.objectives.values():
            for objective_set in self._objective_sets:
                for objective in objective_set.objectives.values():
                    if objective == new_objective:
                        raise self.InputError(f"The objective \"{objective.name}\" was already "
                                    f"added to the study for model \"{self.model.name}\"."
                                    f" Only unique objectives"
                                    " can be added for each model.")

            self._check_for_repeated_objective_names(new_objective.name)

    def _check_for_repeated_objective_names(self, objective_name):
        if objective_name in self._objective_names:
            raise self.InputError("All objectives applied to a "
                                  " model must have unique names. "
                                f"The objective name \"{objective_name}\" is "
                                f"already being used. "
                                " The following objective names "
                                 f"are being applied to the model \"{self._model.name}\": "
                                 f"\n{self._objective_names}" )
        else:
            self._objective_names.append(objective_name)

    def get_cores_required(self):
        total_cores = 0
        for state in self._state_collection.values():
            total_cores += self.model.number_of_local_cores
        return total_cores

    def _set_residual_and_objective_sizes(self):
        residual_vector_length = 0
        number_of_objectives = 0
        for objective_set in self._objective_sets:
            number_of_objectives += objective_set.number_of_objectives
            residual_vector_length += objective_set.residual_vector_length

        if number_of_objectives == 0:
            raise self.ZeroObjectiveError

        self._number_of_objectives = number_of_objectives
        self._residual_vector_length = residual_vector_length

    def evaluate_objectives(self, simulation_results_data_collection):
        eval_set_results = OrderedDict()
        eval_set_qois = OrderedDict()
        for objective_set in self._objective_sets:
            obj_set_results, obj_set_qois = objective_set.calculate_objective_set_results(simulation_results_data_collection)
            eval_set_results.update(obj_set_results)
            eval_set_qois.update(obj_set_qois)

        return eval_set_results, eval_set_qois

    def prepare_model_and_simulators(self, template_dir=None, restart=False):
        for state in self._state_collection.values():
            if not restart:
                self._model.preprocess(state, template_dir)
            if state not in self._simulators.keys():
                state_sim = self._model.build_simulator(state)
                self._simulators[state] = state_sim

    def get_objective_names(self):
        names = []
        for obj_set in self._objective_sets:
            names += obj_set.get_objective_names()

        return names

    @property
    def residual_vector_length(self):
        return self._residual_vector_length

    @property
    def number_of_objectives(self):
        return self._number_of_objectives

    @property
    def model(self):
        return self._model

    @property
    def states(self):
        return self._state_collection

    @property
    def simulators(self):
        return self._simulators

    @property
    def objective_sets(self):
        return  self._objective_sets