import logging
import multiprocessing
from enum import Enum

import numpy as np

class LoggingLevel(Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10

#TODO: don't allow lambdas
class Config:
    # Dictionary of the config attributes and their default values
    # None means no default value, and should be set by the user, unless stated otherwise
    _attributes = {
        # have to be set by the user
        'fitness_function': None,
        'number_of_input_nodes': None,
        'number_of_output_nodes': None,

        # have default values, don't need to be set
        'population_size': 1000,
        'elite_group_size': 50,                                         # number of members who always pass to next generation

        'compatibility_threshold': 6,
        'excess_coefficient': 2,            # TODO: find values and give meaningful documentation
        'disjoint_coefficient': 2,
        'weight_difference_coefficient': 1,
        'change_weight_mutation_distribution': np.random.normal,               # weight to add in change_weight mutation               # TODO: check against paper
        'connection_weight_mutation_distribution': np.random.normal,    # weight to give in add_connection mutation
        'add_connection_probability': 0.3,                              # probability of add_connection mutation occurring      # TODO: think of default value
        'add_node_probability': 0.3,                                    # probability of add_node mutation occurring            # TODO: think of default value
        'change_weight_probability': 0.8,                               # probability of change_weight mutation occurring       # TODO: think of default value
        'reenable_connection_probability': 0.2,                         # probability of re-enable mutation occuring            # TODO: think of default value
        'max_tournament_size': 50,
        # probability of the chance that an inherited connection is disabled if it's disabled in either parent # TODO: think of a default value
        'inherit_disabled_connection_probability': 0.2,
        'processes_in_pool': multiprocessing.cpu_count(),
        'logging_level': LoggingLevel.INFO,

    }

    def __init__(self, params_dict=None):
        #TODO: check values are ok for all attributes
        if params_dict is not None and not isinstance(params_dict, dict):
            raise ValueError("The supplied params_dict is not a dictionary")

        # by iterating over _attributes, we ensure that all of the needed attributes exist and are set (by setattr).
        # get returns None if attribute_name does not exit.
        for attribute_name in Config._attributes.keys():
            self._set(attribute_name, params_dict.get(attribute_name), Config._attributes.get(attribute_name))

        self._init_logger()

        logging.info("NEAT Parameters:")
        self._log_parameters_value()
        # if params_dict:
        #     self._log_missing_parameters(params_dict)
        #     self._log_invalid_parameters(params_dict)

    def _init_logger(self):
        if self.logging_level == LoggingLevel.DEBUG:
            logging.getLogger().setLevel(logging.DEBUG)
        elif self.logging_level == LoggingLevel.INFO:
            logging.getLogger().setLevel(logging.INFO)
        elif self.logging_level == LoggingLevel.WARNING:
            logging.getLogger().setLevel(logging.WARNING)
        elif self.logging_level == LoggingLevel.ERROR:
            logging.getLogger().setLevel(logging.ERROR)
        elif self.logging_level == LoggingLevel.CRITICAL:
            logging.getLogger().setLevel(logging.CRITICAL)
        else:
            raise ValueError("Invalid LoggingLevel")

    def _set(self, attribute_name, provided_value, default_value):
        assert attribute_name in Config._attributes

        # if no value supplied, set the provided default value to attribute
        if provided_value is None:
            setattr(self, attribute_name, default_value)
        else:
            setattr(self, attribute_name, provided_value)
    #TODO: REMOVE COMMENTS
    # def _log_missing_parameters(self, params_dict):
    #     assert isinstance(params_dict, dict)
    #
    #     missing_parameters = set(Config._attributes.keys()) - set(params_dict.keys())
    #     logging.info("The following parameters were not provided by user (and hence set to default):")
    #     self._log_parameters_value(missing_parameters)
    #
    # def _log_invalid_parameters(self, params_dict):
    #     assert isinstance(params_dict, dict)
    #
    #     invalid_parameters = set(params_dict.keys()) - set(Config._attributes.keys())
    #     logging.info("The following parameters given by users are invalid (no attribute by that name exists for Config):")
    #     self._log_parameters_value(invalid_parameters)

    def _log_parameters_value(self):
        for parameter in Config._attributes.keys():
            logging.info(str(parameter) + ": " + str(getattr(self, parameter)))
