import logging

from enum import Enum
from functools import total_ordering

import numpy as np
import multiprocessing

#TODO: config tf.logging
@total_ordering
class LoggingLevel(Enum):
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        else:
            return NotImplemented


#TODO: don't allow lambdas
class Config:
    # Dictionary of the config attributes and their default values
    # None means no default value, and should be set by the user, unless stated otherwise
    _attributes = {
        # have to be set by the user
        #TODO: according to NEAT, fitness function must return only positive values
        'fitness_function': None,
        'number_of_input_nodes': None,
        'number_of_output_nodes': None,

        # have default values, don't need to be set
        'population_size': 1000,
        'elite_group_size': 50,                                         # number of members who always pass to next generation

        'compatibility_threshold': 6.0,
        'excess_coefficient': 2.0,            # TODO: find values and give meaningful documentation
        'disjoint_coefficient': 2.0,
        'weight_difference_coefficient': 1.0,
        'change_weight_mutation_distribution': np.random.normal,               # weight to add in change_weight mutation               # TODO: check against paper
        'connection_weight_mutation_distribution': np.random.normal,    # weight to give in add_connection mutation
        'add_connection_probability': 0.2,                              # probability of add_connection mutation occurring      # TODO: think of default value
        'add_node_probability': 0.2,                                    # probability of add_node mutation occurring            # TODO: think of default value
        'change_weight_probability': 0.1,                               # probability of change_weight mutation occurring       # TODO: think of default value
        #TODO: change it to enable\disable mutation (meteg)
        'reenable_connection_probability': 0.05,                         # probability of re-enable mutation occuring            # TODO: think of default value
        # probability of the chance that an inherited connection is disabled if it's disabled in either parent # TODO: think of a default value
        #TODO: not actually implemented (the inherit attrib) - implement
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

    def _log_parameters_value(self):
        for parameter in Config._attributes.keys():
            logging.info(str(parameter) + ": " + str(getattr(self, parameter)))
