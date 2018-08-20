import logging
import multiprocessing

import numpy as np

class Config:
    # Dictionary of the config attributes and their default values
    _attributes = {
        'population_size': 1000,
        'distance_threshold': 3,
        'fitness_function': lambda neural_net: 1,  # TODO: implement a sensible default?
        'number_of_input_nodes': 256, #TODO: maybe another value
        'number_of_output_nodes': 6, #TODO: same ^
        'c1': 1,
        'c2': 2,
        'c3': 3,
        'weight_mutation_distribution': np.random.normal,               # weight to add in change_weight mutation
        'connection_weight_mutation_distribution': np.random.normal,    # weight to give in add_connection mutation
        'add_connection_probability': 0.3,                              # probability of add_connection mutation occurring      # TODO: think of default value
        'add_node_probability': 0.3,                                    # probability of add_node mutation occurring            # TODO: think of default value
        'change_weight_probability': 0.8,                               # probability of change_weight mutation occurring       # TODO: think of default value
        # probability of the chance that an inherited connection is disabled if it's disabled in either parent # TODO: think of a default value
        'inherit_disabled_connection_probability': 0.2,
        'processes_in_pool': multiprocessing.cpu_count(),
        # the process pool used in neat_map - INITIALIZED BY Config.__init__ BASED ON processes_in_pool
        'pool': None,
    }
    #TODO: maybe use None in _attributes for attributes with no default? (and then log all attributes with no default and no param given)
    def __init__(self, params_dict=None):
        if params_dict is not None and not isinstance(params_dict, dict):
            raise ValueError("The supplied params_dict is not a dictionary")

        # by iterating over _attributes, we ensure that all of the needed attributes exist and are set (by setattr).
        # get returns None if attribute_name does not exit.
        for attribute_name, _ in Config._attributes:
            self.__set(attribute_name, params_dict.get(attribute_name), Config._attributes.get(attribute_name))

        if params_dict:
            self.__log_missing_parameters(params_dict)
            self.__log_invalid_parameters(params_dict)

        self.__init_pool()

    #TODO: neat_map doesn't sound right. maybe a different name?
    def neat_map(self, func, iterable):
        if self.pool:
            return self.pool.map(func, iterable)
        else:
            return map(func, iterable)

    def __set(self, attribute_name, provided_value, default_value):
        assert attribute_name in Config._attributes

        # if no value supplied, set the provided default value to attribute
        if not provided_value:
            setattr(self, attribute_name, default_value)
            logging.info("Config attribute %s set to default value %s" % (attribute_name, default_value))
        else:
            setattr(self, attribute_name, provided_value)
            logging.info("Config attribute %s set to provided value %s" % (attribute_name, provided_value))

    def __init_pool(self):
        assert hasattr(self, 'pool')
        assert hasattr(self, 'processes_in_pool')
        if self.processes_in_pool <= 0:
            raise ValueError("#Processes in pool has to be  > 0. Argument value: %d", self.processes_in_pool)
        elif self.processes_in_pool == 1:
            self.pool = None
        else:  # self.processes_in_pool > 1
            self.pool = multiprocessing.Pool(self.processes_in_pool)
        return self.pool

    @staticmethod
    def __log_missing_parameters(params_dict):
        assert isinstance(params_dict, dict)

        missing_parameters = set(Config._attributes.keys()) - set(params_dict.keys())
        logging.info("The following parameters were not provided by user (and hence set to default): "
                     + str(missing_parameters))

    @staticmethod
    def __log_invalid_parameters(params_dict):
        assert isinstance(params_dict, dict)

        invalid_parameters = set(params_dict.keys()) - set(Config._attributes.keys())
        logging.info("The following parameters given by users are invalid (no attribute by that name exists for Config): "
                     + str(invalid_parameters))

