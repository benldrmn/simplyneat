# SimplyNEAT

An easily-configurable optimized implementation of the genetic algorithm NEAT.

## Getting Started
### Prerequisites

Mandatory: TensorFlow, NumPy
```
pip install TensorFlow
apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
```
Optional: gym
```
pip install gym
pip install gym[atari]
```

### Installing

1. Install SimplyNEAT

```
pip install simplyneat
```

2. Configure

```
from simplyneat.config.config import Config
from simplyneat.neat import Neat

# Make sure to set the mandatory parameters!
config = Config({'fitness_function': lambda x: 0, 'number_of_input_nodes': 3, 'number_of_output_nodes': 3})
neat = Neat(config)

```

3. Run

```
num_of_iterations = 20
statistics = neat.run(num_of_iterations)

print("Average fitness throught iterations: %s" % str(statistics[AVERAGE_FITNESS])
```

## Built With

* [TensorFlow] (https://www.tensorflow.org/)
* [NumPy] (http://www.numpy.org/)
* [OpenAI gym](https://gym.openai.com/)

## Authors

* **Ben Liderman** (https://github.com/benldrmn)
* **Liron Bronfman** (https://github.com/lironbro)

