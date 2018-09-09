# SimplyNEAT

An easily-configurable optimized implementation of the genetic algorithm NEAT.

## Getting Started
### Prerequisites

Mandatory: Theano, scipy
```
pip install Theano
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

* [Theano] (http://deeplearning.net/software/theano/)
* [OpenAI gym](https://gym.openai.com/)

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Ben Liderman** (https://github.com/benldrmn)
* **Liron Bronfman** (https://github.com/lironbro)


See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

