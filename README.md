# Leveraging MARL techniques to speed up learning by decomposing big action spaces

### Master in Artificial Intelligence - Final thesis

This repository contains the code to execute and reproduce the thesis experiments.

## Structure

The repository contains four folders:

* *src*: Contains the source code.
* *requirements*: Contains two requirements files with the packages (and their versions) used to run the experiments with the PettingZoo and MuJoCo environments. 
* *configs*: Contains the configuration files to reproduce the results we have explained in the thesis report. Nonetheless, you cannot use those files to run the code, but you must manually copy each value to the corresponding YAML file inside `src/configs`.
* *gifs*: Contains GIFs with the rendering to show how each method solved the different environments. Nonetheless, we have not created one for all the results we have shown, but only for the best ones (i.e. those we have shown in the respective *Comparison* subsections).

## Usage

You can execute our code using terminal commands or an IDE. In the former case, you can specify the environment and the method to run as follows (assuming you are inside the main folder):

````shell
python src/main.py env_config_name=mujoco_multi algorithm_config_name=jad3
````

You can add extra arguments that will replace the desired parameter's value inside the YAML file of the method you have selected to execute. For example:

```shell
python src/main.py env_config_name=mujoco_multi algorithm_config_name=jad3 lr=0.0001 mixer=qmix
```

You can also modify the arguments defining the environment. The downside of using YAML is that we need to enclose them as a dictionary, with the name `env_args`, and specify the new values for the parameters we might want to change as a dictionary:

```shell
python src/main.py env_config_name=mujoco_multi algorithm_config_name=jad3 env_args={agent_conf:10x2}
```

However, you can modify the YAML files and run the terminal command without those extra arguments. You can even remove those defining the environment and the method and modify the variables `default_env_config_name` and `default_alg_config_name`inside `src/main.py`. Once you change those, you can also run the src/main.py file with an IDE like VisualStudio Code.

We have explained the most essential parameters in the text file `parameters_explanation` inside `src/configs`. We have defined the others in their respective YAML file. 

We do not recommend changing the type of layers the methods use. Methods working with continuous actions are implemented without recurrent layers, so we can ensure the code will work if you change it. We use those parameters to change how many networks you want to train rather than changing the type of layers. You can find the options in the  `__init__` files of `src/modules/agents`and `src/modules/critics`.

We do not recommend activating the environments in the `__init __` file inside `src/envs` that are deactivated. The most recent version of multi-agent MuJoCo, maintained by the people of [Farama](https://robotics.farama.org/envs/MaMuJoCo/), would probably work. The other does not work because it was an environment we wanted to try in the thesis, but we could not make it work (i.e. the one called `adaptive_optics`). Moreover, all the network architectures using CNNs were used in that environment and are currently useless. We have not removed all that code because we might want to try to work with them in the future.

Similarly, we also implemented a continuous version of TransfQMIX, a method we use in our thesis when experimenting with environments with discrete actions. It should work, but we have not checked it because we have not used in a long time. Again, we do not remove it because we might want to work with in the future.

