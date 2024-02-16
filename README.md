# InstanceDACSelector

This repository contains the code for running the SELECTOR methodology [1] on trajectory features extracted from reinforcement agents executed on a Sigmoid and CMAES Benchmark.

Several representations are obtained, using raw trajectories or catch22 features extracted from the actions and rewards observed during the execution.

The requirements.txt file contains a snapshot of the libraries in the environment used to execute the experiments.

The file run_sigmoid_AI.py is used to run selector on features extracted from the actions and/or instance features for the Sigmoid function
The file run_sigmoid_RAI.py is used to run selector on features extracted from the rewards, actions and instance features for the Sigmoid function
The file run_sigmoid_RI.py is used to run selector on features extracted from the rewards and/or instance features for the Sigmoid function
The file run_cmaes.py is used to run selector for the CMAES data.

All of the scripts expect an argument "run" which should be an integer representing the run id of the selector execution.
Which features will be used can be controlled using the use_params and rai variables within each script. The current script version is set to calculate all combinations used in our experiments.

[1] Gjorgjina Cenikj, Ryan Dieter Lang, Andries Petrus Engelbrecht, Carola Doerr, Peter Korošec, and Tome Eftimov. 2022. SELECTOR: selecting a representative benchmark suite for reproducible statistical comparison. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '22). Association for Computing Machinery, New York, NY, USA, 620–629. https://doi.org/10.1145/3512290.3528809
