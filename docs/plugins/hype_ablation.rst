Ablation (HyperSHAP)
====================

Thisn is a method to analyze the performance contributions of individual hyperparameter values (with ablation) using HyperSHAP, 
a game-theoretic framework based on Shapley values and interactions. 
Rather than evaluating a single sequential path of hyperparameter changes, this method calculates the performance contribution 
of each hyperparameter across all possible combinations of a default and incumbent configuration.

This plugin is capable of answering following questions:

* Which hyperparameter changes, when applied to a baseline, lead to the greatest actual improvement in the optimized configuration's performance?
* Which hyperparameters contribute little to no benefit, or even degrade performance, compared to their default baseline values?

To learn more about the underlying game-theoretic framework, please see the paper
`HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization
<https://arxiv.org/abs/2502.01276>`_.


Options
-------
* **Objective**: Choose the objective you wish to calculate the HyperSHAP values for.

* **Budgets**: Filter the results to view ablation scores specific to certain multi-fidelity budgets, allowing you to analyze how the importance of specific configuration changes varies with budget changes.