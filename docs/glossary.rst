Glossary
========

.. glossary::

    Hyperparameter
        Hyperparameters are settings or configurations external to the model that govern the
        learning process and model structure. Unlike parameters learned from the data
        (such as weights in neural networks), hyperparameters need to be set before training and
        can significantly impact model performance. Examples include learning rate, batch size,
        and the number of layers in a neural network.

    Configuration
        A configuration refers to a specific set of hyperparameter values or settings used in an
        algorithm or model. The choice of configuration can significantly affect the performance
        and behavior of the algorithm. Optimization processes involve exploring various
        configurations to identify the most effective ones.

    Objective
        An objective is a metric that is to be minimized or maximized. For example, accuracy or loss
        are objectives, whereas accuracy should be maximized and loss be minimized.

    Budget
        In the context of optimization, a budget represents a constraint on computational resources
        or time allocated to the optimization process. It often refers to limits such as the number
        of iterations, training epochs in neural networks, or the size of data subsets used during
        experimentation.

    BO (Bayesian Optimization)
        Bayesian Optimization is a probabilistic model-based optimization technique used for
        optimizing objective functions that are expensive to evaluate. It balances exploration and
        exploitation to find the minimum of its objective.

    Incumbent
        The incumbent is the best-performing configuration or set of hyperparameter values identified
        up to a given point in the optimization process. It represents the current best solution
        based on the objective metric being optimized.