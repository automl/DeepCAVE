class AbstractEvaluator(object):
    """
    Abstract implementation of Importance evaluator
    """
    def __init__(self, scenario: Scenario,
                 cs: ConfigurationSpace,
                 model: RandomForestWithInstances,
                 to_evaluate: int, rng,
                 verbose: bool=True,
                 **kwargs):
        self._logger = None
        self.scenario = scenario
        self.cs = cs
        self.model = model  # SMAC model
        self.rng = rng
        self.verbose = verbose

        if self.model is not None:
            if 'X' in kwargs and 'y' in kwargs:
                self._train_model(kwargs['X'], kwargs['y'], **kwargs)
            if 'features' in kwargs:
                self.features = kwargs['features']
            else:
                self.features = self.model.instance_features

            self.X = self.model.X
            self.y = self.model.y
            self.types = self.model.types
            self.bounds = self.model.bounds
        self._to_eval = to_evaluate
        if to_evaluate <= 0:
            self.to_evaluate = len(self.cs.get_hyperparameters())
        elif to_evaluate >= len(self.cs.get_hyperparameters()):
            self.to_evaluate = len(self.cs.get_hyperparameters())
        else:
            self.to_evaluate = to_evaluate  # num of parameters to evaluate

        self.evaluated_parameter_importance = OrderedDict()
        self.name = 'Base'

        self.IMPORTANCE_THRESHOLD = 0.05
        self.AXIS_FONT = {'family': 'monospace'}
        self.LABEL_FONT = {'family': 'sans-serif'}
        self.LINE_FONT = {'lw': 4,
                          'color': (0.125, 0.125, 0.125)}
        self.area_color = (0.25, 0.25, 0.45)
        self.unimportant_area_color = (0.125, 0.125, 0.225)
        self.MAX_PARAMS_TO_PLOT = 15

    @abc.abstractclassmethod
    def run(self) -> OrderedDict:
        raise NotImplementedError

    @abc.abstractclassmethod
    def plot_result(self, name=None):
        raise NotImplementedError

    def _train_model(self, X, y, **kwargs):
        self.model.train(X, y, **kwargs)

    def __str__(self):
        tmp = 'Parameter Importance Evaluation Method %s\n' % self.name
        tmp += '{:^15s}: {:<8s}\n'.format('Parameter', 'Value')
        for key in self.evaluated_parameter_importance:
            value = self.evaluated_parameter_importance[key]
            tmp += '{:>15s}: {:<3.4f}\n'.format(key, value)
        return tmp

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = logging.getLogger(value)

    def _refit_model(self, types, bounds, X, y):
        """
        Easily allows for refitting of the model.
        Parameters
        ----------
        types: list
            SMAC EPM types
        X:ndarray
            X matrix
        y:ndarray
            corresponding y vector
        """
        # We need to fake config-space bypass imputation of inactive values in random forest implementation
        fake_cs = ConfigurationSpace(name="fake-cs-for-configurator-footprint")
        # We need to add fake hyperparameters
        fake_cs.add_hyperparameters([
            UniformFloatHyperparameter('fake-%s' % i, lower=0., upper=100000., default_value=0.,
                                       log=False) for i in range(len(types))
        ])

        self.model = RandomForestWithInstances(fake_cs, types, bounds, seed=12345, do_bootstrapping=True)
        self.model.rf_opts.compute_oob_error = True
        self.model.train(X, y)