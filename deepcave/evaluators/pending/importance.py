from typing import Dict, List, Tuple, Union

import glob
import json
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
from pimp.configspace import (
    CategoricalHyperparameter,
    Configuration,
    impute_inactive_values,
)
from pimp.epm.base_epm import RandomForestWithInstances
from pimp.epm.unlogged_epar_x_rfwi import UnloggedEPARXrfi
from pimp.epm.unlogged_rfwi import Unloggedrfwi
from pimp.evaluator.ablation import Ablation
from pimp.evaluator.fanova import fANOVA
from pimp.evaluator.forward_selection import AbstractEvaluator, ForwardSelector
from pimp.evaluator.influence_models import InfluenceModel
from pimp.evaluator.local_parameter_importance import LPI
from pimp.utils import RunHistory, RunHistory2EPM4Cost, RunHistory2EPM4LogCost, Scenario
from smac.epm.rfr_imputator import RFRImputator
from smac.epm.util_funcs import get_types
from smac.tae import StatusType
from smac.utils.io.traj_logging import TrajLogger
from tqdm import tqdm

__author__ = "Andre Biedenkapp"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Andre Biedenkapp"
__email__ = "biedenka@cs.uni-freiburg.de"


class Importance(object):
    def __init__(self, scenario_file: Union[None, str] = None, scenario: Union[None, Scenario] = None,
                 runhistory_file: Union[str, None] = None, runhistory: Union[None, RunHistory] = None,
                 traj_file: Union[None, List[str]] = None, incumbent: Union[None, Configuration] = None,
                 seed: int = 12345, parameters_to_evaluate: int = -1, margin: Union[None, float] = None,
                 save_folder: str = 'PIMP', impute_censored: bool = False, max_sample_size: int = -1,
                 fANOVA_cut_at_default=False, fANOVA_pairwise=True, forwardsel_feat_imp=False,
                 incn_quant_var=True, preprocess=False, forwardsel_cv=False, verbose: bool=True):
        """
        Importance Object. Handles the construction of the data and training of the model. Easy interface to the
        different evaluators.
        :param scenario_file: File to load the scenario from, if scenario is None.
        :param scenario: Scenario Object to use if scenario_file is None
        :param runhistory_file: File to load the runhistory from if runhistory is None.
        :param runhistory: Runhistory Object to use if runhistory_file is None.
        :param traj_file: File to load the trajectory from. If this is None but runhistory_file was specified,
               the trajectory will be read from the same directory as the runhistory. (Will be ignored if incumbent is
               set)
        :param incumbent: Configuration Object to use.
        :param seed: Seed used for the numpy random generator.
        :param parameters_to_evaluate: int that specifies how many parameters have to be evaluated.
               If set to -1 all parameters will be evaluated.
        :param margin: float used in conjunction with influence models. Is the minimal improvement to accept a
                       parameter as important.
        :param save_folder: Folder name to save the output to
        :param impute_censored: boolean that specifies if censored data should be imputed. If not, censored data are
               ignored.
        :param verbose: Toggle output to stdout (not logging, but tqdm-progress bars)
        """
        self.logger = logging.getLogger("pimp.Importance")
        self.rng = np.random.RandomState(seed)
        self._parameters_to_evaluate = parameters_to_evaluate
        self._evaluator = None
        self.margin = margin
        self.threshold = None
        self.seed = seed
        self.impute = impute_censored
        self.cut_def_fan = fANOVA_cut_at_default
        self.pairiwse_fANOVA = fANOVA_pairwise
        self.forwardsel_feat_imp = forwardsel_feat_imp
        self.incn_quant_var = incn_quant_var
        self.preprocess = preprocess
        self._preprocessed = False
        self.X_fanova = None
        self.y_fanova = None
        self.forwardsel_cv = forwardsel_cv
        self.verbose = verbose

        self.evaluators = []

        self._setup_scenario(scenario, scenario_file, save_folder)
        self._load_runhist(runhistory, runhistory_file)
        self._setup_model()
        self.best_dir = None
        self._load_incumbent(traj_file, runhistory_file, incumbent)
        self.logger.info('Best incumbent found in %s' % self.best_dir)
        if 0 < max_sample_size < len(self.X):
            self.logger.warning('Reducing the amount of datapoints!')
            if self.best_dir:
                self.logger.warning('Only using the runhistory that contains the incumbent!')
                self.logger.info('Loading from %s' % self.best_dir)
                self._load_runhist(None, os.path.join(self.best_dir, '*history*'))
                self._convert_data(fit=False)
                self._load_incumbent(glob.glob(os.path.join(self.best_dir, '*traj_aclib2*'), recursive=True)[0], None,
                                     incumbent)
                if max_sample_size < len(self.X):
                    self.logger.warning('Also downsampling as requested!')
                    idx = list(range(len(self.X)))
                    np.random.shuffle(idx)
                    idx = idx[:max_sample_size]
                    self.X = self.X[idx]
                    self.y = self.y[idx]
            else:
                self.logger.warning('Downsampling as requested!')
                idx = list(range(len(self.X)))
                np.random.shuffle(idx)
                idx = idx[:max_sample_size]
                self.X = self.X[idx]
                self.y = self.y[idx]
            self.logger.info('Remaining %d datapoints' % len(self.X))
            self.model.train(self.X, self.y)

    def _preprocess(self, runhistory):
        """
        Method to marginalize over instances such that fANOVA can determine the parameter importance without
        having to deal with instance features.
        :param runhistory: RunHistory that knows all configurations that were run. For all these configurations
                           we have to marginalize away the instance features with which fANOVA will make it's
                           predictions
        """
        self.logger.info('PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING PREPROCESSING')
        self.logger.info('Marginalizing away all instances!')
        configs = runhistory.get_all_configs()
        X_non_hyper, X_prime, y_prime = [], [], []
        for c_id, config in tqdm(enumerate(configs), ascii=True, desc='Completed: ', total=len(configs)):
            config = impute_inactive_values(config).get_array()
            X_prime.append(config)
            X_non_hyper.append(config)
            y_prime.append(self.model.predict_marginalized_over_instances(np.array([X_prime[-1]]))[0].flatten())
            for idx, param in enumerate(self.scenario.cs.get_hyperparameters()):
                if not isinstance(param, CategoricalHyperparameter):
                    X_non_hyper[-1][idx] = param._transform(X_non_hyper[-1][idx])
        X_non_hyper = np.array(X_non_hyper)
        X_prime = np.array(X_prime)
        y_prime = np.array(y_prime)
        # y_prime = np.array(self.model.predict_marginalized_over_instances(X_prime)[0])
        self.X = X_prime
        self.X_fanova = X_non_hyper
        self.y_fanova = y_prime
        self.y = y_prime
        self.logger.info('Size of training X after preprocessing: %s' % str(self.X.shape))
        self.logger.info('Size of training y after preprocessing: %s' % str(self.y.shape))
        self.logger.info('Finished Preprocessing')
        self._preprocessed = True

    def _setup_scenario(self, scenario: Union[None, Scenario], scenario_file: Union[None, str], save_folder: str) -> \
            None:
        """
        Setup for the scenario
        Helper method to have the init method less cluttered.
        For parameter specifications, see __init__
        """
        if scenario is not None:
            self.scenario = scenario
        elif scenario_file is not None:
            self.logger.info('Reading Scenario file and files specified in the scenario')
            self.scenario = Scenario(scenario=scenario_file)
            self.scenario.output_dir = save_folder
            self.scenario.output_dir_for_this_run = save_folder
            written = self.scenario.out_writer.write_scenario_file(self.scenario)
        else:
            raise Exception('Either a scenario has to be given or a file to load it from! Both were set to None!')

    def _load_incumbent(self, traj_file, runhistory_file, incumbent, predict_best=True) -> None:
        """
        Handles the loading of the incumbent according to the given parameters.
        Helper method to have the init method less cluttered.
        For parameter specifications, see __init__
        """
        self.incumbent = (None, None)
        if incumbent is not None:
            self.incumbent = incumbent
        elif traj_file is not None:
            self.logger.info('Reading traj_file: %s' % traj_file)
            self.incumbent = self._read_traj_file(traj_file)[0]
            self.logger.debug('Incumbent %s' % str(self.incumbent))
        elif traj_file is None and runhistory_file is not None:
            traj_files = os.path.join(os.path.dirname(runhistory_file), 'traj_aclib2.json')
            traj_files = sorted(glob.glob(traj_files, recursive=True))
            incumbents = []
            for traj_ in traj_files:
                self.logger.info('Reading traj_file: %s' % traj_)
                incumbents.append(self._read_traj_file(traj_))
                incumbents[-1].extend(self._model.predict_marginalized_over_instances(
                    np.array([impute_inactive_values(incumbents[-1][0]).get_array()])))
                self.logger.debug(incumbents[-1])
            sort_idx = 2 if predict_best else 1
            incumbents = sorted(enumerate(incumbents), key=lambda x: x[1][sort_idx])
            self.best_dir = os.path.dirname(traj_files[incumbents[0][0]])
            self.incumbent = incumbents[0][1][0]
            self.logger.info('Incumbent %s' % str(self.incumbent))
        else:
            raise Exception('No method specified to load an incumbent. Either give the incumbent directly or specify '
                            'a file to load it from!')

    def _setup_model(self) -> None:
        """
        Sets up all the necessary parameters used for the model.
        Helper method to have the init method less cluttered.
        For parameter specifications, see __init__
        """
        self.logger.info('Converting Data and constructing Model')
        self.X = None
        self.y = None
        self.types = None
        self.bounds = None
        self._model = None
        self.logged_y = False
        self._convert_data(fit=True)
        if self.preprocess:
            self._preprocess(self.runhistory)
            if self.scenario.run_obj == "runtime":
                self.y = np.log10(self.y)
            self.model = 'urfi'
            self.model.train(self.X, self.y)

    def _load_runhist(self, runhistory, runhistory_file) -> None:
        """
        Handels loading of the runhistory/runhistories.
        Helper method to have the init method less cluttered.
        For parameter specifications, see __init__
        """
        self.logger.debug(runhistory_file)
        self.logger.debug(runhistory)
        if runhistory is not None:
            self.runhistory = runhistory
        elif runhistory_file is not None:
            self.logger.info('Reading Runhistory')
            self.runhistory = RunHistory()

            globed_files = glob.glob(runhistory_file)
            self.logger.info('#RunHistories found: %d' % len(globed_files))
            if not globed_files:
                self.logger.error('No runhistory files found!')
                sys.exit(1)
            self.runhistory.load_json(globed_files[0], self.scenario.cs)
            for rh_file in globed_files[1:]:
                self.runhistory.update_from_json(rh_file, self.scenario.cs)
        else:
            raise Exception('Either a runhistory or files to load them from have to be specified! Both were set to '
                            'None!')
        self.logger.info('Combined number of Runhistory data points: %d' % len(self.runhistory.data))
        self.logger.info('Number of Configurations: %d' % (len(self.runhistory.get_all_configs())))

    def _read_traj_file(self, fn):
        """
        Simple method to read in a trajectory file in the json format / aclib2 format
        :param fn:
            file name
        :return:
            tuple of (incumbent [Configuration], incumbent_cost [float])
        """
        if not (os.path.exists(fn) and os.path.isfile(fn)):  # File existence check
            raise FileNotFoundError('File %s not found!' % fn)
        with open(fn) as fp:
            # In aclib2, the incumbent is a list of strings, in alljson it's a dictionary.
            fileformat = 'aclib2' if isinstance(json.loads(fp.readline())["incumbent"], list) else 'alljson'

        if fileformat == "aclib2":
            self.logger.info("Format is 'aclib2'. This format has issues with recovering configurations properly. We "
                             "recommend to use the alljson-format.")
            traj = TrajLogger.read_traj_aclib_format(fn, self.scenario.cs)
        else:
            traj = TrajLogger.read_traj_alljson_format(fn, self.scenario.cs)

        incumbent_cost = traj[-1]['cost']
        incumbent = traj[-1]['incumbent']
        return [incumbent, incumbent_cost]

    @property
    def model(self):
        return self._model

    def _get_types(self, scenario, features):
        types, bounds = get_types(scenario, features)
        types = np.array(types, dtype='uint')
        bounds = np.array(bounds, dtype='object')
        return types, bounds

    @model.setter
    def model(self, model_short_name='urfi'):
        if model_short_name not in ['urfi', 'rfi']:
            raise ValueError('Specified model %s does not exist or not supported!' % model_short_name)
        elif model_short_name == 'rfi':
            self.types, self.bounds = self._get_types(self.scenario.cs, self.scenario.feature_array)
            self._model = RandomForestWithInstances(self.scenario.cs, self.types, self.bounds, 12345,
                                                    instance_features=self.scenario.feature_array,
                                                    logged_y=self.logged_y)
        elif model_short_name == 'urfi':
            self.logged_y = True
            if not self._preprocessed:
                self.types, self.bounds = self._get_types(self.scenario.cs, self.scenario.feature_array)
                self._model = UnloggedEPARXrfi(self.scenario.cs, self.types, self.bounds, 12345,
                                               instance_features=self.scenario.feature_array,
                                               cutoff=self.cutoff, threshold=self.threshold,
                                               logged_y=self.logged_y)
            else:
                self.types, self.bounds = self._get_types(self.scenario.cs, None)
                self._model = Unloggedrfwi(self.scenario.cs, self.types, self.bounds, 12345,
                                           instance_features=None,
                                           logged_y=self.logged_y)
        self._model.rf_opts.compute_oob_error = True

    @property
    def evaluator(self) -> AbstractEvaluator:
        """
        Getter of the evaluator property. Returns the set evaluation method.
        :return: AbstractEvaluator
        """
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluation_method: str) -> None:
        """
        Setter of the evaluator property. The wanted evaluation method can be specified as string and the rest is
        handled automatically here
        :param evaluation_method: Name of the evaluation method to use
        :return: None
        """
        if self._model is None:
            self._setup_model()
        self.logger.info('Setting up Evaluation Method')
        if evaluation_method not in ['ablation', 'fanova', 'forward-selection', 'influence-model',
                                     'incneighbor', 'lpi']:
            raise ValueError('Specified evaluation method %s does not exist!' % evaluation_method)
        if evaluation_method == 'ablation':
            if self.incumbent is None:
                raise ValueError('Incumbent is %s!\n \
                                 Incumbent has to be read from a trajectory file before ablation can be used!'
                                 % self.incumbent)
            self.logger.info('Using model %s' % str(self.model))
            self.logger.info('X shape %s' % str(self.model.X.shape))
            evaluator = Ablation(scenario=self.scenario,
                                 cs=self.scenario.cs,
                                 model=self._model,
                                 to_evaluate=self._parameters_to_evaluate,
                                 incumbent=self.incumbent,
                                 logy=self.logged_y,
                                 rng=self.rng,
                                 verbose=self.verbose)
        elif evaluation_method == 'influence-model':
            self.logger.info('Using model %s' % str(self.model))
            self.logger.info('X shape %s' % str(self.model.X.shape))
            evaluator = InfluenceModel(scenario=self.scenario,
                                       cs=self.scenario.cs,
                                       model=self._model,
                                       to_evaluate=self._parameters_to_evaluate,
                                       margin=self.margin,
                                       threshold=self.threshold,
                                       rng=self.rng,
                                       verbose=self.verbose)
        elif evaluation_method == 'fanova':
            self.logger.info('Using model %s' % str(self.model))
            self.logger.info('X shape %s' % str(self.model.X.shape))
            mini = None
            if self.cut_def_fan:
                mini = True         # TODO what about scenarios where we maximize?
            evaluator = fANOVA(scenario=self.scenario,
                               cs=self.scenario.cs,
                               model=self._model,
                               to_evaluate=self._parameters_to_evaluate,
                               runhist=self.runhistory,
                               rng=self.rng,
                               minimize=mini,
                               pairwise=self.pairiwse_fANOVA,
                               preprocessed_X=self.X_fanova,
                               preprocessed_y=self.y_fanova,
                               incumbents=self.incumbent,
                               verbose=self.verbose)
        elif evaluation_method in ['incneighbor', 'lpi']:
            if self.incumbent is None:
                raise ValueError('Incumbent is %s!\n \
                                 Incumbent has to be read from a trajectory file before LPI can be used!'
                                 % self.incumbent)
            self.logger.info('Using model %s' % str(self.model))
            self.logger.info('X shape %s' % str(self.model.X.shape))
            evaluator = LPI(scenario=self.scenario,
                            cs=self.scenario.cs,
                            model=self._model,
                            to_evaluate=self._parameters_to_evaluate,
                            incumbent=self.incumbent,
                            logy=self.logged_y,
                            rng=self.rng,
                            quant_var=self.incn_quant_var,
                            verbose=self.verbose)
        else:
            self.logger.info('Using model %s' % str(self.model))
            evaluator = ForwardSelector(scenario=self.scenario,
                                        cs=self.scenario.cs,
                                        model=self._model,
                                        to_evaluate=self._parameters_to_evaluate,
                                        rng=self.rng,
                                        feature_imp=self.forwardsel_feat_imp,
                                        cv=self.forwardsel_cv,
                                        verbose=self.verbose)
        self._evaluator = evaluator

    def _convert_data(self, fit=True) -> None:  # From Marius
        '''
            converts data from runhistory into EPM format
            Parameters
            ----------
            scenario: Scenario
                smac.scenario.scenario.Scenario Object
            runhistory: RunHistory
                smac.runhistory.runhistory.RunHistory Object with all necessary data
            Returns
            -------
            np.array
                X matrix with configuartion x features for all observed samples
            np.array
                y matrix with all observations
            np.array
                types of X cols -- necessary to train our RF implementation
        '''

        params = self.scenario.cs.get_hyperparameters()
        num_params = len(params)
        self.logger.debug("Counted %d hyperparameters", num_params)

        if self.scenario.run_obj == "runtime":
            self.cutoff = self.scenario.cutoff
            self.threshold = self.scenario.cutoff * self.scenario.par_factor
            self.model = 'urfi'
            self.logged_y = True
            # if we log the performance data,
            # the RFRImputator will already get
            # log transform data from the runhistory
            cutoff = np.log10(self.scenario.cutoff)
            threshold = np.log10(self.scenario.cutoff *
                                 self.scenario.par_factor)
            model = RandomForestWithInstances(self.scenario.cs,
                                              self.types, self.bounds, 12345,
                                              instance_features=self.scenario.feature_array,
                                              )

            imputor = RFRImputator(rng=self.rng,
                                   cutoff=cutoff,
                                   threshold=threshold,
                                   model=model,
                                   change_threshold=0.01,
                                   max_iter=10)
            rh2EPM = RunHistory2EPM4LogCost(scenario=self.scenario,
                                            num_params=num_params,
                                            success_states=[
                                                StatusType.SUCCESS, ],
                                            impute_censored_data=self.impute,
                                            impute_state=[
                                                StatusType.TIMEOUT, StatusType.CAPPED],
                                            imputor=imputor)
        else:
            self.model = 'rfi'
            rh2EPM = RunHistory2EPM4Cost(scenario=self.scenario,
                                         num_params=num_params,
                                         success_states=[StatusType.SUCCESS],
                                         impute_censored_data=self.impute,
                                         impute_state=None)
        self.logger.info('Using model %s' % str(self.model))
        X, Y = rh2EPM.transform(self.runhistory)

        self.X = X
        self.y = Y
        self.logger.info('Size of training X: %s' % str(self.X.shape))
        self.logger.info('Size of training y: %s' % str(self.y.shape))
        self.logger.info('Data was %s imputed' % ('not' if not self.impute else ''))
        if not self.impute:
            self.logger.info('Thus the size of X might be smaller than the datapoints in the RunHistory')
        if fit:
            self.logger.info('Fitting Model')
            self.model.train(X, Y)

    def evaluate_scenario(self, methods, save_folder=None, plot_pyplot=True, plot_bokeh=False) -> Union[
            Tuple[Dict[str, Dict[str, float]], List[AbstractEvaluator]], Dict[str, Dict[str, float]]]:
        """
         the given scenario
        :param evaluation_method: name of the method to use
        :param sort_by: int, determines the order (only used if evaluation_method == all)
            0 => Ablation, fANOVA, Forward Selection
            1 => Ablation, Forward Selection, fANOVA
            2 => fANOVA, Forward Selection, Ablation
            3 => fANOVA, Ablation, Forward Selection
            4 => Forward Selection, Ablation, fANOVA
            5 => Forward Selection, fANOVA, Ablation
        :param plot_pyplot: whether to perform standard matplotlib- plotting
        :param plot_bokeh: whether to perform advanced bokeh plotting
        :return: if evaluation all: Tupel of dictionary[evaluation_method] -> importance values, List ov evaluator
                                    names, ordered according to sort_by
                 else:
                      dict[evalution_method] -> importance values
        """
        # influence-model currently not supported
        if not len(methods) >= 1:
            raise ValueError("Specify at least one method to evaluate the scenario!")
        fn = os.path.join(save_folder, 'pimp_results.json')
        load = os.path.exists(fn)
        dict_ = {}
        for rnd, method in enumerate(methods):
            self.logger.info('Running %s' % method)
            self.evaluator = method
            dict_[self.evaluator.name.lower()] = self.evaluator.run()
            self.evaluators.append(self.evaluator)
            if save_folder and plot_pyplot:
                self.evaluator.plot_result(os.path.join(save_folder, self.evaluator.name.lower()), show=False)
            if save_folder and plot_bokeh:
                self.evaluator.plot_bokeh(os.path.join(save_folder, self.evaluator.name.lower() + "_bokeh"))
            if load:
                with open(fn, 'r') as in_file:
                    doct = json.load(in_file)
                    for key in doct:
                        dict_[key] = doct[key]
            if save_folder:
                with open(fn, 'w') as out_file:
                    json.dump(dict_, out_file, sort_keys=True, indent=4, separators=(',', ': '))
                    load = True
        return dict_, self.evaluators