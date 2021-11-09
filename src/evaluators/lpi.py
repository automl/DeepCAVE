from src.evaluators.epm.rf_with_instances import RandomForestWithInstances


class LPI:
    def __init__(self, run: Run):
        # fidelity=None,
        # states=None,
        # hyperparameter_ids=None):

        self.incumbent = run.get_incumbent(fidelity, hyperparameter_ids)
        #self.transform_config = lambda config, mapping=mapping: run.transform_config(config, mapping)

        X, y, mapping, id_mapping, types, bounds, configspace = run.transform_configs(
            self,
            fidelities,
            states,
            hyperparameters
        )

        self.cs = configspace

        # We need a new configspace

        # Initialize model
        self.model = RandomForestWithInstances(
            configspace,
            types,
            bounds,
            do_bootstrapping=True
        )
        self.model.rf_opts.compute_oob_error = True
        self.model.train(X, y)

        pass

    def _get_one_exchange_neighborhood_by_parameter(self):
        """
        Slight modification of ConfigSpace's get_one_exchange neighborhood. This orders the parameter values and samples
        more neighbors in one go. Further we need to rigorously check each and every neighbor if it is forbidden or not.
        """

        neighborhood_dict = {}
        hp_names = list(self.incumbent.keys())

        for hp_id, hp_name in enumerate(hp_names):
            self.logger.info('Sampling neighborhood of %s' % hp_name)

            array = self.incumbent.get_array()
            # array = self.transform_config(self.incumbent)

            if not np.isfinite(array[hp_id]):
                self.logger.info('>'.join(['-'*50, ' Not active!']))
                continue

            neighbourhood = []
            checked_neighbors = []
            checked_neighbors_non_unit_cube = []
            hp = self.incumbent.configuration_space.get_hyperparameter(hp_name)
            num_neighbors = hp.get_num_neighbors(self.incumbent.get(hp_name))

            self.logger.debug('\t' + str(num_neighbors))
            if num_neighbors == 0:
                self.logger.debug('\tNo neighbors!')
                continue
            elif np.isinf(num_neighbors):  # Continous Parameters
                if hp.log:
                    base = np.e
                    log_lower = np.log(hp.lower) / np.log(base)
                    log_upper = np.log(hp.upper) / np.log(base)
                    neighbors = np.logspace(log_lower, log_upper, self.continous_neighbors,
                                            endpoint=True, base=base)
                else:
                    neighbors = np.linspace(
                        hp.lower, hp.upper, self.continous_neighbors)
                neighbors = list(
                    map(lambda x: hp._inverse_transform(x), neighbors))
            else:
                neighbors = hp.get_neighbors(array[hp_id], self.rng)

            for neighbor in neighbors:
                if neighbor in checked_neighbors:
                    continue

                new_array = array.copy()
                new_array = change_hp_value(
                    self.incumbent.configuration_space,
                    new_array,
                    param,
                    neighbor,
                    index
                )

                try:
                    new_configuration = Configuration(
                        self.incumbent.configuration_space, vector=new_array)
                    neighbourhood.append(new_configuration)
                    new_configuration.is_valid_configuration()
                    check_forbidden(self.cs.forbidden_clauses, new_array)
                    checked_neighbors.append(neighbor)
                    checked_neighbors_non_unit_cube.append(
                        new_configuration[param])
                except (ForbiddenValueError, ValueError) as e:
                    pass

            self.logger.info('>'.join(
                ['-'*50, ' Found {:>3d} valid neighbors'.format(len(checked_neighbors))]))
            self.sampled_neighbors += len(checked_neighbors) + 1
            sort_idx = list(map(lambda x: x[0], sorted(
                enumerate(checked_neighbors), key=lambda y: y[1])))
            if isinstance(self.cs.get_hyperparameter(param), CategoricalHyperparameter):
                checked_neighbors_non_unit_cube = list(
                    np.array(checked_neighbors_non_unit_cube)[sort_idx])
            else:
                checked_neighbors_non_unit_cube = np.array(
                    checked_neighbors_non_unit_cube)[sort_idx]
            neighborhood_dict[param] = [np.array(checked_neighbors)[
                sort_idx], checked_neighbors_non_unit_cube]

        return neighborhood_dict

    def train(self, X, y):
        pass

    def run(self):
        neighborhood_dict = self._get_one_exchange_neighborhood_by_parameter()
