from smac.epm.rf_with_instances import RandomForestWithInstances as RFI


class RandomForestWithInstances(RFI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _impute_inactive(X):
        return X