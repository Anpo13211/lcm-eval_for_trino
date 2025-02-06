import numpy as np
import scipy
from scipy.stats import rankdata

from sklearn.metrics import mean_squared_error


class Metric:
    def __init__(self, metric_name: str, display_name: str, y_max: float, y_min: float = 0, logscale: bool = False,
                 aggregation: str = 'mean', **kwargs):
        self.metric_name = metric_name
        self.display_name = display_name
        self.default_value = None
        self.y_min, self.y_max = y_min, y_max
        self.logscale = logscale
        self.aggregation = aggregation

    def evaluate(self, model=None, metrics_dict=None, **kwargs) -> float:
        metric = self.default_value
        try:
            metric = self.evaluate_metric(**kwargs)
        except ValueError as e:
            print(f"Observed ValueError in metrics calculation {e}")
        return metric

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        return NotImplementedError


class SpeedDown(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='speed_down', display_name="Additional\nRuntime", y_min=None, y_max=None,
                         logscale=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        # Find out where the model think that the runtime is minimal
        minimal_prediction_idx = preds.argmin()

        # Find out how the real runtime at this position
        label_at_minimal_prediction = labels[minimal_prediction_idx]

        # Find the real minimum runtime
        fastest_runtime = min(labels)

        result = label_at_minimal_prediction - fastest_runtime
        return result


class RMSE(Metric):
    y_min = 0.1
    y_max = 1.0

    def __init__(self, **kwargs):
        super().__init__(metric_name='rmse', display_name="RMSE", y_min=0, y_max=None, logscale=False, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        val_mse = np.sqrt(mean_squared_error(labels, preds))
        return val_mse


class SpearmanCorrelation(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='spearman', display_name="Spearmans\nCorr.", y_min=-0.5, y_max=1, **kwargs)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        value = round(scipy.stats.spearmanr(labels, preds).statistic, 2)
        return value


class QError(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='qerror', display_name="Median\nQ-Error", y_min=1, y_max=5, logscale=True,
                         **kwargs)
        self.percentile = 50

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        preds = np.abs(preds)
        # preds = np.clip(preds, self.min_val, np.inf)

        q_errors = np.maximum(labels / preds, preds / labels)
        q_errors = np.nan_to_num(q_errors, nan=np.inf)
        median_q = np.percentile(q_errors, self.percentile)
        return median_q


class MeanAbsoluteError(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='mae', display_name="MAE", y_min=0, y_max=None, logscale=False)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        return np.mean(labels - preds)


class MeanRelativeError(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='mae', display_name="Relative Error", y_min=0, y_max=None, logscale=False)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        return np.mean(preds / labels)


class PickRate(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='pick_rate', display_name="Pick Rate", **kwargs, y_min=0, y_max=1)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        labels = list(labels)
        preds = list(preds)
        return labels.index(min(labels)) == preds.index(min(preds))


class SelectedRuntime(Metric):
    def __init__(self, display_name: str = "Selected Runtime (s)", **kwargs):
        super().__init__(metric_name='runtime', display_name=display_name, **kwargs, y_min=0, y_max=None,
                         logscale=False,
                         aggregation='sum')

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        #index_of_minimal_query = labels.idxmin()
        #runtime_of_minimal_query = preds[index_of_minimal_query]
        index_of_minimal_prediction = preds.idxmin()
        runtime_of_minimal_prediction = labels[index_of_minimal_prediction]
        return runtime_of_minimal_prediction


class MissedPlansFraction(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='missed_plans_fraction', display_name="Surpassed\nPlans (%)", **kwargs,
                         y_min=0, y_max=100)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        index_of_minimal_prediction = preds.idxmin()
        runtime_of_minimal_prediction = labels[index_of_minimal_prediction]
        masks = labels < runtime_of_minimal_prediction
        missed_plans_fraction = 1 - masks.sum() / len(labels)
        return missed_plans_fraction * 100


class MaxUnderestimation(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='max_underestimation', display_name="Max. Rel.\nUnderest.", **kwargs, y_min=1,
                         y_max=None, logscale=True)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        max_underestimation = 1 / min(preds / labels)
        # Cap lower value to 1
        max_underestimation = max(1, max_underestimation)
        return max_underestimation


class MaxOverestimation(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='max_overestimation', display_name="Max. Rel.\nOverest.", **kwargs, y_min=1, y_max=None,
                         logscale=True)

    def evaluate_metric(self, labels=None, preds=None, probs=None):
        max_overestimation = max(preds / labels)
        # Cap lower value to 1
        max_overestimation = max(1, max_overestimation)
        return max_overestimation
