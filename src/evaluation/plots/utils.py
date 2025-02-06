from pathlib import Path
from types import SimpleNamespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import wandb
from matplotlib.ticker import FormatStrFormatter

from classes.classes import ModelConfig, ColorManager, ScaledPostgresModelConfig, ScaledPostgresModelV16Config, \
    ModelName
from classes.paths import LocalPaths
from classes.workloads import JoinOrderEvalWorkload, LiteralEvalWorkload, EvaluationWorkload, JoinImplEvalWorkload
from evaluation.plots.eval import Evaluator
from evaluation.plots.evaluation_metrics import Metric, MaxOverestimation, MaxUnderestimation
from training.dataset.dataset_creation import read_workload_runs

fontsize = 16


class JoinType:
    left_deep = "Left deep"
    right_deep = "Right deep"
    bushy = "Bushy"


def follow_path(plan: SimpleNamespace):
    if "Join" in plan.plan_parameters.op_name or "Nested Loop" in plan.plan_parameters.op_name:
        return "Join"
    elif "Scan" in plan.plan_parameters.op_name:
        return "Scan"
    else:
        assert len(
            plan.children) == 1, f"Found a node of type {plan.plan_parameters.op_name} with {len(plan.children)} children, expected 1."
        return follow_path(plan.children[0])


def get_table_from_wandb_run(run, wandb_user: str, wandb_project: str) -> pd.DataFrame:
    """Extract training scores from a wandb run."""
    api = wandb.Api()
    for artifact in run.logged_artifacts():
        if "test_scores" in artifact._name:
            a = api.artifact(f"{wandb_user}/{wandb_project}/{artifact._name}")
            a.download()
            table = a.get(f"{run.name}/test_scores.table.json")
            if table is not None:
                return table.get_dataframe()
    return None


def load_wandb_runs(wandb_user: str, wandb_project: str, result_dir: Path, model_confs: List[ModelConfig]) -> pd.DataFrame:
    """Load all runs from a wandb project and store them in a csv file."""
    api = wandb.Api()
    if result_dir.exists():
        training_results = pd.read_csv(result_dir)
    else:
        runs = api.runs(wandb_project)
        metric_collection = [dict(model=run.name.split("/")[0],
                                  database=run.name.split("/")[1],
                                  seed=run.config['seed'],
                                  **{row['metric']: row['value']
                                     for _, row in get_table_from_wandb_run(run, wandb_user, wandb_project).iterrows()})
                             for run in runs if get_table_from_wandb_run(run, wandb_user, wandb_project) is not None]
        training_results = pd.DataFrame.from_dict(metric_collection)
    # Filter for models that are in the model_confs
    training_results = training_results[training_results.model.isin([model_config.name.NAME for model_config in model_confs])]

    # Add display name
    mapping = {model_config.name.NAME: model_config.name.DISPLAY_NAME for model_config in model_confs}
    training_results['display_name'] = training_results.model.map(mapping)

    # Order according to model_confs
    order = [model_config.name.NAME for model_config in model_confs]
    training_results = training_results.set_index('model').loc[order].reset_index()
    return training_results

def determine_join_type(plan: SimpleNamespace) -> JoinType:
    assert hasattr(plan, "children")
    assert len(plan.children) == 2
    left_child_type = follow_path(plan.children[0])
    right_child_type = follow_path(plan.children[1])
    if left_child_type == "Join" and right_child_type == "Join":
        return JoinType.bushy
    elif left_child_type == "Join" and right_child_type == "Scan":
        return JoinType.left_deep
    elif left_child_type == "Scan" and right_child_type == "Join":
        return JoinType.right_deep
    elif left_child_type == "Scan" and right_child_type == "Scan":
        return JoinType.bushy
    else:
        raise ValueError(f"Unexpected join type: {left_child_type} and {right_child_type}")


def obtain_join_bushiness(plan: SimpleNamespace, left_joins=0, right_joins=0, bushy_joins=0):
    if "Join" in plan.plan_parameters.op_name or "Nested Loop" in plan.plan_parameters.op_name:
        join_type = determine_join_type(plan)
        if join_type == JoinType.bushy:
            bushy_joins += 1
        elif join_type == JoinType.left_deep:
            left_joins += 1
        elif join_type == JoinType.right_deep:
            right_joins += 1
        else:
            raise ValueError(f"Unexpected join type: {join_type}")
    if hasattr(plan, "children"):
        for children in plan.children:
            left_joins, right_joins, bushy_joins = obtain_join_bushiness(children, left_joins, right_joins, bushy_joins)
    return left_joins, right_joins, bushy_joins


def get_model_results(workload: EvaluationWorkload, model_configs: List[ModelConfig], retrain: bool = False) -> pd.DataFrame:
    parsed_plans = [workload.get_workload_path(LocalPaths().parsed_plans)]
    plans, database_statistics = read_workload_runs(workload_run_paths=parsed_plans)
    cardinalities = Evaluator.readout_workload_information(plans, workload)

    # drop runtime
    cardinalities = cardinalities.drop(columns=['runtime'])

    # Combine model predictions to common dataframe
    predictions = Evaluator.combine_predictions(models=model_configs,
                                                workload=workload,
                                                seeds=[0, 1, 2],
                                                retrain=retrain)

    predictions = predictions.groupby(['model', 'query_index', 'label'])['prediction'].mean().reset_index()
    # Overwrite the label for qpp-net model by using the label of a different model that has the same query_index

    if not isinstance(workload, JoinOrderEvalWorkload):
        source_model = 'Flat Vector'
        for target_model in [ModelName.QPP_NET, ModelName.QPP_NET_ACT_CARDS, ModelName.QPP_NET_NO_COSTS]:
            target_model = target_model.DISPLAY_NAME
            # Get the labels from the source model
            source_labels = predictions[predictions['model'] == source_model][['query_index', 'label']]

            # Update the labels in the target model
            predictions.loc[predictions['model'] == target_model, 'label'] = predictions.loc[predictions['model'] == target_model].merge(
                source_labels, on='query_index', suffixes=('', '_new'))['label']

    if isinstance(workload, LiteralEvalWorkload):
        predictions = pd.merge(predictions, cardinalities, on=["query_index"])

    # Sort results by runtime for join order workloads
    if isinstance(workload, JoinOrderEvalWorkload):
        predictions = predictions.sort_values(by="label")  # .reset_index(drop=False)
        predictions['ranked_runtime'] = predictions['label'].rank(method='dense')
    # Copy colum runtime to label

    return predictions


def draw_metric(results: pd.DataFrame, model_configs: List[ModelConfig], ax: plt.Axes, metric: Metric, fontsize: int) -> pd.DataFrame:
    results_df = pd.DataFrame({
        model.name.DISPLAY_NAME: metric.evaluate_metric(preds=results[results["model"] == model.name.DISPLAY_NAME]["prediction"],
                                                        labels=results[results["model"] == model.name.DISPLAY_NAME]['label']) for model in model_configs},
        index=[metric.metric_name]).T

    seaborn.barplot(data=results_df,
                    hue=results_df.index,
                    y=metric.metric_name,
                    #hue=results_df.index,
                    ax=ax,
                    palette=ColorManager.COLOR_PALETTE,
                    zorder=3,
                    width=1.0,
                    edgecolor='black')

    ax.set(ylim=(metric.y_min, metric.y_max),
           ylabel="",
           xticklabels=[],
           xlabel="")

    ax.set_ylabel(metric.display_name, fontsize=0.8 * fontsize)
    ax.tick_params(axis='y', which='both', rotation=0, labelsize=fontsize)
    ax.yaxis.tick_right()
    ax.grid(axis="y", which='both', linestyle='-', linewidth=0.5)
    ax.get_legend().remove()

    if isinstance(metric, MaxOverestimation) or isinstance(metric, MaxUnderestimation):
        formatter = FormatStrFormatter('%.0f')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(bottom=1)

    if metric.logscale:
        ax.set_yscale("log")
        formatter = FormatStrFormatter('%.0f')
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(formatter)
    for i, model in enumerate(model_configs):
        ax.patches[i].set_facecolor(model.color())

    return results_df


def draw_predictions(workload: EvaluationWorkload,
                     results: pd.DataFrame,
                     model_configs: List[ModelConfig],
                     pred_ax: plt.Axes,
                     fontsize: int,
                     x_index: str = "query_index",
                     plot_cardinality: bool = False):

    if isinstance(workload, JoinOrderEvalWorkload):
        x_index = "ranked_runtime"
    elif isinstance(workload, JoinImplEvalWorkload):
        pred_ax.xaxis.set_ticks(np.arange(0, 3, 1))
        pred_ax.set_xticklabels(workload.yticks, fontsize=fontsize)

    zorder = 10
    # Draw Model Predictions
    for model in model_configs[::-1]:
        label = model.name.DISPLAY_NAME
        model_results = results[results["model"] == model.name.DISPLAY_NAME]
        if model == ScaledPostgresModelConfig() or model == ScaledPostgresModelV16Config():
            linewidth = 4
        else:
            linewidth = 2

        seaborn.lineplot(x=x_index,
                         y="prediction",
                         data=model_results,
                         ax=pred_ax, label=label,
                         color=model.color(),
                         linewidth=linewidth,
                         zorder=zorder++1)

    # Draw Real Runtime
    seaborn.lineplot(x=x_index,
                     y="label",
                     linestyle="-",
                     ax=pred_ax,
                     data=results,
                     label="Real Runtime",
                     color="black",
                     linewidth=4,
                     zorder=100)

    pred_ax.set_yscale("log")
    pred_ax.set_ylabel("Runtime (s)", fontsize=fontsize)
    pred_ax.legend().remove()
    pred_ax.grid()
    pred_ax.grid(axis="y", which='both', linestyle='--', linewidth=0.5)
    if isinstance(workload, JoinOrderEvalWorkload):
        pred_ax.set_xlabel('Join Permutation (sorted by runtime)', fontsize=fontsize)
    elif isinstance(workload, JoinImplEvalWorkload):
        pred_ax.set_xlabel("")
    else:
        pred_ax.set_xlabel(pred_ax.get_xlabel(), fontsize=fontsize)
    pred_ax.tick_params(axis='x', rotation=0, labelsize=fontsize * 0.8)
    pred_ax.tick_params(axis='y', rotation=0, labelsize=fontsize * 0.8)
    handles, labels = pred_ax.get_legend_handles_labels()

    if plot_cardinality:
        card_ax = pred_ax.twinx()
        for card, color, label in zip(["act_card", "est_card"], ["black", "gray"], ["Real Cardinality", "Est. Cardinality"]):
            seaborn.lineplot(x=x_index,
                             y=card,
                             linestyle="--",
                             ax=card_ax,
                             data=results,
                             label=label,
                             color=color,
                             linewidth=2,
                             zorder=100)
        card_ax.set_yscale("log")
        card_ax.tick_params(axis='x', rotation=0, labelsize=fontsize * 0.8)
        card_ax.tick_params(axis='y', rotation=0, labelsize=fontsize * 0.8)
        card_ax.set_ylabel("Cardinality", fontsize=fontsize)
        card_ax.legend().remove()
        handles2, labels2 = card_ax.get_legend_handles_labels()
        handles = handles + handles2
        labels = labels + labels2
    return handles, labels


def draw_bushiness(results: pd.DataFrame, model_configs: List[ModelConfig], ax: plt.Axes):
    results['q_error'] = results.apply(
        lambda row: max(row['prediction'] / row['runtime'], row['runtime'] / row['prediction']), axis=1)
    bushiness_levels = sorted(results['bushiness'].unique())
    models = results['model'].unique()
    model_config_names = [config.display_name for config in model_configs]
    models = sorted(models, key=model_config_names.index)
    # Define the width of the bars
    bar_width = 0.15

    # Define the space between the bar groups
    space_between_groups = 0.55

    # Define an array for the positions of the bars on the x-axis
    r = np.arange(len(bushiness_levels)) * (1 + space_between_groups)

    # For each model, plot a bar for each bushiness level
    for i, model in enumerate(models):
        model_results = results[results['model'] == model]
        grouped = model_results.groupby('bushiness')['q_error'].mean().reindex(
            bushiness_levels)  # Sort grouped data by bushiness levels
        for c in model_configs:
            if c.display_name == model:
                color = c.color
        ax.bar(r + i * bar_width, height=grouped, width=bar_width, label=model, color=color)

    # Calculate and plot the average q-error for each bushiness level
    for i, level in enumerate(bushiness_levels):
        avg_q_error = results[results['bushiness'] == level]['q_error'].mean()
        ax.hlines(y=avg_q_error, xmin=r[i] - bar_width / 2, xmax=r[i] + (len(models) - 0.5) * bar_width, color='black',
                  linewidth=2, linestyle='--')

    ax.set_yscale('log')
    #ax.set_title(f'{workload.wl_name}')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(1, 20)
    # Adjust xticks
    ax.set_xticks(r + (len(models) - 1) * bar_width / 2)
    ax.set_xticklabels(bushiness_levels)
    return