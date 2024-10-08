import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import CAMELS_IDS, CHANGDIAN_IDS, RESULT_DIR, SANXIA_DPL_DIR1
from hydrodhm.utils.results_plot_utils import (
    plot_metrics_1model_trained_with_diffperiods,
)


def _generate_changdian_datalimited_result_dirs(basin_id, model="dpl", cases=None):
    """A literal specification of the result directories of dPL

    Parameters
    ----------
    basin_id : _type_
        _description_
    model : str, optional
        _description_, by default "dpl"

    Returns
    -------
    _type_
        _description_
    """
    if cases is None:
        cases = ["4-year", "3-year", "2-year"]
    if model == "dpl":
        post_fix = ""
    elif model == "dpl_nn":
        post_fix = "_module"
    if len(cases) != 3:
        raise ValueError(
            "cases must be a list with 3 elements, we support 2-4 to 4-4 now only"
        )
    return [
        os.path.join(
            SANXIA_DPL_DIR1,
            "lrchange3" if model == "dpl" else "module",
            basin_id,
        ),
        os.path.join(
            RESULT_DIR,
            "dPL",
            "result",
            "data-limited_analysis",
            f"3to4_1518_1721{post_fix}",
            basin_id,
        ),
        os.path.join(
            RESULT_DIR,
            "dPL",
            "result",
            "data-limited_analysis",
            f"2to4_1618_1721{post_fix}",
            basin_id,
        ),
    ]


def plot_metrics_for_changdian_basins():
    cases = ["4-year", "3-year", "2-year"]
    sanxiabasins_result_dirs = [
        _generate_changdian_datalimited_result_dirs(basin_id, cases=cases)
        for basin_id in CHANGDIAN_IDS
    ]
    plot_metrics_1model_trained_with_diffperiods(
        sanxiabasins_result_dirs,
        CHANGDIAN_IDS,
        cfg_runagain=False,
        cases=cases,
    )
    plot_metrics_1model_trained_with_diffperiods(
        sanxiabasins_result_dirs,
        CHANGDIAN_IDS,
        cfg_runagain=False,
        cases=cases,
        train_or_valid="train",
        legend=False,
    )
    sanxiabasins_nnmodule_result_dirs = [
        _generate_changdian_datalimited_result_dirs(
            basin_id, cases=cases, model="dpl_nn"
        )
        for basin_id in CHANGDIAN_IDS
    ]
    plot_metrics_1model_trained_with_diffperiods(
        sanxiabasins_nnmodule_result_dirs,
        CHANGDIAN_IDS,
        cfg_runagain=False,
        cases=cases,
        legend=False,
    )
    plot_metrics_1model_trained_with_diffperiods(
        sanxiabasins_nnmodule_result_dirs,
        CHANGDIAN_IDS,
        cfg_runagain=False,
        cases=cases,
        train_or_valid="train",
        legend=False,
    )


def _generate_camels_datalimited_result_dirs(basin_id, model="dpl", cases=None):
    """A literal specification of the result directories of dPL

    Parameters
    ----------
    basin_id : _type_
        _description_
    model : str, optional
        _description_, by default "dpl"

    Returns
    -------
    _type_
        _description_
    """
    if cases is None:
        cases = [
            "20y",
            "15y",
            "10y",
            "05y",
            "04y",
            "03y",
            "02y",
        ]
    if model == "dpl":
        post_fix = ""
    elif model == "dpl_nn":
        post_fix = "_module"
    return [
        os.path.join(
            RESULT_DIR,
            "dPL",
            "result",
            "data-limited_analysis",
            f"camels{case}{post_fix}",
            basin_id,
        )
        for case in cases
    ]


def plot_metrics_for_camels_basins():
    camels_result_dirs = [
        _generate_camels_datalimited_result_dirs(basin_id) for basin_id in CAMELS_IDS
    ]
    plot_metrics_1model_trained_with_diffperiods(
        camels_result_dirs,
        CAMELS_IDS,
        cfg_runagain=False,
    )
    plot_metrics_1model_trained_with_diffperiods(
        camels_result_dirs,
        CAMELS_IDS,
        cfg_runagain=False,
        train_or_valid="train",
        legend=False,
    )
    camels_nnmodule_result_dirs = [
        _generate_camels_datalimited_result_dirs(basin_id, model="dpl_nn")
        for basin_id in CAMELS_IDS
    ]
    plot_metrics_1model_trained_with_diffperiods(
        camels_nnmodule_result_dirs,
        CAMELS_IDS,
        cfg_runagain=False,
        legend=False,
    )
    plot_metrics_1model_trained_with_diffperiods(
        camels_nnmodule_result_dirs,
        CAMELS_IDS,
        cfg_runagain=False,
        train_or_valid="train",
        legend=False,
    )


plot_metrics_for_changdian_basins()
plot_metrics_for_camels_basins()
