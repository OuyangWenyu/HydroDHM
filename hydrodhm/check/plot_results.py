import os
from definitions import RESULT_DIR
from hydrodhm.utils.results_utils import (
    get_pbm_params_from_dpl,
    read_dpl_model_q_and_et,
    read_sceua_xaj_streamflow,
)


# sceua_dir = os.path.join(RESULT_DIR, "XAJ", "changdian_61700_4_4")
# dpl_dir = os.path.join(RESULT_DIR, "dPL", "result", "lrchange3", "changdian_61700")
# dpl_nn_dir = os.path.join(RESULT_DIR, "dPL", "result", "module", "changdian_61700")
dpl_dir = os.path.join(RESULT_DIR, "streamflow_prediction", "lrchange3", "changdian_61700")
dpl_nn_dir = os.path.join(RESULT_DIR, "streamflow_prediction", "module", "changdian_61700")
# read_sceua_xaj_streamflow(sceua_dir)
# read_sceua_xaj_streamflow_metric(os.path.join(RESULT_DIR, "XAJ", "changdian_61561"))
# read_sceua_xaj_et(os.path.join(RESULT_DIR, "XAJ", "changdian_61561"))
# read_sceua_xaj_et_metric(
#     os.path.join(RESULT_DIR, "XAJ", "result_old", "changdian_61700")
# )
# get_pbm_params_from_hydromodelxaj(
#     os.path.join(RESULT_DIR, "XAJ", "result_old", "changdian_61700")
# )
read_dpl_model_q_and_et(dpl_dir)
read_dpl_model_q_and_et(dpl_nn_dir)
# read_dpl_model_metric(
#     os.path.join(
#         RESULT_DIR,
#         "dPL",
#         "streamflow_prediction",
#         "streamflow_prediction_50epoch",
#         "changdian_61561",
#     ),
#     os.path.join(
#         RESULT_DIR,
#         "dPL",
#         "streamflow_prediction",
#         "streamflow_prediction_50epoch",
#         "changdian_61561_trainperiod",
#     ),
# )
get_pbm_params_from_dpl(dpl_dir)
