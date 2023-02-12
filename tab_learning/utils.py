import os
import yaml
import copy
import uuid
import pandas as pd
from ray import tune
from pathlib import Path
from sklearn.model_selection import train_test_split

DEFAULT_PARAMS = {"TASK": "Regression",

                  "RUN_NAME": "None",
                  "RUN_RAY_SAMPLES": 5,
                  "RUN_RAY_CPU": 2,
                  "RUN_RAY_GPU": 0.5,
                  "DATA_EPOCHS": 2,
                  "DATA_BATCH_SIZE": 1024,

                  "DATA_TRAIN": "path/to/file.csv",
                  "DATA_TEST": "path/to/file.csv",
                  "DATA_COL_ID": "id",
                  "DATA_COL_TARGET": "target",

                  "TAB_CAT_ENCODER": "Default",
                  "TAB_LAYERS": 8,
                  "TAB_N_STEPS": 3,
                  "TAB_GAMMA": 1.3,
                  "TAB_N_INDEPENDENT": 2,
                  "TAB_N_SHARED": 2,
                  "TAB_MOMENTUM": 0.02,
                  "TAB_LAMBDA_SPARSE": 0.001,
                  "TAB_LOSS_FN": "mse"}


class IO:
    def export_yaml(parameters: dict):
        export_path = Path(os.getcwd())
        export_path = export_path / "output"
        os.makedirs(export_path, exist_ok=True)
        yaml_path = str(export_path / "parameters.yaml")

        task = parameters["TASK"]
        run = {key: value for key, value in parameters.items(
                                                    ) if "RUN_" in key}
        data = {key: value for key, value in parameters.items(
                                                    ) if "DATA_" in key}
        tab = {key: value for key, value in parameters.items(
                                                    ) if "TAB_" in key}

        del data["DATA_COL_FEAT"]
        dict_file = [{"TASK": task}, {"RUN": run},
                     {"DATA": data}, {"TABNET": tab}]

        with open(yaml_path, 'w') as file:
            yaml.dump(dict_file, file, default_flow_style=False)

    def export_preds(idx, preds, parameters):
        model_root = Path(parameters["DATA_OUTPUT"])
        root = model_root.parent.parent
        model_name = model_root.parent.name
        out_path = str(root / f"{model_name}.csv")

        df_submission = pd.DataFrame()
        df_submission[parameters["DATA_COL_ID"]] = idx
        df_submission[parameters["DATA_COL_TARGET"]] = preds

        df_submission.to_csv(out_path, sep=",", index=False)


class DataHandler:
    def split_data(data, parameters, train_size=0.8):
        TASK = parameters["TASK"]
        COL_ID = parameters["DATA_COL_ID"]
        COL_TARGET = parameters["DATA_COL_TARGET"]

        X = data.drop(columns=[COL_ID, COL_TARGET])
        y = data[COL_TARGET]

        if TASK.lower() == "classification":
            X_tr, X_te, y_tr, y_te = train_test_split(X, y,
                                                      train_size=train_size,
                                                      stratify=y,
                                                      random_state=8)

        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y,
                                                      train_size=train_size,
                                                      random_state=8)

        X_tr = X_tr.reset_index()
        X_tr = X_tr.drop(columns=["index"])

        y_tr = y_tr.reset_index()
        y_tr = y_tr.drop(columns=["index"])

        X_te = X_te.reset_index()
        X_te = X_te.drop(columns=["index"])

        y_te = y_te.reset_index()
        y_te = y_te.drop(columns=["index"])

        train = (X_tr, y_tr)
        test = (X_te, y_te)

        return train, test


class ParametersHandler:
    def load_configs(yaml_path: str):

        with open(yaml_path, "r") as f:
            file = yaml.safe_load(f)

        content = {"TASK": file[0]["TASK"]}
        content_list = [file[1]["RUN"], file[2]["DATA"], file[3]["TABNET"]]

        for cfg in content_list:
            for key, value in cfg.items():
                content[key] = value

        # * Auto complete config file
        parameters = copy.deepcopy(DEFAULT_PARAMS)
        for key, value in list(content.items()):
            parameters[key] = value

        if parameters["RUN_NAME"].lower() == "none":
            name = uuid.uuid4()
            name = str(name).split("-")[0]
            parameters["RUN_NAME"] = name

        return parameters

    def get_ray_choices(parameters: dict):
        choices = {}

        for key, value in list(parameters.items()):
            # General parameters
            if ("TAB_" not in key):
                choices[key] = value

            else:
                val_type = type(value)

                if val_type != list:
                    choice_fn = _get_single_choice(value)
                    choices[key] = choice_fn

                else:
                    choice_fn = _get_list_choice(value)
                    choices[key] = choice_fn

        return choices

    def test_parms(parameters):
        params = {}

        for key, value in list(parameters.items()):
            # General parameters
            if ("TAB_" not in key):
                params[key] = value

            if isinstance(value, str):
                if "e-" in value.lower():
                    value = float(value)

            params[key] = value

        return params


def _get_single_choice(value):

    # Check for log
    if isinstance(value, str):
        if "e-" in value.lower():
            value = float(value)

    choice_fn = tune.choice([value])

    return choice_fn


def _get_list_choice(value):

    # TabNet int choices
    if isinstance(value[0], int):
        choice_fn = tune.randint(value[0], value[1])

    # TabNet float choices
    elif isinstance(value[0], float):
        choice_fn = tune.uniform(value[0], value[1])

    # TabNet string and log choices
    elif isinstance(value[0], str):
        if "e-" in value[0].lower():
            value = [float(x) for x in value]
            choice_fn = tune.loguniform(value[0], value[1])
        else:
            choice_fn = tune.choice(value)

    return choice_fn
