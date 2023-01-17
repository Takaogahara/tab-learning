import os
import torch
import warnings
from pathlib import Path
from matplotlib import pyplot as plt

from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier

warnings.simplefilter("ignore", UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


LOSS_FN = {"bceloss": torch.nn.BCELoss(),
           "mse": torch.nn.MSELoss()}
EVAL_METRIC = {"classification": "balanced_accuracy",
               "regression": "rmse"}


class Model:
    def __init__(self, categoricals, parameters):
        self.task = parameters["TASK"]
        self.cat_idxs = categoricals[0]
        self.cat_dims = categoricals[1]

        self.parameters = parameters

    def train(self, train_data, test_data):
        X_train, y_train = train_data
        X_train = X_train.values
        y_train = y_train.values.reshape(-1, 1)

        X_test, y_test = test_data
        X_test = X_test.values
        y_test = y_test.values.reshape(-1, 1)

        parameters = self.parameters

        tab_parameters = {"n_d": parameters["TAB_N_D"],
                          "n_a": parameters["TAB_N_A"],
                          "n_steps": parameters["TAB_N_STEPS"],
                          "gamma": parameters["TAB_GAMMA"],
                          "n_independent": parameters["TAB_N_INDEPENDENT"],
                          "n_shared": parameters["TAB_N_SHARED"],
                          "epsilon": parameters["TAB_EPSILON"],
                          "momentum": parameters["TAB_MOMENTUM"],
                          "lambda_sparse": parameters["TAB_LAMBDA_SPARSE"],
                          "cat_idxs": self.cat_idxs,
                          "cat_dims": self.cat_dims,
                          "cat_emb_dim": parameters["TAB_CAT_EMB_DIM"],
                          "verbose": 0,
                          "seed": 8,
                          "device_name": device}

        eval_metric = EVAL_METRIC[self.task.lower()]
        loss_fn = LOSS_FN[parameters["TAB_LOSS_FN"].lower()]

        if self.task.lower() == "regression":
            model = TabNetRegressor(**tab_parameters)
            model.fit(X_train, y_train,
                      batch_size=parameters["TAB_BATCH_SIZE"],
                      max_epochs=parameters["RUN_RAY_EPOCHS"],
                      loss_fn=loss_fn,
                      eval_set=[(X_test, y_test)],
                      eval_metric=[eval_metric], patience=10,
                      num_workers=0)

        elif self.task.lower() == "classification":
            model = TabNetClassifier(**tab_parameters)
            model.fit(X_train, y_train.flatten(),
                      batch_size=parameters["TAB_BATCH_SIZE"],
                      max_epochs=parameters["RUN_RAY_EPOCHS"],
                      loss_fn=loss_fn,
                      eval_set=[(X_test, y_test.flatten())],
                      eval_metric=[eval_metric], patience=10,
                      num_workers=0)

        loss = model.history["loss"][model.best_epoch]
        self._export_model(model)

        # TODO log model max epoch?

        return loss

    def explain_tabnet(self, data):

        data = data.values
        _, masks = self.model.explain(data)

        fig, axs = plt.subplots(1, len(masks), figsize=(20, 20))
        for i in range(len(masks)):
            axs[i].imshow(masks[i][:25])
            axs[i].set_title(f"Mask {i}")

        plt.show()
        plt.close()

    def _export_model(self, model):

        self.model = model

        export_path = Path(os.getcwd())
        export_path = export_path / "output"
        os.makedirs(export_path, exist_ok=True)
        model_path = str(export_path / "model")

        try:
            model.save_model(model_path)

        except TypeError:
            model.device_name = "cpu"
            model.save_model(model_path)
            model.device_name = model.device
