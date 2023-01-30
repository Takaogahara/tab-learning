import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
from ks_metric import ks_score
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import ConfusionMatrixDisplay as cm_disp
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score, mean_squared_error,
                             mean_absolute_error, r2_score, matthews_corrcoef,
                             balanced_accuracy_score)

mpl.rcParams["figure.figsize"] = [10, 7]
mpl.use("Agg")


class LogMetrics:
    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters

        self.task = parameters["TASK"]
        self.output = parameters["DATA_OUTPUT"]

    def compute(self, train_data, test_data):
        train_X, train_y = train_data
        test_X, test_y = test_data

        train_preds = self.model.predict(train_X)
        test_preds = self.model.predict(test_X)

        train_y = train_y.flatten()
        train_preds = train_preds.flatten()
        test_y = test_y.flatten()
        test_preds = test_preds.flatten()

        if self.task.lower() == "regression":
            train_rmse = mean_squared_error(train_y, train_preds,
                                            squared=False)
            train_mae = mean_absolute_error(train_y, train_preds)
            train_r2 = r2_score(train_y, train_preds)

            test_rmse = mean_squared_error(test_y, test_preds, squared=False)
            test_mae = mean_absolute_error(test_y, test_preds)
            test_r2 = r2_score(test_y, test_preds)

            data = {"true": test_y, "pred": test_preds}
            df_reg = pd.DataFrame.from_dict(data)

            fig = plt.figure(figsize=(10, 7))
            _ = sns.regplot(data=df_reg, x="true",
                            y="pred", fit_reg=True)
            ax = fig.axes[0]
            anchored_text = AnchoredText(f"R2 = {round(test_r2, 4)}", loc=2)
            ax.add_artist(anchored_text)

            path = Path(self.output)
            plt.savefig(str(path / "test_r2.png"), bbox_inches="tight")
            plt.close("all")

            metrics = {"train_rmse": train_rmse,
                       "train_mae": train_mae,
                       "train_r2": train_r2,
                       "test_rmse": test_rmse,
                       "test_mae": test_mae,
                       "test_r2": test_r2}

        else:
            train_acc = accuracy_score(train_y, train_preds)
            train_acc_bal = balanced_accuracy_score(train_y, train_preds)
            train_f1 = f1_score(train_y, train_preds)
            train_prec = precision_score(train_y, train_preds)
            train_rec = recall_score(train_y, train_preds)
            train_mcc = matthews_corrcoef(train_y, train_preds)
            train_roc = roc_auc_score(train_y, train_preds)
            train_ks = ks_score(train_y, train_preds)

            test_acc = accuracy_score(test_y, test_preds)
            test_acc_bal = balanced_accuracy_score(test_y, test_preds)
            test_f1 = f1_score(test_y, test_preds)
            test_prec = precision_score(test_y, test_preds)
            test_rec = recall_score(test_y, test_preds)
            test_mcc = matthews_corrcoef(test_y, test_preds)
            test_roc = roc_auc_score(test_y, test_preds)
            test_ks = ks_score(test_y, test_preds)

            path = Path(self.output)

            _ = cm_disp.from_predictions(test_y, test_preds,
                                         colorbar=False,
                                         cmap="Blues",
                                         normalize=None)
            plt.savefig(str(path / "test_cm.png"), bbox_inches="tight")
            plt.close("all")

            _ = cm_disp.from_predictions(test_y, test_preds,
                                         colorbar=False,
                                         cmap="Blues",
                                         normalize="true")
            plt.savefig(str(path / "test_norm_cm.png"), bbox_inches="tight")
            plt.close("all")

            metrics = {"train_acc": train_acc,
                       "train_acc_bal": train_acc_bal,
                       "train_f1": train_f1,
                       "train_prec": train_prec,
                       "train_rec": train_rec,
                       "train_mcc": train_mcc,
                       "train_roc": train_roc,
                       "train_ks": train_ks,

                       "test_acc": test_acc,
                       "test_acc_bal": test_acc_bal,
                       "test_f1": test_f1,
                       "test_prec": test_prec,
                       "test_rec": test_rec,
                       "test_mcc": test_mcc,
                       "test_roc": test_roc,
                       "test_ks": test_ks}

        return metrics
