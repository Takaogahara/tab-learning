import os
import pickle
import pandas as pd
from pathlib import Path
from sklearn_pandas import DataFrameMapper

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class Dataprocessor:
    def __init__(self, parameters):
        self.parameters = parameters

    def fit_transform(self, data, col_feat):

        int_columns = [c for c in data.columns if data[c].dtypes == int]
        float_columns = [c for c in data.columns if data[c].dtypes == float]
        cat_cols = [c for c in data.columns if data[c].dtypes == object]

        # INT OR FLOAT
        numerical_cols = int_columns + float_columns
        if len(numerical_cols) > 0:
            mapper_int = DataFrameMapper([(numerical_cols, StandardScaler()),
                                          (numerical_cols, SimpleImputer(
                                              strategy="mean"))])

            X_transform = mapper_int.fit_transform(data.copy(deep=True))

            fill_list = ["None"] * (X_transform.shape[1] // 2)
            rename_list = numerical_cols + fill_list

            X_transform = pd.DataFrame(X_transform, columns=rename_list)
            X_transform = X_transform.iloc[:, 0:(X_transform.shape[1] // 2)]

            X_transform = pd.concat([data[cat_cols], X_transform],
                                    axis=1, join="inner")
            self.mapper_int = mapper_int

        # CATEGORICAL
        cat_columns = []
        categorical_dims = {}

        for col in X_transform.columns[X_transform.dtypes == object]:
            l_enc = LabelEncoder()
            X_transform[col] = X_transform[col].fillna("Blank")
            X_transform[col] = l_enc.fit_transform(X_transform[col].values)
            cat_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)

        cat_idxs = [i for i, f in enumerate(col_feat) if f in cat_columns]
        cat_dims = [categorical_dims[f]
                    for i, f in enumerate(col_feat) if f in cat_columns]

        if len(cat_columns) > 0:
            self.label_enc = l_enc
        else:
            self.label_enc = None

        return X_transform, (cat_idxs, cat_dims)

    def export(self):
        mapper_int = self.mapper_int
        label_enc = self.label_enc

        export = {"numerical": mapper_int,
                  "categorical": label_enc}
        self.scaler = export

        export_path = Path(os.getcwd())
        export_path = export_path / "output"
        os.makedirs(export_path, exist_ok=True)
        pkl_path = str(export_path / "dataprocessor.pkl")

        pickle.dump(export, open(pkl_path, "wb"))

        return export_path

    def transform(self, data, load=None):
        # TODO CLIP ON MAX AND MIN FROM FIT

        if load is not None:
            path = Path(load)
            path = str(path / "dataprocessor.pkl")
            with open(path, "rb") as f:
                scal_dict = pickle.load(f)
        else:
            scal_dict = self.scaler

        scaler_num = scal_dict["numerical"]
        scaler_cat = scal_dict["categorical"]
        int_columns = [c for c in data.columns if data[c].dtypes == int]
        float_columns = [c for c in data.columns if data[c].dtypes == float]
        cat_cols = [c for c in data.columns if data[c].dtypes == object]

        # INT OR FLOAT
        numerical_cols = int_columns + float_columns
        X_transform = scaler_num.transform(data.copy(deep=True))

        fill_list = ["None"] * (X_transform.shape[1] // 2)
        rename_list = numerical_cols + fill_list

        X_transform = pd.DataFrame(X_transform, columns=rename_list)
        X_transform = X_transform.iloc[:, 0:(X_transform.shape[1] // 2)]

        X_transform = pd.concat([data[cat_cols], X_transform],
                                axis=1, join="inner")

        # CATEGORICAL
        cat_columns = []
        categorical_dims = {}

        for col in X_transform.columns[X_transform.dtypes == object]:
            X_transform[col] = X_transform[col].fillna("Blank")
            X_transform[col] = scaler_cat.transform(X_transform[col].values)
            cat_columns.append(col)
            categorical_dims[col] = len(scaler_cat.classes_)

        return X_transform
