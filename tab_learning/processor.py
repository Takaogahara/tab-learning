import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn_pandas import DataFrameMapper

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler


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
            mapper_int = DataFrameMapper([(numerical_cols, RobustScaler()),
                                          (numerical_cols, SimpleImputer(
                                              strategy="mean"))])

            X_transform = mapper_int.fit_transform(data.copy(deep=True))

            fill_list = ["None"] * (X_transform.shape[1] // 2)
            rename_list = numerical_cols + fill_list

            X_transform = pd.DataFrame(X_transform, columns=rename_list)
            X_transform = X_transform.iloc[:, 0:(X_transform.shape[1] // 2)]

            self.scale_min = min(X_transform.min())
            self.scale_max = max(X_transform.max())

            X_transform = pd.concat([data[cat_cols], X_transform],
                                    axis=1, join="inner")
            self.mapper_int = mapper_int

        # CATEGORICAL
        cat_columns = []
        categorical_dims = {}
        unique_list = []

        for col in X_transform.columns[X_transform.dtypes == object]:
            unique = list(X_transform[col].unique())
            unique_list = unique_list + unique

        unique_list = list(set(unique_list)) + ["Other"]
        l_enc = LabelEncoder()
        l_enc = l_enc.fit(unique_list)

        for col in X_transform.columns[X_transform.dtypes == object]:
            X_transform[col] = X_transform[col].fillna("Other")

            popular = X_transform[col].value_counts(
                                    ).sort_values(ascending=False).keys()[0]
            chosen = X_transform[X_transform[col] == popular]
            chosen = chosen.sample(1, random_state=8)[col].keys()[0]
            X_transform.loc[chosen, col] = "Other"

            X_transform[col] = l_enc.transform(X_transform[col].values)
            cat_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)

        cat_idxs = [i for i, f in enumerate(col_feat) if f in cat_columns]
        cat_dims = [categorical_dims[f]
                    for i, f in enumerate(col_feat) if f in cat_columns]

        if len(cat_columns) > 0:
            self.label_enc = l_enc
            self.cat_meta = self._get_cat_metadata(X_transform, cat_columns)

        else:
            self.label_enc = None
            self.cat_meta = None

        return X_transform, (cat_idxs, cat_dims)

    def _get_cat_metadata(self, dataframe, cat_columns):
        df_cat = dataframe[cat_columns]
        ranges = df_cat.apply(lambda col: col.unique()).to_dict()
        ranges_dict = {key: list(value) for key, value in ranges.items()}

        return ranges_dict

    def export(self):
        mapper_int = self.mapper_int
        min_max = (self.scale_min, self.scale_max)
        label_enc = self.label_enc
        label_meta = self.cat_meta

        export = {"numerical": mapper_int,
                  "numerical_metadata": min_max,
                  "categorical": label_enc,
                  "categorical_metadata": label_meta}
        self.scaler = export

        export_path = Path(os.getcwd())
        export_path = export_path / "output"
        os.makedirs(export_path, exist_ok=True)
        pkl_path = str(export_path / "dataprocessor.pkl")

        pickle.dump(export, open(pkl_path, "wb"))

        return export_path

    def transform(self, data, load=None):
        if load is not None:
            path = str(load / "dataprocessor.pkl")
            with open(path, "rb") as f:
                scal_dict = pickle.load(f)
        else:
            scal_dict = self.scaler

        scaler_num = scal_dict["numerical"]
        scaler_min, scaler_max = scal_dict["numerical_metadata"]
        scaler_cat = scal_dict["categorical"]
        metadata_cat = scal_dict["categorical_metadata"]

        int_columns = [c for c in data.columns if data[c].dtypes == int]
        float_columns = [c for c in data.columns if data[c].dtypes == float]
        cat_cols = [c for c in data.columns if data[c].dtypes == object]

        # INT OR FLOAT
        numerical_cols = int_columns + float_columns
        X_transform = scaler_num.transform(data.copy(deep=True))

        minmax_scaler = MinMaxScaler((scaler_min, scaler_max))
        X_transform = minmax_scaler.fit_transform(X_transform)

        fill_list = ["None"] * (X_transform.shape[1] // 2)
        rename_list = numerical_cols + fill_list

        X_transform = pd.DataFrame(X_transform, columns=rename_list)
        X_transform = X_transform.iloc[:, 0:(X_transform.shape[1] // 2)]

        X_transform = pd.concat([data[cat_cols], X_transform],
                                axis=1, join="inner")

        # CATEGORICAL
        other = np.where(scaler_cat.classes_ == "Other")[0][0]

        for col in X_transform.columns[X_transform.dtypes == object]:
            X_transform[col] = X_transform[col].fillna("Other")
            X_transform[col] = scaler_cat.transform(X_transform[col].values)

            unique_vals = list(X_transform[col].unique())
            if not set(unique_vals).issubset(metadata_cat[col]):
                diffs = set(unique_vals).difference(set(metadata_cat[col]))
                diffs = list(diffs)
                for diff in diffs:
                    X_transform[X_transform[col] == diff][col] = other

        return X_transform
