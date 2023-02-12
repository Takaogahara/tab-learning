import os
import pickle
import pandas as pd
from pathlib import Path
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer


class Dataprocessor:
    def __init__(self, parameters):
        self.parameters = parameters

    def fit_transform(self, X_train):
        df_x = X_train.copy(deep=True)
        col_feat = self.parameters["DATA_COL_FEAT"]

        imputers = None
        encoders = None
        scalers = None

        int_columns = [c for c in df_x.columns if df_x[c].dtypes == int]
        float_columns = [c for c in df_x.columns if df_x[c].dtypes == float]
        cat_cols = [c for c in df_x.columns if df_x[c].dtypes == object]

        # INT OR FLOAT
        numerical_cols = int_columns + float_columns
        if len(numerical_cols) > 0:
            # Impute data
            numerical_imputer = MeanMedianImputer(imputation_method="median",
                                                  variables=numerical_cols)
            numerical_imputer.fit(df_x)
            df_x = numerical_imputer.transform(df_x)

            # Scale values
            scaler = DataFrameMapper([(cat_cols, None),
                                      (numerical_cols, RobustScaler())],
                                     df_out=False)
            scaler.fit(df_x)
            df_x = scaler.transform(df_x)

            columns = cat_cols + numerical_cols
            df_x = pd.DataFrame(df_x, columns=columns)
            df_x = df_x[list(X_train.columns)]

            # Store objects
            imputers = [numerical_imputer]
            scalers = [scaler]

        # CATEGORICAL
        if len(cat_cols) > 0:
            # Impute data
            categorical_imputer = CategoricalImputer(variables=cat_cols,
                                                     fill_value="Missing")
            categorical_imputer.fit(df_x)
            df_x = categorical_imputer.transform(df_x)

            # Encode rare categories
            rare_encoder = RareLabelEncoder(tol=0.01, n_categories=5,
                                            max_n_categories=10,
                                            variables=cat_cols,
                                            replace_with="Rare")
            rare_encoder.fit(df_x)
            df_x = rare_encoder.transform(df_x)

            # Encode categorical values
            cat_columns = []
            categorical_dims = {}
            unique_list = []

            # Get all unique categorical values
            for col in cat_cols:
                unique = list(df_x[col].unique())
                unique_list = unique_list + unique

            # Fit encoder
            unique_list = list(set(unique_list))
            lbl_encoder = LabelEncoder()
            lbl_encoder.fit(unique_list)

            for col in cat_cols:
                df_x[col] = lbl_encoder.transform(df_x[col].values)
                cat_columns.append(col)
                categorical_dims[col] = len(lbl_encoder.classes_)

            cat_idxs = [i for i, f in enumerate(col_feat) if f in cat_columns]
            cat_dims = [categorical_dims[f]
                        for i, f in enumerate(col_feat) if f in cat_columns]

            # Store objects
            imputers = imputers + [categorical_imputer]
            encoders = [rare_encoder, lbl_encoder]

        # Save objects
        self.imputers = imputers
        self.encoders = encoders
        self.scalers = scalers
        self.ranges = [df_x.min(), df_x.max()]

        export_path = self._export_obj()

        return df_x, (cat_idxs, cat_dims), export_path

    def _export_obj(self):
        export = {"imputers": self.imputers,
                  "encoders": self.encoders,
                  "scalers": self.scalers,
                  "ranges": self.ranges}

        self.fitted = export

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
                fitted_dict = pickle.load(f)
        else:
            fitted_dict = self.fitted

        df_x = data.copy(deep=True)
        imputers = fitted_dict["imputers"]
        encoders = fitted_dict["encoders"]
        scalers = fitted_dict["scalers"]
        ranges = fitted_dict["ranges"]

        int_columns = [c for c in df_x.columns if df_x[c].dtypes == int]
        float_columns = [c for c in df_x.columns if df_x[c].dtypes == float]
        cat_cols = [c for c in df_x.columns if df_x[c].dtypes == object]

        # INT OR FLOAT
        numerical_cols = int_columns + float_columns
        if len(numerical_cols) > 0:
            range_min, range_max = ranges
            numerical_imputer = imputers[0]
            scaler = scalers[0]

            df_x = numerical_imputer.transform(df_x)
            df_x = scaler.transform(df_x)

            columns = cat_cols + int_columns + float_columns
            df_x = pd.DataFrame(df_x, columns=columns)
            df_x = df_x[list(data.columns)]

            for col in numerical_cols:
                sclr = MinMaxScaler(feature_range=(
                    range_min[col], range_max[col]))
                df_x[col] = sclr.fit_transform(df_x[[col]])

        # CATEGORICAL
        if len(cat_cols) > 0:
            categorical_imputer = imputers[1]
            # rare_encoder, woe_encoder = encoders
            rare_encoder, lbl_encoder = encoders

            df_x = categorical_imputer.transform(df_x)
            df_x = rare_encoder.transform(df_x)
            # df_x = lbl_encoder.transform(df_x)

            for col in cat_cols:
                df_x[col] = lbl_encoder.transform(df_x[col].values)

        return df_x
