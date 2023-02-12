import os
import pickle
import pandas as pd
from pathlib import Path
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from feature_engine.preprocessing import MatchVariables
from feature_engine.encoding import RareLabelEncoder, WoEEncoder, OneHotEncoder
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer


class Dataprocessor:
    def __init__(self, parameters):
        self.parameters = parameters

    def fit_transform(self, X_train, y_train):
        df_x = X_train.copy(deep=True)
        df_y = y_train.copy(deep=True)
        col_feat = self.parameters["DATA_COL_FEAT"]
        coder = self.parameters["TAB_CAT_ENCODER"]

        int_columns = [c for c in df_x.columns if df_x[c].dtypes == int]
        float_columns = [c for c in df_x.columns if df_x[c].dtypes == float]
        numerical_cols = int_columns + float_columns
        cat_cols = [c for c in df_x.columns if df_x[c].dtypes == object]
        raw_cols = list(X_train.columns)

        # Pre process dataframe
        cols = [numerical_cols, cat_cols, raw_cols]
        df_x, formater = _fit_features(df_x, cols)

        # Encode categorical data
        cols = [cat_cols, col_feat]
        df_x, cat_params, cat_encoder = _categorical_encoder(df_x, df_y,
                                                             coder, cols)

        # Save objects
        self.num_imputer = formater[0]
        self.scaler = formater[1]
        self.cat_imputer = formater[2]
        self.rare_encoder = formater[3]
        self.match_columns = formater[4]
        self.cat_encoder = cat_encoder
        self.ranges = [df_x.min(), df_x.max()]

        export_path = self._export_obj()

        return df_x, cat_params, export_path

    def _export_obj(self):
        export = {"num_imputer": self.num_imputer,
                  "scaler": self.scaler,
                  "cat_imputer": self.cat_imputer,
                  "rare_encoder": self.rare_encoder,
                  "match_columns": self.match_columns,
                  "cat_encoder": self.cat_encoder,
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
                formater_dict = pickle.load(f)
        else:
            formater_dict = self.fitted

        df_x = data.copy(deep=True)

        int_columns = [c for c in df_x.columns if df_x[c].dtypes == int]
        float_columns = [c for c in df_x.columns if df_x[c].dtypes == float]
        numerical_cols = int_columns + float_columns
        cat_cols = [c for c in df_x.columns if df_x[c].dtypes == object]
        raw_cols = list(data.columns)

        cols = [numerical_cols, cat_cols, raw_cols]
        df_x = _transform_features(df_x, formater_dict, cols)

        return df_x


def _fit_features(df_x, columns):
    numerical_cols = columns[0]
    cat_cols = columns[1]
    raw_cols = columns[2]

    formater = [None] * 5
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
        df_x = df_x[raw_cols]

        # Store objects
        formater[0] = numerical_imputer
        formater[1] = scaler

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

        # Store objects
        formater[2] = categorical_imputer
        formater[3] = rare_encoder

    # Match Train/Test columns
    match_columns = MatchVariables("Missing", verbose=False)
    match_columns.fit(df_x)
    formater[4] = match_columns

    return df_x, formater


def _transform_features(df_x, formater_dict, columns):
    num_imputer = formater_dict["num_imputer"]
    scaler = formater_dict["scaler"]
    cat_imputer = formater_dict["cat_imputer"]
    rare_encoder = formater_dict["rare_encoder"]
    match_columns = formater_dict["match_columns"]
    cat_encoder = formater_dict["cat_encoder"]
    ranges = formater_dict["ranges"]
    range_min, range_max = ranges

    num_cols = columns[0]
    cat_cols = columns[1]
    raw_cols = columns[2]

    if len(num_cols) > 0:
        df_x = num_imputer.transform(df_x)
        df_x = scaler.transform(df_x)

        df_cols = cat_cols + num_cols
        df_x = pd.DataFrame(df_x, columns=df_cols)
        df_x = df_x[raw_cols]

        for col in num_cols:
            sclr = MinMaxScaler(feature_range=(
                range_min[col], range_max[col]))
            df_x[col] = sclr.fit_transform(df_x[[col]])

    if len(cat_cols) > 0:
        df_x = cat_imputer.transform(df_x)
        df_x = rare_encoder.transform(df_x)

    df_x = match_columns.transform(df_x)

    if len(cat_cols) > 0:
        coder = cat_encoder.__class__.__name__

        if coder.lower() == "labelencoder":
            for col in cat_cols:
                df_x[col] = cat_encoder.transform(df_x[col].values)
        else:
            df_x = cat_encoder.transform(df_x)

    return df_x


def _categorical_encoder(df_x, df_y, encoder, columns):
    cat_cols = columns[0]
    col_feat = columns[1]

    cat_idxs = []
    cat_dims = []

    # Encode categorical values
    if encoder.lower() == "default":
        cat_columns = []
        categorical_dims = {}
        unique_list = []

        # Get all unique categorical values
        for col in cat_cols:
            unique = list(df_x[col].unique())
            unique_list = unique_list + unique

        # Fit encoder
        unique_list = list(set(unique_list))
        cat_encoder = LabelEncoder()
        cat_encoder.fit(unique_list)

        for col in cat_cols:
            df_x[col] = cat_encoder.transform(df_x[col].values)
            cat_columns.append(col)
            categorical_dims[col] = len(cat_encoder.classes_)

        cat_idxs = [i for i, f in enumerate(col_feat) if f in cat_columns]
        cat_dims = [categorical_dims[f] for i, f in enumerate(col_feat)
                    if f in cat_columns]

    elif encoder.lower() == "onehot":
        cat_encoder = OneHotEncoder(top_categories=10,
                                    variables=cat_cols)
        cat_encoder.fit(df_x)
        df_x = cat_encoder.transform(df_x)

    elif encoder.lower() == "woe":
        cat_encoder = WoEEncoder(variables=cat_cols)
        cat_encoder.fit(df_x, df_y)
        df_x = cat_encoder.transform(df_x)

    else:
        raise RuntimeError("Select a valid Encoder")

    cat_params = (cat_idxs, cat_dims)
    return df_x, cat_params, cat_encoder
