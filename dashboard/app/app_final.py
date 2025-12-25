import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go


def _try_import_shap():
    try:
        import shap
        return shap
    except Exception:
        return None

@st.cache_resource
def _get_shap_explainer(_model):
    shap = _try_import_shap()
    if shap is None:
        return None
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None

def _xgb_pred_contribs(model, X_df: pd.DataFrame):
    try:
        import xgboost as xgb
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        dm = xgb.DMatrix(X_df, feature_names=list(X_df.columns))
        contrib = booster.predict(dm, pred_contribs=True)
        if contrib is None or contrib.shape[1] < 2:
            return None, None
        base_value = float(contrib[0, -1])
        shap_vals = contrib[:, :-1]
        return base_value, shap_vals
    except Exception:
        return None, None

def _plot_waterfall_custom(base_value: float, contrib: pd.Series, max_display: int = 10):
    contrib = contrib.copy()
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    top = contrib.head(max_display)
    others = contrib.iloc[max_display:].sum() if len(contrib) > max_display else 0.0

    labels = list(top.index)
    values = list(top.values)
    if len(contrib) > max_display:
        labels.append("Other features")
        values.append(float(others))

    measure = ["relative"] * len(values)
    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=measure,
        x=labels,
        y=values,
        connector={"line": {"width": 1}},
    ))
    fig.update_layout(
        title="SHAP waterfall (log-price)",
        height=300,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


try:
    from streamlit_folium import st_folium
    import folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False


st.set_page_config(page_title="Pricing Advisor Dashboard", layout="wide")

DATA_TRAIN_MAP_CSV_CANDIDATES = [
    "train_with_PID_latlon_FINAL.csv",
    "dataset/train_with_PID_latlon_FINAL.csv",
]
DATA_TRAIN_MODEL_CSV_CANDIDATES = [
    "dataset/train.csv",
    "train.csv",
    "train_with_PID_latlon_FINAL.csv",
]
DATA_TEST_CSV_CANDIDATES = [
    "dataset/test.csv",
    "test.csv",
]
MODEL_PKL_CANDIDATES = [
    "models/xgboost_model.pkl",
    "xgboost_model.pkl",
    "cache/best_xgb.pkl",
]
FEATURE_INFO_PKL_CANDIDATES = [
    "models/feature_info.pkl",
    "feature_info.pkl",
]
CACHE_TRAIN_ENG_PKL_CANDIDATES = [
    "cache/train_engineered.pkl",
    "train_engineered.pkl",
]
CACHE_TEST_ENG_PKL_CANDIDATES = [
    "cache/test_engineered.pkl",
    "test_engineered.pkl",
]

MAP_LAT_COL_CANDIDATES = ["Latitude", "Lat", "latitude", "LAT"]
MAP_LON_COL_CANDIDATES = ["Longitude", "Lon", "longitude", "LON", "lng", "Long"]


def _pick_first_existing_path(candidates):
    for p in candidates:
        try:
            if Path(p).exists():
                return p
        except Exception:
            continue
    return candidates[0] if candidates else None

def _pick_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _haversine_km(lat1, lon1, lat2, lon2):

    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R*c

def scroll_to_top():
    st.components.v1.html(
        """
        <script>
        window.parent.document.querySelector('section.main').scrollTo({top: 0, behavior: 'smooth'});
        </script>
        """,
        height=0,
    )

def _safe_mode(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return None
    try:
        return s.mode().iloc[0]
    except Exception:
        return s.iloc[0]

def _get_model_feature_importances(model, feature_names):
    imp = None
    if hasattr(model, "feature_importances_"):
        try:
            imp = np.asarray(model.feature_importances_, dtype=float)
        except Exception:
            imp = None


    if imp is None or (len(imp) != len(feature_names)):
        try:
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")

            vals = np.zeros(len(feature_names), dtype=float)
            for k, v in score.items():
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if 0 <= idx < len(vals):
                        vals[idx] = float(v)
            imp = vals
        except Exception:
            imp = np.ones(len(feature_names), dtype=float)

    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)

def _raw_feature_importance_from_engineered(raw_cols, engineered_importance: pd.Series):
    raw_imp = {}
    for rc in raw_cols:
        if rc in engineered_importance.index:
            raw_imp[rc] = float(engineered_importance.get(rc, 0.0))
            continue

        pref = rc + "_"
        s = engineered_importance[engineered_importance.index.to_series().astype(str).str.startswith(pref)]
        raw_imp[rc] = float(s.sum()) if len(s) else 0.0
    return pd.Series(raw_imp).sort_values(ascending=False)


def _group_raw_feature(feat: str) -> str:
    f = feat.lower()


    if any(k in f for k in ["neighborhood", "condition1", "condition2"]):
        return "📍 Location"


    if any(k in f for k in [
        "lotarea", "lotfrontage", "lotshape", "landcontour", "landslope", "lotconfig",
        "street", "alley", "utilities", "land", "lot"
    ]):
        return "📐 Lot & Land"


    if any(k in f for k in ["mssubclass", "mszoning", "bldgtype", "housestyle"]):
        return "🏠 Building Type & Style"


    if any(k in f for k in [
        "exterior1st", "exterior2nd", "roofstyle", "roofmatl",
        "masvnr", "foundation", "exterqual", "extercond"
    ]):
        return "🧱 Exterior & Structure"


    if any(k in f for k in ["overallqual", "overallcond", "functional", "heatingqc", "kitchenqual", "fireplacequ"]):
        return "⭐ Quality & Condition"


    if any(k in f for k in [
        "grlivarea", "1stflrsf", "2ndflrsf", "lowqualfinsf",
        "totrmsabvgrd", "bedroomabvgr", "kitchenabvgr",
        "fullbath", "halfbath"
    ]):
        return "🛋️ Living Area, Rooms & Baths"


    if "bsmt" in f or "basement" in f or "totalbsmtsf" in f:
        return "🏗️ Basement"


    if "garage" in f:
        return "🚗 Garage"


    if any(k in f for k in ["heating", "centralair", "electrical"]):
        return "⚙️ Systems"


    if any(k in f for k in ["porch", "deck", "pool", "fence", "screenporch", "3ssnporch", "openporch", "enclosedporch", "miscfeature", "miscval"]):
        return "✨ Outdoor & Amenities"


    if any(k in f for k in ["mosold", "yrsold", "saletype", "salecondition"]):
        return "🧾 Sale Info"

    return "📦 Other"


def _safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _value_changed(user_val, default_val, is_numeric: bool) -> bool:
    if is_numeric:
        a = _safe_float(user_val)
        b = _safe_float(default_val)
        if np.isnan(a) and np.isnan(b):
            return False
        if np.isnan(a) != np.isnan(b):
            return True
        return abs(a - b) > 1e-9

    a = None if user_val is None else str(user_val)
    b = None if default_val is None else str(default_val)
    if a in ["nan", "None"]:
        a = None
    if b in ["nan", "None"]:
        b = None
    return a != b


def _compute_similarity_raw(
    df_candidates: pd.DataFrame,
    subject_raw: pd.Series,
    train_raw: pd.DataFrame,
    raw_feature_cols: list[str],
    raw_imp: dict,
    changed_features: list[str] | None = None,
    changed_boost: float = 2.5,
) -> np.ndarray:
    changed_features = changed_features or []

    idx_map = {c: i for i, c in enumerate(raw_feature_cols)}


    num_cols = [c for c in raw_feature_cols if c in train_raw.columns and pd.api.types.is_numeric_dtype(train_raw[c])]
    cat_cols = [c for c in raw_feature_cols if c not in num_cols]


    w_base = np.array([float(raw_imp.get(c, 0.0)) for c in raw_feature_cols], dtype=float)
    if np.all(w_base <= 0):
        w_base = np.ones(len(raw_feature_cols), dtype=float)
    else:

        w_base = w_base / (np.mean(w_base[w_base > 0]) + 1e-12)


    w = w_base.copy()
    if changed_features:
        changed_set = set(map(str, changed_features))
        for i, c in enumerate(raw_feature_cols):
            if c in changed_set:
                w[i] *= (1.0 + float(changed_boost))


    dfc = df_candidates.reindex(columns=raw_feature_cols)


    dist = np.zeros(len(dfc), dtype=float)


    scales = {}
    for c in num_cols:
        s = train_raw[c]
        q75 = float(np.nanpercentile(s.to_numpy(dtype=float), 75)) if s.notna().any() else np.nan
        q25 = float(np.nanpercentile(s.to_numpy(dtype=float), 25)) if s.notna().any() else np.nan
        iqr = q75 - q25 if (not np.isnan(q75) and not np.isnan(q25)) else np.nan
        if np.isnan(iqr) or iqr <= 1e-12:
            sd = float(np.nanstd(s.to_numpy(dtype=float))) if s.notna().any() else np.nan
            scale = sd if (not np.isnan(sd) and sd > 1e-12) else 1.0
        else:
            scale = iqr
        scales[c] = scale


    for c in num_cols:
        x0 = _safe_float(subject_raw.get(c, np.nan))
        x = pd.to_numeric(dfc[c], errors="coerce").to_numpy(dtype=float)
        sc = float(scales.get(c, 1.0))
        d = np.abs(x - x0) / sc


        x_nan = np.isnan(x)
        x0_nan = np.isnan(x0)
        if x0_nan:
            d = np.where(x_nan, 0.0, 1.0)
        else:
            d = np.where(x_nan, 1.0, d)

        wi = float(w[idx_map[c]])
        dist += wi * d


    for c in cat_cols:
        x0 = subject_raw.get(c, None)
        x0 = None if x0 is None else str(x0)
        if x0 in ["nan", "None"]:
            x0 = None

        s = dfc[c]

        s_isna = s.isna().to_numpy()
        x = s.astype(str).to_numpy()

        if x0 is None:
            d = np.where(s_isna, 0.0, 1.0)
        else:
            d = (x != x0).astype(float)
            d = np.where(s_isna, 1.0, d)

        wi = float(w[idx_map[c]])
        dist += wi * d


    w_sum = float(np.sum(w)) + 1e-12
    dist_norm = dist / w_sum
    sim = 1.0 / (1.0 + dist_norm)
    return sim


NA_MEANS_NONE = {
    'Alley': 'No_Alley',
    'BsmtQual': 'No_Basement',
    'BsmtCond': 'No_Basement',
    'BsmtExposure': 'No_Basement',
    'BsmtFinType1': 'No_Basement',
    'BsmtFinType2': 'No_Basement',
    'FireplaceQu': 'No_Fireplace',
    'GarageType': 'No_Garage',
    'GarageFinish': 'No_Garage',
    'GarageQual': 'No_Garage',
    'GarageCond': 'No_Garage',
    'PoolQC': 'No_Pool',
    'Fence': 'No_Fence',
    'MiscFeature': 'No_Misc',
    'MasVnrArea': 0,
    'BsmtFullBath': 0,
    'BsmtHalfBath': 0,
    'BsmtFinSF1': 0,
    'BsmtFinSF2': 0,
    'BsmtUnfSF': 0,
    'TotalBsmtSF': 0,
    'GarageCars': 0,
    'GarageArea': 0
}

ORDINAL_FEATURES = {
    'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtQual': ['No_Basement', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtCond': ['No_Basement', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtExposure': ['No_Basement', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['No_Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['No_Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'FireplaceQu': ['No_Fireplace', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageFinish': ['No_Garage', 'Unf', 'RFn', 'Fin'],
    'GarageQual': ['No_Garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageCond': ['No_Garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'PoolQC': ['No_Pool', 'Fa', 'TA', 'Gd', 'Ex'],
    'Fence': ['No_Fence', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
    'LotShape': ['IR3', 'IR2', 'IR1', 'Reg'],
    'LandSlope': ['Sev', 'Mod', 'Gtl'],
    'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    'PavedDrive': ['N', 'P', 'Y']
}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.copy()


    if all(col in df_eng.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df_eng['TotalSF'] = df_eng['TotalBsmtSF'] + df_eng['1stFlrSF'] + df_eng['2ndFlrSF']


    if all(col in df_eng.columns for col in ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']):
        df_eng['TotalBath'] = (
            df_eng['BsmtFullBath']
            + 0.5 * df_eng['BsmtHalfBath']
            + df_eng['FullBath']
            + 0.5 * df_eng['HalfBath']
        )


    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    if all(col in df_eng.columns for col in porch_cols):
        df_eng['TotalPorchSF'] = df_eng[porch_cols].sum(axis=1)


    if all(col in df_eng.columns for col in ['YrSold', 'YearBuilt']):
        df_eng['HouseAge'] = df_eng['YrSold'] - df_eng['YearBuilt']
        df_eng['HouseAge'] = df_eng['HouseAge'].clip(lower=0)


    if all(col in df_eng.columns for col in ['YrSold', 'YearRemodAdd']):
        df_eng['YearsSinceRemodel'] = df_eng['YrSold'] - df_eng['YearRemodAdd']
        df_eng['YearsSinceRemodel'] = df_eng['YearsSinceRemodel'].clip(lower=0)


    if all(col in df_eng.columns for col in ['YearRemodAdd', 'YearBuilt']):
        df_eng['WasRemodeled'] = (df_eng['YearRemodAdd'] != df_eng['YearBuilt']).astype(int)


    if all(col in df_eng.columns for col in ['YrSold', 'YearBuilt']):
        df_eng['IsNewHouse'] = (df_eng['YrSold'] == df_eng['YearBuilt']).astype(int)


    if '2ndFlrSF' in df_eng.columns:
        df_eng['Has2ndFloor'] = (df_eng['2ndFlrSF'] > 0).astype(int)


    if 'GarageArea' in df_eng.columns:
        df_eng['HasGarage'] = (df_eng['GarageArea'] > 0).astype(int)


    if 'TotalBsmtSF' in df_eng.columns:
        df_eng['HasBasement'] = (df_eng['TotalBsmtSF'] > 0).astype(int)


    if 'Fireplaces' in df_eng.columns:
        df_eng['HasFireplace'] = (df_eng['Fireplaces'] > 0).astype(int)


    if 'PoolArea' in df_eng.columns:
        df_eng['HasPool'] = (df_eng['PoolArea'] > 0).astype(int)


    if all(col in df_eng.columns for col in ['GrLivArea', 'TotRmsAbvGrd']):
        denom = df_eng['TotRmsAbvGrd'].replace(0, np.nan)
        df_eng['AreaPerRoom'] = (df_eng['GrLivArea'] / denom).fillna(0)


    if all(col in df_eng.columns for col in ['LotFrontage', 'LotArea']):
        denom = df_eng['LotArea'].replace(0, np.nan)
        df_eng['FrontageRatio'] = (df_eng['LotFrontage'] / denom).fillna(0)


    if all(col in df_eng.columns for col in ['BsmtFinSF1', 'TotalBsmtSF']):
        denom = df_eng['TotalBsmtSF'].replace(0, np.nan)
        df_eng['BsmtFinishRatio'] = (df_eng['BsmtFinSF1'] / denom).fillna(0)


    if all(col in df_eng.columns for col in ['GarageArea', 'GrLivArea']):
        denom = df_eng['GrLivArea'].replace(0, np.nan)
        df_eng['GarageRatio'] = (df_eng['GarageArea'] / denom).fillna(0)

    return df_eng


@st.cache_resource
def load_assets():

    train_map_csv = _pick_first_existing_path(DATA_TRAIN_MAP_CSV_CANDIDATES)
    train_model_csv = _pick_first_existing_path(DATA_TRAIN_MODEL_CSV_CANDIDATES)
    test_csv = _pick_first_existing_path(DATA_TEST_CSV_CANDIDATES)


    train_full = pd.read_csv(train_map_csv) if train_map_csv and Path(train_map_csv).exists() else pd.read_csv(train_model_csv)


    train_model = pd.read_csv(train_model_csv)
    test_raw = pd.read_csv(test_csv)


    lat_col = _pick_first_existing(train_full.columns, MAP_LAT_COL_CANDIDATES)
    lon_col = _pick_first_existing(train_full.columns, MAP_LON_COL_CANDIDATES)


    raw_cols = list(test_raw.columns)


    keep_cols = [c for c in (raw_cols + ["SalePrice"]) if c in train_model.columns]
    train_raw = train_model[keep_cols].copy()


    model_path = _pick_first_existing_path(MODEL_PKL_CANDIDATES)
    feature_info_path = _pick_first_existing_path(FEATURE_INFO_PKL_CANDIDATES)
    model = joblib.load(model_path)
    feature_info = joblib.load(feature_info_path)
    feature_names = feature_info["feature_names"]

    return train_full, train_raw, test_raw, lat_col, lon_col, model, feature_names


def build_preprocessor(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train_processed = train_raw.copy()
    test_processed = test_raw.copy()


    numeric_features = [c for c in train_processed.select_dtypes(exclude=["object"]).columns if c not in ["SalePrice"]]
    categorical_features = [c for c in train_processed.select_dtypes(include=["object"]).columns if c not in ["SalePrice"]]


    missing = train_processed.isnull().sum()
    missing_pct = 100 * missing / len(train_processed)
    missing_with_values = pd.DataFrame({"Feature": missing.index, "Missing_Percentage": missing_pct.values})
    missing_with_values = missing_with_values[missing_with_values["Missing_Percentage"] > 0].sort_values("Missing_Percentage", ascending=False)


    for feature, fill_value in NA_MEANS_NONE.items():
        if feature in train_processed.columns:
            train_processed[feature] = train_processed[feature].fillna(fill_value)
        if feature in test_processed.columns:
            test_processed[feature] = test_processed[feature].fillna(fill_value)


    for df in (train_processed, test_processed):
        if "GarageYrBlt" in df.columns and "YearBuilt" in df.columns:
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
            if "YrSold" in df.columns:
                df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).clip(lower=0)
            df["GarageAddedLater"] = (df["GarageYrBlt"] > df["YearBuilt"]).astype(int)


    for col in categorical_features:
        if col in train_processed.columns:
            mp = 100 * train_processed[col].isnull().sum() / len(train_processed)
            if 0 < mp < 5 and train_processed[col].nunique(dropna=True) < 20:
                mode_value = train_processed[col].mode(dropna=True)
                mode_value = mode_value.iloc[0] if len(mode_value) > 0 else "Missing"
                train_processed[col] = train_processed[col].fillna(mode_value)
                if col in test_processed.columns:
                    test_processed[col] = test_processed[col].fillna(mode_value)


    for col in numeric_features:
        if col in train_processed.columns:
            mp = 100 * train_processed[col].isnull().sum() / len(train_processed)
            if 0 < mp < 5:
                med = train_processed[col].median()
                train_processed[col] = train_processed[col].fillna(med)
                if col in test_processed.columns:
                    test_processed[col] = test_processed[col].fillna(med)


    high_missing_features = missing_with_values[missing_with_values["Missing_Percentage"] > 15]["Feature"].tolist()
    for feature in high_missing_features:
        if feature in train_processed.columns and feature != "SalePrice":
            train_processed[f"{feature}_Missing"] = train_raw[feature].isnull().astype(int)
            if feature in test_processed.columns:
                test_processed[f"{feature}_Missing"] = test_raw[feature].isnull().astype(int)


    for col in train_processed.columns:
        if train_processed[col].isnull().sum() > 0:
            if train_processed[col].dtype == "object":
                train_processed[col] = train_processed[col].fillna("Unknown")
            else:
                train_processed[col] = train_processed[col].fillna(train_processed[col].median())

    for col in test_processed.columns:
        if test_processed[col].isnull().sum() > 0:
            if test_processed[col].dtype == "object":
                test_processed[col] = test_processed[col].fillna("Unknown")
            else:
                test_processed[col] = test_processed[col].fillna(test_processed[col].median())


    ordinal_mapping = {}
    for feature, categories in ORDINAL_FEATURES.items():
        if feature in train_processed.columns:
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            ordinal_mapping[feature] = mapping
            train_processed[feature] = train_processed[feature].map(mapping).fillna(-1)
            if feature in test_processed.columns:
                test_processed[feature] = test_processed[feature].map(mapping).fillna(-1)


    remaining_categorical = [col for col in train_processed.select_dtypes(include=["object"]).columns
                             if col not in ["Id", "SalePrice"]]
    low_cardinality = [col for col in remaining_categorical if train_processed[col].nunique() < 10]
    high_cardinality = [col for col in remaining_categorical if train_processed[col].nunique() >= 10]


    if low_cardinality:
        train_encoded = pd.get_dummies(train_processed, columns=low_cardinality, prefix=low_cardinality, drop_first=True)
        test_encoded = pd.get_dummies(test_processed, columns=low_cardinality, prefix=low_cardinality, drop_first=True)

        train_cols = set(train_encoded.columns)
        test_cols = set(test_encoded.columns)

        for col in train_cols - test_cols:
            if col != "SalePrice":
                test_encoded[col] = 0
        for col in test_cols - train_cols:
            test_encoded.drop(col, axis=1, inplace=True)


        test_encoded = test_encoded[train_encoded.columns.drop("SalePrice", errors="ignore")]

        train_processed = train_encoded
        test_processed = test_encoded


    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for col in high_cardinality:
        if col in train_processed.columns:
            le = LabelEncoder()
            train_processed[col] = le.fit_transform(train_processed[col].astype(str))
            label_encoders[col] = le
            if col in test_processed.columns:
                test_processed[col] = test_processed[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )


    train_engineered = engineer_features(train_processed)
    test_engineered = engineer_features(test_processed)

    artifacts = {
        "high_missing_features": high_missing_features,
        "ordinal_mapping": ordinal_mapping,
        "low_cardinality": low_cardinality,
        "high_cardinality": high_cardinality,
        "label_encoders": label_encoders,
        "train_processed_cols": list(train_processed.columns),
        "train_engineered_cols": list(train_engineered.columns),
    }
    return artifacts


def transform_single(raw_row: pd.DataFrame, artifacts: dict, train_raw: pd.DataFrame) -> pd.DataFrame:
    assert len(raw_row) == 1
    df = raw_row.copy()


    for feature, fill_value in NA_MEANS_NONE.items():
        if feature in df.columns:
            df[feature] = df[feature].fillna(fill_value)


    if "GarageYrBlt" in df.columns and "YearBuilt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
        if "YrSold" in df.columns:
            df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).clip(lower=0)
        df["GarageAddedLater"] = (df["GarageYrBlt"] > df["YearBuilt"]).astype(int)


    for feature in artifacts["high_missing_features"]:
        if feature in df.columns and feature != "SalePrice":
            df[f"{feature}_Missing"] = df[feature].isnull().astype(int)


    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("Unknown")
            else:

                if col in train_raw.columns and pd.api.types.is_numeric_dtype(train_raw[col]):
                    df[col] = df[col].fillna(train_raw[col].median())
                else:
                    df[col] = df[col].fillna(0)


    for feature, mapping in artifacts["ordinal_mapping"].items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping).fillna(-1)


    low_cardinality = artifacts["low_cardinality"]
    if low_cardinality:
        df_enc = pd.get_dummies(df, columns=[c for c in low_cardinality if c in df.columns],
                                prefix=[c for c in low_cardinality if c in df.columns],
                                drop_first=True)
    else:
        df_enc = df


    train_cols = set(artifacts["train_processed_cols"])
    df_cols = set(df_enc.columns)


    for col in (train_cols - df_cols):
        df_enc[col] = 0

    for col in (df_cols - train_cols):
        df_enc.drop(col, axis=1, inplace=True)

    df_enc = df_enc[artifacts["train_processed_cols"]]


    label_encoders = artifacts["label_encoders"]
    for col, le in label_encoders.items():
        if col in df_enc.columns:
            x = str(df_enc.at[df_enc.index[0], col])
            df_enc.at[df_enc.index[0], col] = le.transform([x])[0] if x in le.classes_ else -1


    for col in df_enc.columns:
        if df_enc[col].dtype == "object":
            df_enc[col] = pd.to_numeric(df_enc[col], errors="coerce").fillna(0)


    df_eng = engineer_features(df_enc)

    return df_eng


@st.cache_resource
def build_calibrator_and_cached_train():
    train_full, train_raw, test_raw, lat_col, lon_col, model, feature_names = load_assets()
    artifacts = build_preprocessor(train_raw, test_raw)


    train_eng = None
    train_eng_path = _pick_first_existing_path(CACHE_TRAIN_ENG_PKL_CANDIDATES)
    if train_eng_path and Path(train_eng_path).exists():
        train_eng = pd.read_pickle(train_eng_path).copy()
    else:

        eng_rows = []
        for i in range(len(train_raw)):
            row = train_raw.iloc[[i]].copy()
            eng = transform_single(row, artifacts, train_raw)
            if "SalePrice" in row.columns:
                eng["SalePrice"] = row["SalePrice"].values
            eng_rows.append(eng)
        train_eng = pd.concat(eng_rows, axis=0, ignore_index=True)


    if "SalePrice" not in train_eng.columns:
        A, B = 1.0, 0.0
        X_train = None
    else:
        y_true_log = np.log1p(train_eng["SalePrice"].astype(float).values)

        X = train_eng.drop(["SalePrice"], axis=1, errors="ignore")
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        pred_raw_all = model.predict(X)
        a, b = np.polyfit(pred_raw_all, y_true_log, 1)
        A, B = float(a), float(b)
        X_train = X


    meta_cols = ["Id", "Neighborhood", "SalePrice"]
    for c in meta_cols:
        if c not in train_full.columns:
            train_full[c] = np.nan
    if lat_col is None or lon_col is None:
        lat_col, lon_col = None, None

    train_meta = train_full.copy()

    return artifacts, (A, B), model, feature_names, lat_col, lon_col, train_full, train_raw, test_raw, train_eng, X_train, train_meta


def make_map(df_points: pd.DataFrame, lat_col: str, lon_col: str,
             subject_lat: float | None, subject_lon: float | None,
             title: str,
             show_subject: bool = True,
             marker_size: int = 10,
             zoom: float = 13):
    fig = go.Figure()

    if df_points is None or len(df_points) == 0:
        fig.update_layout(height=520, title=title)
        return fig

    use_subj_center = bool(show_subject) and (subject_lat is not None) and (subject_lon is not None) and pd.notna(subject_lat) and pd.notna(subject_lon)
    center_lat = float(subject_lat) if use_subj_center else float(df_points[lat_col].median())
    center_lon = float(subject_lon) if use_subj_center else float(df_points[lon_col].median())


    hover = []
    for _, r in df_points.iterrows():
        _id = r.get("Id", "")
        _nbh = r.get("Neighborhood", "")
        _sp = r.get("SalePrice", np.nan)
        if pd.notna(_sp):
            h = f"Id: {_id}<br>Neighborhood: {_nbh}<br>SalePrice: ${float(_sp):,.0f}"
        else:
            h = f"Id: {_id}<br>Neighborhood: {_nbh}"
        hover.append(h)


    if "SalePrice" in df_points.columns and df_points["SalePrice"].notna().any():
        prices = pd.to_numeric(df_points["SalePrice"], errors="coerce")
        cmin = float(np.nanpercentile(prices, 2))
        cmax = float(np.nanpercentile(prices, 98))
        fig.add_trace(go.Scattermapbox(
            lat=df_points[lat_col],
            lon=df_points[lon_col],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=prices,
                colorscale="YlOrRd",
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                colorbar=dict(title="SalePrice"),
                opacity=0.85
            ),
            hovertext=hover,
            hoverinfo="text",
            name="Comps (colored by price)"
        ))
    else:
        fig.add_trace(go.Scattermapbox(
            lat=df_points[lat_col],
            lon=df_points[lon_col],
            mode="markers",
            marker=dict(size=marker_size, color="#d62728", opacity=0.85),
            hovertext=hover,
            hoverinfo="text",
            name="Comps"
        ))


    if show_subject and subject_lat is not None and subject_lon is not None and pd.notna(subject_lat) and pd.notna(subject_lon):
        lat0 = float(subject_lat)
        lon0 = float(subject_lon)


        fig.add_trace(go.Scattermapbox(
            lat=[lat0],
            lon=[lon0],
            mode="markers",
            marker=dict(
                size=marker_size + 8,
                color="#ffffff",
                symbol="circle"
            ),
            hoverinfo="skip",
            showlegend=False,
            name="Subject-halo"
        ))

        fig.add_trace(go.Scattermapbox(
            lat=[lat0],
            lon=[lon0],
            mode="markers",
            marker=dict(
                size=marker_size + 4,
                color="#1f77b4",
                symbol="circle"
            ),
            hovertext=[f"SUBJECT (train Id)<br>Lat: {lat0:.6f}<br>Lon: {lon0:.6f}"],
            hoverinfo="text",
            showlegend=False,
            name="Subject"
        ))

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )
    return fig


def main():
    st.title("🏠 Pricing Advisor Dashboard")


    if st.session_state.get('_scroll_to_results', False):
        try:
            scroll_to_top()
        finally:
            st.session_state['_scroll_to_results'] = False


    (artifacts, (A, B), model, feature_names, lat_col, lon_col,
     train_full, train_raw, test_raw, train_eng, X_train, train_meta) = build_calibrator_and_cached_train()


    raw_feature_cols = [c for c in test_raw.columns if c != "Id"]


    has_latlon = (lat_col is not None and lon_col is not None
                  and lat_col in train_full.columns and lon_col in train_full.columns)


    engineered_imp = _get_model_feature_importances(model, feature_names)
    raw_imp = _raw_feature_importance_from_engineered(raw_feature_cols, engineered_imp)


    col_in, col_out = st.columns([0.4, 0.6], gap="large")


    with col_in:
        input_box = st.container(border=True)
        with input_box:
            st.subheader("Inputs")


            id_list = train_full["Id"].dropna().astype(int).sort_values().tolist() if "Id" in train_full.columns else []
            selected_id = st.selectbox("Select training Id (optional)", options=["(none)"] + id_list)

            eng_ref = None
            disable_inputs = (selected_id != "(none)")

            if selected_id != "(none)":
                row_ref = train_full.loc[train_full["Id"].astype(int) == int(selected_id)].iloc[0]

                if train_eng is not None and "Id" in train_eng.columns:
                    _m = train_eng.loc[train_eng["Id"].astype(int) == int(selected_id)]
                    if len(_m) > 0:
                        eng_ref = _m.iloc[[0]].copy()
            else:
                row_ref = None


            defaults = {}
            if row_ref is not None:
                for c in raw_feature_cols:
                    defaults[c] = row_ref.get(c, None)
            else:
                for c in raw_feature_cols:
                    if c in train_raw.columns:
                        if train_raw[c].dtype == "object":
                            defaults[c] = _safe_mode(train_raw[c].astype(str))
                            if defaults[c] is None:
                                defaults[c] = "Unknown"
                        else:
                            defaults[c] = float(train_raw[c].median()) if train_raw[c].notna().any() else 0.0
                    else:
                        defaults[c] = 0.0


            group_map = {}
            for f in raw_feature_cols:
                g = _group_raw_feature(f)
                group_map.setdefault(g, []).append(f)


            for g in list(group_map.keys()):
                group_map[g] = sorted(group_map[g], key=lambda x: float(raw_imp.get(x, 0.0)), reverse=True)


            group_order = [
                "📍 Location",
                "📐 Lot & Land",
                "🏠 Building Type & Style",
                "🧱 Exterior & Structure",
                "⭐ Quality & Condition",
                "🛋️ Living Area, Rooms & Baths",
                "🏗️ Basement",
                "🚗 Garage",
                "⚙️ Systems",
                "✨ Outdoor & Amenities",
                "🧾 Sale Info",
                "📦 Other",
            ]

            for g in sorted(group_map.keys()):
                if g not in group_order:
                    group_order.append(g)


            feature_container = st.container(height=760, border=True)

            user_inputs = {}
            with feature_container:
                for g in group_order:
                    feats = group_map.get(g, [])
                    if not feats:
                        continue
                    with st.expander(g, expanded=False):
                        cols = st.columns(2)
                        for i, feat in enumerate(feats):
                            col = cols[i % 2]
                            with col:
                                if feat in train_raw.columns and train_raw[feat].dtype == "object":
                                    opts = train_raw[feat].dropna().astype(str).unique().tolist()
                                    opts = sorted(opts)
                                    default_val = str(defaults.get(feat, opts[0] if opts else "Unknown"))
                                    if default_val not in opts and default_val not in ["None", "nan"]:
                                        opts = [default_val] + opts
                                    if not opts:
                                        opts = ["Unknown"]
                                    user_inputs[feat] = st.selectbox(
                                        feat,
                                        options=opts,
                                        index=opts.index(default_val) if default_val in opts else 0
                                    )
                                else:

                                    if feat in train_raw.columns and pd.api.types.is_numeric_dtype(train_raw[feat]):
                                        mn = float(np.nanmin(train_raw[feat].values))
                                        mx = float(np.nanmax(train_raw[feat].values))
                                        val = defaults.get(feat, 0.0)
                                        try:
                                            val = float(val)
                                        except Exception:
                                            val = 0.0
                                        rng = mx - mn
                                        step = 1.0 if rng >= 50 else 0.1
                                        user_inputs[feat] = st.number_input(
                                            feat,
                                            value=float(val),
                                            min_value=float(mn),
                                            max_value=float(mx),
                                            step=float(step),
                                        )
                                    else:
                                        val = defaults.get(feat, 0.0)
                                        try:
                                            val = float(val)
                                        except Exception:
                                            val = 0.0
                                        user_inputs[feat] = st.number_input(feat, value=float(val), disabled=disable_inputs)


            changed_features = []
            for c in raw_feature_cols:
                is_num = (c in train_raw.columns) and pd.api.types.is_numeric_dtype(train_raw[c])
                if _value_changed(user_inputs.get(c, None), defaults.get(c, None), is_num):
                    changed_features.append(c)

            lp_default = float(train_raw["SalePrice"].median()) if "SalePrice" in train_raw.columns else 0.0
            if "listing_price" in st.session_state:
                try:
                    lp_default = float(st.session_state["listing_price"])
                except Exception:
                    pass
            listing_price = st.number_input("Listing price", min_value=0.0, value=float(lp_default), step=1000.0, key="listing_price")


            with st.expander("Comparable settings", expanded=False):
                topk_neigh = st.slider("Map 2: Top-K similar (same neighborhood)", min_value=5, max_value=200, value=50, step=5)
                topk_sim = st.slider("Map 3: Top-K similar (overall)", min_value=5, max_value=30, value=10, step=1)

            btn = st.button("Get recommendation", type="primary", width="stretch")


            if has_latlon:
                if row_ref is not None:
                    subj_lat = float(row_ref[lat_col])
                    subj_lon = float(row_ref[lon_col])
                else:
                    nbh_in = user_inputs.get("Neighborhood", None)
                    if nbh_in is not None and "Neighborhood" in train_full.columns:
                        tmp = train_full[train_full["Neighborhood"].astype(str) == str(nbh_in)]
                        tmp = tmp[tmp[lat_col].notna() & tmp[lon_col].notna()]
                        if len(tmp) > 0:
                            subj_lat = float(tmp[lat_col].median())
                            subj_lon = float(tmp[lon_col].median())
                        else:
                            tmp2 = train_full[train_full[lat_col].notna() & train_full[lon_col].notna()]
                            subj_lat = float(tmp2[lat_col].median()) if len(tmp2) else float(train_full[lat_col].median())
                            subj_lon = float(tmp2[lon_col].median()) if len(tmp2) else float(train_full[lon_col].median())
                    else:
                        tmp2 = train_full[train_full[lat_col].notna() & train_full[lon_col].notna()]
                        subj_lat = float(tmp2[lat_col].median()) if len(tmp2) else float(train_full[lat_col].median())
                        subj_lon = float(tmp2[lon_col].median()) if len(tmp2) else float(train_full[lon_col].median())
            else:
                subj_lat, subj_lon = None, None


            if row_ref is not None and "Neighborhood" in train_full.columns:
                nbh = row_ref.get("Neighborhood", None)
            else:
                nbh = user_inputs.get("Neighborhood", None)


            if btn:

                row = {"Id": int(selected_id) if selected_id != "(none)" else -1}
                for c in raw_feature_cols:
                    row[c] = user_inputs.get(c, defaults.get(c, None))
                raw_one = pd.DataFrame([row])


                eng_one = transform_single(raw_one, artifacts, train_raw)


                X = eng_one.copy()
                for col in feature_names:
                    if col not in X.columns:
                        X[col] = 0
                X = X[feature_names]


                for col in X.columns:
                    if X[col].dtype == "object":
                        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)


                if eng_ref is not None:
                    X = eng_ref.drop(["Id", "SalePrice"], axis=1, errors="ignore").copy()
                    for col in feature_names:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[feature_names]
                    for col in X.columns:
                        if X[col].dtype == "object":
                            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

                pred_raw = float(model.predict(X)[0])
                pred_log = A * pred_raw + B
                pred_price = float(np.expm1(pred_log))

                actual = None
                abs_err = None
                if row_ref is not None and "SalePrice" in train_full.columns and pd.notna(row_ref.get("SalePrice", np.nan)):
                    actual = float(row_ref["SalePrice"])
                    abs_err = float(abs(actual - pred_price))


                shap_mod = _try_import_shap()
                base_value = None
                shap_vals = None
                if shap_mod is not None:
                    explainer = _get_shap_explainer(model)
                    if explainer is not None:
                        try:
                            shap_vals = explainer.shap_values(X)
                            base_value = explainer.expected_value
                            base_value = float(np.array(base_value).ravel()[0])
                        except Exception:
                            shap_vals, base_value = None, None

                if shap_vals is None or base_value is None:
                    base_value, shap_vals = _xgb_pred_contribs(model, X)

                shap_payload = None
                if shap_vals is not None and base_value is not None:
                    base_log = A * float(base_value) + B
                    sv = np.asarray(shap_vals)
                    if sv.ndim > 1:
                        sv = sv[0]
                    shap_log = A * sv
                    shap_payload = {
                        "base_log": float(base_log),
                        "contrib": pd.Series(shap_log, index=X.columns),
                        "X_row": X.iloc[0].copy(),
                    }


                maps_payload = {"enabled": False}
                if has_latlon and subj_lat is not None and subj_lon is not None:
                    df_map = train_full.copy()
                    df_map = df_map[df_map[lat_col].notna() & df_map[lon_col].notna()].copy()


                    if "Neighborhood" not in df_map.columns:
                        df_map["Neighborhood"] = ""
                    if "SalePrice" not in df_map.columns:
                        df_map["SalePrice"] = np.nan
                    if "Id" not in df_map.columns:
                        df_map["Id"] = np.arange(len(df_map))

                    cand1 = pd.DataFrame()
                    if nbh is not None and str(nbh) != "":
                        cand1 = df_map[df_map["Neighborhood"].astype(str) == str(nbh)].copy()

                    cand2 = pd.DataFrame()
                    cand3 = pd.DataFrame()

                    if len(df_map) > 0:


                        subject_raw = pd.Series({c: user_inputs.get(c, defaults.get(c, None)) for c in raw_feature_cols})

                        sim_all = _compute_similarity_raw(
                            df_candidates=df_map,
                            subject_raw=subject_raw,
                            train_raw=train_raw,
                            raw_feature_cols=raw_feature_cols,
                            raw_imp=raw_imp,
                            changed_features=changed_features,
                            changed_boost=2.5,
                        )


                        ids_all = df_map["Id"].to_numpy()
                        if selected_id != "(none)":
                            sim_all = np.where(ids_all == int(selected_id), -np.inf, sim_all)


                        k3 = int(topk_sim)
                        top_idx3 = np.argsort(sim_all)[::-1][:k3]
                        cand3 = df_map.iloc[top_idx3].copy()
                        cand3["Similarity"] = sim_all[top_idx3]


                        if nbh is not None and str(nbh) != "":
                            mask = df_map["Neighborhood"].astype(str) == str(nbh)
                            idxs = np.where(mask.to_numpy())[0]
                            if idxs.size > 0:
                                k2 = min(int(topk_neigh), int(idxs.size))
                                order = np.argsort(sim_all[idxs])[::-1][:k2]
                                top_idx2 = idxs[order]
                                cand2 = df_map.iloc[top_idx2].copy()
                                cand2["Similarity"] = sim_all[top_idx2]

                    maps_payload = {
                        "enabled": True,
                        "lat_col": lat_col,
                        "lon_col": lon_col,
                        "subj_lat": float(subj_lat),
                        "subj_lon": float(subj_lon),
                        "nbh": nbh,
                        "cand1": cand1,
                        "cand2": cand2,
                        "cand3": cand3,
                        "topk_neigh": int(topk_neigh),
                        "topk_sim": int(topk_sim),
                    }

                st.session_state["last_run"] = {
                    "selected_id": selected_id,
                    "pred_price": float(pred_price),
                    "listing_price": float(listing_price),
                    "actual": actual,
                    "abs_err": abs_err,
                    "shap": shap_payload,
                    "maps": maps_payload,
                }

                st.session_state['_scroll_to_results'] = True
                st.rerun()


    with col_out:
        run = st.session_state.get("last_run", None)
        if run is None:
            return

        tabs = st.tabs(["Tab 1: Recommendation & Explanation", "Tab 2: Maps"])


        with tabs[0]:
            st.subheader("Price recommendation")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Recommended price", f"${run['pred_price']:,.0f}")
            with c2:
                st.metric("Listing price", f"${run.get('listing_price', 0.0):,.0f}")

            rp = float(run.get("pred_price", 0.0))
            lp = float(run.get("listing_price", 0.0))
            if rp > 0:
                diff_pct = (lp / rp - 1.0) * 100.0
                if abs(diff_pct) < 1e-9:
                    st.write("Listing price is equal to the recommended price (0.0%).")
                elif diff_pct > 0:
                    st.write(f"Listing price is **{diff_pct:.1f}% higher** than the recommended price.")
                else:
                    st.write(f"Listing price is **{abs(diff_pct):.1f}% lower** than the recommended price.")
                if lp <= rp:
                    st.success("Advice: Should Buy")
                elif lp <= 1.1 * rp:
                    st.info("Advice: Consider Further")
                elif lp <= 1.5 * rp:
                    st.warning("Advice: Negotiate")
                else:
                    st.error("Advice: Look for another home that fits better")

            if run.get("actual") is not None:
                st.caption(f"Actual SalePrice (Id={run['selected_id']}): **${run['actual']:,.0f}**")
                st.caption(f"Absolute error: **${run['abs_err']:,.0f}**")

            st.subheader("Explanation (waterfall)")
            shap_payload = run.get("shap", None)
            if shap_payload is None:
                st.info("SHAP is unavailable for this run.")
            else:
                shap_mod = _try_import_shap()
                base_log = shap_payload["base_log"]
                contrib = shap_payload["contrib"]
                X_row = shap_payload["X_row"]

                if shap_mod is not None:
                    try:
                        import matplotlib.pyplot as plt
                        exp = shap_mod.Explanation(
                            values=contrib.values,
                            base_values=base_log,
                            data=X_row.to_numpy(),
                            feature_names=list(contrib.index),
                        )
                        plt.figure(figsize=(7.2, 2.7))
                        shap_mod.plots.waterfall(exp, max_display=10, show=False)
                        st.pyplot(plt.gcf(), clear_figure=True)
                    except Exception:
                        fig_w = _plot_waterfall_custom(base_log, contrib, max_display=10)
                        st.plotly_chart(fig_w, width="stretch")
                else:
                    fig_w = _plot_waterfall_custom(base_log, contrib, max_display=10)
                    st.plotly_chart(fig_w, width="stretch")


        with tabs[1]:
            maps = run.get("maps", {})
            if not maps.get("enabled", False):
                st.info("Maps are disabled (no latitude/longitude columns found in training data).")
            else:
                latc = maps["lat_col"]
                lonc = maps["lon_col"]
                subj_lat = maps["subj_lat"]
                subj_lon = maps["subj_lon"]

                show_subject = (run.get('selected_id') != '(none)')


                m_tabs = st.tabs([
                    "Map 1: Same neighborhood",
                    "Map 2: Top similar houses in neighborhood",
                    "Map 3: Top similar overall",
                ])

                with m_tabs[0]:
                    cand1 = maps.get("cand1", pd.DataFrame())
                    nbh = maps.get("nbh", None)
                    if cand1 is None or len(cand1) == 0 or nbh is None or str(nbh) == "":
                        st.info("Neighborhood is missing or no mapped rows in this neighborhood.")
                    else:
                        fig = make_map(cand1, latc, lonc, subj_lat, subj_lon, f"Same neighborhood: {nbh}", show_subject=show_subject, zoom=13)
                        st.plotly_chart(fig, width="stretch")

                with m_tabs[1]:
                    cand2 = maps.get("cand2", pd.DataFrame())
                    nbh = maps.get("nbh", None)
                    if cand2 is None or len(cand2) == 0 or nbh is None or str(nbh) == "":
                        st.info("No results for Map 2.")
                    else:
                        k = maps.get("topk_neigh", len(cand2))
                        fig = make_map(cand2, latc, lonc, subj_lat, subj_lon, f"Top {k} similar houses in neighborhood: {nbh}", show_subject=show_subject, marker_size=16, zoom=13)
                        st.plotly_chart(fig, width="stretch")
                        st.dataframe(
                            cand2[["Id", "Neighborhood", "SalePrice"]].reset_index(drop=True),
                            width="stretch"
                        )

                with m_tabs[2]:
                    cand3 = maps.get("cand3", pd.DataFrame())
                    if cand3 is None or len(cand3) == 0:
                        st.info("No results for Map 3.")
                    else:
                        k = maps.get("topk_sim", len(cand3))
                        fig = make_map(cand3, latc, lonc, subj_lat, subj_lon, f"Top {k} similar houses", show_subject=show_subject, marker_size=16, zoom=13)
                        st.plotly_chart(fig, width="stretch")
                        st.dataframe(
                            cand3[["Id", "Neighborhood", "SalePrice"]].reset_index(drop=True),
                            width="stretch"
                        )


if __name__ == "__main__":
    main()
