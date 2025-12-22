import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go

# -------------------------
# SHAP helpers (do NOT affect prediction)
# -------------------------
def _try_import_shap():
    try:
        import shap  # type: ignore
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
    """
    Fallback local explanation without the `shap` package:
    Use XGBoost's built-in TreeSHAP via pred_contribs=True.
    Returns (base_value, shap_values) where:
      pred_raw ≈ base_value + sum(shap_values)
    """
    try:
        import xgboost as xgb  # type: ignore
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
    """
    Simple waterfall plot using Plotly (used if shap plotting isn't available).
    """
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



# Optional (for click-to-pick lat/lon)
try:
    from streamlit_folium import st_folium
    import folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False


# =========================
# Config
# =========================
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


# =========================
# Utilities
# =========================

def _pick_first_existing_path(candidates):
    """Return first existing path string from candidates (relative)."""
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
    # vectorized haversine
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
    """
    Returns a pd.Series (index=feature_names, values>=0) for ordering / similarity.
    Does NOT affect the prediction path (we keep model.predict exactly as-is).
    """
    imp = None
    if hasattr(model, "feature_importances_"):
        try:
            imp = np.asarray(model.feature_importances_, dtype=float)
        except Exception:
            imp = None

    # Fallback for xgboost Booster gain if available
    if imp is None or (len(imp) != len(feature_names)):
        try:
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")
            # xgboost uses f0..fN if feature names not set; map by index
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
    """
    Map engineered importance -> raw importance:
      - direct match: feature == raw_col
      - one-hot: engineered columns starting with raw_col + "_"
      - missing: 0
    """
    raw_imp = {}
    for rc in raw_cols:
        if rc in engineered_importance.index:
            raw_imp[rc] = float(engineered_importance.get(rc, 0.0))
            continue
        # sum one-hot prefixes
        pref = rc + "_"
        s = engineered_importance[engineered_importance.index.to_series().astype(str).str.startswith(pref)]
        raw_imp[rc] = float(s.sum()) if len(s) else 0.0
    return pd.Series(raw_imp).sort_values(ascending=False)


def _group_raw_feature(feat: str) -> str:
    """
    Group raw (Ames) features into a clean, business-friendly UI structure.
    This does NOT affect the model: it's only for input layout.
    """
    f = feat.lower()

    # Location (where the house is)
    if any(k in f for k in ["neighborhood", "condition1", "condition2"]):
        return "📍 Location"

    # Lot / land (the plot itself)
    if any(k in f for k in [
        "lotarea", "lotfrontage", "lotshape", "landcontour", "landslope", "lotconfig",
        "street", "alley", "utilities", "land", "lot"
    ]):
        return "📐 Lot & Land"

    # Building type & style (structure identity)
    if any(k in f for k in ["mssubclass", "mszoning", "bldgtype", "housestyle"]):
        return "🏠 Building Type & Style"

    # Exterior / roof / foundation
    if any(k in f for k in [
        "exterior1st", "exterior2nd", "roofstyle", "roofmatl",
        "masvnr", "foundation", "exterqual", "extercond"
    ]):
        return "🧱 Exterior & Structure"

    # Quality & condition
    if any(k in f for k in ["overallqual", "overallcond", "functional", "heatingqc", "kitchenqual", "fireplacequ"]):
        return "⭐ Quality & Condition"

    # Living area / rooms / baths
    if any(k in f for k in [
        "grlivarea", "1stflrsf", "2ndflrsf", "lowqualfinsf",
        "totrmsabvgrd", "bedroomabvgr", "kitchenabvgr",
        "fullbath", "halfbath"
    ]):
        return "🛋️ Living Area, Rooms & Baths"

    # Basement
    if "bsmt" in f or "basement" in f or "totalbsmtsf" in f:
        return "🏗️ Basement"

    # Garage
    if "garage" in f:
        return "🚗 Garage"

    # Systems / utilities (mechanical)
    if any(k in f for k in ["heating", "centralair", "electrical"]):
        return "⚙️ Systems"

    # Outdoor features & amenities
    if any(k in f for k in ["porch", "deck", "pool", "fence", "screenporch", "3ssnporch", "openporch", "enclosedporch", "miscfeature", "miscval"]):
        return "✨ Outdoor & Amenities"

    # Sale information
    if any(k in f for k in ["mosold", "yrsold", "saletype", "salecondition"]):
        return "🧾 Sale Info"

    return "📦 Other"




# =========================
# Similarity (comps) logic
# =========================

def _safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _value_changed(user_val, default_val, is_numeric: bool) -> bool:
    """Detect if user changed a feature from its default (used only to boost similarity weights)."""
    if is_numeric:
        a = _safe_float(user_val)
        b = _safe_float(default_val)
        if np.isnan(a) and np.isnan(b):
            return False
        if np.isnan(a) != np.isnan(b):
            return True
        return abs(a - b) > 1e-9
    # categorical
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
    """
    Compute similarity between each candidate house and the subject based on RAW features
    that the user sees/chooses in the UI.

    Distance:
      - numeric: |x - x0| / scale   (scale uses IQR, fallback std, fallback 1)
      - categorical: 0 if same, 1 if different (missing handled softly)

    Weights:
      - base weight from raw_imp (aggregated from model importance)
      - if a feature was changed by user, weight is multiplied by (1 + changed_boost)

    similarity = 1 / (1 + normalized_distance)
    """
    changed_features = changed_features or []

    idx_map = {c: i for i, c in enumerate(raw_feature_cols)}

    # Determine numeric vs categorical by train_raw dtype
    num_cols = [c for c in raw_feature_cols if c in train_raw.columns and pd.api.types.is_numeric_dtype(train_raw[c])]
    cat_cols = [c for c in raw_feature_cols if c not in num_cols]

    # Base weights from raw_imp (fallback uniform)
    w_base = np.array([float(raw_imp.get(c, 0.0)) for c in raw_feature_cols], dtype=float)
    if np.all(w_base <= 0):
        w_base = np.ones(len(raw_feature_cols), dtype=float)
    else:
        # keep weights well-behaved: normalize to mean ~1
        w_base = w_base / (np.mean(w_base[w_base > 0]) + 1e-12)

    # Boost for user-changed features
    w = w_base.copy()
    if changed_features:
        changed_set = set(map(str, changed_features))
        for i, c in enumerate(raw_feature_cols):
            if c in changed_set:
                w[i] *= (1.0 + float(changed_boost))

    # Build candidate arrays (aligned to raw_feature_cols)
    dfc = df_candidates.reindex(columns=raw_feature_cols)

    # --- numeric distance ---
    dist = np.zeros(len(dfc), dtype=float)

    # numeric scales
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

    # numeric part (vectorized)
    for c in num_cols:
        x0 = _safe_float(subject_raw.get(c, np.nan))
        x = pd.to_numeric(dfc[c], errors="coerce").to_numpy(dtype=float)
        sc = float(scales.get(c, 1.0))
        d = np.abs(x - x0) / sc
        # handle missing:
        # - if both missing -> 0
        # - if one missing -> 1
        x_nan = np.isnan(x)
        x0_nan = np.isnan(x0)
        if x0_nan:
            d = np.where(x_nan, 0.0, 1.0)
        else:
            d = np.where(x_nan, 1.0, d)
        # apply weight
        wi = float(w[idx_map[c]])
        dist += wi * d

    # categorical part (vectorized)
    for c in cat_cols:
        x0 = subject_raw.get(c, None)
        x0 = None if x0 is None else str(x0)
        if x0 in ["nan", "None"]:
            x0 = None

        s = dfc[c]
        # keep missing as NA
        s_isna = s.isna().to_numpy()
        x = s.astype(str).to_numpy()

        if x0 is None:
            d = np.where(s_isna, 0.0, 1.0)
        else:
            d = (x != x0).astype(float)
            d = np.where(s_isna, 1.0, d)

        wi = float(w[idx_map[c]])
        dist += wi * d

    # normalize by sum weights (only those included)
    w_sum = float(np.sum(w)) + 1e-12
    dist_norm = dist / w_sum
    sim = 1.0 / (1.0 + dist_norm)
    return sim

# =========================
# Preprocess pipeline (from NB1)  ✅ KEEP AS-IS for prediction correctness
# =========================
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
    """Create engineered features (same as NB1)."""
    df_eng = df.copy()

    # 1. Total Square Footage
    if all(col in df_eng.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df_eng['TotalSF'] = df_eng['TotalBsmtSF'] + df_eng['1stFlrSF'] + df_eng['2ndFlrSF']

    # 2. Total Bathrooms
    if all(col in df_eng.columns for col in ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']):
        df_eng['TotalBath'] = (
            df_eng['BsmtFullBath']
            + 0.5 * df_eng['BsmtHalfBath']
            + df_eng['FullBath']
            + 0.5 * df_eng['HalfBath']
        )

    # 3. Total Porch SF
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    if all(col in df_eng.columns for col in porch_cols):
        df_eng['TotalPorchSF'] = df_eng[porch_cols].sum(axis=1)

    # 4. House Age (at time of sale)
    if all(col in df_eng.columns for col in ['YrSold', 'YearBuilt']):
        df_eng['HouseAge'] = df_eng['YrSold'] - df_eng['YearBuilt']
        df_eng['HouseAge'] = df_eng['HouseAge'].clip(lower=0)

    # 5. Years Since Remodel
    if all(col in df_eng.columns for col in ['YrSold', 'YearRemodAdd']):
        df_eng['YearsSinceRemodel'] = df_eng['YrSold'] - df_eng['YearRemodAdd']
        df_eng['YearsSinceRemodel'] = df_eng['YearsSinceRemodel'].clip(lower=0)

    # 6. Remodeling Flag
    if all(col in df_eng.columns for col in ['YearRemodAdd', 'YearBuilt']):
        df_eng['WasRemodeled'] = (df_eng['YearRemodAdd'] != df_eng['YearBuilt']).astype(int)

    # 7. New House Flag
    if all(col in df_eng.columns for col in ['YrSold', 'YearBuilt']):
        df_eng['IsNewHouse'] = (df_eng['YrSold'] == df_eng['YearBuilt']).astype(int)

    # 8. Has 2nd Floor
    if '2ndFlrSF' in df_eng.columns:
        df_eng['Has2ndFloor'] = (df_eng['2ndFlrSF'] > 0).astype(int)

    # 9. Has Garage
    if 'GarageArea' in df_eng.columns:
        df_eng['HasGarage'] = (df_eng['GarageArea'] > 0).astype(int)

    # 10. Has Basement
    if 'TotalBsmtSF' in df_eng.columns:
        df_eng['HasBasement'] = (df_eng['TotalBsmtSF'] > 0).astype(int)

    # 11. Has Fireplace
    if 'Fireplaces' in df_eng.columns:
        df_eng['HasFireplace'] = (df_eng['Fireplaces'] > 0).astype(int)

    # 12. Has Pool
    if 'PoolArea' in df_eng.columns:
        df_eng['HasPool'] = (df_eng['PoolArea'] > 0).astype(int)

    # 15. Area per Room
    if all(col in df_eng.columns for col in ['GrLivArea', 'TotRmsAbvGrd']):
        denom = df_eng['TotRmsAbvGrd'].replace(0, np.nan)
        df_eng['AreaPerRoom'] = (df_eng['GrLivArea'] / denom).fillna(0)

    # 16. Frontage Ratio
    if all(col in df_eng.columns for col in ['LotFrontage', 'LotArea']):
        denom = df_eng['LotArea'].replace(0, np.nan)
        df_eng['FrontageRatio'] = (df_eng['LotFrontage'] / denom).fillna(0)

    # 17. Basement Finish Ratio
    if all(col in df_eng.columns for col in ['BsmtFinSF1', 'TotalBsmtSF']):
        denom = df_eng['TotalBsmtSF'].replace(0, np.nan)
        df_eng['BsmtFinishRatio'] = (df_eng['BsmtFinSF1'] / denom).fillna(0)

    # 18. Garage Ratio
    if all(col in df_eng.columns for col in ['GarageArea', 'GrLivArea']):
        denom = df_eng['GrLivArea'].replace(0, np.nan)
        df_eng['GarageRatio'] = (df_eng['GarageArea'] / denom).fillna(0)

    return df_eng


@st.cache_resource
def load_assets():
    # Load data (NB1/NB2 new notebooks use ./dataset/train.csv and ./dataset/test.csv)
    train_map_csv = _pick_first_existing_path(DATA_TRAIN_MAP_CSV_CANDIDATES)
    train_model_csv = _pick_first_existing_path(DATA_TRAIN_MODEL_CSV_CANDIDATES)
    test_csv = _pick_first_existing_path(DATA_TEST_CSV_CANDIDATES)

    # train_full is used for maps (needs lat/lon if available)
    train_full = pd.read_csv(train_map_csv) if train_map_csv and Path(train_map_csv).exists() else pd.read_csv(train_model_csv)

    # train_model is used for prediction pipeline alignment (no lat/lon needed)
    train_model = pd.read_csv(train_model_csv)
    test_raw = pd.read_csv(test_csv)

    # Map lat/lon cols for comps
    lat_col = _pick_first_existing(train_full.columns, MAP_LAT_COL_CANDIDATES)
    lon_col = _pick_first_existing(train_full.columns, MAP_LON_COL_CANDIDATES)

    # Raw feature columns (Kaggle test defines "raw" features)
    raw_cols = list(test_raw.columns)  # includes Id

    # Keep only raw_cols + SalePrice from train_full (ignore extra cols like PID/lat/lon for modeling)
    keep_cols = [c for c in (raw_cols + ["SalePrice"]) if c in train_model.columns]
    train_raw = train_model[keep_cols].copy()

    # Load model + feature info
    model_path = _pick_first_existing_path(MODEL_PKL_CANDIDATES)
    feature_info_path = _pick_first_existing_path(FEATURE_INFO_PKL_CANDIDATES)
    model = joblib.load(model_path)
    feature_info = joblib.load(feature_info_path)
    feature_names = feature_info["feature_names"]

    return train_full, train_raw, test_raw, lat_col, lon_col, model, feature_names


def build_preprocessor(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    """
    Fit-time artifacts from NB1:
      - high_missing_features list (missing >15%)
      - ordinal_mapping
      - low_cardinality, high_cardinality
      - label_encoders (fit on train)
      - dummy_columns after one-hot alignment
    """
    train_processed = train_raw.copy()
    test_processed = test_raw.copy()

    # Identify numeric/categorical
    numeric_features = [c for c in train_processed.select_dtypes(exclude=["object"]).columns if c not in ["SalePrice"]]
    categorical_features = [c for c in train_processed.select_dtypes(include=["object"]).columns if c not in ["SalePrice"]]

    # Missing stats
    missing = train_processed.isnull().sum()
    missing_pct = 100 * missing / len(train_processed)
    missing_with_values = pd.DataFrame({"Feature": missing.index, "Missing_Percentage": missing_pct.values})
    missing_with_values = missing_with_values[missing_with_values["Missing_Percentage"] > 0].sort_values("Missing_Percentage", ascending=False)

    # Strategy 1: NA means None
    for feature, fill_value in NA_MEANS_NONE.items():
        if feature in train_processed.columns:
            train_processed[feature] = train_processed[feature].fillna(fill_value)
        if feature in test_processed.columns:
            test_processed[feature] = test_processed[feature].fillna(fill_value)

    # GarageYrBlt special
    for df in (train_processed, test_processed):
        if "GarageYrBlt" in df.columns and "YearBuilt" in df.columns:
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
            if "YrSold" in df.columns:
                df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).clip(lower=0)
            df["GarageAddedLater"] = (df["GarageYrBlt"] > df["YearBuilt"]).astype(int)

    # Strategy 2: mode for low missing categorical (<5% and nunique<20)
    for col in categorical_features:
        if col in train_processed.columns:
            mp = 100 * train_processed[col].isnull().sum() / len(train_processed)
            if 0 < mp < 5 and train_processed[col].nunique(dropna=True) < 20:
                mode_value = train_processed[col].mode(dropna=True)
                mode_value = mode_value.iloc[0] if len(mode_value) > 0 else "Missing"
                train_processed[col] = train_processed[col].fillna(mode_value)
                if col in test_processed.columns:
                    test_processed[col] = test_processed[col].fillna(mode_value)

    # Strategy 3: median for low missing numeric (<5%)
    for col in numeric_features:
        if col in train_processed.columns:
            mp = 100 * train_processed[col].isnull().sum() / len(train_processed)
            if 0 < mp < 5:
                med = train_processed[col].median()
                train_processed[col] = train_processed[col].fillna(med)
                if col in test_processed.columns:
                    test_processed[col] = test_processed[col].fillna(med)

    # Strategy 4: missing indicators for >15%
    high_missing_features = missing_with_values[missing_with_values["Missing_Percentage"] > 15]["Feature"].tolist()
    for feature in high_missing_features:
        if feature in train_processed.columns and feature != "SalePrice":
            train_processed[f"{feature}_Missing"] = train_raw[feature].isnull().astype(int)
            if feature in test_processed.columns:
                test_processed[f"{feature}_Missing"] = test_raw[feature].isnull().astype(int)

    # Final: fill any remaining missing
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

    # Ordinal encode
    ordinal_mapping = {}
    for feature, categories in ORDINAL_FEATURES.items():
        if feature in train_processed.columns:
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            ordinal_mapping[feature] = mapping
            train_processed[feature] = train_processed[feature].map(mapping).fillna(-1)
            if feature in test_processed.columns:
                test_processed[feature] = test_processed[feature].map(mapping).fillna(-1)

    # Remaining categorical for OHE / Label
    remaining_categorical = [col for col in train_processed.select_dtypes(include=["object"]).columns
                             if col not in ["Id", "SalePrice"]]
    low_cardinality = [col for col in remaining_categorical if train_processed[col].nunique() < 10]
    high_cardinality = [col for col in remaining_categorical if train_processed[col].nunique() >= 10]

    # One-hot low cardinality with alignment (NB1 logic)
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

        # Reorder to match train (excluding SalePrice if missing)
        test_encoded = test_encoded[train_encoded.columns.drop("SalePrice", errors="ignore")]

        train_processed = train_encoded
        test_processed = test_encoded

    # Label encode high cardinality (fit on train, unseen => -1)
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

    # Engineer features
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
    """Transform a single-row raw dataframe into engineered numeric features (NB1-compatible)."""
    assert len(raw_row) == 1
    df = raw_row.copy()

    # Strategy 1
    for feature, fill_value in NA_MEANS_NONE.items():
        if feature in df.columns:
            df[feature] = df[feature].fillna(fill_value)

    # GarageYrBlt
    if "GarageYrBlt" in df.columns and "YearBuilt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
        if "YrSold" in df.columns:
            df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).clip(lower=0)
        df["GarageAddedLater"] = (df["GarageYrBlt"] > df["YearBuilt"]).astype(int)

    # Missing indicators (>15% list from train)
    for feature in artifacts["high_missing_features"]:
        if feature in df.columns and feature != "SalePrice":
            df[f"{feature}_Missing"] = df[feature].isnull().astype(int)

    # Fill remaining missing (match NB1)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("Unknown")
            else:
                # use train median for stability
                if col in train_raw.columns and pd.api.types.is_numeric_dtype(train_raw[col]):
                    df[col] = df[col].fillna(train_raw[col].median())
                else:
                    df[col] = df[col].fillna(0)

    # Ordinal encode
    for feature, mapping in artifacts["ordinal_mapping"].items():
        if feature in df.columns:
            df[feature] = df[feature].map(mapping).fillna(-1)

    # One-hot low cardinality: create dummies and align to train_processed_cols
    low_cardinality = artifacts["low_cardinality"]
    if low_cardinality:
        df_enc = pd.get_dummies(df, columns=[c for c in low_cardinality if c in df.columns],
                                prefix=[c for c in low_cardinality if c in df.columns],
                                drop_first=True)
    else:
        df_enc = df

    # Align to train_processed_cols (columns after one-hot, before label-encoding in NB1)
    train_cols = set(artifacts["train_processed_cols"])
    df_cols = set(df_enc.columns)

    # add missing
    for col in (train_cols - df_cols):
        df_enc[col] = 0
    # drop extras
    for col in (df_cols - train_cols):
        df_enc.drop(col, axis=1, inplace=True)

    df_enc = df_enc[artifacts["train_processed_cols"]]

    # Label encode high-cardinality
    label_encoders = artifacts["label_encoders"]
    for col, le in label_encoders.items():
        if col in df_enc.columns:
            x = str(df_enc.at[df_enc.index[0], col])
            df_enc.at[df_enc.index[0], col] = le.transform([x])[0] if x in le.classes_ else -1

    # Ensure numeric dtypes where possible
    for col in df_enc.columns:
        if df_enc[col].dtype == "object":
            df_enc[col] = pd.to_numeric(df_enc[col], errors="coerce").fillna(0)

    # Engineer features
    df_eng = engineer_features(df_enc)

    return df_eng


@st.cache_resource
def build_calibrator_and_cached_train():
    """
    ✅ Keep the SAME calibration/prediction logic as app_6.py:
       pred_raw = model.predict(X)
       pred_log = A*pred_raw + B
       pred_price = expm1(pred_log)

    Additionally:
    - Precompute aligned engineered train matrix X_train for fast Map 3.
    """
    train_full, train_raw, test_raw, lat_col, lon_col, model, feature_names = load_assets()
    artifacts = build_preprocessor(train_raw, test_raw)

    # ---- Build engineered train (cache-first, matches NB1_new) ----
    train_eng = None
    train_eng_path = _pick_first_existing_path(CACHE_TRAIN_ENG_PKL_CANDIDATES)
    if train_eng_path and Path(train_eng_path).exists():
        train_eng = pd.read_pickle(train_eng_path).copy()
    else:
        # Fallback: row-wise transform (slower, may drift if NB1 preprocessing changed)
        eng_rows = []
        for i in range(len(train_raw)):
            row = train_raw.iloc[[i]].copy()
            eng = transform_single(row, artifacts, train_raw)
            if "SalePrice" in row.columns:
                eng["SalePrice"] = row["SalePrice"].values
            eng_rows.append(eng)
        train_eng = pd.concat(eng_rows, axis=0, ignore_index=True)

    # ---- Calibration ----
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

    # Minimal meta table for maps/lookup (use train_full for lat/lon, SalePrice)
    meta_cols = ["Id", "Neighborhood", "SalePrice"]
    for c in meta_cols:
        if c not in train_full.columns:
            train_full[c] = np.nan
    if lat_col is None or lon_col is None:
        lat_col, lon_col = None, None

    train_meta = train_full.copy()

    return artifacts, (A, B), model, feature_names, lat_col, lon_col, train_full, train_raw, test_raw, train_eng, X_train, train_meta


# =========================
# Map drawing (improved)
# =========================
def make_map(df_points: pd.DataFrame, lat_col: str, lon_col: str,
             subject_lat: float | None, subject_lon: float | None,
             title: str,
             show_subject: bool = True,
             marker_size: int = 10,
             zoom: float = 13):
    """
    Plotly map:
    - comps colored by SalePrice (continuous colorscale + colorbar)
    - subject shown as STAR (only when a training Id is selected)
    """
    fig = go.Figure()

    if df_points is None or len(df_points) == 0:
        fig.update_layout(height=520, title=title)
        return fig

    use_subj_center = bool(show_subject) and (subject_lat is not None) and (subject_lon is not None) and pd.notna(subject_lat) and pd.notna(subject_lon)
    center_lat = float(subject_lat) if use_subj_center else float(df_points[lat_col].median())
    center_lon = float(subject_lon) if use_subj_center else float(df_points[lon_col].median())

    # Hover
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

    # Comps (colored by price)
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

            # Subject (ONLY when a training Id is selected)
    if show_subject and subject_lat is not None and subject_lon is not None and pd.notna(subject_lat) and pd.notna(subject_lon):
        lat0 = float(subject_lat)
        lon0 = float(subject_lon)

        # Use a simple, highly-visible BLUE circle (scattermapbox does not support marker.line).
        # Add a white halo using a 2-layer circle so it stands out.
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
                color="#1f77b4",  # blue
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


# =========================
# UI
# =========================
def main():
    st.title("🏠 Pricing Advisor Dashboard")

    # Auto-scroll to results after running
    if st.session_state.get('_scroll_to_results', False):
        try:
            scroll_to_top()
        finally:
            st.session_state['_scroll_to_results'] = False


    # Load cached assets (heavy things computed once)
    (artifacts, (A, B), model, feature_names, lat_col, lon_col,
     train_full, train_raw, test_raw, train_eng, X_train, train_meta) = build_calibrator_and_cached_train()

    # Raw input features are based on Kaggle test.csv columns (excluding Id)
    raw_feature_cols = [c for c in test_raw.columns if c != "Id"]

    # We'll use train_full for lookup by Id and for maps (SalePrice + lat/lon)
    has_latlon = (lat_col is not None and lon_col is not None
                  and lat_col in train_full.columns and lon_col in train_full.columns)

    # Feature importance for ordering + similarity (UI only)
    engineered_imp = _get_model_feature_importances(model, feature_names)
    raw_imp = _raw_feature_importance_from_engineered(raw_feature_cols, engineered_imp)

    # ---- Layout: Left = Inputs (Id + Features), Right = Results (2 tabs) ----
    col_in, col_out = st.columns([0.48, 0.52], gap="large")

    # =========================
    # LEFT: Inputs
    # =========================
    with col_in:
        input_box = st.container(border=True)
        with input_box:
            st.subheader("Inputs")

            # Id selection (same box as features)
            id_list = train_full["Id"].dropna().astype(int).sort_values().tolist() if "Id" in train_full.columns else []
            selected_id = st.selectbox("Select training Id (optional)", options=["(none)"] + id_list)

            eng_ref = None
            disable_inputs = (selected_id != "(none)")

            if selected_id != "(none)":
                row_ref = train_full.loc[train_full["Id"].astype(int) == int(selected_id)].iloc[0]
                # Exact engineered row (matches NB1_new) for this Id if available
                if train_eng is not None and "Id" in train_eng.columns:
                    _m = train_eng.loc[train_eng["Id"].astype(int) == int(selected_id)]
                    if len(_m) > 0:
                        eng_ref = _m.iloc[[0]].copy()
            else:
                row_ref = None

            # Defaults: from chosen Id if any; otherwise mode/median from train_raw
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


            # Build groups for ALL raw features (guaranteed complete)
            group_map = {}
            for f in raw_feature_cols:
                g = _group_raw_feature(f)
                group_map.setdefault(g, []).append(f)

            # sort within each group by importance desc
            for g in list(group_map.keys()):
                group_map[g] = sorted(group_map[g], key=lambda x: float(raw_imp.get(x, 0.0)), reverse=True)

            # explicit group order (matches _group_raw_feature outputs)
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
            # add any unexpected groups (shouldn't happen, but safe)
            for g in sorted(group_map.keys()):
                if g not in group_order:
                    group_order.append(g)

            # Scrollable feature panel (bigger)
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
                                    # numeric
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


            # Which features did the user actually change?
            # We only use this to BOOST similarity weights so comps match the user's chosen inputs.
            changed_features = []
            for c in raw_feature_cols:
                is_num = (c in train_raw.columns) and pd.api.types.is_numeric_dtype(train_raw[c])
                if _value_changed(user_inputs.get(c, None), defaults.get(c, None), is_num):
                    changed_features.append(c)

            # Comparable settings
            with st.expander("Comparable settings", expanded=False):
                topk_neigh = st.slider("Map 2: Top-K similar (same neighborhood)", min_value=5, max_value=200, value=50, step=5)
                topk_sim = st.slider("Map 3: Top-K similar (overall)", min_value=5, max_value=30, value=10, step=1)

            btn = st.button("Get recommendation", type="primary", width="stretch")

            # Hidden subject location for maps (not shown in UI)
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

            # Resolve subject neighborhood for maps
            if row_ref is not None and "Neighborhood" in train_full.columns:
                nbh = row_ref.get("Neighborhood", None)
            else:
                nbh = user_inputs.get("Neighborhood", None)

            # Run & cache results (so switching tabs does not recompute heavy things)
            if btn:
                # Build 1-row raw DF (fill non-chosen features with defaults)
                row = {"Id": int(selected_id) if selected_id != "(none)" else -1}
                for c in raw_feature_cols:
                    row[c] = user_inputs.get(c, defaults.get(c, None))
                raw_one = pd.DataFrame([row])

                # Transform -> engineered (KEEP as app_6.py)
                eng_one = transform_single(raw_one, artifacts, train_raw)

                # Align to model features
                X = eng_one.copy()
                for col in feature_names:
                    if col not in X.columns:
                        X[col] = 0
                X = X[feature_names]

                # Ensure numeric
                for col in X.columns:
                    if X[col].dtype == "object":
                        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

                # ✅ Prediction path unchanged (matches app_6.py)
                # If a training Id was selected and we have the exact engineered row (cache/train_engineered.pkl),
                # use it for prediction to guarantee perfect alignment with NB2_new.
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

                # ---- SHAP data for waterfall (do NOT affect prediction) ----
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

                # ---- Precompute comps for maps (to keep tab switching fast) ----
                maps_payload = {"enabled": False}
                if has_latlon and subj_lat is not None and subj_lon is not None:
                    df_map = train_full.copy()
                    df_map = df_map[df_map[lat_col].notna() & df_map[lon_col].notna()].copy()

                    # Ensure required columns exist
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
                        # Similarity is computed in RAW-feature space (what the user actually chose in UI).
                        # This produces comps whose characteristics are closest to the subject inputs.
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

                        # Exclude self (only if a training Id is selected)
                        ids_all = df_map["Id"].to_numpy()
                        if selected_id != "(none)":
                            sim_all = np.where(ids_all == int(selected_id), -np.inf, sim_all)

                        # Map 3 (overall top similar)
                        k3 = int(topk_sim)
                        top_idx3 = np.argsort(sim_all)[::-1][:k3]
                        cand3 = df_map.iloc[top_idx3].copy()
                        cand3["Similarity"] = sim_all[top_idx3]

                        # Map 2 (top similar within neighborhood)
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
                    "actual": actual,
                    "abs_err": abs_err,
                    "shap": shap_payload,
                    "maps": maps_payload,
                }

                st.session_state['_scroll_to_results'] = True
                st.rerun()

    # =========================
    # RIGHT: Results (2 main tabs)
    # =========================
    with col_out:
        run = st.session_state.get("last_run", None)
        if run is None:
            return

        tabs = st.tabs(["Tab 1: Recommendation & Explanation", "Tab 2: Maps"])

        # ---- Tab 1: prediction + actual + SHAP waterfall ----
        with tabs[0]:
            st.subheader("Price recommendation")
            st.metric("Recommended price", f"${run['pred_price']:,.0f}")

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
                        plt.figure(figsize=(7.2, 2.7))  # compact height
                        shap_mod.plots.waterfall(exp, max_display=10, show=False)
                        st.pyplot(plt.gcf(), clear_figure=True)
                    except Exception:
                        fig_w = _plot_waterfall_custom(base_log, contrib, max_display=10)
                        st.plotly_chart(fig_w, width="stretch")
                else:
                    fig_w = _plot_waterfall_custom(base_log, contrib, max_display=10)
                    st.plotly_chart(fig_w, width="stretch")

        # ---- Tab 2: 3 maps ----
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
