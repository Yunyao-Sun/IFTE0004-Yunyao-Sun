import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


NN_ARCHITECTURES = {
    "NN1": [2],
    "NN2": [4, 2],
    "NN3": [8, 4, 2],
    "NN4": [16, 8, 4, 2],
}


def split_by_time(g, train_frac=0.70, val_frac=0.10):
    n = len(g)
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))
    train = g.iloc[:n_train].copy()
    val = g.iloc[n_train:n_train + n_val].copy()
    test = g.iloc[n_train + n_val:].copy()
    return train, val, test


def clean_forecast(pred, train_y):
    pred = np.asarray(pred).copy()
    min_y = np.nanmin(train_y)
    pred[~np.isfinite(pred)] = min_y
    pred[pred <= 0] = min_y
    return pred


def fit_ols_predict(train, val, test, features, target="y"):
    fit_data = pd.concat([train, val], axis=0)
    model = LinearRegression()
    model.fit(fit_data[features], fit_data[target])
    return model.predict(test[features]), model


def fit_loghar_predict(train, val, test, features):
    fit_data = pd.concat([train, val], axis=0)
    model = LinearRegression()
    model.fit(fit_data[features], fit_data["log_y"])
    log_pred = model.predict(test[features])
    resid = fit_data["log_y"].values - model.predict(fit_data[features])
    sigma2 = np.var(resid)
    pred = np.exp(log_pred + 0.5 * sigma2)
    return pred, model


def choose_by_validation(model_grid, train, val, features, target="y", scale=True):
    best_model = None
    best_mse = np.inf
    for model in model_grid:
        if scale:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(model))])
        else:
            pipe = clone(model)
        pipe.fit(train[features], train[target])
        mse = mean_squared_error(val[target], pipe.predict(val[features]))
        if mse < best_mse:
            best_mse = mse
            best_model = pipe
    fit_data = pd.concat([train, val], axis=0)
    best_model.fit(fit_data[features], fit_data[target])
    return best_model


def fit_regularized_predict(train, val, test, features, kind="RR", target="y", random_state=42):
    alphas = np.logspace(-5, 2, 80)
    if kind == "RR":
        grid = [Ridge(alpha=a, random_state=random_state) for a in alphas]
    elif kind == "LA":
        grid = [Lasso(alpha=a, max_iter=50000, random_state=random_state) for a in alphas]
    elif kind == "EN":
        grid = [ElasticNet(alpha=a, l1_ratio=l1, max_iter=50000, random_state=random_state)
                for a in alphas for l1 in np.linspace(0.1, 0.9, 9)]
    else:
        raise ValueError("kind must be RR, LA, or EN")
    model = choose_by_validation(grid, train, val, features, target=target, scale=True)
    return model.predict(test[features]), model


def fit_post_lasso_predict(train, val, test, features, target="y", random_state=42):
    _, lasso_pipe = fit_regularized_predict(train, val, test, features, kind="LA", target=target, random_state=random_state)
    lasso = lasso_pipe.named_steps["model"]
    selected = [f for f, b in zip(features, lasso.coef_) if abs(b) > 1e-12]
    if len(selected) == 0:
        selected = features.copy()
    pred, model = fit_ols_predict(train, val, test, selected, target=target)
    return pred, {"selected": selected, "model": model}


def fit_adaptive_lasso_predict(train, val, test, features, target="y", random_state=42, eps=1e-8):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_val = scaler.transform(val[features])
    X_test = scaler.transform(test[features])
    y_train = train[target].values
    y_val = val[target].values
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    weights = 1.0 / (np.abs(ols.coef_) + eps)
    Xw_train = X_train / weights
    Xw_val = X_val / weights
    Xw_test = X_test / weights
    best_alpha = None
    best_mse = np.inf
    for alpha in np.logspace(-5, 2, 80):
        model = Lasso(alpha=alpha, max_iter=50000, random_state=random_state)
        model.fit(Xw_train, y_train)
        mse = mean_squared_error(y_val, model.predict(Xw_val))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    final = Lasso(alpha=best_alpha, max_iter=50000, random_state=random_state)
    final.fit(X_fit / weights, y_fit)
    return final.predict(Xw_test), {"scaler": scaler, "weights": weights, "alpha": best_alpha, "model": final}


def make_bagging_regressor(base, n_estimators=300, random_state=42):
    try:
        return BaggingRegressor(estimator=base, n_estimators=n_estimators, bootstrap=True, random_state=random_state, n_jobs=-1)
    except TypeError:
        return BaggingRegressor(base_estimator=base, n_estimators=n_estimators, bootstrap=True, random_state=random_state, n_jobs=-1)


def fit_tree_predict(train, val, test, features, kind="RF", target="y", random_state=42):
    fit_data = pd.concat([train, val], axis=0)
    if kind == "BG":
        base = DecisionTreeRegressor(min_samples_leaf=5, random_state=random_state)
        model = make_bagging_regressor(base, n_estimators=300, random_state=random_state)
        model.fit(fit_data[features], fit_data[target])
    elif kind == "RF":
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=max(1, int(np.floor(len(features)/3))), bootstrap=True, random_state=random_state, n_jobs=-1)
        model.fit(fit_data[features], fit_data[target])
    elif kind == "GB":
        grid = []
        for depth in [1, 2]:
            for n_estimators in [50, 100, 200, 300]:
                for lr in [0.01, 0.05, 0.10]:
                    grid.append(GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr, max_depth=depth, min_samples_leaf=5, random_state=random_state))
        best = None
        best_mse = np.inf
        for model in grid:
            model.fit(train[features], train[target])
            mse = mean_squared_error(val[target], model.predict(val[features]))
            if mse < best_mse:
                best_mse = mse
                best = clone(model)
        best.fit(fit_data[features], fit_data[target])
        model = best
    else:
        raise ValueError("kind must be BG, RF, or GB")
    return model.predict(test[features]), model


def build_nn(input_dim, hidden_layers, dropout_rate=0.1, lr=0.001, seed=0):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, kernel_initializer=keras.initializers.GlorotNormal(seed=seed+i), bias_initializer="zeros")(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Dropout(dropout_rate, seed=seed+100+i)(x)
    outputs = layers.Dense(1, kernel_initializer=keras.initializers.GlorotNormal(seed=seed+999))(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def fit_nn_predict(train, val, test, features, arch_name="NN1", ensemble_size=1, n_seeds=5, target="y", epochs=500, patience=100, batch_size=32):
    hidden_layers = NN_ARCHITECTURES[arch_name]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features]).astype("float32")
    X_val = scaler.transform(val[features]).astype("float32")
    X_test = scaler.transform(test[features]).astype("float32")
    y_train = train[target].values.astype("float32")
    y_val = val[target].values.astype("float32")
    ranked = []
    for seed in range(n_seeds):
        model = build_nn(X_train.shape[1], hidden_layers, seed=seed)
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=0)
        val_pred = model.predict(X_val, verbose=0).ravel()
        test_pred = model.predict(X_test, verbose=0).ravel()
        ranked.append({"seed": seed, "val_mse": mean_squared_error(y_val, val_pred), "test_pred": test_pred})
    ranked = sorted(ranked, key=lambda x: x["val_mse"])
    selected = ranked[:min(ensemble_size, len(ranked))]
    pred = np.mean([x["test_pred"] for x in selected], axis=0)
    return pred, {"arch": arch_name, "ensemble_size": ensemble_size, "selected_seeds": [x["seed"] for x in selected]}


def run_model_zoo(panel, feature_sets, config):
    g = panel.copy().replace([np.inf, -np.inf], np.nan)
    ml_features = feature_sets["HAR"] if config.MODEL_SET == "MHAR" else feature_sets["MALL"]
    required_cols = sorted(set(["asset", "date", "y", "log_y"] + ml_features + sum(feature_sets.values(), [])))
    required_cols = [c for c in required_cols if c in g.columns]
    g = g[required_cols].dropna().copy()
    print(f"Final sample size: {len(g)}")
    print(f"ML feature set: {ml_features}")
    train, val, test = split_by_time(g, config.TRAIN_FRAC, config.VAL_FRAC)
    results = {}
    fitted_models = {}
    for name in ["HAR", "HAR-X", "LogHAR", "LevHAR", "SHAR", "HARQ"]:
        if name not in feature_sets:
            continue
        print(f"Fitting {name}")
        cols = feature_sets[name]
        if name == "LogHAR":
            pred, model = fit_loghar_predict(train, val, test, cols)
        else:
            pred, model = fit_ols_predict(train, val, test, cols)
        pred = clean_forecast(pred, pd.concat([train, val])["y"].values)
        results[name] = pred
        fitted_models[name] = model
    for name, kind in [("RR", "RR"), ("LA", "LA"), ("EN", "EN")]:
        print(f"Fitting {name}")
        pred, model = fit_regularized_predict(train, val, test, ml_features, kind=kind, random_state=config.RANDOM_STATE)
        results[name] = clean_forecast(pred, pd.concat([train, val])["y"].values)
        fitted_models[name] = model
    print("Fitting A-LA")
    pred, model = fit_adaptive_lasso_predict(train, val, test, ml_features, random_state=config.RANDOM_STATE)
    results["A-LA"] = clean_forecast(pred, pd.concat([train, val])["y"].values)
    fitted_models["A-LA"] = model
    print("Fitting P-LA")
    pred, model = fit_post_lasso_predict(train, val, test, ml_features, random_state=config.RANDOM_STATE)
    results["P-LA"] = clean_forecast(pred, pd.concat([train, val])["y"].values)
    fitted_models["P-LA"] = model
    for name in ["BG", "RF", "GB"]:
        print(f"Fitting {name}")
        pred, model = fit_tree_predict(train, val, test, ml_features, kind=name, random_state=config.RANDOM_STATE)
        results[name] = clean_forecast(pred, pd.concat([train, val])["y"].values)
        fitted_models[name] = model
    for arch in ["NN1", "NN2", "NN3", "NN4"]:
        layer_id = arch[-1]
        for ensemble_size in [1, 10]:
            print(f"Fitting {arch}, ensemble={ensemble_size}")
            pred, model = fit_nn_predict(train, val, test, ml_features, arch_name=arch, ensemble_size=ensemble_size, n_seeds=config.NN_SEEDS)
            results[f"NN{ensemble_size}{layer_id}"] = clean_forecast(pred, pd.concat([train, val])["y"].values)
            fitted_models[f"NN{ensemble_size}{layer_id}"] = model
    return {"asset": g["asset"].iloc[0], "date": test["date"].values, "y_test": test["y"].values, "predictions": results, "models": fitted_models, "features": feature_sets}
