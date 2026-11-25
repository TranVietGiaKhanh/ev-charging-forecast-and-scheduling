"""
BƯỚC 2 - DỰ BÁO PHỤ TẢI TỔNG HỢP THEO GIỜ (toàn trạm, KHÔNG dự báo từng xe)

Hai mục tiêu:  N = số xe đến trong giờ,  L = tải trạm trung bình (kW)
Bài toán hồi quy: biết đặc trưng của giờ t (lịch + lịch sử cách >= 24 giờ) -> dự đoán y(t).

Nhóm mô hình
  Baseline : Seasonal Naive 24h / 168h, Trung bình lịch sử theo (giờ, thứ)
  Học máy  : Ridge, Decision Tree, Random Forest
  Học sâu  : MLP (mạng nơ-ron truyền thẳng 2 lớp ẩn)
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from . import config as C
from . import timeaxis as T

TARGETS = {"N": "Số xe đến / giờ", "L": "Tải trạm (kW)"}
CAT_COLS = ["hour", "dow"]


def feature_cols(y):
    return ["hour", "dow", "is_weekend", "is_holiday",
            f"{y}_lag24", f"{y}_lag48", f"{y}_lag168", f"{y}_roll7"]


def make_preprocessor(y, onehot):
    """
    onehot=True  (Ridge, MLP): hour và dow -> one-hot, các cột số -> chuẩn hóa (z-score)
    onehot=False (cây)       : giữ nguyên giá trị, cây không cần chuẩn hóa
    """
    cols = feature_cols(y)
    if onehot:
        num = [c for c in cols if c not in CAT_COLS]
        return ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
             ("num", StandardScaler(), num)],
            sparse_threshold=0.0)
    return ColumnTransformer([("keep", "passthrough", cols)], sparse_threshold=0.0)


# ================================================================== MLP
class MLPForecaster:
    """
    Mạng MLP tự viết vòng lặp huấn luyện (dùng partial_fit = đúng 1 epoch mỗi lần gọi):
      1. Tách 20% CUỐI của tập huấn luyện làm tập validation (theo thời gian, không xáo trộn).
      2. Mỗi epoch: cập nhật trọng số bằng Adam trên 80% đầu, đo MSE trên validation.
      3. Dừng sớm (early stopping) nếu val-loss không giảm sau `patience` epoch liên tiếp.
      4. Ghi nhớ số epoch tốt nhất, huấn luyện lại từ đầu trên TOÀN BỘ tập huấn luyện đúng số epoch đó.
      5. Lặp lại với n_seeds hạt giống khác nhau và lấy trung bình dự báo (giảm dao động).
    """

    def __init__(self, target, hidden=(64, 32), lr=1e-3, alpha=1e-3, batch_size=32,
                 max_epochs=400, patience=30, val_frac=0.2, n_seeds=5, seed=C.SEED):
        self.target, self.hidden, self.lr, self.alpha = target, hidden, lr, alpha
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.val_frac, self.n_seeds, self.seed = val_frac, n_seeds, seed

    def _new_net(self, seed):
        return MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu", solver="adam",
                            alpha=self.alpha, learning_rate_init=self.lr,
                            batch_size=self.batch_size, shuffle=True, random_state=seed)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.prep_ = make_preprocessor(self.target, onehot=True)
        Xa = self.prep_.fit_transform(X)
        self.mu_, self.sd_ = y.mean(), (y.std() or 1.0)
        ya = (y - self.mu_) / self.sd_                       # chuẩn hóa nhãn
        n_val = max(1, int(len(Xa) * self.val_frac))
        Xtr, ytr, Xva, yva = Xa[:-n_val], ya[:-n_val], Xa[-n_val:], ya[-n_val:]

        self.nets_, self.histories_, self.best_epochs_ = [], [], []
        for s in range(self.n_seeds):
            net, best, best_ep, wait, hist = self._new_net(self.seed + s), np.inf, 1, 0, []
            for ep in range(1, self.max_epochs + 1):
                net.partial_fit(Xtr, ytr)                     # 1 epoch
                tr = mean_squared_error(ytr, net.predict(Xtr))
                va = mean_squared_error(yva, net.predict(Xva))
                hist.append((ep, tr, va))
                if va < best - 1e-6:
                    best, best_ep, wait = va, ep, 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            final = self._new_net(self.seed + s)              # huấn luyện lại trên toàn bộ tập train
            for _ in range(best_ep):
                final.partial_fit(Xa, ya)
            self.nets_.append(final)
            self.histories_.append(np.array(hist))
            self.best_epochs_.append(best_ep)
        return self

    def predict(self, X):
        Xa = self.prep_.transform(X)
        p = np.mean([n.predict(Xa) for n in self.nets_], axis=0)
        return p * self.sd_ + self.mu_


# ================================================================== các mô hình
def build_sklearn_models(y):
    """(pipeline, lưới siêu tham số) - siêu tham số được chọn bằng TimeSeriesSplit trên tập train."""
    return {
        "Ridge": (
            Pipeline([("prep", make_preprocessor(y, True)), ("m", Ridge())]),
            {"m__alpha": [0.1, 1, 10, 100]}),
        "DecisionTree": (
            Pipeline([("prep", make_preprocessor(y, False)),
                      ("m", DecisionTreeRegressor(random_state=C.SEED))]),
            {"m__max_depth": [3, 4, 5, 6, 8], "m__min_samples_leaf": [2, 5, 10]}),
        "RandomForest": (
            Pipeline([("prep", make_preprocessor(y, False)),
                      ("m", RandomForestRegressor(n_estimators=300, random_state=C.SEED, n_jobs=-1))]),
            {"m__max_depth": [4, 6, 8, None], "m__min_samples_leaf": [1, 3, 5]}),
    }


def baseline_predictions(y, train, test):
    # trung bình lịch sử theo (giờ, thứ) - tính trên tập train
    table = train.groupby(["hour", "dow"])[y].mean().rename("p").reset_index()
    ha = test[["hour", "dow"]].merge(table, how="left", on=["hour", "dow"])["p"]
    ha = ha.fillna(train[y].mean()).values
    return {
        "SeasonalNaive_24h": test[f"{y}_lag24"].values,
        "SeasonalNaive_168h": test[f"{y}_lag168"].values,
        "HistAvg_hour_dow": ha,
    }


def metrics(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)                         # tải và số xe không âm
    return (mean_absolute_error(y_true, y_pred),
            float(np.sqrt(mean_squared_error(y_true, y_pred))))


def fit_target(y, train, test, verbose=True):
    """Huấn luyện mọi mô hình cho mục tiêu y trên `train`, dự báo `test`. Trả về (dự báo, mô hình đã khớp, siêu tham số)."""
    cols = feature_cols(y)
    Xtr, ytr, Xte = train[cols], train[y].values, test[cols]
    preds = baseline_predictions(y, train, test)               # baseline: không cần huấn luyện
    fitted, params = {}, {}
    for name, (pipe, grid) in build_sklearn_models(y).items():
        gs = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(n_splits=C.CV_SPLITS),
                          scoring="neg_mean_absolute_error", refit=True)
        gs.fit(Xtr, ytr)
        fitted[name], preds[name] = gs, gs.predict(Xte)
        params[name] = {k: (None if v is None else v) for k, v in gs.best_params_.items()}
        if verbose:
            print(f"[{y}] {name}: tham số tốt nhất {gs.best_params_} | MAE CV = {-gs.best_score_:.3f}")
    mlp = MLPForecaster(y).fit(Xtr, ytr)
    fitted["MLP"], preds["MLP"] = mlp, mlp.predict(Xte)
    params["MLP"] = {"best_epochs": [int(e) for e in mlp.best_epochs_]}
    if verbose:
        print(f"[{y}] MLP: epoch tốt nhất theo từng seed {mlp.best_epochs_}")
    return {k: np.clip(v, 0, None) for k, v in preds.items()}, fitted, params


# ================================================================== chạy toàn bộ
def run():
    C.RESULTS.mkdir(exist_ok=True)
    C.FIGURES.mkdir(exist_ok=True)
    feats = pd.read_csv(C.DATA_PROC / "features.csv", index_col=0, parse_dates=True)    # chỉ mục = giờ thực UTC
    train = feats.loc[T.utc(C.TRAIN_START):T.utc(C.TRAIN_END)]
    test = feats.loc[T.utc(C.TEST_START):T.utc(C.TEST_END)]
    print(f"Huấn luyện: {len(train)} giờ | Kiểm thử: {len(test)} giờ\n")

    rows, test_preds, insample = [], {}, {}
    best_params = {}
    for y, ylabel in TARGETS.items():
        cols = feature_cols(y)
        Xtr, yte = train[cols], test[y].values
        preds, fitted, params = fit_target(y, train, test)
        best_params.update({f"{y}/{k}": v for k, v in params.items()})
        mlp = fitted["MLP"]

        # đường cong huấn luyện của MLP (seed đầu tiên)
        h = mlp.histories_[0]
        plt.figure(figsize=(6, 3.5))
        plt.plot(h[:, 0], h[:, 1], label="train loss (80% đầu)")
        plt.plot(h[:, 0], h[:, 2], label="validation loss (20% cuối)")
        plt.axvline(mlp.best_epochs_[0], color="gray", ls="--", label="epoch tốt nhất")
        plt.xlabel("epoch"); plt.ylabel("MSE (nhãn đã chuẩn hóa)")
        plt.title(f"MLP - đường cong huấn luyện ({y})"); plt.legend(); plt.tight_layout()
        plt.savefig(C.FIGURES / f"mlp_training_curve_{y}.png", dpi=150); plt.close()

        # ---- đánh giá trên tập kiểm thử
        mae_ref, _ = metrics(yte, preds["SeasonalNaive_24h"])
        for name, p in preds.items():
            mae, rmse = metrics(yte, p)
            rows.append({"target": y, "model": name, "MAE": mae, "RMSE": rmse,
                         "improve_vs_SN24_%": 100 * (mae_ref - mae) / mae_ref})
        test_preds[y] = pd.DataFrame({"actual": yte, **{k: np.clip(v, 0, None) for k, v in preds.items()}},
                                     index=test.index)

        # ---- dự báo trong mẫu (train) cho mô hình dùng để chừa công suất dự phòng
        if y == "N":
            m = fitted[C.RESERVE_MODEL]
            nhat = pd.concat([
                pd.Series(np.clip(m.predict(Xtr), 0, None), index=train.index),
                pd.Series(np.clip(preds[C.RESERVE_MODEL], 0, None), index=test.index)])
            pd.DataFrame({"N_hat": nhat,
                          "split": ["train(in-sample)"] * len(train) + ["test"] * len(test)}
                         ).rename_axis("timestamp_utc").to_csv(C.RESULTS / "N_hat_hourly.csv")

        # ---- hình dự báo
        plt.figure(figsize=(11, 3.8))
        d = test_preds[y]
        x = T.local_naive(d.index)                            # trục hình: giờ địa phương (giờ hiển thị)
        plt.plot(x, d["actual"], "k", lw=1.6, label="Thực tế")
        for name, st in [("SeasonalNaive_24h", ":"), ("Ridge", "-"), ("RandomForest", "-"), ("MLP", "-")]:
            plt.plot(x, d[name], st, lw=1, label=name)
        plt.ylabel(ylabel); plt.title(f"Dự báo {ylabel} trên tập kiểm thử ({x[0]:%d/%m/%Y} - {x[-1]:%d/%m/%Y})")
        plt.legend(ncol=5, fontsize=8); plt.tight_layout()
        plt.savefig(C.FIGURES / f"forecast_{y}.png", dpi=150); plt.close()

    res = pd.DataFrame(rows).round(3)
    res.to_csv(C.RESULTS / "forecast_metrics.csv", index=False)
    out = pd.concat([test_preds[y].add_prefix(f"{y}_") for y in TARGETS], axis=1).rename_axis("timestamp_utc")
    out.insert(0, "local_time", T.local_naive(out.index))     # giờ hiển thị; chỉ mục là giờ thực
    out.to_csv(C.RESULTS / "forecast_predictions_test.csv")
    with open(C.RESULTS / "forecast_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n=== KẾT QUẢ TRÊN TẬP KIỂM THỬ ({len(test)} giờ) ===")
    for y in TARGETS:
        print(f"\nMục tiêu {y} ({TARGETS[y]})")
        print(res[res.target == y].drop(columns="target").to_string(index=False))
    print("\nĐã lưu: results/forecast_metrics.csv, forecast_predictions_test.csv, N_hat_hourly.csv, figures/*.png")
