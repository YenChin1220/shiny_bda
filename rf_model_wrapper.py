# rf_model_wrapper.py（對應你 RF 訓練流程的預測 wrapper）

import numpy as np
import pickle
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fft, fftfreq
import pywt
import json
import sys

# ===== 載入模型與 scaler =====
with open("rf_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("rf_clf.pkl", "rb") as f:
    clf = pickle.load(f)

with open("rf_reg.pkl", "rb") as f:
    reg = pickle.load(f)

# ===== 特徵擷取函數（15 維） =====
def extract_features(signal, fs=4880):
    features = []
    rms = np.sqrt(np.mean(signal**2))
    features.append(rms)
    features.append(skew(signal))
    features.append(kurtosis(signal))
    pdf, _ = np.histogram(signal, bins=100, density=True)
    features.append(entropy(pdf + 1e-12))
    features.append(np.max(np.abs(signal)) / (rms + 1e-12))

    fft_vals = np.abs(fft(signal))[:len(signal)//2]
    freqs = fftfreq(len(signal), d=1/fs)[:len(signal)//2]
    features.append(np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-12))
    features.append(np.sqrt(np.sum((freqs**2) * (fft_vals**2)) / (np.sum(fft_vals**2) + 1e-12)))
    features.append(kurtosis(fft_vals))
    features.append(entropy(fft_vals / (np.sum(fft_vals) + 1e-12)))

    coeffs = pywt.wavedec(signal, 'db4', level=3)
    features.append(kurtosis(coeffs[1]))

    features.append(np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-12))
    features.append(np.max(np.abs(signal)) / ((np.mean(np.sqrt(np.abs(signal)))**2) + 1e-12))
    features.append(np.std(np.diff(signal)) / (np.std(signal) + 1e-12))
    features.append(np.sum(fft_vals**2))
    features.append(np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal)))**2 + 1e-12))
    return features

# ===== 單筆預測主流程 =====
def run_prediction(file_path):
    data = np.loadtxt(file_path, skiprows=1)  # 預期為 1D 序列
    if data.ndim == 2 and data.shape[1] == 3:
        signal = data.mean(axis=1)
    else:
        signal = data.flatten()

    feats = extract_features(signal)
    X = np.array(feats).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred_health = int(clf.predict(X_scaled)[0])
    prob = float(clf.predict_proba(X_scaled)[0, 1] * 100)
    pred_load = float(reg.predict(X_scaled)[0])

    return {
        "Health_Prediction": pred_health,
        "Health_Probability": round(prob, 2),
        "Predicted_Load": round(pred_load, 2)
    }

# ===== 主程式入口：支援 CLI 或 Shiny =====
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            result = run_prediction(file_path)
        else:
            input_json = sys.stdin.read()
            data = json.loads(input_json)
            result = run_prediction(data["file_path"])
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
