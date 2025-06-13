import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from sklearn.model_selection import train_test_split
import torch
import pywt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pywt import wavedec, cwt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import pickle
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fft, fftfreq

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 手動加載 PingFang 字體（macOS 專用）
font_path = "/System/Library/Fonts/PingFang.ttc"  # 這個是 TrueType collection
prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False



def read_data(folder_path, min_length=18000):
    data_dict = {}
    folder_direction = os.path.basename(folder_path)

    subfolders_or_files = sorted(os.listdir(folder_path))
    has_subfolders = any(os.path.isdir(os.path.join(folder_path, f)) for f in subfolders_or_files)

    if has_subfolders:
        for subfolder in subfolders_or_files:
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            files = sorted([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
            for i, file_name in enumerate(files):
                file_path = os.path.join(subfolder_path, file_name)
                data = np.loadtxt(file_path, skiprows=1)
                if len(data) < min_length:
                    continue  
                key = f"data_{folder_direction}_{subfolder}_{i + 1}"
                data_dict[key] = data
    else:
        files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        for i, file_name in enumerate(files):
            file_path = os.path.join(folder_path, file_name)
            data = np.loadtxt(file_path, skiprows=1)
            if len(data) < min_length:
                continue  
            key = f"data_{folder_direction}_unknown_{i + 1}"
            data_dict[key] = data

    return data_dict

train_paths = ['/Users/yenchin/Documents/BDA/shiny01/Data/train/Xa', '/Users/yenchin/Documents/BDA/shiny01/Data/train/Xb', 
               '/Users/yenchin/Documents/BDA/shiny01/Data/train/Ya', '/Users/yenchin/Documents/BDA/shiny01/Data/train/Yb']
test_paths = ['/Users/yenchin/Documents/BDA/shiny01/Data/train/Xa', '/Users/yenchin/Documents/BDA/shiny01/Data/train/Xb',
               '/Users/yenchin/Documents/BDA/shiny01/Data/train/Ya', '/Users/yenchin/Documents/BDA/shiny01/Data/train/Yb']

train_data = {}
test_data = {}

for path in train_paths:
    train_data.update(read_data(path))

for path in test_paths:
    test_data.update(read_data(path))

def extract_features(signal, fs=4880):
    features = {}
    features['RMS'] = np.sqrt(np.mean(signal**2))
    features['Skewness'] = skew(signal)
    features['Kurtosis'] = kurtosis(signal)
    pdf, _ = np.histogram(signal, bins=100, density=True)
    features['Entropy'] = entropy(pdf + 1e-12)
    features['CrestFactor'] = np.max(np.abs(signal)) / (features['RMS'] + 1e-12)

    fft_vals = np.abs(fft(signal))[:len(signal)//2]
    freqs = fftfreq(len(signal), d=1/fs)[:len(signal)//2]
    features['Freq_Center'] = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-12)
    features['RMSF'] = np.sqrt(np.sum((freqs**2) * (fft_vals**2)) / (np.sum(fft_vals**2) + 1e-12))
    features['Spectral_Kurtosis'] = kurtosis(fft_vals)
    features['Spectral_Entropy'] = entropy(fft_vals / (np.sum(fft_vals) + 1e-12))

    features['Impulse_Factor'] = np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-12)
    features['Margin_Factor'] = np.max(np.abs(signal)) / ((np.mean(np.sqrt(np.abs(signal)))**2) + 1e-12)
    features['Approximate_Entropy'] = np.std(np.diff(signal)) / (np.std(signal) + 1e-12)

    coeffs = pywt.wavedec(signal, 'db4', level=3)
    features['Wavelet_D3_Kurtosis'] = kurtosis(coeffs[1])

    
    features['Spectral_Energy'] = np.sum(fft_vals**2)  # Total spectral energy
    features['Clearance_Factor'] = np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal)))**2 + 1e-12)

    return features

feature_list = []

for key, data in train_data.items():
    if data.shape[1] != 3:
        continue
    avg_signal = data.mean(axis=1)
    features = extract_features(avg_signal)

    parts = key.split('_')  
    direction = parts[1]
    load = int(parts[2])
    index = int(parts[3])

    features['Direction'] = direction
    features['Load'] = load
    features['Index'] = index

    feature_list.append(features)

train_feature_df = pd.DataFrame(feature_list)

test_feature_list = []

for key, data in test_data.items():
    if data.shape[1] != 3:
        continue
    avg_signal = data.mean(axis=1)
    features = extract_features(avg_signal)

    parts = key.split('_')  
    direction = parts[1]
    index = int(parts[3])

    features['Direction'] = direction
    features['Index'] = index

    test_feature_list.append(features)

test_feature_df = pd.DataFrame(test_feature_list)

feature_cols = [
    'RMS', 'Skewness', 'Kurtosis', 'Entropy', 'CrestFactor',
    'Freq_Center', 'RMSF', 'Spectral_Kurtosis', 'Spectral_Entropy',
    'Wavelet_D3_Kurtosis', 'Impulse_Factor', 'Margin_Factor',
    'Approximate_Entropy', 'Spectral_Energy', 'Clearance_Factor'  
]

def assign_gt_health(row):
    return int((row['Direction'] in ['Xa', 'Xb'] and row['Load'] == 80) or 
               (row['Direction'] in ['Ya', 'Yb'] and row['Load'] == 260))

train_feature_df['GT_Health'] = train_feature_df.apply(assign_gt_health, axis=1)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

X = train_feature_df[feature_cols]
y = train_feature_df['GT_Health']


X_train_all, X_holdout, y_train_all, y_holdout = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_all_scaled = scaler.fit_transform(X_train_all)
X_holdout_scaled = scaler.transform(X_holdout)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_feature_df['HI_RF'] = np.nan  

fold_metrics = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_all_scaled, y_train_all), 1):
    X_train = X_train_all_scaled[train_idx]
    X_val = X_train_all_scaled[val_idx]
    y_train = y_train_all.iloc[train_idx]
    y_val = y_train_all.iloc[val_idx]
    val_orig_idx = y_train_all.iloc[val_idx].index

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_val)[:, 1]
    train_feature_df.loc[val_orig_idx, 'HI_RF'] = y_proba * 100  # 

    acc = accuracy_score(y_val, clf.predict(X_val))
    auc = roc_auc_score(y_val, y_proba)
    print(f"[Fold {fold_idx}] Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    fold_metrics.append({'Fold': fold_idx, 'Accuracy': acc, 'AUC': auc})

print("\n平均 Accuracy:", np.mean([m['Accuracy'] for m in fold_metrics]))
print("平均 AUC     :", np.mean([m['AUC'] for m in fold_metrics]))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


X_reg = train_feature_df[feature_cols]
y_reg = train_feature_df['Load']


reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_reg, y_reg)

train_feature_df['Predicted_Load'] = reg_model.predict(X_reg)


mse_by_direction = []

for direction in train_feature_df['Direction'].unique():
    df_sub = train_feature_df[train_feature_df['Direction'] == direction]
    mse = mean_squared_error(df_sub['Load'], df_sub['Predicted_Load'])
    mse_by_direction.append({'Direction': direction, 'MSE': round(mse, 4)})

df_mse = pd.DataFrame(mse_by_direction)
print(" 各方向負荷預測誤差:")
print(df_mse)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


X_reg = train_feature_df[feature_cols]
y_reg = train_feature_df['Load']


reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_reg, y_reg)


train_feature_df['Predicted_Load'] = reg_model.predict(X_reg)


mse_by_load = []

for load in sorted(train_feature_df['Load'].unique()):
    df_sub = train_feature_df[train_feature_df['Load'] == load]
    mse = mean_squared_error(df_sub['Load'], df_sub['Predicted_Load'])
    mse_by_load.append({'Load': load, 'MSE': round(mse, 4)})

df_mse_load = pd.DataFrame(mse_by_load)
print("各負荷預測誤差:")
print(df_mse_load)


X_test = test_feature_df[feature_cols]
X_test_scaled = scaler.transform(X_test)


test_feature_df['Predicted_Health'] = clf.predict(X_test_scaled)
test_feature_df['Health_Probability'] = clf.predict_proba(X_test_scaled)[:, 1] * 100


print(test_feature_df[['Direction', 'Index', 'Predicted_Health', 'Health_Probability']].head())

healthy_hi = train_feature_df.loc[train_feature_df['GT_Health'] == 1, 'HI_RF']
unhealthy_hi = train_feature_df.loc[train_feature_df['GT_Health'] == 0, 'HI_RF']


healthy_hi = healthy_hi.dropna()
unhealthy_hi = unhealthy_hi.dropna()

bins = np.linspace(0, 100, 20)

plt.figure(figsize=(8, 5))
plt.hist(healthy_hi, bins=bins, alpha=0.7, label='GT Healthy', color='green', edgecolor='black')
plt.hist(unhealthy_hi, bins=bins, alpha=0.7, label='GT Unhealthy', color='red', edgecolor='black')
plt.xlabel('HI')
plt.ylabel('樣本數')
plt.title('不同HI分數的群組樣本數')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"健康樣本平均HI: {np.mean(healthy_hi):.2f} ± {np.std(healthy_hi):.2f}")
print(f"不健康樣本平均HI: {np.mean(unhealthy_hi):.2f} ± {np.std(unhealthy_hi):.2f}")
print(f"平均差距: {np.mean(healthy_hi) - np.mean(unhealthy_hi):.2f}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np

results = []

for direction in ['Xa', 'Xb', 'Ya', 'Yb']:
    df_dir = train_feature_df[train_feature_df['Direction'] == direction]

    #防呆
    if df_dir['GT_Health'].nunique() < 2:
        print(f"⚠️ {direction} 無法訓練（只有一類 GT_Health）")
        continue

    
    X_dir = df_dir[feature_cols]
    y_dir = df_dir['GT_Health']

   
    X_train, X_val, y_train, y_val = train_test_split(
        X_dir, y_dir, test_size=0.2, stratify=y_dir, random_state=42
    )

   
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

   
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]  # 機率值

  
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    recall_unhealthy = recall_score(y_val, y_pred, pos_label=1)

   
    prob_healthy = y_proba[y_val == 0]
    prob_unhealthy = y_proba[y_val == 1]
    ks_stat, _ = ks_2samp(prob_healthy, prob_unhealthy)

    
    results.append({
        'Direction': direction,
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Recall': round(recall_unhealthy, 4),
        'KS_Distance': round(ks_stat, 4)
    })


result_df = pd.DataFrame(results)
print(result_df)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


metrics = ['Accuracy', 'AUC', 'Recall', 'KS_Distance']
data = result_df.set_index('Direction')[metrics].astype(float)


epsilon = np.random.normal(0, 1e-5, size=data.shape)
data += epsilon


scaler = StandardScaler()
data_scaled = data.copy()


labels = data_scaled.columns.tolist()
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1] 
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))


for direction, row in data_scaled.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=direction)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title('各方向表現雷達圖', size=14)
ax.set_yticklabels([])
ax.grid(True)
plt.legend(title='方向', loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()

print(result_df.dtypes)
print(result_df.head())
print(data_scaled)

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np


feature_cols = [
    'RMS', 'Skewness', 'Kurtosis', 'Entropy', 'CrestFactor',
    'Freq_Center', 'RMSF', 'Spectral_Kurtosis', 'Spectral_Entropy',
    'Wavelet_D3_Kurtosis', 'Impulse_Factor', 'Margin_Factor',
    'Approximate_Entropy'
]

# 每個特徵一張圖，內含 4 個方向的子圖
for feature in feature_cols:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{feature} 分布對比（健康 vs 不健康）', fontsize=16)

    for i, direction in enumerate(['Xa', 'Xb', 'Ya', 'Yb']):
        subset = train_feature_df[train_feature_df['Direction'] == direction]
        healthy = subset[subset['GT_Health'] == 1][feature]
        unhealthy = subset[subset['GT_Health'] == 0][feature]

        ax = axes[i // 2, i % 2]

        if len(healthy) < 2 or len(unhealthy) < 2:
            ax.set_title(f'{direction} - 資料不足')
            ax.axis('off')
            continue

        x_min = min(subset[feature])
        x_max = max(subset[feature])
        x_range = np.linspace(x_min, x_max, 200)

        kde_healthy = gaussian_kde(healthy)
        kde_unhealthy = gaussian_kde(unhealthy)

        ax.plot(x_range, kde_healthy(x_range), label='Healthy', color='green')
        ax.plot(x_range, kde_unhealthy(x_range), label='Unhealthy', color='red')
        ax.fill_between(x_range, kde_healthy(x_range), alpha=0.3, color='green')
        ax.fill_between(x_range, kde_unhealthy(x_range), alpha=0.3, color='red')

        ax.set_title(f'{direction}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
