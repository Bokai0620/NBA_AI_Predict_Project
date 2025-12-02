#================================匯入函數===============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_curve, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import joblib
from sklearn.model_selection import StratifiedKFold
saving_file = "regular_season/XGBoost_image"#檔案儲存名稱
#======================================================================
df = pd.read_csv(r"D:\AI_prediction\python_program\program1\NBA_2021_to_2024_regular_season.csv", encoding="utf-8-sig")
df = np.round(df, 3) # 改變資料只到小數第三位

X = df.drop(columns=["result"], axis=1)  # 特徵欄位，axis=1(欄位)刪掉result欄位。
y = df["result"]                 # 標籤欄位

#================================分割資料為訓練集和測試集===============================
# stratify=y按照y的分布來切分資料，保持訓練和測試資料的勝負比例相同。
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) 
#===================================================================================

#================================建立模型===============================
# 建立基本模型
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)

# 定義要嘗試的參數組合
param_grid = {
    'n_estimators': [100, 200],    # 樹的數量
    'max_depth': [3, 5, 7],        # 樹深度(層數)
    'learning_rate': [0.01, 0.1],  # 學習率
    'subsample': [0.8, 1.0],       #決定每棵樹訓練時使用的樣本比例(樣本抽樣。例:勇士隊...)
    'colsample_bytree': [0.8, 1.0] #控制每棵樹訓練時隨機選取特徵的比例。(特徵抽樣。例:投球數、失誤...)
}

# 建立 GridSearchCV 物件
grid_search = GridSearchCV(
    estimator=xgb_model,      # 要調的模型
    param_grid=param_grid,    # 要嘗試的參數組合
    cv=5,                     # 使用 5 折交叉驗證
    scoring='accuracy',       # 用準確率評估
    n_jobs=-1,                # 用所有 CPU 核心加速(因為多核心需跑多個資料所以需要temp暫存資料夾來存東西，但原本的temp因為在中文路徑下會出錯，所以指定新的暫存資料夾路徑給temp_folder)
    verbose=2                 # 顯示進度
)

# 開始搜尋最佳參數
temp_folder = r"D:\temp_joblib"  # 全英文路徑
with joblib.parallel_backend('loky', temp_folder=temp_folder):
    grid_search.fit(X_train, y_train)

# 顯示最佳結果
print("最佳參數組合：", grid_search.best_params_)

# 取出最佳參數
best_params = grid_search.best_params_

xgb_best_model = XGBClassifier(
    objective='binary:logistic',  # 二元分類
    eval_metric='logloss',        # 損失函數           
    random_state=42,              # 隨機性種子
    **best_params                 # 把最佳參數傳入
)
#======================================================================

#================================訓練模型===============================
xgb_best_model.fit(X_train, y_train)
#======================================================================

#================================印出資訊======================================
# 預測
y_pred = xgb_best_model.predict(X_test)

# 準確率
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred) # 計算混淆矩陣
print("Confusion Matrix:\n", cm)

# 詳細分類報告
print(classification_report(y_test, y_pred, digits=4))
#=============================================================================

#================================5-fold cross-validation======================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

recall_1_list = []
precision_1_list = []
f1_1_list = []
accuracy_list = []
X_np = X.to_numpy()
y_np = y.to_numpy()

for train_idx, test_idx in kf.split(X_np, y_np):
    X_train_fold, X_test_fold = X_np[train_idx], X_np[test_idx]
    y_train_fold, y_test_fold = y_np[train_idx], y_np[test_idx]

    # 訓練模型
    xgb_best_model.fit(X_train_fold, y_train_fold)

    # 預測
    y_pred_fold = xgb_best_model.predict(X_test_fold)

    # 混淆矩陣
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    TN, FP, FN, TP = cm.ravel()

    # Precision、Recall for class 1
    recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0

    recall_1_list.append(recall_1)
    precision_1_list.append(precision_1)

    # F1-score
    if (precision_1 + recall_1) > 0:
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
    else:
        f1_1 = 0
    f1_1_list.append(f1_1)

    # Accuracy（整體準確率）
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy_list.append(accuracy)

print("每一 fold 的 Accuracy:", accuracy_list)
print("平均 Accuracy:", np.mean(accuracy_list))

print("\n每一 fold 的 Recall (class 1):", recall_1_list)
print("平均 Recall_1:", np.mean(recall_1_list))

print("\n每一 fold 的 Precision (class 1):", precision_1_list)
print("平均 Precision_1:", np.mean(precision_1_list))

print("\n每一 fold 的 F1-score (class 1):", f1_1_list)
print("平均 F1-score_1:", np.mean(f1_1_list))
#=============================================================================



#================================混淆矩陣圖======================
plt.figure(figsize=(6,4)) # 寬6吋 、 高4吋
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues") # sns.heatmap()畫熱力圖用 、 annot=True(每個格子中是否顯示數據) 、 fmt="d"(格子中的數字顯示整數) 、 cmap="Blues"(顯示圖為藍色主題)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
os.makedirs(saving_file, exist_ok=True)  # 若資料夾不存在會自動建立
plt.savefig(os.path.join(saving_file, "confusion_matrix.png"), dpi=150)
plt.close()
print("Confusion matrix saved as confusion_matrix.png\n")
#==============================================================

#===========================5-fold 準確率圖==================
plt.figure(figsize=(6,4))
plt.bar(range(1,6), recall_1_list) #plt.bar()畫長條圖用 、 range(1,6)這個是X軸座標 、 scores這個是y軸座標(5次的準確率)
plt.title("5-Fold Cross-Validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0,1)      # 設定 y 軸的範圍（上下限）
plt.tight_layout() # 自動調整圖表的空間配置
os.makedirs(saving_file, exist_ok=True)  # 若資料夾不存在會自動建立
plt.savefig(os.path.join(saving_file, "cross_val_accuracy.png"), dpi=150)
plt.close()
print("Cross-validation plot saved as cross_val_accuracy.png\n")
#=========================================================

#==================================ROC Curve圖=====================================
y_prob = xgb_best_model.predict_proba(X_test)[:, 1] # y_prob模型預測贏的機率

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.4f})")
plt.plot([0,1], [0,1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend() # 顯示圖例
plt.tight_layout() # 自動調整圖表的空間配置
os.makedirs(saving_file, exist_ok=True)  # 若資料夾不存在會自動建立
plt.savefig(os.path.join(saving_file, "ROC_curve.png"), dpi=150)
plt.close()
print("ROC_curve plot saved as ROC_curve.png\n")
#==============================================================================

#==================================特徵重要度圖=====================================
importance = xgb_best_model.feature_importances_ # 取得每個特徵的重要度分數（0~1之間）
features = X.columns # 取得特徵名稱

# 排序後畫圖
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(len(features)), importance[indices]) # range(len(features))特徵數量 、 importance[indices]按照排序後的重要度索引取出數值畫出長條圖。
plt.xticks(range(len(features)), features[indices], rotation=90) # range(len(features))特徵數量 、 features[indices]由特徵權重大到小排序到X軸(特徵名稱) 、 rotation=90把文字旋轉 90 度（直立顯示）
plt.title("XGBoost Feature Importance")
plt.tight_layout()
os.makedirs(saving_file, exist_ok=True)  # 若資料夾不存在會自動建立
plt.savefig(os.path.join(saving_file, "Feature_importance.png"), dpi=150)
plt.close()
print("Feature_importance plot saved as Feature_importance.png\n")
#=================================================================================

#==================================SHAP解釋=====================================
explainer = shap.TreeExplainer(xgb_best_model)
shap_values = explainer.shap_values(X_test)

# ----------------------------
# 1. 全域特徵重要度 (bar plot)。 Bar plot 是 取平均絕對值，所以只顯示「影響力大小」，不會顯示增加或降低勝率。
#假設特徵 𝑗 的 5 個樣本 SHAP 值如下：
#ϕj​=[0.2,−0.3,0.1,−0.1,0.5] 相同特徵的SHAP值，∣0.2∣ + ∣-0.3∣ + ∣0.1∣ + ∣-0.1∣ + ∣0.5∣​ / 5 = 0.24
# ----------------------------
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)  # show=False 不直接顯示
os.makedirs(saving_file, exist_ok=True)  # 若資料夾不存在會自動建立
plt.savefig(os.path.join(saving_file, "shap_summary_bar.png"), dpi=150, bbox_inches='tight')
plt.close()  # 釋放圖形資源
print("shap_summary_bar plot saved as shap_summary_bar.png\n")

# ----------------------------
# 2. 詳細 SHAP 分佈 (dot plot)
# ----------------------------
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
os.makedirs(saving_file, exist_ok=True)  # 若資料夾不存在會自動建立
plt.savefig(os.path.join(saving_file, "shap_summary_dot.png"), dpi=150, bbox_inches='tight')
plt.close()  # 釋放圖形資源
print("shap_summary_dot plot saved as shap_summary_dot.png\n")
#===============================================================================