#================================匯入函數===============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import joblib
from sklearn.model_selection import StratifiedKFold
saving_file = "regular_season/MLP_image"#檔案儲存名稱
#======================================================================

#================================取得csv檔===============================================================================
df = pd.read_csv(r"D:\AI_prediction\python_program\program1\NBA_2021_to_2024_regular_season.csv", encoding="utf-8-sig")
df = np.round(df, 3) # 改變資料只到小數第三位
#=======================================================================================================================

#================================特徵工程取得===================================================================================
def prepare_features_labels(df):

    X_list = []
    y_list = []

    for i in range(0, 7380, 2):
        # 選擇第 i 行，然後從該行中選取 'fg' 欄位的值
        home_count = i #主隊的計數
        away_count = i + 1 #客隊的計數

#============================================主隊特徵=================================   
        home_2FGM_np       = np.array([df.iloc[home_count]['fg2']])
        home_3FGM_np       = np.array([df.iloc[home_count]['fg3']])
        home_FTM_np        = np.array([df.iloc[home_count]['ft']])
        home_2FGA_np       = np.array([df.iloc[home_count]['fga2']])
        home_3FGA_np       = np.array([df.iloc[home_count]['fg3a']])
        home_FTA_np        = np.array([df.iloc[home_count]['fta']])
        home_OReb_np       = np.array([df.iloc[home_count]['orb']])
        home_Assists_np    = np.array([df.iloc[home_count]['ast']])
        home_Fouls_Rv_np   = np.array([df.iloc[away_count]['pf']])  # 對手犯規
        home_Turnovers_np  = np.array([df.iloc[home_count]['tov']])
        home_Blocks_Ag_np  = np.array([df.iloc[away_count]['blk']]) # 被蓋
        home_DReb_np       = np.array([df.iloc[home_count]['drb']])
        home_Steals_np     = np.array([df.iloc[home_count]['stl']])
        home_Blocks_Fv_np  = np.array([df.iloc[home_count]['blk']]) # 自己蓋
        home_Fouls_Com_np  = np.array([df.iloc[home_count]['pf']])  # 自己犯規
        home_Result_np     = np.array([df.iloc[home_count]['result']])  # 主隊的輸贏值(0:輸 、 1:贏)
#=====================================================================================================     

#===========================================客隊特徵================================= 
        away_2FGM_np      = np.array([df.iloc[away_count]['fg2']])
        away_3FGM_np      = np.array([df.iloc[away_count]['fg3']])
        away_FTM_np       = np.array([df.iloc[away_count]['ft']])
        away_2FGA_np      = np.array([df.iloc[away_count]['fga2']])
        away_3FGA_np      = np.array([df.iloc[away_count]['fg3a']])
        away_Turnovers_np = np.array([df.iloc[away_count]['tov']])
        away_FTA_np       = np.array([df.iloc[away_count]['fta']])
        away_OReb_np      = np.array([df.iloc[away_count]['orb']])
        away_DReb_np      = np.array([df.iloc[away_count]['drb']])
#=========================================================================================================      

#==========================================特徵工程計算================================= 
        #進攻指標
        home_i_offense = home_2FGM_np + home_3FGM_np + home_FTM_np - (home_2FGA_np - home_2FGM_np + home_3FGA_np - home_3FGM_np + home_FTA_np - home_FTM_np) + home_OReb_np + home_Assists_np + home_Fouls_Rv_np - home_Turnovers_np - home_Blocks_Ag_np 
        
        #防守指標
        home_i_defense = -(away_2FGM_np + away_3FGM_np + away_FTM_np) + (away_2FGA_np - away_2FGM_np + away_3FGA_np - away_3FGM_np + away_FTA_np - away_FTM_np) + home_DReb_np + home_Steals_np + home_Blocks_Fv_np - away_OReb_np - home_Fouls_Com_np 
        
        #有效命中率
        home_offense_efgp = (home_2FGM_np + home_3FGM_np + (0.5 * home_3FGM_np))/(home_2FGA_np + home_3FGA_np) 
        
        #失誤率
        home_offense_tp = home_Turnovers_np / (home_2FGA_np + home_3FGA_np + (0.44 * home_FTA_np) - home_OReb_np + home_Turnovers_np)
        
        #進攻籃板率
        home_offense_orp = (home_OReb_np) / (home_OReb_np + away_DReb_np)

        #罰球率
        home_offense_ftr = (home_FTA_np) / (home_2FGA_np + home_3FGA_np)

        #允許有效命中率
        home_defense_efgp = (away_2FGM_np + away_3FGM_np + (0.5 * away_3FGM_np)) / (away_2FGA_np + away_3FGA_np) 
        
        #迫使失誤率
        home_defense_tp = away_Turnovers_np / (away_2FGA_np + away_3FGA_np + (0.44 * away_FTA_np) - away_OReb_np + away_Turnovers_np) 
        
        #允許進攻籃板率
        home_defense_orp = (away_OReb_np) / (away_OReb_np + home_DReb_np) 
        
        #允許罰球率
        home_defense_ftr = (away_FTA_np) / (away_2FGA_np + away_3FGA_np) 
#=========================================================================================================        

#=====================整合成一場比賽的特徵值(features) & 加入陣列==========
        features = np.array([
        home_i_offense,
        home_i_defense,
        home_offense_efgp,
        home_offense_tp,
        home_offense_orp,
        home_offense_ftr,
        home_defense_efgp,
        home_defense_tp,
        home_defense_orp,
        home_defense_ftr
        ]).astype(float).flatten()
#=====================================================================    

#====================把每場比賽加入陣列中================================
        X_list.append(features)
        y_list.append(int(home_Result_np[0]))   # 主隊結果（0 或 1）   
#=====================================================================    

#====================把numpy陣列轉換成真正二維陣列========================        
    X = np.array(X_list)  # shape = (樣本數, 特徵數)
    y = np.array(y_list)  # shape = (樣本數,)
#=====================================================================    
    
    return X, y
#=====================================================================================================================================

#=========================資料標準化==================================
X, y = prepare_features_labels(df) #取得特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#====================================================================

#=========================資料分割========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.12, # 88% train, 12% test
    random_state=42, # 確保結果可重現
    shuffle=True, 
    stratify=y
)

print(f"訓練集大小: {len(X_train)} 個樣本")
print(f"測試集大小: {len(X_test)} 個樣本")
#========================================================================

#=========================模型訓練========================================
mlp_model = MLPClassifier(
    hidden_layer_sizes=(10, 5),  
    activation='relu',           
    solver='adam',               
    max_iter=500,                
    random_state=42,
    early_stopping=True,         
    n_iter_no_change=20          
)

print("\n開始訓練 MLP 模型...")
# 訓練 MLP 模型
mlp_model.fit(X_train, y_train)
print("MLP 模型訓練完成。")

# 評估 MLP 模型
y_pred_mlp = mlp_model.predict(X_test)
# 取得預測機率，用於計算 AUC-ROC
y_pred_proba_mlp = mlp_model.predict_proba(X_test)[:, 1]

# 計算指標
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
auc_roc_mlp = roc_auc_score(y_test, y_pred_proba_mlp)

print("\n--- MLP 模型在測試集上的性能 ---")
print(f"準確率 (Accuracy): {accuracy_mlp:.4f}")
print(f"AUC-ROC: {auc_roc_mlp:.4f}")

print("\n混淆矩陣 (Confusion Matrix):") 
cm = confusion_matrix(y_test, y_pred_mlp)
print(cm)

print("\n分類報告 (Classification Report):")
print(classification_report(y_test, y_pred_mlp,digits=4))
#========================================================================

#================================5-fold cross-validation======================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

recall_1_list = []
precision_1_list = []
f1_1_list = []
accuracy_list = []

for train_idx, test_idx in kf.split(X, y):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    # 訓練模型
    mlp_model.fit(X_train_fold, y_train_fold)

    # 預測
    y_pred_fold = mlp_model.predict(X_test_fold)

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

#================================混淆矩陣圖(88% train)======================
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

#===========================5-fold 準確率圖(類別1的Recall)==================
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

#================================== SHAP for MLP (KernelExplainer) =====================================

#=============================使用資料===========================
# 1. 限制背景資料（通常 50～100 就很夠)(訓練資料)
background_size = 100
background_X = X_train[:background_size]

# 2. 限制 SHAP 計算樣本（避免太慢)(測試資料)
shap_sample_size = 200
shap_sample_X = X_test[:shap_sample_size]

# background_X = X_train  # 不限制大小
# shap_sample_X = X_test  # 不限制大小
#================================================================

# 2. 建立 KernelExplainer
explainer = shap.KernelExplainer(
    lambda data: mlp_model.predict_proba(data)[:, 1],
    background_X
)

# 3. 計算 SHAP 值
shap_values = explainer.shap_values(shap_sample_X)

# 4. 特徵名稱
feature_names = [
    "i_offense", "i_defense",
    "off_efgp", "off_tp", "off_orp", "off_ftr",
    "def_efgp", "def_tp", "def_orp", "def_ftr"
]

# 5. 畫圖
plt.figure()
shap.summary_plot(shap_values, shap_sample_X, feature_names=feature_names, plot_type="bar", show=False)
plt.savefig(os.path.join(saving_file, "mlp_shap_summary_bar.png"), dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
shap.summary_plot(shap_values, shap_sample_X, feature_names=feature_names, show=False)
plt.savefig(os.path.join(saving_file, "mlp_shap_summary_dot.png"), dpi=150, bbox_inches='tight')
plt.close()
#=========================================================================================================





