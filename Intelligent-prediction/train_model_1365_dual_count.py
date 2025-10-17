# 导入需要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import numpy as np

# 主程序
if __name__ == "__main__":
    # 1. 加载数据
    # 加载包含双重参与方计数的最终版数据文件
    input_filename = "project_data_dual_count_final.csv"
    print(f"正在加载最终版数据: {input_filename}")
    df = pd.read_csv(input_filename)
    
    # 2. 数据预处理
    print("正在进行数据预处理...")

    if 'degree' in df.columns:
        df = df.drop(columns=['degree'])
        print("'degree'特征已成功移除。")

    # --- 独热编码流程 ---
    # 使用 pd.get_dummies 对所有类别特征进行独热编码
    categorical_features = ['projectCategory', 'projectScale', 'projectRegion', 'hostCountryName']
    print(f"正在对以下类别特征进行独热编码: {categorical_features}")
    df_processed = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=int)
    
    # 分离特征 (X) 和目标 (y)
    X = df_processed.drop(columns=['projectId', 'label'])
    y = df_processed['label']
    
    print("所有特征列已成功转换为数值型。")

    # 3.【已修改】采用标准的80/20随机划分方式
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"数据集划分完成：训练集 {len(X_train)} 条, 测试集 {len(X_test)} 条。")

    # --- 导出文件 ---
    print("\n正在导出训练集和测试集 (包含projectId)...")
    
    # 导出训练集
    X_train_export = X_train.copy()
    X_train_export['projectId'] = df_processed.loc[X_train.index, 'projectId']
    X_train_export.to_csv('X_train_dual_count.csv', index=False)
    y_train.to_csv('y_train_dual_count.csv', index=False, header=True)
    
    # 导出测试集
    X_test_export = X_test.copy()
    X_test_export['projectId'] = df_processed.loc[X_test.index, 'projectId']
    X_test_export.to_csv('X_test_dual_count.csv', index=False)
    y_test.to_csv('y_test_dual_count.csv', index=False, header=True)
    
    print("训练集和测试集已成功导出。")
    # --- 导出完成 ---

    # --- 特征缩放 ---
    # X_train 和 X_test 已不包含 projectId，可直接用于缩放
    print("\n正在进行特征缩放...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("特征缩放完成。")

    # --- 模型定义 ---
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }

    best_model = None
    best_model_name = ""
    best_accuracy = 0.0

    # 8. 依次训练、评估并找出最佳模型
    for name, model in models.items():
        print("\n" + "="*30)
        print(f"开始训练模型: {name}")
        print("="*30)
        
        model.fit(X_train_scaled, y_train)
        print("模型训练完成！")
        
        print("\n--- 模型性能评估 ---")
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  -> 准确率 (Accuracy): {accuracy:.4f}")
        print("\n  -> 分类报告 (Classification Report):")
        print(classification_report(y_test, y_pred))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    # 9. 保存最佳模型及其相关文件
    print("\n" + "="*30)
    print(f"所有模型评估完成。最佳模型是: {best_model_name} (准确率: {best_accuracy:.4f})")
    
    if best_model:
        joblib.dump(best_model, 'project_risk_model_dual_count.joblib')
        print("最佳模型已成功保存为: project_risk_model_dual_count.joblib")
        
        with open('model_columns_dual_count.pkl', 'wb') as f:
            pickle.dump(X_train.columns.tolist(), f)
        print("模型特征列表已成功保存为: model_columns_dual_count.pkl")
        
        joblib.dump(scaler, 'scaler_dual_count.joblib')
        print("特征缩放器已成功保存为: scaler_dual_count.joblib")
    
    print("="*30)