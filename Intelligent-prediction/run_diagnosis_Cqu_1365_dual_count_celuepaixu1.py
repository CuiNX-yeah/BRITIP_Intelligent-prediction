# -*- coding: utf-8 -*-
# 智能诊断 + 策略推荐（输出中尽量不出现“风险”二字；匹配度仅用于内部排序、不显示）

import pandas as pd
import joblib
import pickle
from neo4j import GraphDatabase

# --- 1. 全局设置 ---
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "12345678")     # 请确保密码正确
RISK_THRESHOLD = 0.549           # 与流程图一致：>= 0.549 触发“核心应对策略”，否则给“预防性策略”


# --- 2. 加载因素-策略匹配评分（仅用于内部排序，输出不显示） ---
def load_factor_strategy_scores(filepath='96.csv'):
    """
    加载第五章的因素-策略匹配评分。
    评分只用于内部排序，输出中不显示具体分值。
    """
    try:
        df = pd.read_csv(filepath)
        scores_dict = {}
        for _, row in df.iterrows():
            factor_id = row['factor_id']
            strategy_id = row['strategy_id']
            score = row['matching_score']
            if factor_id not in scores_dict:
                scores_dict[factor_id] = {}
            scores_dict[factor_id][strategy_id] = float(score)
        print(f"成功加载 {len(df)} 条因素-策略匹配关系")
        return scores_dict
    except FileNotFoundError:
        print(f"提示：未找到因素-策略评分文件 {filepath}；将按默认顺序推荐策略（功能不受影响，仅无排序依据）")
        return {}


# --- 3. 获取项目的社群ID（兼容多种字段名） ---
def get_current_community_id(project_id, uri, auth):
    """获取项目的社群ID，兼容多种字段名"""
    query = """
    MATCH (p:Project {`项目ID`: $project_id})
    RETURN coalesce(p.communityId_v2, p.communityId, p.communityId_baseline) AS cid
    """
    driver = GraphDatabase.driver(uri, auth=auth)
    try:
        records, _, _ = driver.execute_query(query, project_id=project_id)
        if records:
            return records[0]["cid"]
        return None
    finally:
        driver.close()


# --- 4. 策略推荐（基于中断倾向阈值与内部排序；输出不显示分数，且尽量不出现“风险”二字） ---
def get_strategies_with_priority(project_id, disruption_probability, uri, auth, scores_dict):
    """
    基于中断倾向和评分排名推荐策略：
      - 概率 >= 阈值：Top-3 因素，每个因素取 2 条“核心应对策略”
      - 概率 <  阈值：Top-2 因素，每个因素取 1 条“预防性策略”
    评分仅用于排序，不在输出中显示。
    """
    driver = GraphDatabase.driver(uri, auth=auth)

    # 获取社群ID
    community_id = get_current_community_id(project_id, uri, auth)
    if not community_id:
        print(f"提示：项目 {project_id} 未找到社群ID")
        return {}, "未知"

    print(f"项目社群ID: {community_id}")

    # 控制推荐规模
    if disruption_probability >= RISK_THRESHOLD:
        n_factors = 3
        strategies_per_factor = 2
        strategy_type = "核心应对策略"
    else:
        n_factors = 2
        strategies_per_factor = 1
        strategy_type = "预防性策略"

    # 社群主要影响因素（兼容字符串和数字）
    find_factors_query = """
    MATCH (p:Project)
    WHERE toString(coalesce(p.communityId_v2, p.communityId, p.communityId_baseline)) = toString($cid)
    MATCH (p)-[:AFFECTED_BY_SUBFACTOR]->(sf:SubFactor)<-[:HAS_SUBFACTOR]-(f:Factor)
    WHERE f.`影响因素` IS NOT NULL 
      AND trim(coalesce(f.`影响因素`, '')) <> '' 
      AND f.`影响因素` <> '-'
    WITH f, count(DISTINCT p) AS projectCount
    ORDER BY projectCount DESC
    LIMIT $n_factors
    RETURN f.`影响因素` AS factor_name, 
           f.`影响因素ID` AS factor_id
    """

    # 因素对应策略（兼容不同关系名）
    find_strategies_query = """
    MATCH (f:Factor {`影响因素ID`: $factor_id})-[r]-(s:Strategy)
    WHERE type(r) IN ['LEADS_TO_STRATEGY', 'MOTIVATES']
      AND s.`策略` IS NOT NULL 
      AND trim(coalesce(s.`策略`, '')) <> ''
      AND s.`策略` <> '-'
    RETURN DISTINCT s.`策略` AS strategy_name, 
           s.`策略ID` AS strategy_id
    """

    recommendations = {'core': [], 'supplementary': []}

    try:
        # 获取主要因素
        records, _, _ = driver.execute_query(
            find_factors_query,
            cid=str(community_id),
            n_factors=n_factors
        )

        # 若按社群查不到，则退化为直接从该项目的记录中找 Top 因素
        if not records:
            print(f"社群 {community_id} 暂未识别出主要影响因素，改用项目自身关联因素")
            backup_query = """
            MATCH (p:Project {`项目ID`: $project_id})-[:AFFECTED_BY_SUBFACTOR]->(sf:SubFactor)<-[:HAS_SUBFACTOR]-(f:Factor)
            WHERE f.`影响因素` IS NOT NULL AND f.`影响因素` <> '-'
            WITH f, count(*) AS cnt
            ORDER BY cnt DESC
            LIMIT $n_factors
            RETURN f.`影响因素` AS factor_name, f.`影响因素ID` AS factor_id
            """
            records, _, _ = driver.execute_query(
                backup_query,
                project_id=project_id,
                n_factors=n_factors
            )

        top_factors = [(r["factor_name"], r["factor_id"]) for r in records]
        if top_factors:
            print(f"找到Top-{len(top_factors)}影响因素: {[f[0] for f in top_factors]}")

        # 对每个因素获取策略并按评分排序（不显示分值；输出中将“风险”替换为“不确定性”）
        for factor_name, factor_id in top_factors:
            s_records, _, _ = driver.execute_query(
                find_strategies_query, factor_id=factor_id
            )

            strategies = []
            for rec in s_records:
                s_name = rec["strategy_name"]
                s_id = rec["strategy_id"]
                score = scores_dict.get(factor_id, {}).get(s_id, 0.0)  # 仅排序用
                strategies.append({
                    'factor': factor_name,
                    'strategy': s_name,
                    'score': float(score)
                })

            # 评分高的优先
            strategies.sort(key=lambda x: x['score'], reverse=True)

            # 选择前 N 条作为核心策略，额外留出 2 条作为补充
            if strategies:
                recommendations['core'].extend(strategies[:strategies_per_factor])
                if len(strategies) > strategies_per_factor:
                    recommendations['supplementary'].extend(
                        strategies[strategies_per_factor:strategies_per_factor + 2]
                    )

    except Exception as e:
        print(f"查询错误: {e}")
    finally:
        driver.close()

    return recommendations, strategy_type


# --- 5. 主程序 ---
def main():
    """主诊断程序（输出中尽量不出现“风险”二字；匹配度不显示）"""
    print("正在加载模型、数据和因素-策略评分...")
    try:
        model = joblib.load('project_risk_model_dual_count.joblib')
        scaler = joblib.load('scaler_dual_count.joblib')
        with open('model_columns_dual_count.pkl', 'rb') as f:
            model_columns = pickle.load(f)

        X_test = pd.read_csv("X_test_dual_count.csv")
        y_test = pd.read_csv("y_test_dual_count.csv").squeeze("columns")
        df_full = pd.read_csv("project_data_dual_count_final.csv")

        scores_dict = load_factor_strategy_scores('96.csv')
        print("模型和数据加载完成！")

    except FileNotFoundError as e:
        print(f"错误：找不到必需的文件: {e.filename}")
        return

    # 指定要分析的项目
    target_project_id = "C1307"

    # 查找项目
    if target_project_id not in X_test['projectId'].values:
        print(f"错误：项目ID '{target_project_id}' 不在测试集中")
        return

    project_data = X_test[X_test['projectId'] == target_project_id]
    true_label = y_test.loc[project_data.index[0]]
    project_features = project_data.drop(columns=['projectId'])[model_columns]

    # 预测概率
    project_features_scaled = scaler.transform(project_features)
    probabilities = model.predict_proba(project_features_scaled)[0]
    disruption_probability = probabilities[1]

    # 生成诊断报告
    print("\n" + "=" * 20 + " 项目可持续性智能诊断报告 " + "=" * 20)
    original_info = df_full[df_full['projectId'] == target_project_id].iloc[0]

    print(f"项目编号: {original_info.get('projectId', 'N/A')}")
    print(f"项目信息: {original_info.get('hostCountryName', 'N/A')}, "
          f"{original_info.get('projectCategory', 'N/A')}, "
          f"{original_info.get('projectScale', 'N/A')}")
    print(f"参与方数量: {original_info.get('preciseParticipantCount', 'N/A')}")
    print(f"真实结果: {'发生中断' if true_label == 1 else '未发生中断'}")
    print("-" * 65)

    # —— 可持续性 / 中断倾向评估（不出现“风险”）——
    print("\n【可持续性评估】")
    tendency = "高中断倾向" if disruption_probability >= RISK_THRESHOLD else "低中断倾向"
    print(f"  预测结果：{tendency}")
    print(f"  中断概率：{disruption_probability:.2%}")
    print(f"  预测结果({RISK_THRESHOLD:.3f}阈值): {'建议重点防控' if disruption_probability >= RISK_THRESHOLD else '常规管理即可'}")

    # 策略推荐（按内部排序，不显示分值；文本中将“风险”替换为“不确定性”以保持一致表述）
    print("\n【策略推荐】")
    recommendations, strategy_type = get_strategies_with_priority(
        target_project_id, disruption_probability, URI, AUTH, scores_dict
    )

    def _sanitize_text(s: str) -> str:
        # 去换行 + 替换“风险”为“不确定性”
        return s.replace('\n', ' ').replace('风险', '不确定性').strip()

    if recommendations.get('core'):
        print(f"\n  {strategy_type}（必须执行）：")
        for i, item in enumerate(recommendations['core'], 1):
            strategy_clean = _sanitize_text(item['strategy'])
            factor_clean = _sanitize_text(item['factor'])
            print(f"    {i}. [{factor_clean}] {strategy_clean}")
            # 不显示匹配度

    if recommendations.get('supplementary'):
        print(f"\n  补充策略（建议参考）：")
        for i, item in enumerate(recommendations['supplementary'], 1):
            strategy_clean = _sanitize_text(item['strategy'])
            factor_clean = _sanitize_text(item['factor'])
            print(f"    {i}. [{factor_clean}] {strategy_clean}")
            # 不显示匹配度

    if (not recommendations.get('core')) and (not recommendations.get('supplementary')):
        print("  未能从知识图谱中找到相关策略，建议咨询专家。")

    # —— 决策建议（不出现“风险”）——
    print("\n【决策建议】")
    if disruption_probability >= RISK_THRESHOLD:
        print("  该项目存在较高中断倾向，建议：")
        if recommendations.get('core'):
            print("  1. 立即实施上述核心策略")
            print("  2. 建立中断监测与预警机制")
            print("  3. 准备应急预案并开展演练")
        else:
            print("  1. 邀请专家制定针对性方案")
            print("  2. 加强中断监测与动态评估")
            print("  3. 准备应急预案并开展演练")
    else:
        print("  该项目中断倾向较低，建议：")
        print("  1. 实施预防性策略")
        print("  2. 保持常规监测")
        print("  3. 定期复评项目可持续性")

    print("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n按 Enter 键退出...")
