# 导入需要的库
import pandas as pd
from neo4j import GraphDatabase

# --- Neo4j连接信息 ---
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "12345678")
# --------------------

def get_data_from_neo4j_final(uri, auth):
    """
    连接到Neo4j，执行最终版的、带有双重参与方计数逻辑的查询，并返回DataFrame。
    该查询使用全角'｜'作为分隔符，并已兼容'参与方名称'为列表的情况。
    """
    driver = GraphDatabase.driver(uri, auth=auth)
    
    # 【最终版Cypher查询 - 已修正列表处理逻辑】
    query = """
    MATCH (p:Project)
    WHERE p.communityId IS NOT NULL 
      AND p.degree IS NOT NULL 
      AND p.pageRank IS NOT NULL
      AND p.结果 IS NOT NULL
    
    OPTIONAL MATCH (p)-[:LOCATED_IN]->(c:Country)
    OPTIONAL MATCH (participant:Participant)-[r]->(p)
    WHERE type(r) IN ['FUNDS', 'IMPLEMENTS', 'RECEIVES_FUNDING_FOR']

    // 步骤1: 针对每个项目，收集其所有核心参与方节点
    WITH p, c, collect(participant) as participants

    // 步骤2: 在聚合阶段，同时进行两种计数
    WITH p, c, participants,
        // 【精确计数 - 已修正】使用嵌套的REDUCE函数来处理'参与方名称'为列表的情况
        // 外层REDUCE遍历每个参与方节点(part)
        // 内层REDUCE遍历该节点'参与方名称'列表中的每个名字(name)，对其进行split并累加
        REDUCE(total = 0, part IN participants | total + 
            REDUCE(innerTotal = 0, name IN part.`参与方名称` | innerTotal + size(split(name, '｜')))
        ) as preciseParticipantCount,

        // 【节点计数】计算各类参与方节点的数量
        size([part IN participants WHERE part.`参与方类型` = 'T2-中国机构 + 东道国机构']) AS count_T2,
        size([part IN participants WHERE part.`参与方类型` = 'T6-其他国家机构']) AS count_T6,
        size([part IN participants WHERE part.`参与方类型` = 'S4-东道国公司']) AS count_S4,
        size([part IN participants WHERE part.`参与方类型` = 'T5-中国机构+其他国家机构']) AS count_T5,
        size([part IN participants WHERE part.`参与方类型` = 'T1-中国机构']) AS count_T1,
        size([part IN participants WHERE part.`参与方类型` = 'S2-其他合资企业/特殊目的载体']) AS count_S2,
        size([part IN participants WHERE part.`参与方类型` = 'T3-东道国机构']) AS count_T3,
        size([part IN participants WHERE part.`参与方类型` = 'T7-东道国机构+其他国家机构']) AS count_T7,
        size([part IN participants WHERE part.`参与方类型` = 'S3-东道主合资企业/特殊目的载体']) AS count_S3,
        size([part IN participants WHERE part.`参与方类型` = 'R7-东道国企业']) AS count_R7,
        size([part IN participants WHERE part.`参与方类型` = 'T4-中国机构+东道国机构+其他国家机构']) AS count_T4,
        size([part IN participants WHERE part.`参与方类型` = 'S1-东道国政府机构']) AS count_S1,
        size([part IN participants WHERE part.`参与方类型` = 'R6-其他国外企业']) AS count_R6,
        size([part IN participants WHERE part.`参与方类型` = 'R11-东道国政府机构']) AS count_R11,
        size([part IN participants WHERE part.`参与方类型` = 'S5-中国公司']) AS count_S5,
        size([part IN participants WHERE part.`参与方类型` = 'R4-中国企业']) AS count_R4,
        size([part IN participants WHERE part.`参与方类型` = 'R2-中国政府机构']) AS count_R2,
        size([part IN participants WHERE part.`参与方类型` = 'S6-外国合资企业/特殊目的载体']) AS count_S6,
        size([part IN participants WHERE part.`参与方类型` = 'R3-中国商业银行']) AS count_R3,
        size([part IN participants WHERE part.`参与方类型` = 'R1-中国政策性银行']) AS count_R1,
        size([part IN participants WHERE part.`参与方类型` = 'R12-中国政策性银行|中国政府机构']) AS count_R12,
        size([part IN participants WHERE part.`参与方类型` = 'S7-东道国银行']) AS count_S7,
        size([part IN participants WHERE part.`参与方类型` = 'S11-东道国政府|东道国公司']) AS count_S11,
        size([part IN participants WHERE part.`参与方类型` = 'R5-国际银行']) AS count_R5,
        size([part IN participants WHERE part.`参与方类型` = 'S15-受援方合资企业/特殊目的载体']) AS count_S15,
        size([part IN participants WHERE part.`参与方类型` = 'S14-其他私营部门|其他合资企业/特殊目的公司']) AS count_S14,
        size([part IN participants WHERE part.`参与方类型` = 'S9-东道国银行|东道国基金']) AS count_S9,
        size([part IN participants WHERE part.`参与方类型` = '未知机构']) AS count_unknown,
        size([part IN participants WHERE part.`参与方类型` = 'S13-东道国政府机构|东道国有企业|东道国有银行']) AS count_S13,
        size([part IN participants WHERE part.`参与方类型` = 'S12-东道国政府|其他国际机构']) AS count_S12,
        size([part IN participants WHERE part.`参与方类型` = 'S8-外国合资企业/特殊目的载体｜东道国政府机构']) AS count_S8,
        size([part IN participants WHERE part.`参与方类型` = 'S10-中国银行']) AS count_S10,
        size([part IN participants WHERE part.`参与方类型` = 'R9-中国政策性银行|中国商业银行']) AS count_R9,
        size([part IN participants WHERE part.`参与方类型` = 'R8-中国政策性银行|东道国政府机构']) AS count_R8

    RETURN
      p.项目ID AS projectId,
      p.communityId AS communityId,
      p.degree AS degree,
      p.pageRank AS pageRank,
      size(participants) as totalCoreParticipantNodeCount, // 参与方节点总数
      preciseParticipantCount, // 精确的参与方总数
      count_T2, count_T6, count_S4, count_T5, count_T1, count_S2, count_T3, count_T7, count_S3, count_R7,
      count_T4, count_S1, count_R6, count_R11, count_S5, count_R4, count_R2, count_S6, count_R3, count_R1,
      count_R12, count_S7, count_S11, count_R5, count_S15, count_S14, count_S9, count_unknown, count_S13,
      count_S12, count_S8, count_S10, count_R9, count_R8,
      p.项目类别 AS projectCategory,
      p.项目规模 AS projectScale,
      c.`项目东道国所属地区` AS projectRegion,
      c.`项目东道国名称` AS hostCountryName,
      CASE WHEN TRIM(p.结果) IN ['取消', '中断，三个月以内', '中断，三个月以上'] THEN 1 ELSE 0 END AS label
    """
    
    with driver.session() as session:
        result = session.run(query)
        df = pd.DataFrame([r.data() for r in result], columns=result.keys())
    
    driver.close()
    return df

if __name__ == "__main__":
    print("正在从Neo4j数据库导出特征数据 (使用最终版双重参与方计数)...")
    project_df_final = get_data_from_neo4j_final(URI, AUTH)
    
    output_filename = "project_data_dual_count_final.csv"
    project_df_final.to_csv(output_filename, index=False)
    
    print("-" * 30)
    print(f"数据导出成功！共导出 {len(project_df_final)} 条项目数据。")
    print(f"文件已保存为: {output_filename}")
    print("导出的数据前5行预览:")
    print(project_df_final.head())