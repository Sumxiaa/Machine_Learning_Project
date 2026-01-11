# coach_effect.py
#
# 目的: 基于固定效应模型的残差，为指定国家识别最具潜力的投资项目，
#       并估算“伟大教练”可能带来的奖牌增量。

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("--- 1. 加载上一阶段的分析结果 ---")

try:
    # 读取 analysis.py 保存的、包含残差的结果文件
    panel_df = pd.read_csv('analysis_results_with_residuals.csv')
    print("结果文件 'analysis_results_with_residuals.csv' 加载成功。")
except FileNotFoundError:
    print("错误: 未找到 'analysis_results_with_residuals.csv'。")
    print("请先运行更新版的 analysis.py 以生成该文件。")
    exit()

print("\n--- 2. 定义分析参数与计算核心指标 ---")

# a. 定义我们关心的国家和分析的时间范围
TARGET_COUNTRIES = ['CHN', 'USA', 'GBR']
ANALYSIS_START_YEAR = 2000 # 我们更关心近现代奥运的表现

# b. 筛选分析所需的数据
df_recent = panel_df[panel_df['Year'] >= ANALYSIS_START_YEAR].copy()

# c. 计算每个项目的平均小项数 (项目重要性)
df_avg_events = df_recent.groupby('Sport')['Num_Events'].mean().reset_index()
df_avg_events.rename(columns={'Num_Events': 'Avg_Num_Events'}, inplace=True)

# d. 计算每个“国家-项目”的平均表现差距 (G_c,s)
df_gaps = df_recent.groupby(['NOC', 'Sport'])['Residuals'].mean().reset_index()
df_gaps.rename(columns={'Residuals': 'Performance_Gap'}, inplace=True)

# e. 合并数据，为计算CIS分数做准备
df_analysis = pd.merge(df_gaps, df_avg_events, on='Sport')

# f. 只筛选表现低于预期的项目 (潜力项目)
df_potential = df_analysis[df_analysis['Performance_Gap'] < 0].copy()

# g. 计算“教练投资优先级分数”(CIS) 和 预估影响
# CIS = -表现差距 * 项目重要性
df_potential['CIS'] = -df_potential['Performance_Gap'] * df_potential['Avg_Num_Events']
# 预估影响 = 表现差距的绝对值
df_potential['Estimated_Impact_per_Olympics'] = abs(df_potential['Performance_Gap'])

print("核心指标计算完成。")


print("\n--- 3. 为指定国家生成投资建议报告 ---")

def generate_report(country_noc, data, n=5):
    """为单个国家生成报告并可视化"""
    country_report = data[data['NOC'] == country_noc].sort_values(by='CIS', ascending=False).head(n)
    
    if country_report.empty:
        print(f"\n--- {country_noc} 投资建议报告 ---")
        print("未找到表现低于预期的潜力项目。该国在所有项目上均表现良好或符合预期。")
        return

    print(f"\n--- {country_noc} “伟大教练”投资建议报告 (Top {n}) ---")
    print("以下项目表现显著低于预期，具有最大的提升潜力。")
    print(country_report[[
        'Sport', 'CIS', 'Estimated_Impact_per_Olympics', 'Performance_Gap'
    ]].to_string(formatters={
        'CIS': '{:.2f}'.format,
        'Estimated_Impact_per_Olympics': '{:.2f} medals'.format,
        'Performance_Gap': '{:.2f}'.format
    }))

    # 可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(x='CIS', y='Sport', data=country_report, palette='magma')
    plt.title(f'Top {n} Coaching Investment Priorities for {country_noc}', fontsize=16)
    plt.xlabel('Coaching Investment Score (Higher is Better)')
    plt.ylabel('Sport')
    plt.tight_layout()
    plt.show()

# 为每个目标国家生成报告
for country in TARGET_COUNTRIES:
    generate_report(country, df_potential)

print("\n分析脚本运行结束。")