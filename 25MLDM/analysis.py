import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

print("--- 1. 加载真实数据文件 ---")

try:
    df_programs = pd.read_csv('summerOly_programs.csv', encoding='latin1')
    df_athletes = pd.read_csv('summerOly_athletes.csv', encoding='latin1')
except FileNotFoundError:
    print("错误: 请确保 'summerOly_programs.csv' 和 'summerOly_athletes.csv' 文件在此脚本的同一目录下。")
    exit()

print("数据加载完成。")


print("\n--- 2. 数据准备与面板数据构建 ---")

# a. 计算因变量: 各国各项目每年的奖牌数
df_medal_counts_panel = df_athletes[df_athletes['Medal'].notna()].groupby(['Year', 'NOC', 'Sport']).size().reset_index(name='Medal_Count')
print("面板数据 (因变量) 构建完成。")

# b. 准备自变量: 各项目每年的小项数
print("\n正在动态识别所有年份并转换赛事项目数据...")
year_columns = [col for col in df_programs.columns if col.isdigit() and len(col) == 4]
print(f"已自动识别到 {len(year_columns)} 个年份列, 从 {year_columns[0]} 到 {year_columns[-1]}")
df_programs_long = pd.melt(
    df_programs,
    id_vars=['Sport'],
    value_vars=year_columns,
    var_name='Year',
    value_name='Num_Events'
)
df_programs_long['Year'] = pd.to_numeric(df_programs_long['Year'])
df_programs_long['Num_Events'] = pd.to_numeric(df_programs_long['Num_Events'], errors='coerce').fillna(0).astype(int)
print("赛事项目数据已成功转换为长表格式。")

# c. 合并为最终的面板数据集
panel_df = pd.merge(
    df_medal_counts_panel,
    df_programs_long,
    on=['Year', 'Sport'],
    how='left'
)
panel_df['Num_Events'] = panel_df['Num_Events'].fillna(0)
# 筛选，只保留有赛事项目的记录
panel_df = panel_df[panel_df['Num_Events'] > 0].copy()
print("\n最终面板数据集构建完成。")


print("\n--- 3. 固定效应模型实现 ---")
formula = "Medal_Count ~ Num_Events + C(NOC) + C(Sport) + C(Year)"
model = smf.ols(formula, data=panel_df).fit()
# 使用对异方差稳健的标准误，使结果更可靠
robust_model = model.get_robustcov_results(cov_type='HC1')
print("模型拟合完成。")


print("\n--- 4. 结果解读与分析 ---")
print("\n固定效应模型回归结果摘要 (使用稳健标准误):")
print(robust_model.summary())


# 计算残差用于后续分析
panel_df['Residuals'] = robust_model.resid

# --- 5. 新增可视化模块 ---
print("\n--- 5. 生成可视化图表 ---")

def plot_strongest_sports(country_noc, panel_df_with_residuals, n=10):
    """
    为指定国家绘制其表现最好(超出预期)和最差(低于预期)的运动项目。
    """
    country_df = panel_df_with_residuals[panel_df_with_residuals['NOC'] == country_noc].copy()
    if country_df.empty:
        print(f"错误: 在数据中未找到国家代码 {country_noc}。")
        return

    # 计算每个项目的平均残差
    mean_residuals = country_df.groupby('Sport')['Residuals'].mean().sort_values()
    
    # 分为表现最好和最差的
    strongest = mean_residuals.tail(n)
    weakest = mean_residuals.head(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    sns.set_style("whitegrid")

    # 绘制表现最好的项目
    sns.barplot(x=strongest.values, y=strongest.index, ax=ax1, palette="viridis")
    ax1.set_title(f'Top {n} Strongest Sports for {country_noc}\n(Performance Above Expectation)', fontsize=14)
    ax1.set_xlabel('Average Medals Won MORE Than Predicted (Residuals)')
    ax1.set_ylabel('Sport')

    # 绘制表现最差的项目
    sns.barplot(x=weakest.values, y=weakest.index, ax=ax2, palette="plasma")
    ax2.set_title(f'Top {n} Weakest Sports for {country_noc}\n(Performance Below Expectation)', fontsize=14)
    ax2.set_xlabel('Average Medals Won FEWER Than Predicted (Residuals)')
    ax2.set_ylabel('') # 共享Y轴，不重复标签

    fig.suptitle(f'Performance Analysis for {country_noc}', fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_added_variable(fitted_model, feature_to_plot):
    """
    绘制增量贡献图的最终通用版本, 能自动兼容新旧 statsmodels 库。
    """
    print(f"\n正在为变量 '{feature_to_plot}' 生成一组诊断图...")
    # 调整画布大小以获得更好的布局
    fig = plt.figure(figsize=(12, 9))

    try:
        # 方案A: 首先尝试使用新版 statsmodels 的 'exog_var' 参数
        print("正在尝试使用新版API (exog_var)...")
        sm.graphics.plot_regress_exog(fitted_model, exog_var=feature_to_plot, fig=fig)

    except TypeError:
        # 方案B: 如果方案A失败 (说明是旧版), 则自动回退到使用 'exog_idx' 参数
        print("检测到旧版 statsmodels API, 正在切换到兼容模式 (exog_idx)...")
        try:
            # 1. 从模型中获取所有变量的名称列表
            variable_names = fitted_model.model.exog_names
            # 2. 找到我们目标的变量的索引位置
            feature_index = variable_names.index(feature_to_plot)
            # 3. 使用索引来调用绘图函数
            sm.graphics.plot_regress_exog(fitted_model, exog_idx=feature_index, fig=fig)
        except ValueError:
            print(f"错误: 无法在模型变量中找到 '{feature_to_plot}'。")
            # 清理未使用的Figure
            plt.close(fig)
            return
        except Exception as e:
            print(f"在使用兼容模式时发生未知错误: {e}")
            plt.close(fig)
            return
            
    # 调整布局和标题
    # y=1.00 让标题稍微向上移动，避免与子图标题重叠
    fig.suptitle(f'Diagnostic Plots for "{feature_to_plot}"', y=1.00, fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()

# --- 调用绘图函数 ---
# 1. 分析中国的优势项目
plot_strongest_sports('CHN', panel_df)
# 2. 分析美国的优势项目
plot_strongest_sports('USA', panel_df)
# 3. 可视化 Num_Events 的纯影响
plot_added_variable(robust_model, 'Num_Events')

print("\n--- 6. 保存分析结果以供后续使用 ---")
# 将包含残差的面板数据保存到CSV文件，以便 coach_effect.py 调用
try:
    panel_df.to_csv('analysis_results_with_residuals.csv', index=False)
    print("分析结果已成功保存到 'analysis_results_with_residuals.csv'")
except Exception as e:
    print(f"保存文件时出错: {e}")

#寻找存在伟大教练证据
def plot_coach_effect_evidence(country_noc, sport_name, panel_df):
    """
    为指定的“国家-运动项目”组合绘制奖牌数和模型残差的时间序列图，
    用以寻找“伟大教练效应”的证据。
    """
    # 筛选出目标数据
    df_target = panel_df[(panel_df['NOC'] == country_noc) & (panel_df['Sport'] == sport_name)].sort_values('Year')

    if df_target.empty:
        print(f"\n未找到 {country_noc} 在 {sport_name} 项目上的数据。")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    sns.set_style("whitegrid")

    # --- 图1: 真实奖牌数的时间序列 ---
    ax1.plot(df_target['Year'], df_target['Medal_Count'], marker='o', linestyle='-', color='dodgerblue', label='Actual Medals')
    ax1.set_title(f'Historical Medal Counts for {country_noc} in {sport_name}', fontsize=14)
    ax1.set_ylabel('Number of Medals')
    ax1.legend()

    # --- 图2: 模型残差的时间序列 ---
    ax2.plot(df_target['Year'], df_target['Residuals'], marker='o', linestyle='--', color='orangered', label='Model Residuals')
    # 添加0线作为参考
    ax2.axhline(0, color='black', linestyle=':', lw=1)
    ax2.set_title(f'Performance vs. Expectation (Residuals) for {country_noc} in {sport_name}', fontsize=14)
    ax2.set_ylabel('Medals Above/Below Expectation')
    ax2.set_xlabel('Olympic Year')
    ax2.legend()
    
    fig.suptitle(f'Searching for "Great Coach Effect" Evidence:\n{country_noc} - {sport_name}', fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# --- 调用函数进行案例分析 ---
# 假设 panel_df 已经加载并包含 'Residuals' 列

print("\n--- 正在寻找'伟大教练效应'的历史证据 ---")

# 案例1: 郎平与排球 (Volleyball)
# 注意: 'Volleyball' 在数据中可能分为 'Volleyball' 和 'Beach Volleyball'，这里我们假设分析的是室内排球
# 我们可以分析中国(CHN)和美国(USA)
plot_coach_effect_evidence('CHN', 'Volleyball', panel_df)
plot_coach_effect_evidence('USA', 'Volleyball', panel_df)

# 案例2: 贝拉·卡罗伊与体操 (Gymnastics)
# 我们可以分析罗马尼亚(ROU)和美国(USA)
plot_coach_effect_evidence('ROU', 'Gymnastics', panel_df)
plot_coach_effect_evidence('USA', 'Gymnastics', panel_df)

print("\n分析脚本运行结束。")