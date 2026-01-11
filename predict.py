# 1. 环境设置与数据加载
# ==================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
#from io import StringIO

print("--- 步骤1: 正在加载虚拟数据... ---")

# 加载数据
df_medals = pd.read_csv('summerOly_medal_counts.csv')
df_hosts = pd.read_csv('summerOly_hosts.csv')
df_athletes_raw = pd.read_csv('summerOly_athletes.csv')
print("数据加载完成。")

# 2. 特征工程
# ==================================
print("\n--- 步骤2: 正在进行特征工程... ---")

# --- 步骤 2a 和 2b  ---
print("正在创建 '国家名称' -> 'NOC' 的映射关系...")
df_athletes_raw['Country_Name'] = df_athletes_raw['Team'].apply(lambda x: x.split('-')[0])
country_to_noc_map = df_athletes_raw[['Country_Name', 'NOC']].drop_duplicates().set_index('Country_Name')['NOC'].to_dict()
print("正在处理主办国数据，提取NOC...")
df_hosts['Host_Country_Name'] = df_hosts['Host'].str.split(', ').str[-1]
df_hosts['Host_NOC'] = df_hosts['Host_Country_Name'].map(country_to_noc_map)

# --- 步骤 2c  ---
print("正在从原始运动员数据计算每国每年的参赛人数...")
df_athlete_counts = df_athletes_raw.groupby(['Year', 'NOC'])['Name'].nunique().reset_index()
df_athlete_counts.rename(columns={'Name': 'Num_Athletes'}, inplace=True)

# --- 步骤 2d  ---
print("正在构建最终特征矩阵...")
# 结合运动员名单和奖牌名单，确保覆盖所有参赛国
all_nocs_medals = df_medals['NOC'].unique()
all_nocs_athletes = df_athletes_raw['NOC'].unique()
all_nocs = np.union1d(all_nocs_medals, all_nocs_athletes)
# ----------------------------------------------
print(f"已识别到共 {len(all_nocs)} 个参赛国家/地区。")

all_years = np.unique(np.append(df_medals['Year'].unique(), df_hosts['Year'].unique()))
future_year = np.array([2028])
all_years_ext = np.unique(np.concatenate([all_years, future_year]))
df_base = pd.DataFrame([(year, noc) for year in all_years_ext for noc in all_nocs], columns=['Year', 'NOC'])

# 依次合并所有数据源
df_full = pd.merge(df_base, df_medals, on=['Year', 'NOC'], how='left')
df_full = pd.merge(df_full, df_hosts[['Year', 'Host_NOC']], on='Year', how='left')
df_full = pd.merge(df_full, df_athlete_counts, on=['Year', 'NOC'], how='left')

print("\n合并后的数据框列名:", df_full.columns.tolist())
df_full[['Gold', 'Total']] = df_full[['Gold', 'Total']].fillna(0)
df_full['Num_Athletes'] = df_full['Num_Athletes'].fillna(df_full.groupby('NOC')['Num_Athletes'].transform('mean')).fillna(5)
df_full['Is_Host'] = (df_full['NOC'] == df_full['Host_NOC']).astype(int)
df_full = df_full.sort_values(by=['NOC', 'Year'])
for lag in [4, 8]:
    df_full[f'Gold_lag_{lag}'] = df_full.groupby('NOC')['Gold'].shift(int(lag/4))
    df_full[f'Total_lag_{lag}'] = df_full.groupby('NOC')['Total'].shift(int(lag/4))
for col in ['Gold_lag_4', 'Total_lag_4', 'Gold_lag_8', 'Total_lag_8']:
    df_full[col] = df_full[col].fillna(0)
print("\n特征工程完成。")

# 3. 模型训练与预测 
# ==================================
print("\n--- 步骤3: 正在训练模型并进行预测... ---")

features = ['Year', 'Num_Athletes', 'Gold_lag_4', 'Total_lag_4', 'Gold_lag_8', 'Total_lag_8', 'Is_Host']
df_full['NOC_cat'] = df_full['NOC'].astype('category').cat.codes
features_with_cat = features + ['NOC_cat']

X_train = df_full[df_full['Year'] < 2028].dropna(subset=features)
y_train_gold = df_full.loc[X_train.index]['Gold']
y_train_total = df_full.loc[X_train.index]['Total']
X_pred_2028 = df_full[df_full['Year'] == 2028][features_with_cat]

lgbm_gold = lgb.LGBMRegressor(random_state=42)
lgbm_gold.fit(X_train[features_with_cat], y_train_gold, feature_name=features_with_cat, categorical_feature=['NOC_cat'])

lgbm_total = lgb.LGBMRegressor(random_state=42)
lgbm_total.fit(X_train[features_with_cat], y_train_total, feature_name=features_with_cat, categorical_feature=['NOC_cat'])

pred_gold_2028 = lgbm_gold.predict(X_pred_2028)
pred_total_2028 = lgbm_total.predict(X_pred_2028)
pred_gold_2028[pred_gold_2028 < 0] = 0
pred_total_2028[pred_total_2028 < 0] = 0
print("模型训练和初步预测完成。")


# 4. 不确定性量化 
# ==================================
print("\n--- 步骤4: 正在量化不确定性... ---")
def train_quantile_model(alpha, X_train, y_train):
    model = lgb.LGBMRegressor(objective='quantile', alpha=alpha, random_state=42)
    model.fit(X_train[features_with_cat], y_train, feature_name=features_with_cat, categorical_feature=['NOC_cat'])
    return model

total_lower_model = train_quantile_model(0.05, X_train, y_train_total)
total_upper_model = train_quantile_model(0.95, X_train, y_train_total)
pred_total_lower = total_lower_model.predict(X_pred_2028)
pred_total_upper = total_upper_model.predict(X_pred_2028)
print("预测区间计算完成。")


# 5. 模型解释 (SHAP) 
# ==================================
print("\n--- 步骤5: 正在生成模型解释... ---")
explainer = shap.TreeExplainer(lgbm_total)
shap_values = explainer.shap_values(X_train[features_with_cat])
print("正在显示SHAP特征重要性图...")
plt.title("SHAP Feature Importance for Total Medals Model")
shap.summary_plot(shap_values, X_train[features_with_cat], show=False)
#plt.savefig("SHAP Feature Importance for Total Medals Model.png",dpi=300)
plt.show()
print("模型解释完成。")

# 6. 结果展示 
# ==================================
print("\n--- 步骤6: 正在整理并展示最终结果... ---")

# --- 修正：将预测区间添加到最终结果DataFrame ---
df_results_2028 = df_full[df_full['Year'] == 2028][['NOC']].copy().reset_index(drop=True)
df_results_2028['Predicted_Gold'] = np.round(pred_gold_2028).astype(int)
df_results_2028['Predicted_Total'] = np.round(pred_total_2028).astype(int)
df_results_2028['Total_Lower_Bound'] = np.round(pred_total_lower).astype(int)
df_results_2028['Total_Upper_Bound'] = np.round(pred_total_upper).astype(int)

# 确保下界不高于预测值，上界不低于预测值
df_results_2028['Total_Lower_Bound'] = np.minimum(df_results_2028['Total_Lower_Bound'], df_results_2028['Predicted_Total'])
df_results_2028['Total_Upper_Bound'] = np.maximum(df_results_2028['Total_Upper_Bound'], df_results_2028['Predicted_Total'])

# 合并2024年数据以计算变化
df_2024 = df_medals[df_medals['Year']==2024][['NOC', 'Total']].rename(columns={'Total':'Total_2024'})
df_results_2028 = pd.merge(df_results_2028, df_2024, on='NOC', how='left').fillna(0)
df_results_2028['Change_vs_2024'] = df_results_2028['Predicted_Total'] - df_results_2028['Total_2024']

# --- 最终结果表格输出 ---
print("\n--- 2028年洛杉矶奥运会奖牌榜预测 (Top 10) ---")
# 创建更清晰的区间列
df_results_2028['90%_Prediction_Interval'] = df_results_2028.apply(
    lambda row: f"[{row['Total_Lower_Bound']}, {row['Total_Upper_Bound']}]", axis=1
)
print(df_results_2028.sort_values(by='Predicted_Total', ascending=False).head(10)[
    ['NOC', 'Predicted_Gold', 'Predicted_Total', '90%_Prediction_Interval', 'Change_vs_2024']
])


# --- 新增可视化函数 ---
def plot_top_n_predictions(results_df, n=15):
    """绘制Top N国家奖牌总数预测的条形图，并附上预测区间作为误差棒"""
    top_n = results_df.sort_values('Predicted_Total', ascending=False).head(n)
    plt.figure(figsize=(12, 8))
    
    # 计算误差
    lower_error = top_n['Predicted_Total'] - top_n['Total_Lower_Bound']
    upper_error = top_n['Total_Upper_Bound'] - top_n['Predicted_Total']
    error = [lower_error, upper_error]
    
    plt.bar(top_n['NOC'], top_n['Predicted_Total'], yerr=error, capsize=5, color='skyblue', ecolor='gray')
    
    plt.ylabel('Predicted Total Medals')
    plt.title(f'Top {n} Countries Medal Prediction for LA 2028 (with 90% Prediction Intervals)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    #plt.savefig("Top 15 Countries Medal Prediction for LA 2028 (with 90% Prediction Intervals).png", dpi=300)
    plt.show()

def plot_biggest_movers(results_df, n=5):
    """绘制奖牌数变化最大（上升和下降）国家的条形图"""
    sorted_df = results_df.sort_values('Change_vs_2024', ascending=False)
    
    improvers = sorted_df.head(n)
    decliners = sorted_df.tail(n).sort_values('Change_vs_2024', ascending=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.bar(improvers['NOC'], improvers['Change_vs_2024'], color='green')
    ax1.set_title(f'Top {n} Most Improved Countries')
    ax1.set_ylabel('Change in Total Medals vs 2024')
    
    ax2.bar(decliners['NOC'], decliners['Change_vs_2024'], color='red')
    ax2.set_title(f'Top {n} Most Declined Countries')
    ax2.set_ylabel('Change in Total Medals vs 2024')
    
    fig.suptitle('Projected Change in Medal Counts: 2028 vs 2024', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.savefig("Projected Change in Medal Counts for LA 2028 vs 2024.png", dpi=300)
    plt.show()

# --- 调用新的绘图函数 ---
print("\n正在生成可视化图表...")
plot_top_n_predictions(df_results_2028)
plot_biggest_movers(df_results_2028)

# 7. 模型表现评估 (新增模块)
# ==================================
print("\n--- 步骤7: 正在评估模型表现 (Backtesting on 2024 data)... ---")
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 回测设置 ---
# 训练数据：2024年之前的所有数据
# 验证数据：2024年的真实数据
validation_year = 2024
X_train_eval = df_full[df_full['Year'] < validation_year][features_with_cat]
y_train_eval = df_full.loc[X_train_eval.index]['Total']

X_validation = df_full[df_full['Year'] == validation_year][features_with_cat]
y_actual_validation = df_full.loc[X_validation.index]['Total']

# --- 重新训练一个仅使用2024年之前数据的模型 ---
eval_model = lgb.LGBMRegressor(random_state=42)
eval_model.fit(X_train_eval, y_train_eval, feature_name=features_with_cat, categorical_feature=['NOC_cat'])

# --- 对2024年进行预测 ---
y_pred_validation = eval_model.predict(X_validation)
y_pred_validation[y_pred_validation < 0] = 0

# --- 计算性能指标 ---
mae = mean_absolute_error(y_actual_validation, y_pred_validation)
rmse = np.sqrt(mean_squared_error(y_actual_validation, y_pred_validation))

print(f"\n模型在2024年数据上的回测表现:")
print(f"平均绝对误差 (MAE): {mae:.2f} medals")
print(f"均方根误差 (RMSE): {rmse:.2f} medals")
print(f"解读: MAE表示，模型对2024年各国奖牌数的预测平均偏差为 {mae:.2f} 枚奖牌。")

# --- 可视化模型表现 ---
def plot_actual_vs_predicted(y_actual, y_predicted, year):
    """绘制一个散点图，比较模型的预测值与真实值"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_predicted, alpha=0.6, edgecolors='k')
    
    # 绘制完美的预测线 (y=x)
    perfect_line = np.linspace(min(y_actual.min(), y_predicted.min()), 
                               max(y_actual.max(), y_predicted.max()), 100)
    plt.plot(perfect_line, perfect_line, 'r--', label='Perfect Prediction (y=x)')
    
    plt.xlabel("Actual Total Medals")
    plt.ylabel("Predicted Total Medals")
    plt.title(f"Model Performance: Actual vs. Predicted Medals for {year} Olympics")
    plt.grid(True)
    plt.legend()
    plt.axis('equal') # 确保x和y轴的比例相同
    plt.show()

print("\n正在生成模型表现的可视化图表...")
# 创建一个包含真实值和预测值的DataFrame用于绘图
eval_results = pd.DataFrame({
    'Actual': y_actual_validation,
    'Predicted': y_pred_validation
})
plot_actual_vs_predicted(eval_results['Actual'], eval_results['Predicted'], validation_year)

# 8. “零的突破”概率预测模型 
# ============================================
print("\n--- 步骤8: 正在为'零的突破'构建概率预测模型... ---")

# --- 1, 2, 3. 准备和训练模型 ---
df_full['Won_Medal'] = (df_full['Total'] > 0).astype(int)
features = ['Year', 'Num_Athletes', 'Gold_lag_4', 'Total_lag_4', 'Gold_lag_8', 'Total_lag_8', 'Is_Host']
df_full['NOC_cat'] = df_full['NOC'].astype('category').cat.codes
features_with_cat = features + ['NOC_cat']
X_train_class = df_full[df_full['Year'] < 2028].dropna(subset=features)
y_train_class = df_full.loc[X_train_class.index]['Won_Medal']
clf_model = lgb.LGBMClassifier(random_state=42, is_unbalance=True)
clf_model.fit(X_train_class[features_with_cat], y_train_class, feature_name=features_with_cat, categorical_feature=['NOC_cat'])

# --- 4. 识别并准备预测数据 ---
historical_medals = df_full[df_full['Year'] < 2028].groupby('NOC')['Total'].sum()
zero_medal_nocs = historical_medals[historical_medals == 0].index
print(f"\n数据库中共有 {len(zero_medal_nocs)} 个国家历史上从未获得奖牌。")

# 先筛选出目标国家在2028年的完整数据
df_2028_zero_medal_full = df_full[
    (df_full['Year'] == 2028) & (df_full['NOC'].isin(zero_medal_nocs))
]

# --- 5. 执行预测并展示结果 ---
# 增加防御性判断
if not df_2028_zero_medal_full.empty:
    # 从完整数据中提取模型需要的特征列
    X_pred_2028_zero_medal = df_2028_zero_medal_full[features_with_cat]
    
    probabilities = clf_model.predict_proba(X_pred_2028_zero_medal)[:, 1]

    # 从完整数据中提取NOC列来构建结果，而不是从特征矩阵中提取
    df_first_medal_prob = pd.DataFrame({
        'NOC': df_2028_zero_medal_full['NOC'].values,
        'Probability_to_Win_First_Medal': probabilities
    })
    df_first_medal_prob['Odds'] = df_first_medal_prob['Probability_to_Win_First_Medal'] / (1 - df_first_medal_prob['Probability_to_Win_First_Medal'])

    print("\n--- 2028年最有可能实现奖牌'零的突破'的国家预测 ---")
    print(df_first_medal_prob.sort_values(by='Probability_to_Win_First_Medal', ascending=False).head(10).to_string(formatters={
        'Probability_to_Win_First_Medal': '{:.2%}'.format,
        'Odds': '{:.2f}:1'.format
    }))
    
    # 绘图函数定义
    def plot_first_medal_probabilities(df_prob, n=10):
        # 修正: 'Probability_to_Win_First_Meda_First_Medal' -> 'Probability_to_Win_First_Medal'
        top_n = df_prob.sort_values('Probability_to_Win_First_Medal', ascending=False).head(n)
        plt.figure(figsize=(10, 6))
        bars = plt.barh(top_n['NOC'], top_n['Probability_to_Win_First_Medal'], color='mediumpurple')
        for bar in bars:
            width = bar.get_width()
            plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}', va='center')
        plt.xlabel('Probability')
        plt.title(f'Top {n} Most Likely Countries to Win Their First Medal in LA 2028')
        plt.gca().invert_yaxis()
        plt.xlim(0, max(top_n['Probability_to_Win_First_Medal'].max() * 1.2, 0.1))
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    print("\n正在生成'零的突破'概率的可视化图表...")
    plot_first_medal_probabilities(df_first_medal_prob)
else:
    print("\n未找到历史上从未获奖牌的国家，无法进行'零的突破'预测。")