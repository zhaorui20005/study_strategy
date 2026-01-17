import pandas as pd  # 导入 pandas 库，用于数据处理和分析
import numpy as np   # 导入 numpy 库，用于数值计算（虽然在这个版本中主要用了 pandas 的功能）

# 定义一个常量，表示初始投资本金，用于百分比计算
# 假设初始资金为 1,000,000 USD (参考您图片中的持仓价值)
INITIAL_EQUITY = 1000000 

def calculate_performance_metrics(df):
    """
    计算交易日志的整体绩效指标（包括百分比指标）
    按照PineScript逻辑计算最大回撤：考虑Adverse excursion %与基于累计盈亏的回撤的较大值
    注意：df只包含Exit行，已经过滤掉Entry行
    """
    # 1. 整体收益率 (Net Profit %) - 按照复利计算
    # 使用每笔交易的Net P&L %进行累乘计算
    # 获取每笔交易的Net P&L %（已经是百分比形式，如-0.74表示-0.74%）
    profit_pct_list = df['Net P&L %'].values
    
    # 将百分比转换为小数形式（除以100），然后计算 (1 + profit%1) * (1 + profit%2) * ...
    # 例如：-0.74% -> -0.0074 -> (1 + (-0.0074)) = 0.9926
    cumulative_return = np.prod(1 + profit_pct_list / 100)
    
    # 最终收益率 = (累乘结果 - 1) * 100，转换为百分比
    total_net_profit_pct = (cumulative_return - 1) * 100
    
    # 2. 交易总数 (Total Trades)
    # df只包含Exit行，所以交易总数就是行数
    total_trades = len(df)
    
    # 3. 胜率 (Win Rate)
    # 计算胜率：盈利交易数 / 总交易数
    winning_trades_count = (df['Net P&L USD'] > 0).sum()
    # 计算胜率百分比：赢的次数 / 总次数 * 100
    win_rate = (winning_trades_count / total_trades) * 100 if total_trades > 0 else 0
    
    # 4. 最大回撤 (Maximum Drawdown) - 只计算交易close之后的最大回撤
    # 使用 Net P&L % 进行复利计算，基于equity曲线计算最大回撤
    max_drawdown_pct = calculate_advanced_dynamic_dd(df)
    
    # 返回一个字典，包含所有计算出的指标（只显示百分比格式）
    return {
        "Total Net Profit (%)": total_net_profit_pct,
        "Total Trades": total_trades,
        "Win Rate (%)": win_rate,
        "Maximum Drawdown (%)": max_drawdown_pct
    }


def calculate_simple_drawdown(df):
    """
    计算交易close之后的最大回撤（基于equity曲线）
    只考虑每笔交易结算后的资金曲线，不考虑交易过程中的Adverse excursion
    """
    # 创建df的副本，避免修改原始数据
    df = df.copy()
    
    # 1. 计算每笔交易结算后的账户资金 (Equity)
    # 使用Net P&L %进行复利计算，假设初始资金为 1.0
    # Net P&L %已经是百分比形式（如-0.74表示-0.74%），需要除以100转换为小数
    net_pnl_pct = df['Net P&L %'].values / 100  # 转换为小数形式（如-0.74 -> -0.0074）
    df['cum_pnl_factor'] = np.cumprod(1 + net_pnl_pct)
    df['equity_end'] = df['cum_pnl_factor']  # 每笔交易结算后的资金
    
    # 2. 计算资金曲线的峰值（到目前为止达到的最高点）序列
    peak = df['equity_end'].cummax()
    
    # 3. 计算回撤（峰值减去当前资金曲线值）
    drawdown = peak - df['equity_end']
    
    # 4. 计算回撤百分比：回撤 / 峰值 * 100
    drawdown_pct = (drawdown / peak) * 100
    
    # 5. 最大回撤百分比：所有行中回撤百分比的最大值
    max_drawdown_pct = drawdown_pct.max()
    
    return max_drawdown_pct


def calculate_advanced_dynamic_dd(df):
    """
    计算动态最大回撤
    按照用户描述的逻辑：
    1. 收益率10%, adverse_excursion = -5% -> 最大回撤 = 5%
    2. 收益率-5%, adverse_excursion = -5.1% -> 最大回撤 = 5.1%
    3. 收益率-0.5%, adverse_excursion = -1% -> 最大回撤 = 6%
    
    注意：
    - Favorable excursion % 永远为非负数，已经是百分比形式（如13.24表示13.24%）
    - Adverse excursion % 永远为非正数，已经是百分比形式（如-4.41表示-4.41%）
    
    逻辑：对于每一笔交易，计算动态最大回撤：
    - 前一天的最大回撤 + 当前交易的Adverse excursion %（绝对值）
    - 当前equity曲线的回撤%
    - 当前交易的Adverse excursion %（绝对值）
    取这些值的最大值，然后整个序列取最大值
    """
    # 创建df的副本，避免修改原始数据
    df = df.copy()
    
    # 1. 计算每笔交易开始前的账户资金 (Equity)
    # 使用Net P&L %进行复利计算，假设初始资金为 1.0
    # Net P&L %已经是百分比形式（如-0.74表示-0.74%），需要除以100转换为小数
    net_pnl_pct = df['Net P&L %'].values / 100  # 转换为小数形式（如-0.74 -> -0.0074）
    df['cum_pnl_factor'] = np.cumprod(1 + net_pnl_pct)
    df['equity_start'] = df['cum_pnl_factor'].shift(1).fillna(1.0)
    
    # 2. 计算结算后的 账户资金
    df['equity_end'] = df['equity_start'].values * (1 + net_pnl_pct)
    
    # 3. 更新 历史全局最高点 (Running Peak)
    # 只计算到当前交易开始前（equity_start），不考虑当前交易
    df['running_peak'] = df['equity_start'].cummax()
    
    # 4. 计算 动态最低点 (Intra-trade Low)
    # 账户在此笔交易中达到的最低点 = 交易前资金 * (1 + 最大不利变动率)
    # Adverse excursion %已经是百分比形式（如-4.41表示-4.41%），需要除以100转换为小数
    adverse_excursion_pct = df['Adverse excursion %'].values / 100  # 转换为小数形式（已经是负数）
    df['intra_trade_low'] = df['equity_start'].values * (1 + adverse_excursion_pct)
    
    # 5. 计算基于equity曲线的回撤百分比
    # 回撤 = (峰值 - 当前最低点) / 峰值 * 100
    drawdown_from_equity = (df['running_peak'] - df['intra_trade_low']) / df['running_peak'] * 100
    df['drawdown_from_equity'] = drawdown_from_equity
    
    # 6. 获取Adverse excursion %（取绝对值，因为它是负数）
    # Adverse excursion %已经是百分比形式，直接使用
    adverse_excursion_pct_abs = df['Adverse excursion %'].abs()
    
    # 7. 按照用户描述的逻辑：当前的最大回撤 = max(当前equity曲线的回撤%, 当前交易的Adverse excursion %)
    # 对于每一行，取基于equity曲线的回撤百分比和Adverse excursion %的较大值
    combined_drawdown_pct = pd.concat([drawdown_from_equity, adverse_excursion_pct_abs], axis=1).max(axis=1)
    
    # # 将combined_drawdown_pct添加到df中作为新列
    # df['combined_drawdown_pct'] = combined_drawdown_pct
    
    # # 8. 计算动态最大回撤：整个序列的最大值
    max_dynamic_dd = combined_drawdown_pct.max()
    # pd.set_option('display.max_rows', None)

    # print(df[['Net P&L %','Adverse excursion %','equity_start','equity_end','running_peak', 'drawdown_from_equity', 'combined_drawdown_pct']])
    # # 打印完成后，如果需要恢复默认设置（可选）
    # pd.reset_option('display.max_rows')
    return max_dynamic_dd


def merge_alternating_trades(file_a_path, file_b_path):
    """
    按每两行（一笔完整交易）交替合并 A 和 B 的文件，并进行分析
    """
    # 加载数据，pandas 智能识别 CSV 格式
    df_a = pd.read_csv(file_a_path) 
    df_b = pd.read_csv(file_b_path) 
    
    # 确保我们处理的是偶数行数据，因为是以 2 行为一组（Entry/Exit pair）
    len_a = len(df_a) // 2 * 2
    len_b = len(df_b) // 2 * 2
    
    # 取两者最短的长度来对齐合并，确保一一对应
    min_len = min(len_a, len_b) 
    
    merged_list = [] # 初始化一个列表，用于存放交替合并后的交易对
    for i in range(0, min_len, 2): # 循环，步长为 2
        # 提取 A 文件的 2 行交易数据 (一笔完整的交易)
        trade_a = df_a.iloc[i:i+2]
        # 提取 B 文件的 2 行交易数据
        trade_b = df_b.iloc[i:i+2]
        
        merged_list.append(trade_a) # 将 A 的交易对加入列表
        merged_list.append(trade_b) # 将 B 的交易对加入列表
        
    # 使用 concat 将列表中的所有 DataFrame 片段合并成一个大的 DataFrame
    combined_df = pd.concat(merged_list).reset_index(drop=True)
    
    # 过滤掉Entry行，只保留Exit行
    combined_df = combined_df[combined_df['Type'].str.contains('Exit', na=False)].reset_index(drop=True)

    # --- 打印绩效分析 ---
    print("--- 综合整体绩效分析 ---")
    # 调用函数计算合并后的指标并打印
    overall_metrics = calculate_performance_metrics(combined_df)
    for key, value in overall_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\n--- A 策略独立绩效分析 ---")
    # 独立分析 A 策略的绩效
    # 过滤掉Entry行，只保留Exit行
    df_a_subset = df_a.head(min_len).copy()
    df_a_subset = df_a_subset[df_a_subset['Type'].str.contains('Exit', na=False)].reset_index(drop=True)
    
    metrics_a = calculate_performance_metrics(df_a_subset)
    for key, value in metrics_a.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\n--- B 策略独立绩效分析 ---")
    # 独立分析 B 策略的绩效
    # 过滤掉Entry行，只保留Exit行
    df_b_subset = df_b.head(min_len).copy()
    df_b_subset = df_b_subset[df_b_subset['Type'].str.contains('Exit', na=False)].reset_index(drop=True)
    
    metrics_b = calculate_performance_metrics(df_b_subset)
    for key, value in metrics_b.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
        
    return combined_df # 返回合并后的 DataFrame (可选)

# --- 运行程序 ---
# 替换为您的实际 CSV 文件路径
file_A = 'trades_strategy_A.csv' 
file_B = 'trades_strategy_B.csv'

# 执行合并与分析 (如果您有文件，取消下一行的注释并运行)
combined_results_df = merge_alternating_trades(file_A, file_B)
