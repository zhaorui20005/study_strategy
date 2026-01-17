import pandas as pd  # 导入 pandas 库，用于数据处理和分析
import numpy as np   # 导入 numpy 库，用于数值计算（虽然在这个版本中主要用了 pandas 的功能）

# 定义一个常量，表示初始投资本金，用于百分比计算
# 假设初始资金为 1,000,000 USD (参考您图片中的持仓价值)
INITIAL_EQUITY = 1000000 

def calculate_performance_metrics(df):
    """
    计算交易日志的整体绩效指标（包括百分比指标）
    按照PineScript逻辑计算最大回撤：考虑Adverse excursion %与基于累计盈亏的回撤的较大值
    注意：每笔交易有2行（Entry和Exit），且两行的profit值相同
    """
    # 1. 整体收益率 (Net Profit %) - 按照复利计算
    # 使用每笔交易的Net P&L %进行累乘计算
    # 注意：由于Entry和Exit的profit相同，我们只取Exit行来计算
    exit_rows = df[df['Type'].str.contains('Exit', na=False)]
    if len(exit_rows) == 0:
        # 如果没有Exit行，则取每2行的最后一行（Exit应该在Entry之后）
        exit_rows = df.iloc[1::2] if len(df) > 1 else df
    
    # 获取每笔交易的Net P&L %（已经是百分比形式，如-0.74表示-0.74%）
    profit_pct_list = exit_rows['Net P&L %'].values
    
    # 将百分比转换为小数形式（除以100），然后计算 (1 + profit%1) * (1 + profit%2) * ...
    # 例如：-0.74% -> -0.0074 -> (1 + (-0.0074)) = 0.9926
    cumulative_return = np.prod(1 + profit_pct_list / 100)
    
    # 最终收益率 = (累乘结果 - 1) * 100，转换为百分比
    total_net_profit_pct = (cumulative_return - 1) * 100
    
    # 2. 交易总数 (Total Trades)
    # 每笔交易有2行（Entry和Exit），所以交易总数是行数除以2
    total_trades = len(df) // 2
    
    # 3. 胜率 (Win Rate)
    # 只计算Exit行的胜率，因为Entry和Exit的profit值相同
    winning_trades_count = (exit_rows['Net P&L USD'] > 0).sum()
    # 计算胜率百分比：赢的次数 / 总次数 * 100
    win_rate = (winning_trades_count / total_trades) * 100 if total_trades > 0 else 0
    
    # 4. 最大回撤 (Maximum Drawdown) - 按照PineScript逻辑
    # 使用 'Cumulative P&L USD'（累计盈亏美元）列来计算
    cumulative_pnl = df['Cumulative P&L USD']
    
    # 将累计盈亏加上初始本金，得到完整的资金曲线
    equity_curve = INITIAL_EQUITY + cumulative_pnl
    
    # 计算资金曲线的峰值（到目前为止达到的最高点）序列
    peak = equity_curve.cummax()
    
    # 计算基于累计盈亏曲线的回撤（峰值减去当前资金曲线值）
    drawdown = peak - equity_curve
    
    # 计算基于累计盈亏曲线的回撤百分比
    drawdown_pct_from_equity = (drawdown / peak) * 100
    
    # 获取Adverse excursion %列
    # 注意：Adverse excursion %是负数，表示不利偏移百分比
    # 在PineScript中，Adverse excursion %是相对于entry时的equity的百分比
    # 我们需要将其转换为相对于当前equity的百分比，或者直接使用绝对值
    adverse_excursion_pct = df['Adverse excursion %'].abs()
    
    # 按照PineScript逻辑：对于每一行，取基于累计盈亏的回撤百分比和Adverse excursion %的较大值
    # 这样可以考虑持仓期间可能经历的最大回撤
    # 注意：我们需要确保两个Series的长度相同
    if len(drawdown_pct_from_equity) == len(adverse_excursion_pct):
        combined_drawdown_pct = pd.concat([drawdown_pct_from_equity, adverse_excursion_pct], axis=1).max(axis=1)
    else:
        # 如果长度不同，使用numpy的maximum函数
        combined_drawdown_pct = pd.Series(np.maximum(drawdown_pct_from_equity.values, adverse_excursion_pct.values))
    
    # 最大回撤百分比：所有行中回撤百分比的最大值
    max_drawdown_pct = combined_drawdown_pct.max()
    
    # 返回一个字典，包含所有计算出的指标（只显示百分比格式）
    return {
        "Total Net Profit (%)": total_net_profit_pct,
        "Total Trades": total_trades,
        "Win Rate (%)": win_rate,
        "Maximum Drawdown (%)": max_drawdown_pct
    }

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
    
    # 重新计算合并后的"累计盈亏"，因为原始的累计盈亏是独立的
    # 注意：由于Entry和Exit的profit值相同，我们需要只对Exit行计算累计盈亏
    # 然后使用前向填充将值赋给Entry行
    exit_mask = combined_df['Type'].str.contains('Exit', na=False)
    if exit_mask.sum() == 0:
        # 如果没有Exit行，则假设每2行的第二行是Exit（索引1, 3, 5...）
        exit_mask = pd.Series([False] * len(combined_df))
        if len(combined_df) > 0:
            exit_mask.iloc[1::2] = True
    
    # 只对Exit行的profit进行cumsum（避免重复计算）
    if exit_mask.sum() > 0:
        exit_profits = combined_df.loc[exit_mask, 'Net P&L USD'].copy()
        cumulative_pnl_exit = exit_profits.cumsum()
        
        # 将累计盈亏值赋给所有行（使用前向填充）
        combined_df['Cumulative P&L USD'] = 0.0
        combined_df.loc[exit_mask, 'Cumulative P&L USD'] = cumulative_pnl_exit.values
        # 使用前向填充，将Exit行的累计盈亏值填充到对应的Entry行
        combined_df['Cumulative P&L USD'] = combined_df['Cumulative P&L USD'].replace(0, pd.NA).ffill().fillna(0)
    else:
        # 如果没有Exit行，直接使用cumsum（虽然会重复计算，但至少能运行）
        combined_df['Cumulative P&L USD'] = combined_df['Net P&L USD'].cumsum()

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
    # 直接使用原始CSV中的累计盈亏列（如果存在）
    df_a_subset = df_a.head(min_len).copy()
    # 确保累计盈亏列存在且有效
    if 'Cumulative P&L USD' in df_a_subset.columns:
        # 使用原始CSV中的累计盈亏，但需要确保Entry和Exit行的累计盈亏相同
        # 由于Entry和Exit的profit相同，累计盈亏也应该相同，我们只取Exit行的累计盈亏
        exit_mask_a = df_a_subset['Type'].str.contains('Exit', na=False)
        if exit_mask_a.sum() == 0:
            exit_mask_a = pd.Series([False] * len(df_a_subset))
            if len(df_a_subset) > 0:
                exit_mask_a.iloc[1::2] = True
        # 只使用Exit行的累计盈亏，然后前向填充到Entry行
        if exit_mask_a.sum() > 0:
            exit_cumulative = df_a_subset.loc[exit_mask_a, 'Cumulative P&L USD'].copy()
            df_a_subset.loc[exit_mask_a, 'Cumulative P&L USD'] = exit_cumulative.values
            df_a_subset['Cumulative P&L USD'] = df_a_subset['Cumulative P&L USD'].replace(0, pd.NA).ffill().fillna(0)
    
    metrics_a = calculate_performance_metrics(df_a_subset)
    for key, value in metrics_a.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\n--- B 策略独立绩效分析 ---")
    # 独立分析 B 策略的绩效
    # 直接使用原始CSV中的累计盈亏列（如果存在）
    df_b_subset = df_b.head(min_len).copy()
    # 确保累计盈亏列存在且有效
    if 'Cumulative P&L USD' in df_b_subset.columns:
        # 使用原始CSV中的累计盈亏，但需要确保Entry和Exit行的累计盈亏相同
        # 由于Entry和Exit的profit相同，累计盈亏也应该相同，我们只取Exit行的累计盈亏
        exit_mask_b = df_b_subset['Type'].str.contains('Exit', na=False)
        if exit_mask_b.sum() == 0:
            exit_mask_b = pd.Series([False] * len(df_b_subset))
            if len(df_b_subset) > 0:
                exit_mask_b.iloc[1::2] = True
        # 只使用Exit行的累计盈亏，然后前向填充到Entry行
        if exit_mask_b.sum() > 0:
            exit_cumulative = df_b_subset.loc[exit_mask_b, 'Cumulative P&L USD'].copy()
            df_b_subset.loc[exit_mask_b, 'Cumulative P&L USD'] = exit_cumulative.values
            df_b_subset['Cumulative P&L USD'] = df_b_subset['Cumulative P&L USD'].replace(0, pd.NA).ffill().fillna(0)
    
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
