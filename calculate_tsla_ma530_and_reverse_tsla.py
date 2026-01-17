import pandas as pd  # 导入 pandas 库，用于数据处理和分析
import numpy as np   # 导入 numpy 库，用于数值计算（虽然在这个版本中主要用了 pandas 的功能）

# 定义一个常量，表示初始投资本金，用于百分比计算
# 假设初始资金为 1,000,000 USD (参考您图片中的持仓价值)
INITIAL_EQUITY = 1000000 

def calculate_performance_metrics(df):
    """
    计算交易日志的整体绩效指标（包括百分比指标）
    """
    # 1. 整体收益率 (Net Profit USD)
    # 对 'Net P&L USD'（净盈亏美元）列求和
    total_net_profit_usd = df['Net P&L USD'].sum()
    
    # 转换为百分比：总利润 / 初始本金 * 100
    total_net_profit_pct = (total_net_profit_usd / INITIAL_EQUITY) * 100
    
    # 2. 交易总数 (Total Trades)
    # 计算 DataFrame 的行数即为交易总数
    total_trades = len(df)
    
    # 3. 胜率 (Win Rate)
    # 筛选出净盈亏大于 0 的行，并计算它们的数量
    winning_trades_count = df[df['Net P&L USD'] > 0].shape[0] 
    # 计算胜率百分比：赢的次数 / 总次数 * 100
    win_rate = (winning_trades_count / total_trades) * 100 if total_trades > 0 else 0
    
    # 4. 最大回撤 (Maximum Drawdown)
    # 使用 'Cumulative P&L USD'（累计盈亏美元）列来计算
    cumulative_pnl = df['Cumulative P&L USD']
    
    # 将累计盈亏加上初始本金，得到完整的资金曲线
    equity_curve = INITIAL_EQUITY + cumulative_pnl
    
    # 计算资金曲线的峰值（到目前为止达到的最高点）序列
    peak = equity_curve.cummax()
    
    # 计算回撤（峰值减去当前资金曲线值）
    drawdown = peak - equity_curve
    
    # 最大回撤（美元值）
    max_drawdown_usd = drawdown.max()

    # 最大回撤百分比：最大回撤美元 / 达到峰值时的资金 * 100
    # 我们找到最大回撤发生时的峰值（用 idxmax 找到最大回撤的索引，再找到对应的峰值资金）
    max_drawdown_pct = (max_drawdown_usd / peak[drawdown.idxmax()]) * 100
    
    # 返回一个字典，包含所有计算出的指标
    return {
        "Total Net Profit (USD)": total_net_profit_usd,
        "Total Net Profit (%)": total_net_profit_pct,
        "Total Trades": total_trades,
        "Win Rate (%)": win_rate,
        "Maximum Drawdown (USD)": max_drawdown_usd,
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
    
    # 重新计算合并后的“累计盈亏”，因为原始的累计盈亏是独立的
    # 使用 cumsum() 方法进行累加
    combined_df['Cumulative P&L USD'] = combined_df['Net P&L USD'].cumsum()

    # --- 打印绩效分析 ---
    print("--- 综合整体绩效分析 ---")
    # 调用函数计算合并后的指标并打印
    overall_metrics = calculate_performance_metrics(combined_df)
    for key, value in overall_metrics.items():
        print(f"{key}: {value:.2f}")

    print("\n--- A 策略独立绩效分析 ---")
    # 独立分析 A 策略的绩效
    metrics_a = calculate_performance_metrics(df_a.head(min_len))
    for key, value in metrics_a.items():
        print(f"{key}: {value:.2f}")

    print("\n--- B 策略独立绩效分析 ---")
    # 独立分析 B 策略的绩效
    metrics_b = calculate_performance_metrics(df_b.head(min_len))
    for key, value in metrics_b.items():
        print(f"{key}: {value:.2f}")
        
    return combined_df # 返回合并后的 DataFrame (可选)

# --- 运行程序 ---
# 替换为您的实际 CSV 文件路径
file_A = 'trades_strategy_A.csv' 
file_B = 'trades_strategy_B.csv'

# 执行合并与分析 (如果您有文件，取消下一行的注释并运行)
combined_results_df = merge_alternating_trades(file_A, file_B)
