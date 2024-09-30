import numpy as np

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float) -> float:
    """
    计算夏普比率（Sharpe Ratio）

    参数:
    returns (numpy.ndarray): 投资组合或资产的回报率数组（例如日回报率或月回报率）
    risk_free_rate (float): 无风险利率（例如年化国债收益率）

    返回:
    float: 投资组合或资产的夏普比率
    """
    # 计算平均回报率
    mean_return = np.mean(returns)
    
    # 计算超额回报（超过无风险利率的回报）
    excess_return = mean_return - risk_free_rate
    
    # 计算投资组合回报的标准差（波动率）
    return_std = np.std(returns)
    
    # 计算夏普比率
    sharpe_ratio_value = excess_return / return_std
    
    return sharpe_ratio_value

# 示例使用:
# 创建一个包含日回报率的示例数据集
daily_returns = np.array([0.001, 0.002, -0.0015, 0.0025, 0.003, -0.002])

# 假设无风险利率为 0.02（即年化2%的无风险利率，假设是年化回报率）
# 如果日回报率，则需要将无风险利率按时间进行缩放
annual_risk_free_rate = 0.02
daily_risk_free_rate = annual_risk_free_rate / 252  # 假设一年252个交易日

# 计算夏普比率
sharpe = sharpe_ratio(daily_returns, daily_risk_free_rate)
print(f"夏普比率: {sharpe:.4f}")