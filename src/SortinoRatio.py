import numpy as np

def sortino_ratio(returns: np.ndarray, risk_free_rate: float, target_return: float = 0.0) -> float:
    """
    计算索提诺比率（Sortino Ratio）

    参数:
    returns (numpy.ndarray): 投资组合或资产的回报率数组（例如日回报率或月回报率）
    risk_free_rate (float): 无风险利率（例如年化国债收益率）
    target_return (float): 目标回报率，默认为0（可以用作最小回报率）

    返回:
    float: 投资组合或资产的索提诺比率
    """
    # 计算平均回报率
    mean_return = np.mean(returns)
    
    # 计算超额回报（超过无风险利率的回报）
    excess_return = mean_return - risk_free_rate
    
    # 计算下行风险（即回报率低于目标回报的负波动）
    downside_deviation = np.sqrt(np.mean(np.minimum(returns - target_return, 0) ** 2))
    
    # 如果下行风险为0，返回无穷大，避免除以零的错误
    if downside_deviation == 0:
        return np.inf
    
    # 计算索提诺比率
    sortino_ratio_value = excess_return / downside_deviation
    
    return sortino_ratio_value

# 示例使用:
# 创建一个包含日回报率的示例数据集
daily_returns = np.array([0.001, 0.002, -0.0015, 0.0025, 0.003, -0.002])

# 假设无风险利率为 0.02（即年化2%的无风险利率，假设是年化回报率）
# 如果日回报率，则需要将无风险利率按时间进行缩放
annual_risk_free_rate = 0.02
daily_risk_free_rate = annual_risk_free_rate / 252  # 假设一年252个交易日

# 计算索提诺比率
sortino = sortino_ratio(daily_returns, daily_risk_free_rate)
print(f"索提诺比率: {sortino:.4f}")