import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import minimize
from typing import List, Dict

from config.config import FONT_PATH, BASIC_IMAGE_DIR, BASIC_DATA_DIR, TICKERS

from util.log_utils import logger

font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = 'SimHei'

class PortfolioOptimization:
    """投资组合优化类，用于计算最优资产配置。

    Attributes:
        tickers (List[str]): 股票代码列表。
        data (pd.DataFrame): 获取的股票价格数据。
        returns (pd.DataFrame): 计算的股票收益率。
    """

    def __init__(self, tickers: List[str], period='5y'):
        """初始化投资组合优化类。

        :Parameters:
            tickers : str, list
                List of tickers to download
            period : str
                Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                Either Use period parameter or use start and end
        """
        self.tickers = tickers
        self.period = period
        logger.log_info(f"对于{period}, 我们开始获取股票价格数据，股票代码：{self.tickers}")
        self.data = self._download_data()
        logger.log_info(f"股票价格数据获取完成")
        self.returns = self._calculate_returns()
        logger.log_info(f"股票收益率计算完成")

    def _download_data(self) -> pd.DataFrame:
        """下载股票价格数据。

        Returns:
            pd.DataFrame: 收盘价数据。
        """
        # 下载最近3个月的收盘价数据
        data = yf.download(self.tickers, period=self.period)
        data.to_csv(
            os.path.join(
                BASIC_DATA_DIR,
               f'stock_price_{self.period}.csv'
            )
        )
        return data['Adj Close']

    def _calculate_returns(self) -> pd.DataFrame:
        """计算股票的每日收益率。

        Returns:
            pd.DataFrame: 每日收益率数据。
        """
        returns = self.data.pct_change().dropna()
        return returns

    def optimize_portfolio(self) -> Dict[str, float]:
        """优化投资组合以最小化风险。

        Returns:
            Dict[str, float]: 最优资产权重。
        """
        num_assets = len(self.tickers)
        returns_mean = self.returns.mean() * 252
        returns_cov = self.returns.cov() * 252
        args = (returns_cov)
        constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重之和为1
                {'type': 'eq', 'fun': lambda x: np.sum(returns_mean * x) - 0.5},  # 预期收益为0.5
                
            )
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets]

        # 最小化投资组合的方差
        result = minimize(self._portfolio_variance, initial_weights, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

        self.optimal_weights = result.x  # 保存最优权重
        self.min_variance = result.fun   # 保存最小方差

        optimal_weights_dict = dict(zip(self.tickers, self.optimal_weights))
        return optimal_weights_dict

    def _portfolio_variance(self, weights: np.ndarray, returns_cov: pd.DataFrame) -> float:
        """计算投资组合的方差。

        Args:
            weights (np.ndarray): 资产权重。
            returns_cov (pd.DataFrame): 资产的协方差矩阵。

        Returns:
            float: 投资组合的方差。
        """
        portfolio_variance = np.dot(weights.T, np.dot(returns_cov, weights))
        return portfolio_variance

    def plot_efficient_frontier(self):
        """绘制有效边界。"""
        num_portfolios = 100000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        returns_mean = self.returns.mean() * 252
        returns_cov = self.returns.cov() * 252

        for i in range(num_portfolios):
            # 生成随机权重，确保权重之和为1
            weights = np.random.dirichlet(np.ones(len(self.tickers)))
            weights_record.append(weights)
            # 计算投资组合的预期收益和标准差
            portfolio_return = np.dot(weights, returns_mean)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns_cov, weights)))
            # 存储结果
            results[0, i] = portfolio_std
            results[1, i] = portfolio_return
            results[2, i] = portfolio_return / portfolio_std  # 夏普比率

        # 绘制散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10)
        plt.colorbar(label='夏普比率')
        plt.xlabel('年化标准差')
        plt.ylabel('年化收益')
        plt.title('有效边界')

        # 计算最优投资组合的收益和标准差
        optimal_return = np.dot(self.optimal_weights, returns_mean)
        optimal_std = np.sqrt(np.dot(self.optimal_weights.T, np.dot(returns_cov, self.optimal_weights)))

        # 在图中标记最优点
        plt.scatter(optimal_std, optimal_return, marker='*', color='r', s=500, label='最优点')

        # 添加注释，显示最优权重
        weight_text = '\n'.join([f"{ticker}: {weight:.2%}" for ticker, weight in zip(self.tickers, self.optimal_weights)])
        plt.annotate('最优点', xy=(optimal_std, optimal_return), xytext=(optimal_std + 0.005, optimal_return + 0.005),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=12, ha='left')
        plt.text(optimal_std + 0.005, optimal_return - 0.01, weight_text, fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        plt.legend()
        plt.savefig(
            os.path.join(
                BASIC_IMAGE_DIR,
                f'efficient_frontier_{self.period}.png'
            )
        )
        plt.close()


if __name__ == "__main__":
    for period in ['5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max']:
        portfolio = PortfolioOptimization(
                TICKERS, 
                period=period
            )
        # 获取最优权重
        optimal_weights = portfolio.optimize_portfolio()
        logger.log_info("最优资产配置权重：")
        for ticker, weight in optimal_weights.items():
            logger.log_info(f"{ticker}: {weight:.2%}")

        # 绘制有效边界
        portfolio.plot_efficient_frontier()
        logger.log_info("有效边界绘制完成")