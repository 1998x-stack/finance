import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import List

from util.log_utils import logger
from config.config import FONT_PATH, BASIC_IMAGE_DIR, BASIC_DATA_DIR

font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = 'SimHei'

class CapitalMarketLine:
    """资本市场线类，用于计算并绘制资本市场线。

    Attributes:
        tickers (List[str]): 股票代码列表。
        data (pd.DataFrame): 股票价格数据。
        returns (pd.DataFrame): 股票收益率数据。
        risk_free_rate (float): 无风险利率。
    """

    def __init__(self, tickers: List[str], risk_free_rate: float):
        """初始化资本市场线类。

        Args:
            tickers (List[str]): 股票代码列表。
            risk_free_rate (float): 无风险利率，年化利率（如0.01表示1%）。
        """
        self.tickers = tickers
        self.risk_free_rate = risk_free_rate
        self.data = self._download_data()
        self.returns = self._calculate_returns()
        self.num_assets = len(tickers)
        self.weights = np.zeros(self.num_assets)
        self.optimal_portfolio = {}
        self.portfolios = pd.DataFrame()

    def _download_data(self) -> pd.DataFrame:
        """下载股票价格数据。

        Returns:
            pd.DataFrame: 收盘价数据。
        """
        data = yf.download(self.tickers, period='1y')['Adj Close']
        return data

    def _calculate_returns(self) -> pd.DataFrame:
        """计算股票的每日收益率。

        Returns:
            pd.DataFrame: 每日收益率数据。
        """
        returns = self.data.pct_change().dropna()
        return returns

    def simulate_portfolios(self, num_portfolios: int = 50000):
        """模拟随机投资组合，计算预期收益、风险和夏普比率。

        Args:
            num_portfolios (int, optional): 模拟的投资组合数量。默认为50000。
        """
        results = np.zeros((num_portfolios, 3))
        weights_record = []

        for i in trange(num_portfolios):
            # 随机生成权重，确保权重之和为1
            weights = np.random.dirichlet(np.ones(self.num_assets))
            weights_record.append(weights)

            # 计算组合的预期收益和标准差
            portfolio_return = np.sum(self.returns.mean() * weights) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std

            # 保存结果
            results[i, 0] = portfolio_std
            results[i, 1] = portfolio_return
            results[i, 2] = sharpe_ratio

        # 创建数据框保存结果
        self.portfolios = pd.DataFrame(results, columns=['Standard Deviation', 'Expected Return', 'Sharpe Ratio'])
        self.portfolios['Weights'] = weights_record

    def find_optimal_portfolio(self):
        """找到具有最高夏普比率的投资组合，即市场组合。"""
        max_sharpe_idx = self.portfolios['Sharpe Ratio'].idxmax()
        self.optimal_portfolio = self.portfolios.loc[max_sharpe_idx]
        self.weights = self.optimal_portfolio['Weights']
        logger.log_info("市场组合的权重：")
        for ticker, weight in zip(self.tickers, self.weights):
            logger.log_info(f"{ticker}: {weight:.2%}")

    def plot_cml(self):
        """绘制资本市场线、有效边界和投资组合分布。"""
        plt.figure(figsize=(12, 8))
        # 绘制散点图
        plt.scatter(self.portfolios['Standard Deviation'], self.portfolios['Expected Return'],
                    c=self.portfolios['Sharpe Ratio'], cmap='viridis', marker='o', s=10, alpha=0.3)
        plt.colorbar(label='夏普比率')
        plt.xlabel('年化标准差')
        plt.ylabel('年化预期收益')
        plt.title('资本市场线（CML）与有效边界')

        # 绘制资本市场线
        max_sharpe_ratio = self.optimal_portfolio['Sharpe Ratio']
        slope = max_sharpe_ratio
        intercept = self.risk_free_rate
        x = np.linspace(0, self.portfolios['Standard Deviation'].max(), 100)
        y = intercept + slope * x
        plt.plot(x, y, 'r--', linewidth=2, label='资本市场线（CML）')

        # 标记市场组合
        plt.scatter(self.optimal_portfolio['Standard Deviation'], self.optimal_portfolio['Expected Return'],
                    marker='*', color='r', s=500, label='市场组合')
        plt.xlabel('年化标准差')
        plt.ylabel('年化预期收益')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(BASIC_IMAGE_DIR, f'capital_market_line.png')
        )
        plt.close()

    def run(self):
        """执行完整的 CML 分析流程。"""
        self.simulate_portfolios()
        logger.log_info("CML 分析已完成。")
        self.find_optimal_portfolio()
        logger.log_info("市场组合已找到。")
        self.plot_cml()
        logger.log_info("CML 图表已保存。")

if __name__ == "__main__":
    # 设置参数
    tickers_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    risk_free_rate = 0.01  # 无风险利率

    cml = CapitalMarketLine(tickers=tickers_list, risk_free_rate=risk_free_rate)
    cml.run()