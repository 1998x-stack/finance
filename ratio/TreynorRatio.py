import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import List
import warnings

warnings.filterwarnings('ignore')


class TreynorRatioCalculator:
    """特雷诺比率计算器，用于计算投资组合的特雷诺比率并进行可视化。

    Attributes:
        tickers (List[str]): 股票代码列表。
        weights (List[float]): 投资组合中各资产的权重。
        market_ticker (str): 市场指数代码。
        start_date (str): 数据起始日期。
        end_date (str): 数据结束日期。
        risk_free_rate (float): 无风险利率，年化利率（如0.01表示1%）。
        data (pd.DataFrame): 股票和市场指数的收盘价数据。
        returns (pd.DataFrame): 股票和市场指数的收益率数据。
        portfolio_returns (pd.Series): 投资组合的收益率序列。
        portfolio_beta (float): 投资组合的β系数。
        treynor_ratio (float): 投资组合的特雷诺比率。
    """

    def __init__(self, tickers: List[str], weights: List[float], market_ticker: str,
                 start_date: str, end_date: str, risk_free_rate: float):
        """初始化特雷诺比率计算器。

        Args:
            tickers (List[str]): 股票代码列表。
            weights (List[float]): 投资组合中各资产的权重，权重之和应为1。
            market_ticker (str): 市场指数代码。
            start_date (str): 数据起始日期，格式为'YYYY-MM-DD'。
            end_date (str): 数据结束日期，格式为'YYYY-MM-DD'。
            risk_free_rate (float): 无风险利率，年化利率（如0.01表示1%）。
        """
        self.tickers = tickers
        self.weights = weights
        self.market_ticker = market_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.data = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.portfolio_returns = pd.Series()
        self.portfolio_beta = 0.0
        self.treynor_ratio = 0.0

        self._validate_inputs()

    def _validate_inputs(self):
        """验证输入参数的有效性。"""
        if len(self.tickers) != len(self.weights):
            raise ValueError("股票代码列表和权重列表的长度必须相同。")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("权重之和必须等于1。")
        if self.risk_free_rate < 0 or self.risk_free_rate > 1:
            raise ValueError("无风险利率必须在0和1之间。")

    def fetch_data(self):
        """获取股票和市场指数的收盘价数据。"""
        # 获取股票数据
        stock_data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        # 获取市场指数数据
        market_data = yf.download(self.market_ticker, start=self.start_date, end=self.end_date)['Adj Close']

        # 合并数据
        self.data = pd.concat([stock_data, market_data.rename(self.market_ticker)], axis=1)
        self.data.dropna(inplace=True)
        print("已成功获取股票和市场指数数据。")

    def calculate_returns(self):
        """计算股票和市场指数的收益率。"""
        self.returns = self.data.pct_change().dropna()
        # 计算投资组合收益率
        self.portfolio_returns = (self.returns[self.tickers] * self.weights).sum(axis=1)
        print("已计算收益率。")

    def calculate_portfolio_beta(self):
        """计算投资组合的β系数。"""
        # 准备回归模型的数据
        X = self.returns[self.market_ticker]
        Y = self.portfolio_returns

        # 添加常数项用于回归
        X = sm.add_constant(X)

        # 线性回归
        model = sm.OLS(Y, X).fit()
        self.portfolio_beta = model.params[1]
        print(f"投资组合的β系数为: {self.portfolio_beta:.4f}")

    def calculate_treynor_ratio(self):
        """计算投资组合的特雷诺比率。"""
        # 计算平均收益率（年化）
        portfolio_mean_return = self.portfolio_returns.mean() * 252
        # 无风险利率
        risk_free_rate = self.risk_free_rate
        # 检查β系数是否为零
        if self.portfolio_beta == 0:
            print("投资组合的β系数为零，无法计算特雷诺比率。")
            self.treynor_ratio = None
        else:
            # 计算特雷诺比率
            self.treynor_ratio = (portfolio_mean_return - risk_free_rate) / self.portfolio_beta
            print(f"投资组合的特雷诺比率为: {self.treynor_ratio:.4f}")

    def plot_performance(self):
        """绘制投资组合与市场指数的收益率比较。"""
        cumulative_portfolio_returns = (1 + self.portfolio_returns).cumprod()
        cumulative_market_returns = (1 + self.returns[self.market_ticker]).cumprod()

        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_portfolio_returns, label='投资组合')
        plt.plot(cumulative_market_returns, label=f'市场指数 ({self.market_ticker})')
        plt.title('投资组合与市场指数的累计收益率比较')
        plt.xlabel('日期')
        plt.ylabel('累计收益率')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        """执行完整的特雷诺比率计算流程。"""
        self.fetch_data()
        self.calculate_returns()
        self.calculate_portfolio_beta()
        self.calculate_treynor_ratio()
        self.plot_performance()


if __name__ == "__main__":
    # 设置参数
    tickers_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    weights_list = [0.25, 0.25, 0.25, 0.25]  # 等权重分配
    market_ticker = '^GSPC'  # 标普500指数
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    risk_free_rate = 0.01  # 无风险利率

    treynor_calculator = TreynorRatioCalculator(
        tickers=tickers_list,
        weights=weights_list,
        market_ticker=market_ticker,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate
    )
    treynor_calculator.run()