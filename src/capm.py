import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import statsmodels.api as sm

from config.config import FONT_PATH, BASIC_IMAGE_DIR

font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = 'SimHei'
class CAPMModel:
    """CAPM模型类，用于计算股票的β系数和预期回报率。

    Attributes:
        stock_ticker (str): 个股的股票代码。
        market_ticker (str): 市场指数的股票代码。
        risk_free_rate (float): 无风险利率。
        start_date (str): 数据起始日期。
        end_date (str): 数据结束日期。
        stock_returns (pd.Series): 个股的收益率序列。
        market_returns (pd.Series): 市场指数的收益率序列。
        beta (float): 计算得到的β系数。
        expected_return (float): 预期回报率。
    """

    def __init__(self, stock_ticker: str, market_ticker: str, risk_free_rate: float,
                 start_date: str, end_date: str):
        """初始化 CAPM 模型类。

        Args:
            stock_ticker (str): 个股的股票代码。
            market_ticker (str): 市场指数的股票代码。
            risk_free_rate (float): 无风险利率，年化利率（如0.02表示2%）。
            start_date (str): 数据起始日期，格式为'YYYY-MM-DD'。
            end_date (str): 数据结束日期，格式为'YYYY-MM-DD'。
        """
        self.stock_ticker = stock_ticker
        self.market_ticker = market_ticker
        self.risk_free_rate = risk_free_rate
        self.start_date = start_date
        self.end_date = end_date
        self.stock_returns = pd.Series()
        self.market_returns = pd.Series()
        self.beta = 0.0
        self.expected_return = 0.0

    def fetch_data(self):
        """获取个股和市场指数的收盘价数据。"""
        # 获取个股数据
        stock_data = yf.download(self.stock_ticker, start=self.start_date, end=self.end_date)
        # 获取市场指数数据
        market_data = yf.download(self.market_ticker, start=self.start_date, end=self.end_date)

        # 计算每日收益率
        self.stock_returns = stock_data['Adj Close'].pct_change().dropna()
        self.market_returns = market_data['Adj Close'].pct_change().dropna()

        # 对齐日期索引
        combined_data = pd.concat([self.stock_returns, self.market_returns], axis=1, join='inner')
        self.stock_returns = combined_data.iloc[:, 0]
        self.market_returns = combined_data.iloc[:, 1]

    def calculate_beta(self):
        """计算个股的β系数。"""
        # 添加常数项用于回归
        X = sm.add_constant(self.market_returns)
        Y = self.stock_returns

        # 线性回归
        model = sm.OLS(Y, X).fit()
        self.beta = model.params[1]
        print(f"计算得到的β系数为: {self.beta:.4f}")

    def calculate_expected_return(self):
        """使用 CAPM 公式计算预期回报率。"""
        market_premium = self.market_returns.mean() * 252 - self.risk_free_rate
        self.expected_return = self.risk_free_rate + self.beta * market_premium
        print(f"根据 CAPM 计算的预期年化回报率为: {self.expected_return:.2%}")
        
    def plot_fitted_curve_with_regression_line(self):
        """绘制拟合曲线和回归线。"""
        # 添加常数项用于回归
        X = sm.add_constant(self.market_returns)
        Y = self.stock_returns

        # 线性回归
        model = sm.OLS(Y, X).fit()

        # 绘制拟合曲线
        plt.figure(figsize=(10, 6))
        plt.scatter(self.market_returns, self.stock_returns, label='数据点')

        # 绘制回归线
        plt.plot(self.market_returns, model.fittedvalues, color='red', label='回归线')

        plt.title('拟合曲线和回归线')
        plt.xlabel('市场指数收益率')
        plt.ylabel('个股收益率')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(
                BASIC_IMAGE_DIR,
                f'fitted_curve_with_regression_line_{self.stock_ticker}.png'
            )
        )
        plt.close()

    def plot_security_market_line(self):
        """绘制证券市场线（SML）。"""
        # 定义β的范围
        beta_values = np.linspace(-1, 2, 100)
        market_premium = self.market_returns.mean() * 252 - self.risk_free_rate
        expected_returns = self.risk_free_rate + beta_values * market_premium

        # 绘制SML
        plt.figure(figsize=(10, 6))
        plt.plot(beta_values, expected_returns, label='证券市场线 (SML)')
        plt.scatter(self.beta, self.expected_return, color='red', label=f'{self.stock_ticker}')
        plt.title('证券市场线 (SML)')
        plt.xlabel('β系数')
        plt.ylabel('预期年化回报率')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(
                BASIC_IMAGE_DIR,
                f'security_market_line_{self.stock_ticker}.png'
            )
        )
        plt.close()

    def run(self):
        """执行CAPM分析的完整流程。"""
        self.fetch_data()
        self.calculate_beta()
        self.calculate_expected_return()
        self.plot_security_market_line()
        self.plot_fitted_curve_with_regression_line()

if __name__ == "__main__":
    # 设置参数
    stock_ticker = 'AAPL'          # 个股代码
    market_ticker = '^GSPC'        # 标普500指数代码
    risk_free_rate = 0.02          # 无风险利率（2%）
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.today() - pd.DateOffset(months=3)).strftime('%Y-%m-%d')

    capm = CAPMModel(stock_ticker, market_ticker, risk_free_rate, start_date, end_date)
    capm.run()