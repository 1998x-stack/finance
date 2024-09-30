import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import yfinance as yf
import numpy as np
import scipy.stats as stats
import datetime
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config.config import FONT_PATH, BASIC_IMAGE_DIR

font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams['font.family'] = 'SimHei'

class BlackScholesModel:
    """布莱克-舒尔茨期权定价模型类，用于定价欧式期权。

    Attributes:
        stock_ticker (str): 股票代码。
        option_type (str): 期权类型，'call'或'put'。
        strike_price (float): 期权执行价格。
        maturity_date (datetime.date): 期权到期日。
        risk_free_rate (float): 无风险利率。
        volatility (Optional[float]): 波动率，如果为None则计算历史波动率。
        current_stock_price (float): 当前股票价格。
        time_to_maturity (float): 距离到期日的时间（以年计）。
    """

    def __init__(self, stock_ticker: str, option_type: str, strike_price: float,
                 maturity_date: datetime.date, risk_free_rate: float,
                 volatility: Optional[float] = None):
        """初始化Black-Scholes模型。

        Args:
            stock_ticker (str): 股票代码。
            option_type (str): 期权类型，'call'或'put'。
            strike_price (float): 期权执行价格。
            maturity_date (datetime.date): 期权到期日。
            risk_free_rate (float): 无风险利率，年化利率（如0.01表示1%）。
            volatility (Optional[float], optional): 波动率。如果为None，则计算历史波动率。默认为None。
        """
        self.stock_ticker = stock_ticker
        self.option_type = option_type.lower()
        self.strike_price = strike_price
        self.maturity_date = maturity_date
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.current_stock_price = 0.0
        self.time_to_maturity = 0.0

    def fetch_data(self):
        """获取股票价格数据并计算必要参数。"""
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=90)
        stock_data = yf.download(self.stock_ticker, start=start_date, end=end_date)

        # 计算当前股票价格
        self.current_stock_price = stock_data['Adj Close'][-1]

        # 计算距离到期日的时间（以年计）
        self.time_to_maturity = (self.maturity_date - end_date).days / 365

        # 如果未提供波动率，则计算历史波动率
        if self.volatility is None:
            log_returns = np.log(stock_data['Adj Close'] / stock_data['Adj Close'].shift(1)).dropna()
            self.volatility = np.std(log_returns) * np.sqrt(252)

    def calculate_option_price(self) -> float:
        """计算期权价格。

        Returns:
            float: 期权价格。
        """
        S = self.current_stock_price
        K = self.strike_price
        r = self.risk_free_rate
        T = self.time_to_maturity
        sigma = self.volatility

        # 处理到期时间为负的情况
        if T <= 0:
            if self.option_type == 'call':
                return max(0.0, S - K)
            elif self.option_type == 'put':
                return max(0.0, K - S)
            else:
                raise ValueError("期权类型必须为 'call' 或 'put'。")
        else:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if self.option_type == 'call':
                option_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            elif self.option_type == 'put':
                option_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            else:
                raise ValueError("期权类型必须为 'call' 或 'put'。")
            return option_price

    def plot_option_values(self):
        """绘制不同股票价格下的期权价值曲线。"""
        S = np.linspace(0.5 * self.current_stock_price, 1.5 * self.current_stock_price, 100)
        option_values = []

        for s in S:
            d1 = (np.log(s / self.strike_price) + (self.risk_free_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (self.volatility * np.sqrt(self.time_to_maturity))
            d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)
            if self.option_type == 'call':
                price = s * stats.norm.cdf(d1) - self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * stats.norm.cdf(d2)
            else:
                price = self.strike_price * np.exp(-self.risk_free_rate * self.time_to_maturity) * stats.norm.cdf(-d2) - s * stats.norm.cdf(-d1)
            option_values.append(price)

        plt.figure(figsize=(10, 6))
        plt.plot(S, option_values, label='期权价值曲线')
        plt.xlabel('股票价格')
        plt.ylabel('期权价值')
        plt.title(f'{self.stock_ticker} 期权价值曲线 ({self.option_type.capitalize()} Option)')
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(BASIC_IMAGE_DIR, f'{self.stock_ticker}_{self.option_type}_option_value_curve.png')
        )

    def run(self):
        """执行模型计算和可视化。"""
        self.fetch_data()
        option_price = self.calculate_option_price()
        print(f"当前股票价格: {self.current_stock_price:.2f}")
        print(f"计算得到的期权价格为: {option_price:.2f}")
        self.plot_option_values()

if __name__ == "__main__":
    # 设置参数
    stock_ticker = 'AAPL'  # 股票代码
    option_type = 'put'   # 期权类型 'call' 或 'put'
    strike_price = 150.0   # 期权执行价格
    maturity_date = datetime.date.today() + datetime.timedelta(days=90)  # 到期日
    risk_free_rate = 0.01  # 无风险利率

    bs_model = BlackScholesModel(
        stock_ticker=stock_ticker,
        option_type=option_type,
        strike_price=strike_price,
        maturity_date=maturity_date,
        risk_free_rate=risk_free_rate,
        volatility=None  # 如果为None，则计算历史波动率
    )

    bs_model.run()