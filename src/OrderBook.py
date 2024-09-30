import bisect
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass(order=True)
class Order:
    """
    订单类，表示一个买入或卖出的订单。

    Attributes:
        price (float): 订单价格。
        quantity (int): 订单数量。
        order_type (str): 订单类型，'buy' 或 'sell'。
        timestamp (datetime): 订单时间戳。
        order_id (int): 订单唯一标识符。
    """
    price: float
    timestamp: datetime
    order_id: int
    quantity: int = field(compare=False)
    order_type: str = field(compare=False)

class OrderBook:
    """
    订单簿类，维护买入和卖出订单，并执行撮合逻辑。

    Attributes:
        buy_orders (List[Order]): 买入订单列表，按价格从高到低排序。
        sell_orders (List[Order]): 卖出订单列表，按价格从低到高排序。
        order_id_counter (int): 订单ID计数器。
    """
    def __init__(self):
        """初始化订单簿。"""
        self.buy_orders: List[Order] = []
        self.sell_orders: List[Order] = []
        self.order_id_counter: int = 0

    def add_order(self, price: float, quantity: int, order_type: str, timestamp: datetime) -> None:
        """
        添加新订单到订单簿。

        Args:
            price (float): 订单价格。
            quantity (int): 订单数量。
            order_type (str): 订单类型，'buy' 或 'sell'。
            timestamp (datetime): 订单时间戳。
        """
        self.order_id_counter += 1
        order = Order(price=price, quantity=quantity, order_type=order_type,
                      timestamp=timestamp, order_id=self.order_id_counter)
        if order_type == 'buy':
            # 插入买入订单，保持列表按价格从高到低排序
            bisect.insort_left(self.buy_orders, order)
            print(f"添加买入订单：{order}")
        elif order_type == 'sell':
            # 插入卖出订单，保持列表按价格从低到高排序
            bisect.insort_left(self.sell_orders, order)
            print(f"添加卖出订单：{order}")
        else:
            raise ValueError("订单类型必须是 'buy' 或 'sell'。")

    def match_orders(self) -> None:
        """根据价格和时间优先原则撮合买卖订单。"""
        while self.buy_orders and self.sell_orders:
            highest_buy = self.buy_orders[-1]  # 最高买价
            lowest_sell = self.sell_orders[0]  # 最低卖价
            if highest_buy.price >= lowest_sell.price:
                traded_quantity = min(highest_buy.quantity, lowest_sell.quantity)
                trade_price = lowest_sell.price  # 按卖出价格成交

                print(f"成交：买方订单ID {highest_buy.order_id} 与 卖方订单ID {lowest_sell.order_id}，"
                      f"价格 {trade_price}，数量 {traded_quantity}")

                # 更新订单数量或移除已完成的订单
                highest_buy.quantity -= traded_quantity
                lowest_sell.quantity -= traded_quantity
                if highest_buy.quantity == 0:
                    self.buy_orders.pop()
                if lowest_sell.quantity == 0:
                    self.sell_orders.pop(0)
            else:
                break  # 无法匹配更多订单

    def call_auction(self) -> None:
        """集合竞价，确定能最大化成交量的价格。"""
        all_orders = self.buy_orders + self.sell_orders
        prices = sorted({order.price for order in all_orders})

        max_volume = 0
        final_price = 0.0

        for price in prices:
            buy_volume = sum(order.quantity for order in self.buy_orders if order.price >= price)
            sell_volume = sum(order.quantity for order in self.sell_orders if order.price <= price)
            traded_volume = min(buy_volume, sell_volume)

            if traded_volume > max_volume or (traded_volume == max_volume and abs(price - final_price) < 1e-5):
                max_volume = traded_volume
                final_price = price

        print(f"集合竞价成交价：{final_price}，成交量：{max_volume}")
        # 按照最终价格撮合订单
        self._execute_call_auction(final_price)

    def _execute_call_auction(self, price: float) -> None:
        """执行集合竞价撮合逻辑。

        Args:
            price (float): 集合竞价确定的成交价格。
        """
        buy_orders = [order for order in self.buy_orders if order.price >= price]
        sell_orders = [order for order in self.sell_orders if order.price <= price]

        buy_orders.sort(key=lambda o: (o.price, o.timestamp), reverse=True)
        sell_orders.sort(key=lambda o: (o.price, o.timestamp))

        while buy_orders and sell_orders:
            buy_order = buy_orders[0]
            sell_order = sell_orders[0]
            traded_quantity = min(buy_order.quantity, sell_order.quantity)

            print(f"集合竞价成交：买方订单ID {buy_order.order_id} 与 卖方订单ID {sell_order.order_id}，"
                  f"价格 {price}，数量 {traded_quantity}")

            buy_order.quantity -= traded_quantity
            sell_order.quantity -= traded_quantity

            if buy_order.quantity == 0:
                buy_orders.pop(0)
                self.buy_orders.remove(buy_order)
            if sell_order.quantity == 0:
                sell_orders.pop(0)
                self.sell_orders.remove(sell_order)

def simulate_trading(data_stream: List[Dict]) -> None:
    """
    模拟交易，处理数据流中的订单。

    Args:
        data_stream (List[Dict]): 包含订单数据的列表。
    """
    order_book = OrderBook()
    for data in data_stream:
        timestamp = data['timestamp']
        if timestamp.time() >= datetime.strptime('09:15', '%H:%M').time() and \
           timestamp.time() <= datetime.strptime('09:25', '%H:%M').time():
            # 集合竞价时间段
            order_book.add_order(price=data['price'], quantity=data['quantity'],
                                 order_type=data['order_type'], timestamp=timestamp)
        else:
            # 连续竞价时间段
            order_book.add_order(price=data['price'], quantity=data['quantity'],
                                 order_type=data['order_type'], timestamp=timestamp)
            order_book.match_orders()

    # 在集合竞价结束时执行集合竞价
    order_book.call_auction()

# 示例数据流
data_stream = [
    {'price': 10.0, 'quantity': 100, 'order_type': 'buy',
     'timestamp': datetime.strptime('09:16:00', '%H:%M:%S')},
    {'price': 10.5, 'quantity': 50, 'order_type': 'sell',
     'timestamp': datetime.strptime('09:17:00', '%H:%M:%S')},
    {'price': 9.8, 'quantity': 150, 'order_type': 'buy',
     'timestamp': datetime.strptime('09:18:00', '%H:%M:%S')},
    {'price': 10.2, 'quantity': 80, 'order_type': 'sell',
     'timestamp': datetime.strptime('09:19:00', '%H:%M:%S')},
    {'price': 10.0, 'quantity': 70, 'order_type': 'buy',
     'timestamp': datetime.strptime('09:30:00', '%H:%M:%S')},
    {'price': 9.9, 'quantity': 60, 'order_type': 'sell',
     'timestamp': datetime.strptime('09:31:00', '%H:%M:%S')},
]

if __name__ == "__main__":
    simulate_trading(data_stream)