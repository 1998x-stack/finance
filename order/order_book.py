import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from dataclasses import dataclass, field
from datetime import datetime
import redis
import json

@dataclass(order=True)
class Order:
    """
    订单类，表示一个买入或卖出的订单。

    Attributes:
        price (float): 订单价格。
        quantity (int): 订单数量。
        order_type (str): 订单类型，'buy' 或 'sell'。
        timestamp (datetime): 订单时间戳。
        order_id (str): 订单唯一标识符。
    """
    price: float
    timestamp: datetime
    order_id: str
    quantity: int = field(compare=False)
    order_type: str = field(compare=False)

    def to_dict(self) -> dict:
        """
        将订单转换为字典格式，便于存储。

        Returns:
            dict: 订单的字典表示。
        """
        return {
            'price': self.price,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
        }

    @staticmethod
    def from_dict(data: dict) -> 'Order':
        """
        从字典数据创建订单对象。

        Args:
            data (dict): 订单的字典数据。

        Returns:
            Order: 订单对象。
        """
        return Order(
            price=data['price'],
            quantity=data['quantity'],
            order_type=data['order_type'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            order_id=data['order_id']
        )

class OrderBook:
    """
    订单簿类，维护买入和卖出订单，并执行撮合逻辑。

    Attributes:
        redis_conn (redis.Redis): Redis 连接实例。
    """
    def __init__(self, redis_conn: redis.Redis):
        """初始化订单簿。"""
        self.redis_conn = redis_conn

    def add_order(self, order: Order) -> None:
        """
        添加新订单到订单簿。

        Args:
            order (Order): 要添加的订单对象。
        """
        order_data = json.dumps(order.to_dict())
        if order.order_type == 'buy':
            # 使用有序集合存储买入订单，按价格从高到低排序
            self.redis_conn.zadd('buy_orders', {order_data: -order.price})
            print(f"添加买入订单：{order}")
        elif order.order_type == 'sell':
            # 使用有序集合存储卖出订单，按价格从低到高排序
            self.redis_conn.zadd('sell_orders', {order_data: order.price})
            print(f"添加卖出订单：{order}")
        else:
            raise ValueError("订单类型必须是 'buy' 或 'sell'。")

    def match_orders(self) -> None:
        """根据价格和时间优先原则撮合买卖订单。"""
        while True:
            buy_order_data = self.redis_conn.zrange('buy_orders', 0, 0)
            sell_order_data = self.redis_conn.zrange('sell_orders', 0, 0)
            if not buy_order_data or not sell_order_data:
                break
            buy_order = Order.from_dict(json.loads(buy_order_data[0]))
            sell_order = Order.from_dict(json.loads(sell_order_data[0]))
            if buy_order.price >= sell_order.price:
                traded_quantity = min(buy_order.quantity, sell_order.quantity)
                trade_price = sell_order.price  # 按卖出价格成交

                print(f"成交：买方订单ID {buy_order.order_id} 与 卖方订单ID {sell_order.order_id}，"
                      f"价格 {trade_price}，数量 {traded_quantity}")

                # 更新订单数量或移除已完成的订单
                if buy_order.quantity > traded_quantity:
                    buy_order.quantity -= traded_quantity
                    self.redis_conn.zrem('buy_orders', buy_order_data[0])
                    self.add_order(buy_order)
                else:
                    self.redis_conn.zrem('buy_orders', buy_order_data[0])

                if sell_order.quantity > traded_quantity:
                    sell_order.quantity -= traded_quantity
                    self.redis_conn.zrem('sell_orders', sell_order_data[0])
                    self.add_order(sell_order)
                else:
                    self.redis_conn.zrem('sell_orders', sell_order_data[0])
            else:
                break  # 无法匹配更多订单

    def call_auction(self) -> None:
        """集合竞价，确定能最大化成交量的价格。"""
        # 获取所有订单
        buy_orders_data = self.redis_conn.zrange('buy_orders', 0, -1)
        sell_orders_data = self.redis_conn.zrange('sell_orders', 0, -1)
        buy_orders = [Order.from_dict(json.loads(data)) for data in buy_orders_data]
        sell_orders = [Order.from_dict(json.loads(data)) for data in sell_orders_data]
        prices = sorted({order.price for order in buy_orders + sell_orders})

        max_volume = 0
        final_price = 0.0

        for price in prices:
            buy_volume = sum(order.quantity for order in buy_orders if order.price >= price)
            sell_volume = sum(order.quantity for order in sell_orders if order.price <= price)
            traded_volume = min(buy_volume, sell_volume)

            if traded_volume > max_volume:
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
        # 获取符合条件的订单
        buy_orders_data = self.redis_conn.zrangebyscore('buy_orders', min=-float('inf'), max=-price)
        sell_orders_data = self.redis_conn.zrangebyscore('sell_orders', min=price, max=float('inf'))
        buy_orders = [Order.from_dict(json.loads(data)) for data in buy_orders_data]
        sell_orders = [Order.from_dict(json.loads(data)) for data in sell_orders_data]

        # 按价格和时间排序
        buy_orders.sort(key=lambda o: (-o.price, o.timestamp))
        sell_orders.sort(key=lambda o: (o.price, o.timestamp))

        # 开始撮合
        while buy_orders and sell_orders:
            buy_order = buy_orders[0]
            sell_order = sell_orders[0]
            traded_quantity = min(buy_order.quantity, sell_order.quantity)

            print(f"集合竞价成交：买方订单ID {buy_order.order_id} 与 卖方订单ID {sell_order.order_id}，"
                  f"价格 {price}，数量 {traded_quantity}")

            buy_order.quantity -= traded_quantity
            sell_order.quantity -= traded_quantity

            # 更新或移除买方订单
            self.redis_conn.zrem('buy_orders', json.dumps(buy_order.to_dict()))
            if buy_order.quantity > 0:
                self.add_order(buy_order)
                buy_orders[0] = buy_order
            else:
                buy_orders.pop(0)

            # 更新或移除卖方订单
            self.redis_conn.zrem('sell_orders', json.dumps(sell_order.to_dict()))
            if sell_order.quantity > 0:
                self.add_order(sell_order)
                sell_orders[0] = sell_order
            else:
                sell_orders.pop(0)