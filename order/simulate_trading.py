import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from datetime import datetime
from typing import List, Dict
import uuid
from order_book import Order, OrderBook
from redis_client import get_redis_connection

def simulate_trading(data_stream: List[Dict]) -> None:
    """
    模拟交易，处理数据流中的订单。

    Args:
        data_stream (List[Dict]): 包含订单数据的列表。
    """
    redis_conn = get_redis_connection()
    # 清空 Redis 中的订单簿数据
    redis_conn.delete('buy_orders', 'sell_orders')
    order_book = OrderBook(redis_conn)
    for data in data_stream:
        timestamp = data['timestamp']
        order = Order(
            price=data['price'],
            quantity=data['quantity'],
            order_type=data['order_type'],
            timestamp=timestamp,
            order_id=str(uuid.uuid4())
        )
        order_book.add_order(order)

        if datetime.strptime('09:15:00', '%H:%M:%S').time() <= timestamp.time() <= datetime.strptime('09:25:00', '%H:%M:%S').time():
            # 集合竞价时间段，暂不撮合
            continue
        else:
            # 连续竞价时间段，尝试撮合
            order_book.match_orders()

    # 集合竞价结束，执行集合竞价撮合
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