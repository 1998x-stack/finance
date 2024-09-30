import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import redis

def get_redis_connection():
    """
    获取 Redis 连接。

    Returns:
        redis.Redis: Redis 连接实例。
    """
    return redis.Redis(host='localhost', port=6379, db=0)
