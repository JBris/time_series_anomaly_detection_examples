import os

def set(k, v):
    try:
        r = getClient()
        res = r.set(k, v)
        r.close()
        return res
    except:
        return False

def get(k):
    try:
        r = getClient()
        v = r.get(k)
        r.close()
        return v
    except:
        return None

def getClient():
    redis_host = os.getenv('PYTHON_REDIS_HOST')
    if redis_host is None: redis_host = 'localhost'
    redis_port= os.getenv('PYTHON_REDIS_PORT')
    if redis_port is None: redis_port = 6379
    
    import redis
    return redis.Redis(host=redis_host, port=redis_port, decode_responses=True)  
    