#!/usr/bin/env python

import redis
r = redis.Redis(host='redis', port=6379, decode_responses=True)
r.set('foo', 'bar')

res = r.get('foo')
print(str(res))
r.close()