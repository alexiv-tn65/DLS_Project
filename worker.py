import os
import redis
from rq import Connection, Queue, SimpleWorker

import platform
import signal
# if platform.system() != 'Linux':
#     signal.SIGHUP = 1
#     signal.SIGALRM = 1

listen = ["default"]

# REDIS_ENDPOINT = os.getenv("REDIS_ENDPOINT", default="localhost")
# REDIS_PORT = os.getenv("REDIS_PORT", default="6379")

REDIS_ENDPOINT = "localhost"
REDIS_PORT = "6379"

redis_url = "redis://" + REDIS_ENDPOINT + ":" + REDIS_PORT

connect = redis.from_url(redis_url)

if __name__ == "__main__":
    with Connection(connect):
        if platform.system() != 'Linux':
            import rq_win
            worker = rq_win.WindowsWorker(list(map(Queue, listen)))
            worker.work()
        else:
            worker = SimpleWorker(list(map(Queue, listen)))
            worker.work()