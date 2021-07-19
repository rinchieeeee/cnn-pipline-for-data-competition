import math
import time
from contextlib import contextmanager

class AverageMeter(object):
    """
    objectは別に書かなくてもOK. ただし, recommendされているらしいぞ...
    Computes and stores the average and current value.
    """
    
    def __init__(self):
        self.reset()  # 最初に呼び出されるごとに初期化
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def Second2Minutes(second):
    """
    return example ...
    if m = 3, s = 34 then, return  3m, 34s.
    """
    m = math.floor(second / 60)
    second -= m * 60
    return "%dm %ds" % (m, second)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s/(percent)  # elapsed time
    rs = es - s   # remain time
    return "%s (remain %s)" % (Second2Minutes(second = s), Second2Minutes(second = rs))


# 時間管理
@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(config):
    """
    config is equal to config_general
    """
    log_file = config["output_dir"] + "/" + "train.log"
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename = log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger