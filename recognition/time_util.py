import time
import logging

class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_t = time.time()

    def __exit__(self, t, v, traceback):
        self.end_t = time.time()
        print("== {} exec time is {} m-seconds ======".format(self.name, 1000*(self.end_t-self.start_t)))
        #logging.info("== {} exec cost time {} seconds ===".format(self.name, self.end_t-self.start_t))
