import io
from threading import RLock
import logging
import os
import sys


class Stream:
    """
    Simple stream with write and flush operations
    """

    def __init__(self, log, lvl):
        self.buf = io.StringIO()
        self.log = log
        self.lvl = lvl
        self.lock = RLock()

    def write(self, msg):
        with self.lock:
            if msg == '\n':
                self.flush()
            else:
                self.buf.write(msg)

    def flush(self):
        with self.lock:
            self.buf.flush()
            self.log.log(self.lvl, self.buf.getvalue())
            self.buf.truncate(0)
            self.buf.seek(0)


class Log:
    """
    * The idea of this class is to set some log verbosity level to control the output stream
    * This class can have n loggers named vi where i is a number in the range [0, n-1]
    * They are used in the python print statements by setting the 'file' parameter to one of these loggers
    * Each logger has a stdout stream handle by default
    * If the log_verbosity is set to some number x, then the output of all the loggers i where i > x is suppressed
    * Thus, let's say we put log_verbosity 0, this means that all print statements that has:
            print(..., file=self.v1) will not show any output (same with self.v2, self.v3, etc if exists)
    """

    def __init__(self):
        self.v0 = None
        self.v1 = None

    def initialize(self, log_verbosity, log_file=None):
        """
        Initialize the loggers and their handlers (output destination)

        :param log_verbosity: An integer, the log verbosity level
        :param log_file: A string, the log file path
        :return:
        """
        if not 0 <= log_verbosity <= 1:
            raise ValueError('log_verbosity should be either 0 or 1. Received: {}', format(log_verbosity))
        v = [logging.getLogger('v' + str(i)) for i in range(2)]
        logs = []
        if log_file:
            logs.append(log_file)
        logs.append('stdout')
        for l in logs:
            if l == 'stdout':
                handler = logging.StreamHandler(sys.stdout)
            elif os.path.isdir(os.path.dirname(l)):
                handler = logging.FileHandler(l)
            else:
                raise Exception('Invalid log: {}'.format(l))
            handler.setLevel(logging.DEBUG)
            for i in range(log_verbosity + 1):
                if handler not in v[i].handlers:
                    v[i].addHandler(handler)

        # suppress the logs of these loggers
        null = logging.FileHandler(os.devnull)
        for i in range(len(v)):
            v[i].setLevel(logging.DEBUG)
            if not v[i].handlers:
                v[i].addHandler(null)

        # python 'print(...)' expects a Steam for 'file' parameter
        self.v0 = Stream(v[0], logging.DEBUG)
        self.v1 = Stream(v[1], logging.DEBUG)

