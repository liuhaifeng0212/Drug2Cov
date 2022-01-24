'''
    author: Haifeng Liu
    e-mail: liuhaifeng0212@qq.com
    released date: 24/01/2022
'''

import os, shutil
#import ConfigParser as cp
try:
    import configparser as cp
except:
    from six.moves import configparser as cp


class Logging():
    def __init__(self, filename):
        self.filename = filename
    
    def record(self, str_log):
        filename = self.filename
        print(str_log)
        with open(filename, 'a') as f:
            f.write("%s\r\n" % str_log)
            f.flush()
