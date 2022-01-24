'''
    author: Haifeng Liu
    e-mail: liuhaifeng0212@qq.com
    released date: 24/01/2022
'''

import os
from time import time
from DataModule import DataModule

class DataUtil():
    def __init__(self,args):

        self.args = args
        #print('DataUtil, Line12, test- conf data_dir:%s' % self.conf.data_dir)

    def initializeRankingHandle(self):
        #t0 = time()
        self.createTrainHandle()
        self.createEvaluateHandle()
        #t1 = time()
        #print('Prepare data cost:%.4fs' % (t1 - t0))
    
    def createTrainHandle(self):
        data_dir = os.path.join(os.getcwd(), 'dataset/%s' % self.args.data_name)
        train_filename = "%s/%s.train.rating" % (data_dir, self.args.data_name)
        val_filename = "%s/%s.val.rating" % (data_dir, self.args.data_name)
        if self.args.case_study_flag:
            test_filename = "%s/%s_%s.%s.test.rating" % (data_dir, self.args.sparse_left,self.args.sparse_right,self.args.data_name)
        else:
            test_filename = "%s/%s.test.rating" % (data_dir, self.args.data_name)

        self.train = DataModule(self.args, train_filename)
        self.val = DataModule(self.args, val_filename)
        self.test = DataModule(self.args, test_filename)

    def createEvaluateHandle(self):
        # data_dir = self.conf.data_dir
        data_dir = os.path.join(os.getcwd(), 'dataset/%s' % self.args.data_name)

        val_filename = "%s/%s.val.rating" % (data_dir, self.args.data_name)
        if self.args.case_study_flag:
            test_filename = "%s/%s_%s.%s.test.rating" % (data_dir, self.args.sparse_left,self.args.sparse_right,self.args.data_name)
        else:
            test_filename = "%s/%s.test.rating" % (data_dir, self.args.data_name)

        self.val_eva = DataModule(self.args, val_filename)
        self.test_eva = DataModule(self.args, test_filename)
