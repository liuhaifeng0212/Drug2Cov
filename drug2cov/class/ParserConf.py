'''
    author: Haifeng Liu
    e-mail: liuhaifeng0212@qq.com
    released date: 24/01/2022
'''



#import configParser as cp
import re, os
try:
    import configparser as cp
except:
    from six.moves import configparser as cp

class ParserConf():

    def __init__(self, args):
        self.args = args
        # self.data_name = 'yelp'
    def processValue(self, key, value):
        #print(key, value)
        tmp = value.split(' ')
        dtype = tmp[0]
        value = tmp[1:]
        #print(dtype, value)

        if value != None:
            if dtype == 'string':
                self.conf_dict[key] = vars(self)[key] = value[0]
            elif dtype == 'int':
                self.conf_dict[key] = vars(self)[key] = int(value[0])
            elif dtype == 'float':
                self.conf_dict[key] = vars(self)[key] = float(value[0])
            elif dtype == 'list':
                self.conf_dict[key] = vars(self)[key] = [i for i in value]
            elif dtype == 'int_list':
                self.conf_dict[key] = vars(self)[key] = [int(i) for i in value]
            elif dtype == 'float_list':
                self.conf_dict[key] = vars(self)[key] = [float(i) for i in value]
        else:
            print('%s value is None' % key)

    def parserConf(self):  # get model parameters

        self.data_dir = os.path.join(os.getcwd(), 'data/%s' % self.args.data_name)
        self.links_filename = os.path.join(os.getcwd(), 'data/%s/%s.links' % (self.args.data_name, self.args.data_name))
        self.user_review_vector_matrix = os.path.join(os.getcwd(), 'data/%s/user_vector.npy' % self.args.data_name)
        self.item_review_vector_matrix = os.path.join(os.getcwd(), 'data/%s/item_vector.npy' % self.args.data_name)
        self.pre_model = os.path.join(os.getcwd(), 'pretrain/%s/%s' % (self.args.data_name, self.pre_model))
