'''
    author: Haifeng Liu
    e-mail: liuhaifeng0212@qq.com
    released date: 24/01/2021
'''

import sys, os, argparse

sys.path.append(os.path.join(os.getcwd(), 'class'))

from ParserConf import ParserConf
from DataUtil import DataUtil
from Evaluate import Evaluate

from drug2cov import drug2cov

def executeTrainModel(args, model_name):


    #print('System start to load TensorFlow graph...')
    model = eval(model_name)

    model = model(args)

    #print('System start to load data...')
    data = DataUtil(args)
    evaluate = Evaluate(args)

    import train as starter
    starter.start(args, data, model, evaluate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Welcome to the Experiment Platform Entry')
    parser.add_argument('--data_name', nargs='?',default='covid', help='dataset dirs')
    parser.add_argument('--model_name', nargs='?',default='drug2cov', help='diffnet, matrix model name')
    parser.add_argument('--gpu', nargs='?',default='3',help='available gpu id')
    parser.add_argument('--dimension', type=int, default=64, help='embedding dim')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs')
    parser.add_argument('--num_evaluate', type=int, default=0, help='evaluate sample size')
    parser.add_argument('--num_procs', type=int, default=16, help='multiprocess')
    parser.add_argument('--topk', type=int, default=1, help='recommendation list')
    parser.add_argument('--evaluate_batch_size', type=int, default=2560, help='evaluate batch size')
    parser.add_argument('--training_batch_size', type=int, default=512, help='train batch size')
    parser.add_argument('--epoch_notice', type=int, default=10, help='  ')
    parser.add_argument('--report', type=int, default=1, help='evaluate report epoch ')
    parser.add_argument('--log_dir', type=str, default='./dataset/pycharm_env2/drug2cov/', help='log dirs')
    # parser.add_argument('--pretrain_flag', type=int, default=0, help='1 载入预训练模型，0 重新训练')
    parser.add_argument('--case_study_flag', type=int, default=0, help='case study')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--fusion_weight', type=float, default=0.1, help='fusion weight')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha weight')
    parser.add_argument('--gcn_weight', type=float, default=1.0, help='drug disease gcn layer weight')



    args = parser.parse_args()

    data_name = args.data_name
    model_name = args.model_name
    device_id = args.gpu
    for arg in vars(args):
        print(arg,getattr(args,arg))
    # print(args)
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = device_id   #gpu id
    # config_path = os.path.join(os.getcwd(), 'conf/%s_%s.ini' % (data_name, model_name))

    executeTrainModel(args, model_name)

