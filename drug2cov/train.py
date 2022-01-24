'''
    author: Haifeng Liu
    e-mail: liuhaifeng0212@qq.com
    released date: 24/01/2022
'''

import os, sys, shutil

from time import time
import numpy as np
import tensorflow as tf
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #ignore the warnings 

from Logging import Logging

def start(args, data, model, evaluate):
    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # define log name 
    log_path = os.path.join(os.getcwd(), 'log/%s_%s.log' % (args.data_name, args.model_name))

    # start to prepare data for training and evaluating
    data.initializeRankingHandle()

    d_train, d_val, d_test, d_test_eva = data.train, data.val, data.test, data.test_eva

    print('System start to load data...')
    t0 = time()
    d_train.initializeRankingTrain()
    d_val.initializeRankingVT()
    d_test.initializeRankingVT()
    d_test_eva.initalizeRankingEva()
    t1 = time()
    print('Data has been loaded successfully, cost:%.4fs' % (t1 - t0))

    # prepare model necessary data.
    data_dict = d_train.prepareModelSupplement(model)
    model.inputSupply(data_dict)                            # generate sparse Tensor
    model.startConstructGraph()
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    # standard tensorflow running environment initialize
    tf_conf = tf.ConfigProto()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    tf_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_conf)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(args.log_dir + '/board/' +args.data_name+'/'+TIMESTAMP+'/', sess.graph)
    sess.run(model.init)


    #if conf.pretrain_flag == 1:       # reload model
        #model.saver.restore(sess, conf.pre_model)

    # set debug_flag=0, doesn't print any results
    log = Logging(log_path)
    print()
    log.record('Following will output the evaluation of the model:') # record

    def read_idmap(file):
        re_dict = {}
        with open(file, 'r') as f:
            for line in f:
                a, b = line.strip().split('\t')
                re_dict[a] = int(b)
        return re_dict, len(re_dict)
    def read_drug_engname(file):
        re_dict = {}
        with open(file, 'r') as f:
            for line in f:
                a, b = line.strip().split('\t')
                re_dict[a] = b
        return re_dict
    drug_repo2name = read_drug_engname(os.path.join(os.getcwd(), 'dataset/%s/drug_2namemap.csv' % args.data_name))

    disease_dict, disease_num = read_idmap(
        os.path.join(os.getcwd(), 'dataset/%s/disease_idmap.csv' % args.data_name))
    drug_dict, drug_num = read_idmap(os.path.join(os.getcwd(), 'dataset/%s/drug_idmap.csv' % args.data_name))
    covid_id = disease_dict['COVID-19']

    best_auc = 0
    # Start Training !!!
    for epoch in range(1, args.epochs+1):
        # optimize model with training data and compute train loss
        tmp_train_loss = []
        t0 = time()
        tmp_ssl_loss = []
        #tmp_total_list = []
        while d_train.terminal_flag:
            d_train.getTrainRankingBatch()
            d_train.linkedMap()

            train_feed_dict = {}
            for (key, value) in model.map_dict['train'].items():


                train_feed_dict[key] = d_train.data_dict[value]
            [sub_train_loss, _,summary] = sess.run( \
                [model.map_dict['out']['train'], model.opt,merged], feed_dict=train_feed_dict)
            tmp_train_loss.append(sub_train_loss[0])
            ssl_loss = sub_train_loss[1]
            writer.add_summary(summary,epoch)
            tmp_ssl_loss.append(ssl_loss)
        print('ssl_loss',np.mean(tmp_ssl_loss))
        train_loss = np.mean(tmp_train_loss)
        t1 = time()

        if epoch % args.report==0:

            # compute val loss and test loss
            d_val.getVTRankingOneBatch()
            d_val.linkedMap()
            val_feed_dict = {}
            for (key, value) in model.map_dict['val'].items():

                val_feed_dict[key] = d_val.data_dict[value]
            val_loss = sess.run(model.map_dict['out']['val'], feed_dict=val_feed_dict)
            d_test.getVTRankingOneBatch()
            d_test.linkedMap()
            test_feed_dict = {}
            for (key, value) in model.map_dict['test'].items():
                test_feed_dict[key] = d_test.data_dict[value]
            test_loss = sess.run(model.map_dict['out']['test'], feed_dict=test_feed_dict)
            t2 = time()
            # start evaluate model performance, hr and ndcg
            def getPositivePredictions():
                d_test_eva.getEvaPositiveBatch()
                d_test_eva.linkedRankingEvaMap()
                eva_feed_dict = {}
                for (key, value) in model.map_dict['eva'].items():
                    eva_feed_dict[key] = d_test_eva.data_dict[value]
                    # print('postive eva_feed_dict',key,value,eva_feed_dict[key].shape)
                positive_predictions = sess.run(
                    model.map_dict['out']['eva'],
                    feed_dict=eva_feed_dict
                )

                return positive_predictions  #positive score

            def out_cov_drug(covid_drug_sim,cov_drug_id,out_file):
                drug_id_name = dict(zip(drug_dict.values(),drug_dict.keys()))
                drug_real_id = [drug_id_name[x] for x in cov_drug_id]
                drug_sim = dict(zip(drug_real_id,covid_drug_sim))
                sort_drug = sorted(drug_sim.items(),key=lambda x:x[1],reverse=True)
                print([(x[0],drug_repo2name[x[0]],round(x[1],4)) for x in sort_drug[:4]])
                with open(out_file,'w',encoding='utf8') as f:
                    f.write('drugbankid\tdrugname\tsimilarity\n')
                    for i in sort_drug:
                        if i[0] in drug_repo2name.keys():
                            f.write('{}\t{}\t{}\n'.format(i[0],drug_repo2name[i[0]],i[1]))
                        else:
                            f.write('{}\tneed check it\t{}\n'.format(i[0], i[1]))

            def getNegativePredictions():
                disease_neg_pre = {}
                terminal_flag = 1
                while terminal_flag:
                    batch_disease_list, terminal_flag,cov_drug,neg_index_dict = d_test_eva.getEvaRankingBatch()
                    d_test_eva.linkedRankingEvaMap()
                    eva_feed_dict = {}
                    for (key, value) in model.map_dict['eva'].items():
                        eva_feed_dict[key] = d_test_eva.data_dict[value]

                    index = 0
                    negative_predictions = sess.run(
                            model.map_dict['out']['eva'],
                            feed_dict=eva_feed_dict
                        )
                    for  disease in batch_disease_list:
                        disease_neg_pre[disease] = list(np.concatenate(negative_predictions[neg_index_dict[disease]]))
                        if disease == covid_id:
                            covid_drug_sim = disease_neg_pre[disease]
                            cov_drug = cov_drug
                            out_file = os.path.join(os.getcwd(), './covid_rec/Epoch_{}_COVID推荐药物.csv'.format(epoch))
                            out_cov_drug(covid_drug_sim, cov_drug, out_file)

                return disease_neg_pre                 # {user:[neg_pre]}

            tt2 = time()

            pos_index_dict = d_test_eva.eva_pos_index_dict

            positive_predictions = getPositivePredictions()     #
            negative_predictions = getNegativePredictions()     #
            d_test_eva.index = 0    # !!!important, prepare for new batch

            auc,aupr = evaluate.evaluateaucPerformance(pos_index_dict,positive_predictions,negative_predictions)

            tt3 = time()

            # print log to console and log_file
            log.record('Epoch:%d, compute loss cost:%.4fs, train loss:%.4f, val loss:%.4f, test loss:%.4f' % \
                (epoch, (t2-t0), train_loss, val_loss, test_loss))
            log.record('Evaluate cost:%.4fs, auc:%.4f, aupr:%.4f' % ((tt3-tt2), auc, aupr))
            # train_writer.close()
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                best_aupr  = aupr
        print("best auc epoch is {} auc score {} aupr score {}".format(best_epoch,best_auc,best_aupr))

        ## reset train data pointer, and generate new negative data
        d_train.generateTrainNegative()         # sample negative
