'''
    author: Haifeng Liu
    e-mail: liuhaifeng0212@qq.com
    released date: 24/01/2022
'''

import math
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics

class Evaluate():
    def __init__(self, args):
        self.args = args

    def getIdcg(self, length):
        idcg = 0.0
        for i in range(length):
            idcg = idcg + math.log(2) / math.log(i + 2)
        return idcg

    def getDcg(self, value):
        dcg = math.log(2) / math.log(value + 2)
        return dcg

    def getHr(self, value):
        hit = 1.0
        return hit

    def evaluateaucPerformance(self,pos_index_dict,positive_predictions,negative_predictions):
        num_procs = 4
        disease_list = list(pos_index_dict.keys())
        batch_size = int(len(disease_list) / num_procs)
        hr_list, ndcg_list = [], []
        index = 0
        for _ in range(num_procs+1):
            if index + batch_size < len(disease_list):
                batch_disease_list = disease_list[index:index+batch_size]
                index = index + batch_size
            else:
                batch_disease_list = disease_list[index:len(disease_list)+1]
            # print(454 in batch_disease_list)
            tmp_hr_list, tmp_ndcg_list = self.getaucpro(pos_index_dict, positive_predictions, \
                negative_predictions, batch_disease_list)
            hr_list.extend(tmp_hr_list)
            ndcg_list.extend(tmp_ndcg_list)
        return np.mean(hr_list),np.mean(ndcg_list)

    def getaucpro(self,pos_index_dict, positive_predictions, \
                negative_predictions, batch_disease_list):


        tmp_auc_list, tmp_aupr_list = [], []

        for disease in batch_disease_list:
            real_disease_index_list = pos_index_dict[disease]
            real_disease_rating_list = list(np.concatenate(positive_predictions[real_disease_index_list])) # 数组拆分拼接到一起
            real_disease_rating_list = np.array(real_disease_rating_list)
            real_disease_rating_list[np.isnan(real_disease_rating_list)] = 0

            neg_disease_rating_list = negative_predictions[disease]
            neg_disease_rating_list = np.array(neg_disease_rating_list)
            neg_disease_rating_list[np.isnan(neg_disease_rating_list)] = 0

            positive_length = len(real_disease_rating_list)            #positive num


            eva_label,eva_pre= [],[]

            eva_label.extend([1] *len(real_disease_rating_list))
            eva_label.extend([0]* len(neg_disease_rating_list))
            eva_pre.extend(real_disease_rating_list)
            eva_pre.extend(neg_disease_rating_list)
            try:
                fpr, tpr, thresholds = metrics.roc_curve(eva_label, eva_pre, pos_label=1)
            except:
                print('neg_disease_rating_list',neg_disease_rating_list)
                print(eva_pre)
            tmp_auc = metrics.auc(fpr, tpr)
            tmp_aupr = average_precision_score(eva_label, eva_pre)
            # tmp_auc = roc_auc_score(eva_label,eva_pre)
            # tmp_aupr = average_precision_score(eva_label,eva_pre)
            tmp_auc_list.append(tmp_auc)
            tmp_aupr_list.append(tmp_aupr)
        return  tmp_auc_list,tmp_aupr_list

    def evaluateRankingPerformance(self, evaluate_pos_index_dict,evaluate_neg_index_dict, evaluate_real_rating_matrix, \
        evaluate_predict_rating_matrix, topK, num_procs, exp_flag=0, sp_name=None, result_file=None):
        disease_list = list(evaluate_pos_index_dict.keys())            #
        batch_size = int(len(disease_list) / num_procs)            # 采用多进程 的 batch size 460

        hr_list, ndcg_list = [], []
        index = 0
        for _ in range(num_procs+1):
            if index + batch_size < len(disease_list):
                batch_user_list = disease_list[index:index+batch_size]
                index = index + batch_size
            else:
                # batch_user_list = user_list[index:len(user_list)]
                batch_user_list = disease_list[index:]



            # print(evaluate_pos_index_dict)

            tmp_hr_list, tmp_ndcg_list = self.getHrNdcgProc(evaluate_pos_index_dict,evaluate_neg_index_dict, evaluate_real_rating_matrix, \
                evaluate_predict_rating_matrix, topK, batch_user_list)

            hr_list.extend(tmp_hr_list)
            ndcg_list.extend(tmp_ndcg_list)
        return np.mean(hr_list), np.mean(ndcg_list)


    def getAUCAUPR(self,evaluate_pos_index_dict,
        evaluate_neg_index_dict,
        evaluate_real_rating_matrix,
        evaluate_predict_rating_matrix,
        topK,
        disease_list):
        pass


    def getHrNdcgProc(self, 
        evaluate_pos_index_dict,
        evaluate_neg_index_dict,
        evaluate_real_rating_matrix,
        evaluate_predict_rating_matrix,
        topK, 
        disease_list):

        tmp_hr_list, tmp_ndcg_list = [], []

        for disease in disease_list:

            real_pos_drug_index_list = evaluate_pos_index_dict[disease]
            # print(evaluate_real_rating_matrix.shape)  # 2025 * 1
            real_drug_rating_list = list(np.concatenate(evaluate_real_rating_matrix[real_pos_drug_index_list]))
            positive_length = len(real_drug_rating_list)
            target_length = min(positive_length, topK)
            predcite_neg_drug_index_list = evaluate_neg_index_dict[disease]
            # print(disease,predcite_neg_drug_index_list)
            predict_drug_rating_list = list(np.concatenate(evaluate_predict_rating_matrix[predcite_neg_drug_index_list]))
            negative_length = len(predict_drug_rating_list)
            # predict_rating_list = evaluate_predict_rating_matrix[disease]     #1000 number

            real_drug_rating_list.extend(predict_drug_rating_list)            #

            sort_index = np.argsort(real_drug_rating_list)
            sort_index = sort_index[::-1]


            user_hr_list = []
            user_ndcg_list = []
            hits_num = 0
            for idx in range(topK):
                ranking = sort_index[idx]
                if ranking < positive_length:
                    hits_num += 1
                    user_hr_list.append(self.getHr(idx))
                    user_ndcg_list.append(self.getDcg(idx))

            idcg = self.getIdcg(target_length)          # 最优idcg

            tmp_hr = np.sum(user_hr_list) / target_length
            tmp_ndcg = np.sum(user_ndcg_list) / idcg
            tmp_hr_list.append(tmp_hr)
            tmp_ndcg_list.append(tmp_ndcg)

        return tmp_hr_list, tmp_ndcg_list
