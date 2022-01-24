'''
    author: Haifeng Liu
    e-mail: liuhaifeng0212@qq.com
    released date: 24/01/2022
'''
import os
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from time import time
import random
import itertools
from collections import defaultdict
from sklearn.metrics import pairwise
import pickle
# from entry import para_args



def loadpickle(name):
    pkl_file = open('%s.pkl'%(name), 'rb')
    data1 = pickle.load(pkl_file)
    return data1

def savepickle(name,data):
    output=open('%s.pkl'%(name),'wb')
    pickle.dump(data,output)
    output.close()



class DataModule():
    def __init__(self,args, filename):
        self.args = args
        self.data_dict = {}
        self.terminal_flag = 1  #finish flag
        self.filename = filename
        self.index = 0

        def read_idmap(file):
            re_dict = {}
            with open(file,'r') as f:
                for line in f:
                    a,b = line.strip().split('\t')
                    re_dict[a] = int(b)
            return re_dict,len(re_dict)
        drug_dict,drug_num = read_idmap(os.path.join(os.getcwd(), 'dataset/%s/drug_idmap.csv' % self.args.data_name))
        disease_dict,disease_num = read_idmap(os.path.join(os.getcwd(), 'dataset/%s/disease_idmap.csv' % self.args.data_name))
        protein_dict,protein_num = read_idmap(os.path.join(os.getcwd(), 'dataset/%s/protein_idmap.csv' % self.args.data_name))
        self.disease_num = disease_num
        self.drug_num = drug_num
        self.protein_num = protein_num
        self.covid = disease_dict['COVID-19']
        print('covid19 d',self.covid)

###########################################  Initalize Procedures ############################################
    def prepareModelSupplement(self, model):
        data_dict = {}
        if 'CONSUMED_DRUGS_SPARSE_MATRIX' in model.supply_set:
            self.generateConsumedDrugSparseMatrix()
            data_dict['CONSUMED_DRUGS_INDICES_INPUT'] = self.consumed_drugs_indices_list
            data_dict['CONSUMED_DRUGS_VALUES_INPUT'] = self.consumed_drugs_values_list

            data_dict['DRUG_CONSUMED_DISEASE_INDICES_INPUT'] = self.drug_consumed_disease_indices_list
            data_dict['DRUG_CONSUMED_DISEASE_VALUES_INPUT'] = self.drug_consumed_disease_values_list
        if 'SOCIAL_NEIGHBORS_SPARSE_MATRIX' in model.supply_set:
            self.readDrugSocialNeighbors()
            self.generateDrugSocialNeighborsSparseMatrix()
            data_dict['DRUG_SOCIAL_NEIGHBORS_INDICES_INPUT'] = self.drug_social_neighbors_indices_list
            data_dict['DRUG_SOCIAL_NEIGHBORS_VALUES_INPUT'] = self.drug_social_neighbors_values_list
        if 'DRUG_NEIGHBORS_SPARSE_MATRIX' in model.supply_set:
            self.generate_diseases_neighbors_matrix()
            data_dict['DISEASES_FRIEND_INDICES_INPUT'] = self.diseases_neighbors_indices_list
            data_dict['DISEASES_FRIEND_VALUES_INPUT'] = self.diseases_neighbors_values_list

        if 'DRUG_PROTEIN_SPARSE_MATRIX' in model.supply_set:
            self.readDrug2Protein()
            self.generateDrugProteinNeiboursSparseMatrix()
            data_dict['DRUG2PROTEIN_INDCES_INPUT'] = self.drug2protein_indices_list
            data_dict['DRUG2PROTEIN_VALUES_INPUT'] = self.drug2protein_values_list
        return data_dict

    def initializeRankingTrain(self):
        self.readData()
        self.arrangePositiveData()
        self.generateTrainNegative()

    def initializeRankingVT(self):
        self.readData()
        self.arrangePositiveData()
        self.generateTrainNegative()

    def initalizeRankingEva(self):
        self.readData()
        self.getEvaPositiveBatch()
        self.generateEvaNegative()

    def linkedMap(self):
        self.data_dict['DISEASE_LIST'] = self.disease_list
        self.data_dict['DRUG_LIST'] = self.drug_list
        self.data_dict['LABEL_LIST'] = self.labels_list
    
    def linkedRankingEvaMap(self):
        self.data_dict['EVA_DISEASE_LIST'] = self.eva_disease_list
        self.data_dict['EVA_DRUG_LIST'] = self.eva_drug_list

###########################################  Ranking ############################################
    def readData(self):  #
        f = open(self.filename) ## May should be specific for different subtasks
        total_disease_list = set()
        hash_data = defaultdict(int)   #   {：int()}
        for _, line in enumerate(f):
            arr = line.split("\t")
            dis,drug,label = int(arr[0]),int(arr[1]),int(arr[2])
            hash_data[(int(arr[0]), int(arr[1]))] = label
            total_disease_list.add(int(arr[0]))
        self.total_disease_list = list(total_disease_list)
        self.hash_data = hash_data


    def arrangePositiveData(self):
        positive_data = defaultdict(set)        # {:set()}
        total_data = set()
        hash_data = self.hash_data
        for (disease, drug) in hash_data:
            if hash_data[(disease, drug)]==1:
                total_data.add((disease, drug))
                positive_data[disease].add(drug)
        self.positive_data = positive_data
        self.total_pos_data = len(total_data)
    
    '''
        This function designes for the train/val/test negative generating section
    '''
    def generateTrainNegative(self):
        negative_data = defaultdict(set)          # {：set()}
        total_data = set()
        hash_data = self.hash_data
        for (disease, drug) in hash_data:
            total_data.add((disease, drug))
            if hash_data[(disease,drug)] ==0:
                negative_data[disease].add(drug)
        self.negative_data = negative_data
        self.terminal_flag = 1

    '''
        This function designes for the val/test section, compute loss
    '''
    def getVTRankingOneBatch(self):
        positive_data = self.positive_data       #{dis:{drug1,drug2}}
        negative_data = self.negative_data
        total_disease_list = self.total_disease_list
        disease_list = []
        drug_list = []
        labels_list = []
        # print('total_disease_list',len(total_disease_list))
        for disease in total_disease_list:
            disease_list.extend([disease] * len(positive_data[disease]))
            drug_list.extend(positive_data[disease])
            labels_list.extend([1] * len(positive_data[disease]))
            disease_list.extend([disease] * len(negative_data[disease]))
            drug_list.extend(negative_data[disease])
            labels_list.extend([0] * len(negative_data[disease]))

        self.disease_list = np.reshape(disease_list, [-1, 1])
        self.drug_list = np.reshape(drug_list, [-1, 1])
        self.labels_list = np.reshape(labels_list, [-1, 1])
        # print('self.disease_list',self.disease_list.shape)
    '''
        This function designes for the training process
    '''
    def getTrainRankingBatch(self):
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_disease_list = self.total_disease_list
        index = self.index
        batch_size = self.args.training_batch_size         # batch_size 512

        disease_list, drug_list, labels_list = [], [], []
        
        if index + batch_size < len(total_disease_list):
            target_disease_list = total_disease_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            target_disease_list = total_disease_list[index:len(total_disease_list)]
            self.index = 0
            self.terminal_flag = 0

        for disease in target_disease_list:
            disease_list.extend([disease] * len(positive_data[disease]))
            drug_list.extend(list(positive_data[disease]))
            labels_list.extend([1] * len(positive_data[disease]))
            disease_list.extend([disease] * len(negative_data[disease]))
            drug_list.extend(list(negative_data[disease]))
            labels_list.extend([0] * len(negative_data[disease]))
        
        self.disease_list = np.reshape(disease_list, [-1, 1])     #shape = (batch,1)
        self.drug_list = np.reshape(drug_list, [-1, 1])     #shape = (batch,1)
        self.labels_list = np.reshape(labels_list, [-1, 1]) #shape = (batch,1)
    
    '''
        This function designes for the positive data in rating evaluate section
    '''
    def getEvaPositiveBatch(self):
        hash_data = self.hash_data
        disease_list = []
        drug_list = []
        pos_index_dict = defaultdict(list)    # {:[]}
        neg_index_dict = defaultdict(list)
        index = 0
        for (disease, drug) in hash_data:
            if hash_data[(disease, drug)]==1:
                disease_list.append(disease)
                drug_list.append(drug)
                pos_index_dict[disease].append(index)    # {user:[]}
                index = index + 1

        self.eva_disease_list = np.reshape(disease_list, [-1, 1])
        self.eva_drug_list = np.reshape(drug_list, [-1, 1])
        self.eva_pos_index_dict = pos_index_dict                       # {disease:{indexs}}

        # print('eva_disease_list',self.eva_disease_list.shape)
        # print('eva_drug_list',self.eva_drug_list.shape)
    '''
        This function designes for the negative data generation process in rating evaluate section
    '''

    def generateEvaNegative(self):
        hash_data = self.hash_data
        eva_negative_data = defaultdict(list)   #{:[]}
        for (disease,drug) in hash_data:
            if hash_data[(disease,drug)] ==0:
                eva_negative_data[disease].append(drug)
        self.eva_negative_data = eva_negative_data

    '''
        This function designs for the rating evaluate section, generate negative batch
    '''
    def getEvaRankingBatch(self):
        batch_size = self.args.evaluate_batch_size
        num_evaluate = self.args.num_evaluate
        eva_negative_data = self.eva_negative_data
        total_disease_list = self.total_disease_list
        index = self.index   # index = 0
        terminal_flag = 1
        total_diseases = len(total_disease_list)
        disease_list = []
        drug_list = []
        cov_drug = []
        neg_ind = 0
        neg_index_dict = defaultdict(list)
        if index + batch_size < total_diseases:
            batch_disease_list = total_disease_list[index:index+batch_size]
            self.index = index + batch_size
        else:
            terminal_flag = 0
            batch_disease_list = total_disease_list[index:total_diseases+1]
            self.index = 0
        # print('batch_disease_list',len(batch_disease_list))
        for disease in batch_disease_list:
            if disease==self.covid:
                # print('COVID',len(eva_negative_data[disease]))
                cov_drug = eva_negative_data[disease]
                # print('eva_negative covid drug',cov_drug)
            for neg_drug in eva_negative_data[disease]:
                disease_list.append(disease)
                drug_list.append(neg_drug)
                neg_index_dict[disease].append(neg_ind)
                neg_ind += 1
        neg_ind = 0
        self.eva_disease_list = np.reshape(disease_list, [-1, 1])
        self.eva_drug_list = np.reshape(drug_list, [-1, 1])

        return batch_disease_list, terminal_flag,cov_drug, neg_index_dict

##################################################### Supplement for Sparse Computation ############################################
    def readDrugSocialNeighbors(self, friends_flag=0):
        drug_social_neighbors = defaultdict(set)
        links_filename = os.path.join(os.getcwd(), 'dataset/%s/%s_drug2drug.links' % (self.args.data_name, self.args.data_name))
        links_file = open(links_filename)
        for _, line in enumerate(links_file):
            tmp = line.split('\t')
            drug1, drug2,label = int(tmp[0]), int(tmp[1]),int(tmp[2])
            drug_social_neighbors[drug1].add(drug2)
            if friends_flag == 1:
                drug_social_neighbors[drug2].add(drug1)
        self.drug_social_neighbors = drug_social_neighbors


    def readDrug2Protein(self):
        drug2protein = defaultdict(set)
        links_filename = os.path.join(os.getcwd(),'dataset/%s/%s_drug2protein.links'% (self.args.data_name, self.args.data_name))
        links_file = open(links_filename)
        for _, line in enumerate(links_file):
            tmp = line.split('\t')
            drug, protein,label = int(tmp[0]), int(tmp[1]),int(tmp[2])
            drug2protein[drug].add(protein)
        self.drug_2_protein = drug2protein
    '''
        Generate Social Neighbors Sparse Matrix Indices and Values
    '''


    def generate_diseases_neighbors_matrix(self):
        positive_data = self.positive_data
        diseases_neighbors_indices_list = []
        diseases_neighbors_values_list = []
        diseases_neighbors_dict = defaultdict(list)
        diseases_social = defaultdict(set)
        disease_count = {}
        alpha = 1
        for disease1, drug1 in positive_data.items():
            for disease2,drug2 in positive_data.items():
                if len(set(drug1) & set(drug2)) >=alpha:
                    diseases_social[disease1].add(disease2)

        for disease in diseases_social:
            diseases_neighbors_dict[disease] = sorted(diseases_social[disease])

        disease_list = sorted(list(diseases_neighbors_dict.keys()))
        print('训练集共计有disease 个',len(disease_list))
        for disease in disease_list:

            for friend_disease in diseases_neighbors_dict[disease]:
                diseases_neighbors_indices_list.append([disease, friend_disease])
                #########################
                diseases_neighbors_values_list.append(1.0/len(diseases_social[disease]))

        self.diseases_neighbors_indices_list = np.array(diseases_neighbors_indices_list).astype(np.int64)   #转换成numpy int64
        self.diseases_neighbors_values_list = np.array(diseases_neighbors_values_list).astype(np.float32)   #转换成numpy float32  b

    def generateDrugSocialNeighborsSparseMatrix(self):
        social_neighbors = self.drug_social_neighbors
        social_neighbors_indices_list = []
        social_neighbors_values_list = []
        social_neighbors_dict = defaultdict(list)       # {：[]}
        for drug in social_neighbors:
            social_neighbors_dict[drug] = sorted(social_neighbors[drug])
        drug_list = sorted(list(social_neighbors.keys()))
        for drug in drug_list:
            for friend_drug in social_neighbors_dict[drug]:
                social_neighbors_indices_list.append([drug, friend_drug])
                #########################
                social_neighbors_values_list.append(1.0/len(social_neighbors_dict[drug]))
                #########################social_neighbors_indices_list

        self.drug_social_neighbors_indices_list = np.array(social_neighbors_indices_list).astype(np.int64)   #转换成numpy int64
        self.drug_social_neighbors_values_list = np.array(social_neighbors_values_list).astype(np.float32)   #转换成numpy float32

    def generateDrugProteinNeiboursSparseMatrix(self):
        Drug_protein = self.drug_2_protein
        Drug2Protein_indices_list = []
        Drug2Protein_values_list = []
        Drug2Protein_neighbors_dict = defaultdict(list)  # {：[]}
        for drug in Drug_protein:
            Drug2Protein_neighbors_dict[drug] = sorted(Drug_protein[drug])  #
        drug_list = sorted(list(Drug_protein.keys()))
        for drug in drug_list:
            for protein in Drug2Protein_neighbors_dict[drug]:
                Drug2Protein_indices_list.append([drug, protein])  #
                #########################
                Drug2Protein_values_list.append(1.0 / len(Drug2Protein_neighbors_dict[drug]))
                #########################
        self.drug2protein_indices_list = np.array(Drug2Protein_indices_list).astype(np.int64)  # 转换成numpy int64
        self.drug2protein_values_list = np.array(Drug2Protein_values_list).astype(np.float32)  # 转换成numpy float32

    '''
        Generate Consumed Items Sparse Matrix Indices and Values
    '''
    def generateConsumedDrugSparseMatrix(self):
        positive_data = self.positive_data  #正例 disease:set(drug)
        consumed_drugs_indices_list = []
        consumed_drugs_values_list = []
        drug_consumed_disease_indices_list = []
        drug_consumed_disease_values_list = []
        consumed_drugs_dict = defaultdict(list)     #{:[]}

        for disease in positive_data:
            consumed_drugs_dict[disease] = sorted(positive_data[disease])
        disease_list = sorted(list(positive_data.keys()))
        for disease in disease_list:
            for drug in consumed_drugs_dict[disease]:
                consumed_drugs_indices_list.append([disease, drug])
                drug_consumed_disease_indices_list.append([drug, disease])
                consumed_drugs_values_list.append(1.0/len(consumed_drugs_dict[disease]))
                drug_consumed_disease_values_list.append(1.0)

        self.consumed_drugs_indices_list = np.array(consumed_drugs_indices_list).astype(np.int64)
        self.consumed_drugs_values_list = np.array(consumed_drugs_values_list).astype(np.float32)
        self.drug_consumed_disease_indices_list = np.array(drug_consumed_disease_indices_list).astype(np.int64)
        self.drug_consumed_disease_values_list = np.array(drug_consumed_disease_values_list).astype(np.float32)

