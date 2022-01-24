'''
    author: Haifeng Liu
    e-mail: liuhaifeng0212@qq.com
    released date: 24/01/2021
'''
import tensorflow as tf
import numpy as np
import os
import scipy.sparse as sp

class drug2cov():
    def __init__(self, args):
        self.args = args
        self.weight = 0.7
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_DRUGS_SPARSE_MATRIX',
            'DRUG_NEIGHBORS_SPARSE_MATRIX',
            'DRUG_PROTEIN_SPARSE_MATRIX'
        )


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




    def startConstructGraph(self):      #初始化 计算图
        self.initializeNodes()          #
        self.constructTrainGraph()      #
        self.saveVariables()            #
        self.defineMap()                #

    def inputSupply(self, data_dict):
        self.drug_social_neighbors_indices_input = data_dict['DRUG_SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.drug_social_neighbors_values_input = data_dict['DRUG_SOCIAL_NEIGHBORS_VALUES_INPUT']

        self.consumed_drugs_indices_input = data_dict['CONSUMED_DRUGS_INDICES_INPUT']   # 药物
        self.consumed_drugs_values_input = data_dict['CONSUMED_DRUGS_VALUES_INPUT']     # 值

        self.drug_consumed_disease_indices_input = data_dict['DRUG_CONSUMED_DISEASE_INDICES_INPUT']
        self.drug_consumed_disease_values_input = data_dict['DRUG_CONSUMED_DISEASE_VALUES_INPUT']

        self.diseases_neighbors_indices_list = data_dict['DISEASES_FRIEND_INDICES_INPUT']
        self.diseases_neighbors_values_list = data_dict['DISEASES_FRIEND_VALUES_INPUT']

        self.drug2protein_indices_list = data_dict['DRUG2PROTEIN_INDCES_INPUT']
        self.drug2protein_values_list = data_dict['DRUG2PROTEIN_VALUES_INPUT']

        self.drug_social_neighbors_dense_shape = np.array([self.drug_num, self.drug_num]).astype(np.int64)
        self.drug_treat_disease_dense_shape = np.array([self.drug_num, self.disease_num]).astype(np.int64)
        self.disease_social_dense_shape = np.array([self.disease_num, self.disease_num]).astype(np.int64)
        self.disease_treat_drug_dense_shape = np.array([self.disease_num, self.drug_num]).astype(np.int64)
        self.drug2protein_dense_shape = np.array([self.drug_num,self.protein_num]).astype(np.int64)


        self.drug_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices = self.drug_social_neighbors_indices_input,
            values = self.drug_social_neighbors_values_input,
            dense_shape=self.drug_social_neighbors_dense_shape
        )
        self.consumed_drugs_sparse_matrix = tf.SparseTensor(
            indices = self.consumed_drugs_indices_input,
            values = self.consumed_drugs_values_input,
            dense_shape=self.disease_treat_drug_dense_shape
        )
        #########################################################################################################
        self.disease_neighbors_spars_matrix = tf.SparseTensor(
            indices = self.diseases_neighbors_indices_list,
            values = self.diseases_neighbors_values_list,
            dense_shape=self.disease_social_dense_shape
        )

        self.drug_consumed_disease_sparse_matrix = tf.SparseTensor(
            indices = self.drug_consumed_disease_indices_input,
            values = self.drug_consumed_disease_values_input,
            dense_shape=self.drug_treat_disease_dense_shape
        )

        self.drug2protein_sparse_matrix =  tf.SparseTensor(

            indices = self.drug2protein_indices_list,
            values = self.drug2protein_values_list,
            dense_shape= self.drug2protein_dense_shape
        )

    def convertDistribution(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1])
        y = (x - mean) * 0.2 / tf.sqrt(var)
        return y
    #############
    def geberateDrugEmbedding_from_DrugNeighbors(self,current_drug_embedding):
        drug_embedding_from_friend_drugs = tf.sparse_tensor_dense_matmul(
            self.drug_social_neighbors_sparse_matrix, current_drug_embedding,adjoint_a=True
        )
        return  drug_embedding_from_friend_drugs
    #############
    def generateDiseaseEmbeddingFromDiseaseSocialNeighbors(self, current_disease_embedding):
        disease_embedding_from_social_neighbors = tf.sparse_tensor_dense_matmul(
            self.disease_neighbors_spars_matrix, current_disease_embedding
        )
        return disease_embedding_from_social_neighbors

    def generateDiseaseEmebddingFromConsumedDrugs(self, current_drug_embedding):
        Disease_embedding_from_consumed_Drugs = tf.sparse_tensor_dense_matmul(
            self.consumed_drugs_sparse_matrix, current_drug_embedding
        )
        return Disease_embedding_from_consumed_Drugs

    def generateDrugEmbeddingFromProein(self,current_protein_embedding):
        Drug_embedding_from_protein = tf.sparse_tensor_dense_matmul(
            self.drug2protein_sparse_matrix,current_protein_embedding
        )
        return Drug_embedding_from_protein

    def generate_Drug_EmebddingFromConsumedDiseases(self, current_disease_embedding):
        drug_embedding_from_consumed_diseases = tf.sparse_tensor_dense_matmul(
            self.drug_consumed_disease_sparse_matrix, current_disease_embedding
        )
        return drug_embedding_from_consumed_diseases

    def initializeNodes(self):

        with tf.name_scope('Input_layer'):
            self.disease_input = tf.placeholder("int32", [None, 1],name='disease_input')
            self.drug_input = tf.placeholder("int32", [None, 1],name='drug_input')
            self.labels_input = tf.placeholder("float32", [None, 1],name='label_input')   # shape (batch_size,1)
            self.protein_input =tf.placeholder('int32',[None,1],name='protein_input')
            self.drug_biases = tf.Variable(tf.zeros([self.drug_num], dtype=tf.float32))
            self.disease_biases = tf.Variable(tf.zeros([self.disease_num], dtype=tf.float32))




        self.disease_embedding = tf.Variable(tf.nn.l2_normalize(
            tf.random_normal([self.disease_num, self.args.dimension], stddev=1 / (self.args.dimension ** 0.5))), name='disease_embedding')
        self.drug_embedding = tf.Variable(tf.nn.l2_normalize(
            tf.random_normal([self.drug_num, self.args.dimension], stddev=1 /(self.args.dimension ** 0.5))), name='drug_embedding')
        self.protein_embedding = tf.Variable(tf.nn.l2_normalize(
            tf.random_normal([self.protein_num,self.args.dimension],stddev= 1/ (self.args.dimension ** 0.5))), name='protein_embedding')
        self.reduce_dimension_layer = tf.layers.Dense(\
            self.args.dimension, activation=tf.nn.relu, name='reduce_dimension_layer')
        self.item_fusion_layer = tf.layers.Dense(\
            self.args.dimension, activation=tf.nn.relu, name='drug_fusion_layer')
        self.user_fusion_layer = tf.layers.Dense(\
            self.args.dimension, activation=tf.nn.relu, name='disease_fusion_layer')

    def constructTrainGraph(self):
        #with tf.name_scope('Convert_layer'):
            #first_disease_review_vector_matrix = self.convertDistribution(self.drug_embedding)
            #first_drug_review_vector_matrix = self.convertDistribution(self.disease_embedding)

            #self.disease_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_disease_review_vector_matrix)
            #self.item_reduce_dim_vector_matrix = self.reduce_dimension_layer(first_drug_review_vector_matrix)

            #second_disease_matrix = self.convertDistribution(self.disease_reduce_dim_vector_matrix)
            #second_drug_matrix = self.convertDistribution(self.drug_reduce_dim_vector_matrix)      #

        ########################

        self.fusion_drug_embedding = self.drug_embedding
        self.fusion_disease_embedding = self.disease_embedding
        self.final_protein_embedidng = self.protein_embedding
        self.final_protein_embedidng = tf.nn.dropout(self.final_protein_embedidng,rate=self.args.dropout)
        self.pos_drug_bias = tf.nn.embedding_lookup(self.drug_biases, self.drug_input)
        self.pos_disease_bias = tf.nn.embedding_lookup(self.disease_biases, self.disease_input)


        with tf.name_scope('Drug_consumed_Disease_layer'):
            drug_embedding_from_consumed_diseases = self.generate_Drug_EmebddingFromConsumedDiseases(self.fusion_disease_embedding)
            print('drug_embedding_from_consumed_diseases 尺寸',drug_embedding_from_consumed_diseases.shape)

        with tf.name_scope('Drug2Protein_layer'):
            drug_embedding_from_protein = self.generateDrugEmbeddingFromProein(self.protein_embedding)
            print('drug_embedding_from_protein 尺寸',drug_embedding_from_protein)
        with tf.name_scope('Disease_consumed_Drug_layer'):
            disease_embedding_from_consumed_drugs = self.generateDiseaseEmebddingFromConsumedDrugs(self.fusion_drug_embedding)     # 生成user embedding：item embed 与 consumed 矩阵相乘，可改进
            print('disease_embedding_from_consumed_drugs 尺寸',drug_embedding_from_consumed_diseases.shape)


        ######################
        with tf.name_scope('Drug_gcn_layer'):
            first_gcn_drug_embedding = self.geberateDrugEmbedding_from_DrugNeighbors(self.fusion_drug_embedding)
            second_gcn_drug_embedding = self.geberateDrugEmbedding_from_DrugNeighbors(first_gcn_drug_embedding)
            third_gcn_drug_embedding = self.geberateDrugEmbedding_from_DrugNeighbors(second_gcn_drug_embedding)
            four_gcn_drug_embedding = self.geberateDrugEmbedding_from_DrugNeighbors(third_gcn_drug_embedding)

        with tf.name_scope('Result_drug_layer'):
            self.final_drug_embedding = self.args.fusion_weight * self.fusion_drug_embedding + drug_embedding_from_consumed_diseases + tf.multiply(four_gcn_drug_embedding, self.weight) + drug_embedding_from_protein

            self.final_drug_embedding = tf.nn.dropout(self.final_drug_embedding,rate=self.args.dropout)
            print('self.pos_drug_bias',self.pos_drug_bias.shape)
            print('drug emb size', self.final_drug_embedding.shape)


        with tf.name_scope('disease_gcn_layer'):
            first_gcn_disease_embedding = self.generateDiseaseEmbeddingFromDiseaseSocialNeighbors(self.fusion_disease_embedding)
            #second_gcn_disease_embedding = self.generateDiseaseEmbeddingFromDiseaseSocialNeighbors(first_gcn_disease_embedding)

        with tf.name_scope('Result_drug_layer'):
            self.final_disease_embedding = self.args.fusion_weight * self.fusion_disease_embedding + disease_embedding_from_consumed_drugs

            self.final_disease_embedding = tf.nn.dropout(self.final_disease_embedding,rate=self.args.dropout)
            print(' disease emb size',self.final_disease_embedding.shape)

#############################################################################################################################################
        latest_disease_latent = tf.gather_nd(self.final_disease_embedding, self.disease_input)

        latest_drug_latent = tf.gather_nd(self.final_drug_embedding, self.drug_input)
        latest_protein_latent = tf.gather_nd(self.final_protein_embedidng,self.protein_input)

        predict_vector = tf.multiply(latest_disease_latent, latest_drug_latent)
        predic_drug2protein_vector = tf.multiply(latest_drug_latent,latest_protein_latent)
        self.prediction = tf.sigmoid(tf.reduce_sum(predict_vector, 1, keepdims=True))             # (batch,1)
        self.pre_drug2protein = tf.sigmoid(tf.reduce_sum(predic_drug2protein_vector,1,keep_dims=True))
        def hierarchical_mutual_information_maximization(em, adj):
            def row_shuffle(embedding):
                return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))

            def row_column_shuffle(embedding):
                corrupted_embedding = tf.transpose(tf.gather(tf.transpose(embedding), tf.random.shuffle(
                    tf.range(tf.shape(tf.transpose(embedding))[0]))))
                corrupted_embedding = tf.gather(corrupted_embedding,
                                                tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
                return corrupted_embedding

            def score(x1, x2):
                return tf.sigmoid(tf.reduce_sum(tf.multiply(x1, x2), 1))

            disease_embeddings = em
            edge_embeddings = tf.sparse_tensor_dense_matmul(adj, disease_embeddings)
            # Local MIM
            pos = score(disease_embeddings, edge_embeddings)
            neg1 = score(row_shuffle(disease_embeddings), edge_embeddings)
            neg2 = score(row_column_shuffle(edge_embeddings), disease_embeddings)
            local_loss = tf.reduce_mean(-tf.log(tf.sigmoid(pos - neg1)) - tf.log(tf.sigmoid(neg1 - neg2)))
            # Global MIM
            graph = tf.reduce_mean(edge_embeddings, 0)
            pos = score(edge_embeddings, graph)
            neg1 = score(row_column_shuffle(edge_embeddings), graph)
            global_loss = tf.reduce_mean(-tf.log(tf.sigmoid(pos - neg1)))
            return global_loss + local_loss
        def matrix2A(dis2drug,tag=None):
            def sp_mat_to_sp_tensor(sp_mat):
                coo = sp_mat.tocoo().astype(np.float32)
                indices = np.array(list(zip(coo.row,coo.col)))
                values = coo.data
                shape = coo.shape
                return tf.sparse.SparseTensor(indices,values,shape)

            def normalize_adj_matrix(sp_mat, norm_method="left"):
                """Normalize adjacent matrix

                Args:
                    sp_mat: A sparse adjacent matrix
                    norm_method (str): The normalization method, can be 'symmetric'
                        or 'left'.

                Returns:
                    sp.spmatrix: The normalized adjacent matrix.

                """

                d_in = np.asarray(sp_mat.sum(axis=1))  # indegree
                if norm_method == "left":
                    rec_d_in = np.power(d_in, -1).flatten()  # reciprocal
                    rec_d_in[np.isinf(rec_d_in)] = 0.  # replace inf
                    rec_d_in = sp.diags(rec_d_in)  # to diagonal matrix
                    norm_sp_mat = rec_d_in.dot(sp_mat)  # left matmul
                elif norm_method == "symmetric":
                    rec_sqrt_d_in = np.power(d_in, -0.5).flatten()
                    rec_sqrt_d_in[np.isinf(rec_sqrt_d_in)] = 0.
                    rec_sqrt_d_in = sp.diags(rec_sqrt_d_in)

                    mid_sp_mat = rec_sqrt_d_in.dot(sp_mat)  # left matmul
                    norm_sp_mat = mid_sp_mat.dot(rec_sqrt_d_in)  # right matmul
                else:
                    raise ValueError(f"'{norm_method}' is an invalid normalization method.")

                return norm_sp_mat
            users_np, items_np = dis2drug[:, 0], dis2drug[:, 1]
            ratings = np.ones_like(users_np, dtype=np.float32)
            if tag=='dis2drug':
                n_nodes = self.disease_num + self.drug_num
                up_left_adj = sp.csr_matrix((ratings, (users_np, items_np + self.disease_num)),
                                            shape=(n_nodes, n_nodes))
            elif tag=='drug2dis':
                n_nodes =  self.drug_num + self.disease_num
                up_left_adj = sp.csr_matrix((ratings, (users_np, items_np + self.drug_num)), shape=(n_nodes, n_nodes))
            elif tag=='drug2pro':
                n_nodes = self.drug_num + self.protein_num
                up_left_adj = sp.csr_matrix((ratings, (users_np, items_np + self.protein_num)), shape=(n_nodes, n_nodes))
            elif tag=='drug2drug':
                n_nodes = self.drug_num + self.drug_num
                up_left_adj = sp.csr_matrix((ratings, (users_np, items_np + self.drug_num)), shape=(n_nodes, n_nodes))
            elif tag=='dis2dis':
                n_nodes = self.disease_num + self.disease_num
                up_left_adj = sp.csr_matrix((ratings, (users_np, items_np + self.disease_num)), shape=(n_nodes, n_nodes))
            adj_mat = up_left_adj + up_left_adj.T
            adj_matrix = normalize_adj_matrix(adj_mat + sp.eye(adj_mat.shape[0]), norm_method="left")
            adj_matrix = sp_mat_to_sp_tensor(adj_matrix)
            return adj_matrix

        A_adj = matrix2A(self.consumed_drugs_indices_input,tag='dis2drug')
        ego_emd = tf.concat([self.final_disease_embedding,self.final_drug_embedding],0) # 8596, 64
        self.ssl_loss = hierarchical_mutual_information_maximization(ego_emd,A_adj) #

        drug2drug = matrix2A(self.drug_consumed_disease_indices_input,tag='drug2dis')
        drugdis_emd = tf.concat([self.final_drug_embedding, self.final_disease_embedding], 0)   #
        self.drug2drug_loss =  hierarchical_mutual_information_maximization(drugdis_emd,drug2drug)
        drug2pro= matrix2A(self.drug2protein_indices_list,tag='drug2pro')
        drugpro_emd = tf.concat([self.final_drug_embedding, self.final_protein_embedidng], 0)
        self.drug2pro_loss =  hierarchical_mutual_information_maximization(drugpro_emd,drug2pro)


        with tf.name_scope('Loss_layer'):
            tv = tf.trainable_variables()
            regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
            self.loss = tf.nn.l2_loss(self.labels_input - self.prediction + self.pos_drug_bias) +  self.args.alpha *( self.ssl_loss + self.drug2pro_loss +  self.drug2drug_loss) + regularization_cost
            self.opt_loss = tf.nn.l2_loss(self.labels_input - self.prediction + self.pos_drug_bias) + self.args.alpha *( self.ssl_loss + self.drug2pro_loss +  self.drug2drug_loss) + regularization_cost
            tf.summary.scalar('loss', self.opt_loss)
        with tf.name_scope('Optimizer_layer'):
            self.opt = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.opt_loss)
        self.init = tf.global_variables_initializer()
    def saveVariables(self):
        ############################# Save Variables #################################
        variables_dict = {}
        variables_dict[self.disease_embedding.op.name] = self.disease_embedding
        variables_dict[self.drug_embedding.op.name] = self.drug_embedding

        for v in self.reduce_dimension_layer.variables:
            variables_dict[v.op.name] = v

        self.saver = tf.train.Saver(variables_dict)
        ############################# Save Variables #################################

    def defineMap(self):

        map_dict = {}
        map_dict['train'] = {
            self.disease_input: 'DISEASE_LIST',
            self.drug_input: 'DRUG_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['val'] = {
            self.disease_input: 'DISEASE_LIST',
            self.drug_input: 'DRUG_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['test'] = {
            self.disease_input: 'DISEASE_LIST',
            self.drug_input: 'DRUG_LIST',
            self.labels_input: 'LABEL_LIST'
        }

        map_dict['eva'] = {
            self.disease_input: 'EVA_DISEASE_LIST',
            self.drug_input: 'EVA_DRUG_LIST'
        }

        map_dict['out'] = {
            'train': (self.loss,self.ssl_loss),             #train
            'val': self.loss,               #val
            'test': self.loss,              #test
            'eva': self.prediction,
            'drug2protein':self.pre_drug2protein
        }

        self.map_dict = map_dict

