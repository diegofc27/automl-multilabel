import openml
import numpy as np
import math
import time
from sklearn import preprocessing
from config import SEED


def dataset_num_to_id(dataset_number):
    dataset_ids = [40590, 40591, 40592, 40593, 40594, 40595, 40596, 40597, 40588, 40589]
    return dataset_ids[dataset_number]


class OpenMlDataset():
    def __init__(self, dataset_id, threshold = 0.9, normalization_type="standard", split=0.75, seed=0) -> None:
        self.dataset_id = dataset_id
        self.random_generator = np.random.RandomState(SEED)
        self.split = split
        self.load_dataset()
        self.pred_preprocessing = PredictorsPreprocessingModule(self.pred_data, self.pred_categorical_indicator, correlation_threshold=threshold, normalization_type=normalization_type)
        #self.target_preprocessing = TargetsPreprocessingModule(self.target_data, transformation_name=transformation)
        self.pred_data = self.pred_preprocessing.get_data()
        self.train_test_split()
        #self.target_data = self.target_preprocessing.get_data()
    
    def load_dataset(self):
        dataset = openml.datasets.get_dataset(self.dataset_id)
        self.targets_name = dataset.default_target_attribute.split(",")
        
        X, _, categorical_indicator, features_name = dataset.get_data(dataset_format="dataframe")
        self.features_name = features_name
        self.categorical_indicator = categorical_indicator
        
        for i, name in enumerate(self.features_name):
            if not self.categorical_indicator[i]:
                continue
            encoder = preprocessing.LabelEncoder()
            encoder.fit(X[name])
            column = encoder.transform(X[name])
            X.loc[:, name] = column.astype('float32')
        
        X = X.astype('float32')
        self.data = X.to_numpy()
        
        print("Chosen dataset is ",dataset.name, " (id:"+str(self.dataset_id)+")")
        print("Original dataset shape is ",self.data.shape)
        print("Number of predictors : ", len(self.features_name) - len(self.targets_name), "Number of targets : ", len(self.targets_name))

        # Split data into predictors and targets
        target_inds = []
        pred_inds = []
        for i in range(self.data.shape[1]):
            if self.features_name[i] in self.targets_name:
                target_inds.append(i)
            else:
                pred_inds.append(i)
        self.target_categorical_indicator = [self.categorical_indicator[i] for i in target_inds]
        self.pred_categorical_indicator = [self.categorical_indicator[i] for i in pred_inds]
        self.pred_data = self.data[:, pred_inds]
        self.target_data = self.data[:, target_inds]


    def get_predictors(self):
        return self.train_pred_data, self.pred_categorical_indicator
        
    
    def get_targets(self):
        return self.train_tar_data, self.target_categorical_indicator
        
        
    def get_test_predictors(self):
        return self.test_pred_data, self.pred_categorical_indicator
        
    
    def get_test_targets(self):
        return self.test_tar_data, self.target_categorical_indicator

    
    def get_dataset(self):
        return self.pred_data, self.target_data
    

    def train_test_split(self):
        #print(type(self.pred_data))
        #print(type(self.target_data))
        perm_idxs = np.arange(self.data.shape[0])
        self.random_generator.shuffle(perm_idxs)
        split_idx = int(self.data.shape[0]*self.split)  
        train_idxs = perm_idxs[:split_idx]
        valid_idxs = perm_idxs[split_idx:]
        self.train_pred_data = self.pred_data[train_idxs]
        self.train_tar_data = self.target_data[train_idxs]
        self.test_pred_data = self.pred_data[valid_idxs]
        self.test_tar_data = self.target_data[valid_idxs] 



class PredictorsPreprocessingModule():
    def __init__(self, data, categorical_indicator, correlation_threshold = 0.8, normalization_type = "standard") -> None:
        self.data = data
        self.categorical_indicator = categorical_indicator
        self.correlation_threshold = correlation_threshold
        self.normalization_type = normalization_type
        assert self.data.shape[1] == len(self.categorical_indicator), "Categorical indicator should have length equal to the number of features"
        self.features_to_remove_list = []
        self.scaler_list = []
        self.preprocessing()
    
    def preprocessing(self):
        self.transform_2_value_numeric_to_binary()
        self.check_for_single_value_parameters()
        #self.check_for_duplicated_samples()
        self.check_for_values_with_one_occurance()
        self.check_for_pseudocorrelation()
        self.check_for_correlation()
        # Must be the last
        self.normalize_dataset()
        self.dataset_one_hot_encoding()
    
    
    def transform_2_value_numeric_to_binary(self):
        transformed_features_num = 0
        for i in range(self.data.shape[1]):
            if self.categorical_indicator[i]:
                pass
            unique_values = np.unique(self.data[:, i])
            if unique_values.shape[0] == 2:
                encoder = preprocessing.LabelEncoder()
                encoder.fit(self.data[:, i])
                
                self.data[:, i] = (encoder.transform(self.data[:, i])).astype('float32')
                self.categorical_indicator[i] = True
                transformed_features_num = transformed_features_num + 1

        print(transformed_features_num, " numeric features were transformed into binary")
        
        
    def check_for_duplicated_samples(self):
        
        old = self.data.shape[0]
        self.data = np.unique(self.data, axis=0)
        new = self.data.shape[0]
        print("Removed ",old - new," samples due to being duplicates")
        
       
    def check_for_values_with_one_occurance(self):
        
        params_to_delete = []
        for i in np.arange(self.data.shape[1]-1, -1, -1):
            if not self.categorical_indicator[i]:
                continue
            temp = (self.data[:,i]).astype(int)
            temp = temp + abs(int(np.amin(temp)))+1
            occurances = np.bincount(temp)
            if 1 in occurances:
                params_to_delete.append(i)

        old = self.data.shape[1]
        self.data = np.delete(self.data, params_to_delete, 1)
        self.clear_feature_data(params_to_delete)
        new = self.data.shape[1]
        print("Removed ",old - new," parameters with 1 occurance of value")
        #print(params_to_delete)
        
        
    def check_for_single_value_parameters(self):
        
        params_to_delete = []
        for i in np.arange(self.data.shape[1]-1, -1, -1):
            unique_values = np.unique(self.data[:,i])
            if unique_values.shape[0] == 1:
                params_to_delete.append(i)

        old = self.data.shape[1]
        self.data = np.delete(self.data, params_to_delete, 1)
        self.clear_feature_data(params_to_delete)
        new = self.data.shape[1]
        print("Removed ",old - new," parameters with one unique value")
        #print(params_to_delete)
    
    def check_for_pseudocorrelation(self):
        
        params_to_delete = []
        for i in np.arange(self.data.shape[1]-1, -1, -1):
            if not self.categorical_indicator[i]:
                continue
            for j in np.arange(i-1, -1, -1):
                if not self.categorical_indicator[j]:
                    continue
                # Check for pseudocorrelation
                temp_i = self.data[:,i]
                temp_j = self.data[:,j]
                uniq_val_num_i = np.unique(self.data[:,i]).shape[0]
                uniq_val_num_j = np.unique(self.data[:,j]).shape[0]
                if uniq_val_num_i > uniq_val_num_j:
                    lower = j
                else:
                    lower = i
                temp_i = temp_i * uniq_val_num_j
                sum = temp_i + temp_j
                uniq_val_sum = np.unique(sum)
                if uniq_val_sum.shape[0] == uniq_val_num_i or uniq_val_sum.shape[0] == uniq_val_num_j:
                    if not(lower in params_to_delete):
                        params_to_delete.append(lower)

        old = self.data.shape[1]
        self.data = np.delete(self.data, params_to_delete, 1)
        self.clear_feature_data(params_to_delete)
        new = self.data.shape[1]
        print("Removed ",old - new," parameters with pseudocorrelated values")
              
              
    def check_for_correlation(self):
        params_to_delete = []
        for i in range(self.data.shape[1]):
            if self.categorical_indicator[i]:
                continue
            for j in range(i+1, self.data.shape[1]):
                if self.categorical_indicator[j]:
                    continue
                
                sum_i = np.sum(self.data[:, i])
                sum_j = np.sum(self.data[:, j])
                sum_i2 = np.sum(self.data[:, i]**2)
                sum_j2 = np.sum(self.data[:, j]**2)
                sum_ij = np.sum(self.data[:, i] * self.data[:, j])
                n = self.data.shape[0]
                
                correlation = (n*sum_ij - sum_i * sum_j)/(math.sqrt((n * sum_i2 - (sum_i)**2)*(n * sum_j2 - (sum_j)**2)))
                correlation = abs(correlation)
                if correlation>= self.correlation_threshold:
                    #print("correlation ",correlation)
                    #print(" i ",i," j ",j)
                    if np.unique(self.data[:, i]).shape[0] >= np.unique(self.data[:, j]).shape[0]:
                        if not(j in params_to_delete):
                            params_to_delete.append(j)
                    else:
                        if not(i in params_to_delete):
                            params_to_delete.append(i)
                
        old = self.data.shape[1]
        self.data = np.delete(self.data, params_to_delete, 1)
        self.clear_feature_data(params_to_delete)
        new = self.data.shape[1]
        print("Removed ",old - new," parameters with correlated values")
        #print(params_to_delete)
            
                 
    def clear_feature_data(self, indices:list):
        indices.sort()
        self.features_to_remove_list.append(indices)
        indices = indices[::-1]
        for i in indices:
            self.categorical_indicator.pop(i)
       
       
    def normalize_dataset(self):
        for i in range(self.data.shape[1]):
            if self.categorical_indicator:
                self.scaler_list.append(None)
                continue
            
            if self.normalization_type == "standard":
                self.scaler_list.append(preprocessing.StandardScaler())
                self.scaler_list[i].fit(self.data[:, i].reshape(-1, 1))
                self.data[:, i] = self.scaler_list[i].transform(self.data[:, i].reshape(-1, 1)).reshape((-1))
            elif self.normalization_type == "robust":
                self.scaler_list.append(preprocessing.RobustScaler())
                self.scaler_list[i].fit(self.data[:, i].reshape(-1, 1))
                self.data[:, i] = self.scaler_list[i].transform(self.data[:, i].reshape(-1, 1)).reshape((-1))
            else:
                modes = ["standard", "robust"]
                print("Wrong normalization mode. Choose one of the following: ", modes)
                
                
    def one_hot_encoding(self, x):
        length = x.shape[0]
        max_value = int(np.max(x))
        new_array = np.zeros((length, max_value+1))
        new_array[np.arange(length), x.astype(int)] = 1.0
        return new_array 
               
                
    def dataset_one_hot_encoding(self):
        for i in range(self.data.shape[1]-1, -1, -1):
            if np.unique(self.data[:, i]).shape[0] == 2 or not self.categorical_indicator[i]:
                temp_ar = np.reshape(self.data[:, i], (-1, 1))
            else:
                temp_ar = self.one_hot_encoding(self.data[:,i])
            
            if i+1 == self.data.shape[1]:
                self.data = np.concatenate((self.data[:,:i], temp_ar), axis=1)
            else:
                self.data = np.concatenate((self.data[:,:i], temp_ar, self.data[:,i+1:]), axis=1)
    
                
    def get_data(self):
        return self.data
            

    # def transform_validation(self, valid_data):
    #     for i in range(len(self.features_to_remove_list)):
    #         valid_data = np.delete(valid_data, self.features_to_remove_list[i], axis = 1)
    #     for i in range(len(self.scaler_list)):
    #         if self.scaler_list[i] is None:
    #             continue
    #         valid_data[:,i] = self.scaler_list[i].transform(valid_data[:, i].reshape(-1, 1)).reshape((-1))
    #     return valid_data
        

class TargetsPreprocessingModule():
    def __init__(self, data, transformation_name = "pt3") -> None:
        self.data = data
        self.transformation_name = transformation_name
        self.choose_transformation()
        self.dataset_one_hot_encoding()
    
    
    def choose_transformation(self):
        if self.transformation_name == "pt3":
            self.pt3()
        else:
            assert False, "Invalid transformation name"        
            
    def pt3(self):
        self.unique_vectors = np.unique(self.data, axis=0)
        print("Found ", self.unique_vectors.shape[0], "unique target vectors")
        indices_list = []
        
        for i in range(self.unique_vectors.shape[0]):
            temp = abs(self.data - self.unique_vectors[i])
            sum = np.sum(temp, axis=1)
            indices = np.nonzero(sum == 0)[0]
            self.data[indices] = i
        
        self.data = self.data[:,0].reshape(-1,1)
    
    # def pt3_val(self, val_data):
    #     for i in range(self.unique_vectors.shape[0]):
    #         temp = abs(val_data - self.unique_vectors[i])
    #         sum = np.sum(temp, axis=1)
    #         indices = np.nonzero(sum == 0)[0]
    #         if indices.size == 0:
    #             continue
    #         val_data[indices] = i
        
    #     val_data = val_data[:,0].reshape(-1,1)
    #     val_data = self.dataset_one_hot_encoding(val_data)
    #     return val_data
                
    
    def one_hot_encoding(self, x):
        length = x.shape[0]
        max_value = int(np.max(x))
        new_array = np.zeros((length, max_value+1))
        new_array[np.arange(length), x.astype(int)] = 1.0
        return new_array 
               
                
    def dataset_one_hot_encoding(self):
        for i in range(self.data.shape[1]-1, -1, -1):
            if np.unique(self.data[:, i]).shape[0] == 2:
                temp_ar = np.reshape(self.data[:, i], (-1, 1))
            else:
                temp_ar = self.one_hot_encoding(self.data[:,i])
            
            if i+1 == self.data.shape[1]:
                self.data = np.concatenate((self.data[:,:i], temp_ar), axis=1)
            else:
                self.data = np.concatenate((self.data[:,:i], temp_ar, self.data[:,i+1:]), axis=1)
        return self.data
    
    def get_data(self):
        return self.data
    
        

class CrossValidation():
    def __init__(self, pred_data, target_data, split_num) -> None:
        self.pred_data = pred_data
        self.target_data = target_data
        self.split_num = split_num
        self.make_splits()
        self.iter = 0    
    
    def make_splits(self):
        samples_per_split = self.pred_data.shape[0]//self.split_num + 1
        self.pred_data_per_split = []
        self.target_data_per_split = []
        for i in range(self.split_num):
            self.pred_data_per_split.append(self.pred_data[i*samples_per_split : min((i+1)*samples_per_split, self.pred_data.shape[0])])
            self.target_data_per_split.append(self.target_data[i*samples_per_split : min((i+1)*samples_per_split, self.target_data.shape[0])])

    
    def __iter__(self):
        return self
    
    
    def __next__(self):
        if self.iter == self.split_num:
            self.iter = 0
            raise StopIteration
        
        valid_pred = self.pred_data_per_split[self.iter]
        valid_target = self.target_data_per_split[self.iter]
        train_pred = None
        train_target = None
        for i in range(self.split_num):
            if i == self.iter:
                continue
            if train_pred is None:
                train_pred = self.pred_data_per_split[i]
                train_target = self.target_data_per_split[i]
            else:
                train_pred = np.concatenate((train_pred, self.pred_data_per_split[i]), axis=0)
                train_target = np.concatenate((train_target, self.target_data_per_split[i]), axis=0)
        self.iter = self.iter + 1
        return train_pred, train_target, valid_pred, valid_target
    
    
    def __getitem__(self, index):
        valid_pred = self.pred_data_per_split[index]
        valid_target = self.target_data_per_split[index]
        train_pred = None
        train_target = None
        for i in range(self.split_num):
            if i == index:
                continue
            if train_pred is None:
                train_pred = self.pred_data_per_split[i]
                train_target = self.target_data_per_split[i]
            else:
                train_pred = np.concatenate((train_pred, self.pred_data_per_split[i]), axis=0)
                train_target = np.concatenate((train_target, self.target_data_per_split[i]), axis=0)
        return train_pred, train_target, valid_pred, valid_target
    
    
    def __len__(self):
        return self.split_num


class Split:
  def __init__(self, x, y):
    self.X = x
    self.y = y

dataset_names = {    40588:"birds",
40589:"emotions",
40590:"enron",
40591:"genbase",
40592:"image",
40593:"langLog",
40594:"reuters",
40595:"scene",
40596:"slashdot",
40597:"yeast"}
