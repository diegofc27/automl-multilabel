import os
import numpy as np
import sys
import time
import re
import math

np.set_printoptions(threshold=sys.maxsize)

class Dataset():
    def __init__(self, dataset_number, threshold=0.7, seed = 0) -> None:
        self.features_name = []
        self.features_type = []
        self.features_possible_values = []
        self.features_unique_value = []
        self.data = []
        self.random_generator = np.random.RandomState(seed)
        self.threshold = threshold
        self.dataset_number_to_name(dataset_number)
        
        self.initialization()
    
    def load_dataset(self):
        
        # Get path
        absolute_path = os.getcwd()
        full_path = absolute_path + "\\datasets\\" + self.dataset_name + ".arff"
        
        # Read file
        with open(full_path, 'r') as f:
            lines = f.readlines()
        
        # Load dataset by reading file line by line
        state = 0
        for line in lines:
            strings = line[:-1].split(" ", 2)
            if strings[0] == "@relation":
                print("Begin to read dataset")
                
            elif strings[0] == "@attribute":
                self.features_name.append(strings[1][1:-1])
                if strings[2][0] == "{" and strings[2][-1] == "}":
                    if len(strings[2].split(",")) == 2:
                        self.features_type.append("binary")
                    else:
                        self.features_type.append("categorical")
                else:
                    self.features_type.append(strings[2])
                
                if self.features_type[-1] == "categorical" or self.features_type[-1] == "binary":
                    values = strings[2][1:-1].replace(" ", "")
                    values = values.replace("'", "")
                    values = values.split(",")
                    self.features_possible_values.append(values)
                else :
                    self.features_possible_values.append("endless")
                
            elif strings[0] == "@data":
                # From now Read data
                state = 1
            
            elif state == 1:
                tmp_line = line[:-1].replace('"', "")
                tmp_line = tmp_line.replace(' ', "")
                values = tmp_line.split(",")
                sample = []
                for i in range(len(values)):
                    value = values[i]
                    if self.features_type[i] == "numeric":
                        sample.append(float(value))
                    
                    elif self.features_type[i] == "binary" or self.features_type[i] == "categorical":
                        value = self.features_possible_values[i].index(value)
                        sample.append(float(value))
                self.data.append(sample)
            else:
                assert False, "Got to invalid case: state "+str(state)+" first string "+strings[0]                                
                
                
        self.data = np.array(self.data)
        for i in np.arange(self.data.shape[1]):
            self.features_unique_value.append(np.unique(self.data[:,i]))
            if len(self.features_unique_value[i]) == 2 and self.features_type == "numeric":
                self.features_type[i] = "binary"
                self.features_possible_values = self.features_unique_value
                for j in range(self.data.shape[0]):
                    self.data[j,i] = float(self.features_possible_values[i].index(value))
                

        
    def check_for_single_value_parameters(self):
        
        params_to_delete = []
        for i in np.arange(self.data.shape[1]-1, -1, -1):
            if self.features_name[i] in self.target_features:
                continue
            unique_values = self.features_unique_value[i]
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
            if self.features_name[i] in self.target_features or self.features_type[i] == "numeric":
                continue
            for j in np.arange(i-1, -1, -1):
                if self.features_name[j] in self.target_features or self.features_type[j] == "numeric":
                    continue
                # Check for pseudocorrelation
                temp_i = self.data[:,i]
                temp_j = self.data[:,j]
                uniq_val_num_i = len(self.features_unique_value[i])
                uniq_val_num_j = len(self.features_unique_value[j])
                if uniq_val_num_i > uniq_val_num_j:
                    lower = j
                else:
                    lower = i
                temp_i = temp_i * uniq_val_num_j
                sum = temp_i + temp_j
                uniq_val = np.unique(sum)
                if uniq_val.shape[0] == uniq_val_num_i or uniq_val.shape[0] == uniq_val_num_j:
                    if not(lower in params_to_delete):
                        params_to_delete.append(lower)
                        #print(self.features_name[i]," ",self.features_name[j]," ",lower)
                        
                    #break

        old = self.data.shape[1]
        self.data = np.delete(self.data, params_to_delete, 1)
        self.clear_feature_data(params_to_delete)
        new = self.data.shape[1]
        print("Removed ",old - new," parameters with pseudocorrelated values")
        #print(params_to_delete)
              
              
    def check_for_correlation(self):
        params_to_delete = []
        for i in range(len(self.features_type)):
            if self.features_name[i] in self.target_features or self.features_type[i] != "numeric":
                continue
            for j in range(len(self.features_type)):
                if self.features_name[j] in self.target_features or self.features_type[j] == "numeric":
                    continue
                
                sum_i = np.sum(self.data[:, i])
                sum_j = np.sum(self.data[:, j])
                sum_i2 = np.sum(self.data[:, i]**2)
                sum_j2 = np.sum(self.data[:, j]**2)
                sum_ij = np.sum(self.data[:, i] * self.data[:, j])
                n = self.data.shape[0]
                
                correlation = (n*sum_ij - sum_i * sum_j)/(math.sqrt((n * sum_i2 - (sum_i)**2)*(n * sum_j2 - (sum_j)**2)))
                correlation = abs(correlation)
                if correlation>= self.threshold:
                    if len(self.features_unique_value[i])>=len(self.features_unique_value[j]):
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
        indices = indices[::-1]
        for i in indices:
            self.features_name.pop(i)
            self.features_type.pop(i)
            self.features_possible_values.pop(i)
            self.features_unique_value.pop(i)   
        
        
    def find_targets(self):
        
        if self.dataset_name == "birds":
            pattern = r"^[A-Z][a-z]"
            number_of_targets = 19
        elif self.dataset_name == "emotions":
            pattern = r"[a-z]\.[a-z]"
            number_of_targets = 6
        elif self.dataset_name == "enron":
            pattern = r"[A-Z]\.[A-Z][0-9]+"
            number_of_targets = 53
        elif self.dataset_name == "genbase":
            pattern = r"PDOC"
            number_of_targets = 27
        elif self.dataset_name == "image":
            pattern = r"^((?!Feature).)*$"
            number_of_targets = 5
        elif self.dataset_name == "langLog":
            pattern = r"^((?!tok[0-9]+).)*$"
            number_of_targets = 75
        elif self.dataset_name == "reuters":
            pattern = r"^label[1-7]"
            number_of_targets = 7
        elif self.dataset_name == "scene":
            pattern = "^((?!Att[0-9]).)*$"
            number_of_targets = 6
        elif self.dataset_name == "slashdot":
            pattern = "^[A-Z][a-zA-Z]"
            number_of_targets = 22
        elif self.dataset_name == "yeast":
            pattern = r"Class[1-9]"
            number_of_targets = 14
        else: 
            assert False, "Wrong dataset name : "+self.dataset_name
        
        self.target_features = []
        for i in range(self.data.shape[1]-1, -1, -1):
            decision = re.search(pattern, self.features_name[i])
            if decision:
                self.target_features.append(self.features_name[i])
                self.features_name
        self.number_of_targets = number_of_targets
        assert len(self.target_features) == self.number_of_targets, "Found number of targets "+str(len(self.target_features))+" differs from true number "+str(self.number_of_targets) 
        
        
    def check_for_values_with_one_occurance(self):
        
        params_to_delete = []
        for i in np.arange(self.data.shape[1]-1, -1, -1):
            if self.features_name[i] in self.target_features:
                continue
            if self.features_type[i] == "numeric":
                continue
            temp = (self.data[:,i]).astype(int)
            occurances = np.bincount(temp)
            if 1 in occurances:
                params_to_delete.append(i)

        old = self.data.shape[1]
        self.data = np.delete(self.data, params_to_delete, 1)
        self.clear_feature_data(params_to_delete)
        new = self.data.shape[1]
        print("Removed ",old - new," parameters with 1 occurance of value")
        #print(params_to_delete)
    
    
    def check_for_duplicated_samples(self):
        
        old = self.data.shape[0]
        self.data = np.unique(self.data, axis=0)
        new = self.data.shape[0]
        print("Removed ",old - new," samples due to being duplicates")
    
    
    def normalize_dataset(self):
        
        for i in range(self.data.shape[1]):
            if self.features_name in self.target_features:
                continue
            if not self.features_type == "numeric":
                continue
            
            average = np.mean(self.data[:, i])
            
            self.data[:, i] = self.data[:, i] - average
            
            maximum = np.amax(np.absolute(self.data[:,i]))
            
            self.data[:, i] = self.data[:, i]/maximum
            
    
    def dataset_split(self, train_ratio : float):
        
        perm_idxs = np.arange(self.data.shape[0])
        self.random_generator.shuffle(perm_idxs)
        split_idx = int(self.data.shape[0]*train_ratio)
        train_idxs = perm_idxs[:split_idx]
        valid_idxs = perm_idxs[split_idx:]
        
        train_pred_data = self.predictors[train_idxs]
        train_tar_data = self.targets[train_idxs]
        valid_pred_data = self.predictors[valid_idxs]
        valid_tar_data = self.targets[valid_idxs]
        return train_pred_data, train_tar_data, valid_pred_data, valid_tar_data
        
       
    def dataset_to_predictors_targets(self):
        
        self.predictors = []
        self.targets = []
        self.predictors_features_type = []
        self.targets_features_type = []
        
        for i in range(self.data.shape[1]):
            if self.features_name[i] in self.target_features:
                self.targets.append(self.data[:,i])
                self.targets_features_type.append(self.features_type[i])
            else:
                self.predictors.append(self.data[:,i])
                self.predictors_features_type.append(self.features_type[i])
                
        self.predictors = np.array(self.predictors)
        self.targets = np.array(self.targets)
        self.targets = np.swapaxes(self.targets, 0, 1)
        self.predictors = np.swapaxes(self.predictors, 0, 1)
        
    
    def dataset_number_to_name(self, number):
        dataset_names = ["birds", "emotions", "enron", "genbase",
                         "image", "langLog", "reuters", "scene",
                         "slashdot", "yeast"]
        
        assert number < len(dataset_names), "Chosen number "+str(number)+" is outside possible values 0-"+str(len(dataset_names)-1)
        self.dataset_name = dataset_names[number]
        print("Chosen dataset is "+dataset_names[number])
            
        
    def one_hot_encoding(self, x):
        length = x.shape[0]
        max_value = int(np.max(x))
        new_array = np.zeros((length, max_value+1))
        new_array[np.arange(length), x.astype(int)] = 1.0
        return new_array
    
    
    def dataset_to_one_hot_encoding(self):
        for i in range(len(self.predictors_features_type)-1, -1, -1):
            if self.predictors_features_type[i] == "categorical":
                ar = self.one_hot_encoding(self.predictors[:,i])
            else:
                ar = self.predictors[:,i]
                ar = np.reshape(ar, (-1, 1))

            if i+1 == len(self.predictors_features_type):
                self.predictors = np.concatenate((self.predictors[:,:i], ar),axis=1)
            else:
                self.predictors = np.concatenate((self.predictors[:,:i], ar, self.predictors[:,i+1:]),axis=1)
        
        for i in range(len(self.targets_features_type)-1, -1, -1):
            if self.targets_features_type[i] == "categorical":
                ar = self.one_hot_encoding(self.targets[:,i])
            else:
                ar = self.targets[:,i]
                ar = np.reshape(ar, (-1, 1))
            if i+1 == len(self.targets_features_type):
                self.targets = np.concatenate((self.targets[:,:i], ar),axis=1)
            else:
                self.targets = np.concatenate((self.targets[:,:i], ar, self.targets[:,i+1:]),axis=1)
            
    def initialization(self):
        start = time.process_time()
        self.load_dataset()
        self.find_targets()
        self.check_for_single_value_parameters()
        self.check_for_values_with_one_occurance()
        self.check_for_duplicated_samples()
        self.check_for_pseudocorrelation()
        self.check_for_correlation()
        self.normalize_dataset()
        self.dataset_to_predictors_targets()
        self.dataset_to_one_hot_encoding()
        
        print("It took ",time.process_time() - start)
        print("Final shape of the dataset (predictors + tagrets) is ",self.data.shape)
        print("Number of predictors : ", self.data.shape[1] - len(self.target_features), " and number of targets : ",len(self.target_features))





class DatasetSampler():
    def __init__(self, train_pred, train_target, valid_pred, valid_target, train_bs = 64, valid_bs = 64, seed = 0, reshuffle = False, mode = "train") -> None:
        self.train_pred = train_pred
        self.train_target = train_target
        self.train_bs = train_bs
        
        self.valid_pred = valid_pred
        self.valid_target = valid_target
        self.valid_bs = valid_bs
        
        self.random_generator = np.random.RandomState(seed)
        self.reshuffle = reshuffle
        
        self.train_iter = 0
        self.valid_iter = 0
        
        self.train_iter_max = math.ceil(self.train_pred.shape[0] / self.train_bs)
        self.valid_iter_max = math.ceil(self.valid_pred.shape[0] / self.valid_bs)
        
        assert mode == "train" or mode == "valid", "Chosen mode is not train or valid. Chosen mode is"+mode
        self.mode = mode
        
        
    def sample_batch(self, pred, target, bs, iter):
             
        pred_batch = pred[iter*bs : min((iter+1)*bs ,pred.shape[0])]
        target_batch = target[iter*bs : min((iter+1)*bs ,pred.shape[0])]
        return pred_batch, target_batch 
    
    
    def get_batch(self):
                
        if self.mode == "train":
            if self.train_iter == self.train_iter_max :
                if self.reshuffle:
                    perm = np.arange(self.train_pred.shape[0])
                    self.random_generator.shuffle(perm)
                    self.train_pred = self.train_pred[perm]
                    self.train_target = self.train_target[perm]
                self.train_iter = 0
                raise StopIteration
                
            pred_batch, target_batch = self.sample_batch(self.train_pred, self.train_target, self.train_bs, self.train_iter)  
            self.train_iter = self.train_iter + 1
            
        elif self.mode == "valid":
            if self.valid_iter == self.valid_iter_max :
                if self.reshuffle:
                    perm = np.arange(self.valid_pred.shape[0])
                    self.random_generator.shuffle(perm)
                    self.valid_pred = self.valid_pred[perm]
                    self.valid_target = self.valid_target[perm]
                self.valid_iter = 0
                raise StopIteration
            
            pred_batch, target_batch = self.sample_batch(self.valid_pred, self.valid_target, self.valid_bs, self.valid_iter)
            self.valid_iter = self.valid_iter + 1
            
        return pred_batch, target_batch
        
    
    def reset_iterators(self):
        self.train_iter = 0
        self.valid_iter = 0
        
    
    def set_mode(self, mode):
        assert mode == "train" or mode == "valid", "Chosen mode is not train or valid. Chosen mode is"+mode
        self.mode = mode
        
    def __iter__(self):
        return self
    
    
    def __next__(self):
        return self.get_batch()
    
    
    def __getitem__(self, index):
        
        assert self.mode != "train" or self.train_iter_max >= index, "Index outside range :"+str(self.train_iter_max)+" got "+str(index)
        assert self.mode != "valid" or self.valid_iter_max >= index, "Index outside range :"+str(self.valid_iter_max)+" got "+str(index)
        
        if self.mode == "train":
            temp = self.train_iter
            to_return = self.get_batch()
            self.train_iter = temp
            
        elif self.mode == "valid":
            temp = self.valid_iter
            to_return = self.get_batch()
            self.valid_iter = temp
            
        else:
            assert False, "Invalid mode :"+self.mode
            
        return to_return
            
    def __len__(self):
        if self.mode == "train":
            to_return = self.train_iter_max
            
        elif self.mode == "valid":
            to_return = self.valid_iter_max
            
        else:
            assert False, "Invalid mode :"+self.mode
    
        return to_return
        
# TESTING
#for i in range(10):
dataset = Dataset(1)
train_bs = 16
valid_bs = 32
tr_pred, tr_tar, val_pred, val_tar = dataset.dataset_split(0.7)
#print(tr_pred.shape)
#print(tr_tar.shape)
#print(val_pred.shape)
#print(val_tar.shape)
data_sampler = DatasetSampler(tr_pred, tr_tar, val_pred, val_tar, train_bs, valid_bs, 0 , True, "train")

for i, (x,y) in enumerate(data_sampler):
    print(i," ",x.shape, " ", y.shape)

data_sampler.set_mode("valid")

for i, (x,y) in enumerate(data_sampler):
    print(i," ",x.shape, " ", y.shape)

for i, (x,y) in enumerate(data_sampler):
    print(i," ",x.shape, " ", y.shape)
# pred_batch, target_batch = data_sampler.get_batch("valid")
# print(target_batch)
# pred_batch, target_batch = data_sampler.get_batch("train")
# print(target_batch)
# pred_batch, target_batch = data_sampler.get_batch("valid")
# print(target_batch)
# pred_batch, target_batch = data_sampler.get_batch("valid")
# print(target_batch)
# pred_batch, target_batch = data_sampler.get_batch("valid")
# print(target_batch)
# pred_batch, target_batch = data_sampler.get_batch("valid")
# print(target_batch)
# data_sampler.reset_iterators()
# pred_batch, target_batch = data_sampler.get_batch("valid")
# print(target_batch)
# pred_batch, target_batch = data_sampler.get_batch("valid")
# print(target_batch)
#print(target_batch)


# dataset.load_dataset()
# #print(dataset.features_name)
# #print(dataset.features_type)
# #print(dataset.data[0:2])
# start = time.process_time()
# dataset.find_targets()
# dataset.check_for_single_value_parameters()
# dataset.check_for_values_with_one_occurance()
# dataset.check_for_duplicated_samples()
# #print(dataset.features_unique_value_num[997])
# #print(dataset.features_unique_value_num[934])
# #print(np.unique(dataset.data[:,155+2]))
# #print(np.unique(dataset.data[:,152+1]))
# #print(np.unique(dataset.data[:,155+2] + dataset.data[:,152+1]*2))
# #print(dataset.features_unique_value_num)
# dataset.check_for_correlation()
# dataset.normalize_dataset()


# print(dataset.target_features)
# print(len(dataset.target_features))
# print("It took ",time.process_time() - start)
# dataset.dataset_split(0.7)
# print(dataset.data.shape)
# #print(dataset.data) 
# #print(dataset.features_unique_value_num)
# #print(dataset.features_name)                          