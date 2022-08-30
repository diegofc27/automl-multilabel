
import dataset
import numpy as np
import time
import torch
from torch import nn
from dehb import DEHB,ConfigVectorSpace,ConfigSpace
from dataset import OpenMlDataset, CrossValidation,Split
from data import MyDataset
from utils import train,test
from torch.utils.data import DataLoader
import rtdl
import wandb
import argparse


device = "cuda" if torch.cuda.is_available() else "cpu"


def obj(x,budget,**kwargs):

    seed = 42
    dataset = kwargs['dataset']
    
    cross_validation = CrossValidation(pred_data=dataset.train_pred_data, target_data=dataset.train_tar_data, split_num=3)

    score_tests = np.array([])

    for i, (tp, tt, vp, vt) in enumerate(cross_validation):
        loss_fn = nn.BCEWithLogitsLoss()


        train_split = Split(x=tp,y=tt)
        validation_split = Split(x=vp,y=vt)

        
        train_dataset = MyDataset(train_split)
        test_dataset = MyDataset(validation_split)

        
        train_dataloader = DataLoader(train_dataset, batch_size=256)
        test_dataloader = DataLoader(test_dataset, batch_size=256)

        n_inputs, n_outputs = tp.shape[1], tt.shape[1]

        config = { "d_in":n_inputs,
        "n_blocks":x["n_blocks"],
        "d_main":x["d_main"],
        "d_hidden":x["d_hidden"],
        "dropout_first":x["dropout_first"],
        "dropout_second":x["dropout_second"],
        "d_out":n_outputs}

        model = rtdl.ResNet.make_baseline(**config).to(device)  

        config["batch_size"] = 256
        config["epoch"] = budget
        config["weight_decay"] = x['weight_decay']

        optimizer = torch.optim.AdamW(model.parameters(), lr=x["lr"], weight_decay=x['weight_decay'])
        #Only save the first run
        if i==0:
          with wandb.init(
                  project="multilabel",
                  config=config,
                  group=str(dataset.dataset_id),
                  reinit=True,
                  mode="offline",
                  settings=wandb.Settings(start_method="thread")):

              for t in range(budget):
                  train(train_dataloader, model, loss_fn, optimizer,ftt=False,dataset_test=test_dataloader)
                  score_test = test(test_dataloader,model, loss_fn,ftt=False)
                  wandb.log({"val/f1_score": score_test})

              score_tests = np.append(score_tests,score_test)
        else:
            for t in range(budget):
                  train(train_dataloader, model, loss_fn, optimizer,ftt=False,dataset_test=test_dataloader)
                  score_test = test(test_dataloader,model, loss_fn,ftt=False)

            score_tests = np.append(score_tests,score_test)

    print(config)
    print("Mean f1 score of configuration: " + str(np.mean(score_tests)))
    return {'mean_F1':np.mean(score_tests), 'std_F1':np.std(score_tests)}

SEED = 42

#Get params
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=40597,  help="Select dataset id")
parser.add_argument("--min_budget", default=20,  help="Select min_budget")
parser.add_argument("--max_budget", default=400,  help="Select min_budget")
parser.add_argument("--min", default=2,  help="Select minutes")
args = parser.parse_args()

#Select dataset
dataset_id =args.dataset
current_dataset = OpenMlDataset(dataset_id,seed=SEED)
current_dataset.train_test_split()

#Run DEHB
rs = np.random.RandomState(seed=SEED)
space = ConfigVectorSpace(
    name="neuralnetwork",
    seed=SEED,
    space={
        "lr": ConfigSpace.UniformFloatHyperparameter("lr", lower=1e-4, upper=1e-3, log=True, default_value=1e-3),
        "dropout_first": ConfigSpace.Float('dropout_first', bounds=(0, 0.9), default=0.34, distribution=ConfigSpace.Normal(mu=0.5, sigma=0.35)),
        "dropout_second": ConfigSpace.Float('dropout_second', bounds=(0, 0.9), default=0.34, distribution=ConfigSpace.Normal(mu=0.5, sigma=0.35)),
        "weight_decay": ConfigSpace.Float("weight_decay", bounds=(0, 2), default=0.1),
        "n_blocks": ConfigSpace.UniformIntegerHyperparameter("n_blocks", lower=4, upper=5, default_value=5),
        "d_main": ConfigSpace.OrdinalHyperparameter("d_main", sequence=[256,512], default_value=512),
        "d_hidden" : ConfigSpace.OrdinalHyperparameter("d_hidden", sequence=[ 256, 512], default_value=512)
    },
)

dehb = DEHB(space, metric='mean_F1',min_budget=int(args.min_budget), max_budget=int(args.max_budget), rs=rs)

start_time = time.process_time()

#Print best configuration
print(f"\n Best configuration  {dehb.optimize(obj, limit=int(args.min),  unit='min', dataset=current_dataset)}")
print(f"Time elapsed (CPU time): {(time.process_time() - start_time):.4f} seconds")
print(f"Validation F1: {dehb.inc_score}")

#save data
dehb.save_data()

#Train best configuration with the whole training set 
train_split = Split(x=current_dataset.train_pred_data,y=current_dataset.train_tar_data)
test_split = Split(x=current_dataset.test_pred_data,y=current_dataset.test_tar_data)
loss_fn = nn.BCEWithLogitsLoss()
n_inputs, n_outputs = current_dataset.train_pred_data.shape[1], current_dataset.train_tar_data.shape[1]
train_dataset = MyDataset(train_split)
test_dataset = MyDataset(test_split)
train_dataloader = DataLoader(train_dataset, batch_size=256)
test_dataloader = DataLoader(test_dataset, batch_size=256)

config = { "d_in":n_inputs,
"n_blocks":dehb.inc_config['n_blocks'] ,
"d_main": dehb.inc_config['d_main'],
"d_hidden":dehb.inc_config['d_hidden'] ,
"dropout_first":dehb.inc_config['dropout_first'],
"dropout_second":dehb.inc_config['dropout_second'],
"d_out":n_outputs}

model = rtdl.ResNet.make_baseline(**config).to(device)  

optimizer = torch.optim.AdamW(model.parameters(), lr= dehb.inc_config['lr'], weight_decay=dehb.inc_config['weight_decay'])
budget = dehb.inc_config['budget']
for t in range(budget):
  train(train_dataloader, model, loss_fn, optimizer,ftt=False,dataset_test=test_dataloader)

#Calculate final F1 test score
score_test = test(test_dataloader,model, loss_fn,ftt=False)
print(f"Final Test F1: {score_test}")

