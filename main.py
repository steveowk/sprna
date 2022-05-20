import argparse
import random
import os
import random
from os import path
from utils import *
from copy import deepcopy
from RNADataLoader import CustomDataset
from torch.utils.data import DataLoader
from NNet import RnaNet
from TrainModel import Model
from RNAGymEnv import RNAEnv


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--root_path', help = "root dir", type = str, default = "./sprna/")
parser.add_argument('-f', '--model_file', type = str, help = "path to value network", default = "./sprna/model.ckpt")
parser.add_argument('-e', '--episodes', type = int, help = "number of episodes to play", default = 20000)
parser.add_argument('-b', '--batch_size', type = int, help = "batch size", default = 64)
parser.add_argument('-x', '--mx_epochs', type = int, help = "Max epochs to play", default = 30)
parser.add_argument('-w', '--num_workers', type = int, help = "number of workers", default = 8)
parser.add_argument('-a', '--replay_size', type = int, help = "replay memeory size", default = 100000)
parser.add_argument('-s', '--test_size', type = float, help = "test size per cent", default = 0.25)
parser.add_argument('-r', '--W', type = int, help = "the feature parameter W", default = 4)
parser.add_argument('-k', '--sample_train', type = int, help = "replay sample size to train on", default = 10000)
parser.add_argument('-z', '--sample_test', type = int, help = "replay sample size to test on", default = 10000)
parser.add_argument('-l', '--batch_playout_size', type = int, help = "batch size for the playout", default = 10)
parser.add_argument('-d', '--train_dir', type = str, help = "results directory", default = './sprna/train/')
parser.add_argument('-i', '--eval_dir', type = str, help = "eval results directory", default = './sprna/eval/')
parser.add_argument('-t', '--log_train', type = bool, help = "log train results", default = True)
parser.add_argument('-o', '--optimizer', type = str, help = "optimizer", default = "adam")
parser.add_argument('-c', '--loss', type = str, help = "loss", default = "mse")
parser.add_argument('-g', '--lr', type = float, help = "learnig rate", default = 0.0001)
parser.add_argument('-v', '--print_every', type = int, help = "freq of  printing the logs", default = 25)
args = parser.parse_args()


model = Model(RnaNet(), 
              args.mx_epochs, 
              args.optimizer,
              args.loss,
              args.model_file,
              args.batch_size,
              args.lr,
              args.print_every,
              0
              )
track_evaluation = trackEvalResults()
eval_data = loadValidation()
positive_train = deque([], maxlen = args.replay_size)
negative_train = deque([], maxlen = args.replay_size)
positive_test =  deque([], maxlen = args.replay_size)
negative_test =  deque([], maxlen = args.replay_size)

def value(state, actions, eval = False):
    status = True if len(list(actions[0])) == 2 else False
    global steps_done
    action_value, best_action_data  = [], []
    current_state_value = state.designedSequence()
    for a in actions:
        input_sample = state.code(a)
        input_tensor = torch.tensor(input_sample)
        input_tensor = input_tensor.view(1, 1, input_tensor.size(0),input_tensor.size(1))
        action_value.append(model.predict(input_tensor))
        best_action_data.append(input_sample)
        state.undoMove(current_state_value)
    action_value = torch.tensor(action_value).view(-1, 1)  
    action = decayingEgreedy(action_value) if not eval else evalArgmax(action_value)
    best_action = single_sites_ix[action] if not status else paired_sites_ix[action]
    steps_done+=1
    best_action_data = torch.tensor(best_action_data[action])
    state.coded_state_value.append(best_action_data)
    return best_action

def trainModel():
    xtrain, ytrain = sampleReplay(positive_train, negative_train, args.sample_train)
    xtest, ytest =  sampleReplay(positive_test, negative_test, args.sample_test)
    train_loader = DataLoader(CustomDataset(xtrain, ytrain), batch_size = args.batch_size, num_workers = args.num_workers)
    valid_loader = DataLoader(CustomDataset(xtest, ytest), batch_size = args.batch_size, num_workers = args.num_workers)
    model.trainer(train_loader, valid_loader)  

def logSummaries():
    keys = list(track_evaluation.keys())
    for k in keys:print(f"{k} score {len(track_evaluation[k])}/{len(eval_data[k])}")

def trainOrtest():
    if random.random() < float(args.test_size):return True
    return False

def labelStates(state, reward):
    if reward==1:
        for input_tensor in state.coded_state_value:
            if not trainOrtest():
                if len(positive_train)>=args.replay_size:positive_train.popleft()
                positive_train.append([input_tensor, torch.tensor([1.0])])
            else:
                if len(positive_test)>=args.replay_size:positive_test.popleft()
                positive_test.append([input_tensor, torch.tensor([1.0])])
    else:
        for input_tensor in state.coded_state_value:
                if not trainOrtest():
                    if len(negative_train)>=args.replay_size:negative_train.popleft()
                    negative_train.append([input_tensor, torch.tensor([-1.0])])
                else:
                    if len(negative_test)>=args.replay_size:negative_test.popleft()
                    negative_test.append([input_tensor, torch.tensor([-1.0])])  
                    
def SPRNABatchPlayout(B, P, U):#batch playout function
    for seq_id, seq in enumerate(B):
        print(f"Playing Seq. {seq_id+1}")
        state = RNAEnv(seq, args.W, P, U)
        sites = state.availableSites()
        sites = state.shuffleSites()
        for s in sites:
            state.current_site = s
            if state.currentPaired():best_action = value(state, P)
            else:best_action = value(state, U)
            state.applyMove(best_action)
        if state.hammingLoss() > 0:state.localSearch()
        labelStates(state, state.reward())
        if args.log_train:
            if not path.exists(args.train_dir):os.mkdir(args.train_dir)
        print(f"pos_test{len(positive_test)} pos_train {len(positive_train)} neg_test {len(negative_test)} neg_train {len(negative_train)}")

def evaluate(iteration, P, U):
    if not path.exists(args.eval_dir):os.mkdir(args.eval_dir)
    data = list(eval_data.keys())
    dirs = [f"{args.eval_dir}{x}/" for x in data]
    for p in dirs:
        if not path.exists(p):os.mkdir(p)
    for _ , (d, loc) in enumerate(zip(data,dirs)):
        print(f"Inference on {d} iteration {iteration}")
        for seq_id, seq in enumerate(eval_data[d]): 
            state = RNAEnv(seq, args.W, P, U)
            sites = state.availableSites()
            sites = state.shuffleSites()
            for s in sites:
                state.current_site = s
                if state.currentPaired():best_action = value(state, P)
                else:best_action = value(state, U)
            state.applyMove(best_action)
            if state.hammingLoss() > 0 : state.localSearch()
            if state.reward()== 1.0:track_evaluation[d].add(seq_id)
            writeSummary(seq_id, iteration, state.hammingLoss(), state.getDesigned(), loc+"summary_"+str(iteration)+".csv")
            if seq_id%args.print_every==0:logSummaries()

if __name__ == "__main__":
    if not path.exists(args.root_path):os.mkdir(args.root_path)
    P = ["GC","CG","AU","UA","UG","GU"] # paired sites
    U = ["G","A","U","C"] #unpaired sites
    load_train = loadCandD("train") # train data
    for iteration in range(args.episodes):
        B = random.sample(load_train, args.batch_playout_size)
        SPRNABatchPlayout(B, P, U) #batch playout
        trainModel()
        evaluate(iteration, P, U) #evaluate on A, B, C, & D
    print(f"Done! ...")
