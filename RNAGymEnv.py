from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from copy import deepcopy
from random import shuffle
from sklearn.metrics import hamming_loss as norm_hamming
from utils import single_sites, paired_sites, all_sites_ix, getAllPairings, binaryCodings, generateW, loadCandD
import RNA
from itertools import product
from distance import hamming as raw_hamming

class RNAEnv(Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, target_seq, W, P, U):
        self.target_seq = list(target_seq)
        self.single_sites = U
        self.paired_sites  = P
        self.paired_loc, self.unpaired_loc = getAllPairings(target_seq)
        self.designed_seq = deepcopy(self.target_seq)
        self.mutation_threshold = 10
        self.move_locations = self.paired_loc+self.unpaired_loc 
        self.binary_codes = binaryCodings(single_sites+paired_sites)
        self.current_site = None
        self.coded_state_value = []
        self.W = W        
        # Gym interface constraints
        self.train = loadCandD("train") 
        self.feature_list = list(generateW(self.move_locations, self.W))
        self.action_space = Discrete(len(all_sites_ix.keys()))
        self.observation_space = Box(low = 0,  high = 1, shape=(len(self.feature_list)*4,), dtype = np.int8)
    
    def availableSites(self):
        return deepcopy(self.move_locations)
    
    def designedSequence(self):
        return deepcopy(self.designed_seq)
    
    def shuffleSites(self):
        shuffle(self.move_locations)
        return self.availableSites()
    
    def clone(self):
        return deepcopy(self)

    def currentPaired(self):
        if self.current_site in self.paired_loc:return True
        return False 

    def undoMove(self, seq):
        self.designed_seq = seq

    def applyMove(self, action):self.step(action)

    def terminal(self):
        if "." not in self.designed_seq and "(" not in self.designed_seq:return True
        return False

    def step(self, action):#can be used as Gym interface standalone
        if self.currentPaired():          
            move_loc_x, move_loc_y = self.current_site
            base_tox, base_toy  = action
            self.designed_seq[move_loc_x] = base_tox
            self.designed_seq[move_loc_y] = base_toy
        else: 
            move_loc_x = self.current_site
            self.designed_seq[move_loc_x] = action
        
        if self.terminal():done=True
        else:done=False
        if done:reward = 1.0
        else:reward=0.0
        #uncomment to experimenet with the Gym interface
        #return self.coded_state_value[-1], reward, done, {}
    
    def code(self, action):
        self.applyMove(action)
        binary_codes = self.binary_codes
        Array = []
        for feature in self.feature_list:
            X = []
            for location in feature:
                if isinstance(location, int):
                    charx = self.designed_seq[location]
                    if charx == ".": X.extend(list(binary_codes['unknown_single']))#unknown single              
                    else:X.extend(list(binary_codes[charx])) #known single
                else:
                    locx, locy = location
                    charx, chary = self.designed_seq[locx],self.designed_seq[locy]
                    if charx == "(" and chary ==")" : X.extend(list(binary_codes['unknown_pair']))#unknown pair
                    else: X.extend(list(binary_codes[charx+chary]))
            X = list(map(int, X))
            Array.append(X)
        assert(Array)
        return Array
      
    def hammingLoss(self):
        pred = RNA.fold(''.join(self.designed_seq))[0]
        assert(len(list(pred))==len(self.target_seq))
        return norm_hamming(list(pred), self.target_seq)
     
    def getDiff(self):
       pred = list(self.getDesignedFold())
       target = list(self.getTarget())
       assert(len(pred)==len(target))
       return [index for index in range(len(target)) if target[index]!=pred[index]]
  
    def getMutated(self, mutations, sites):
       mutated_primary = self.designedSequence()
       for site, mutation in zip(sites, mutations):
           mutated_primary[site] = mutation
       return "".join(mutated_primary)
 
    def hamming_distance(self, seqA, seqB):return raw_hamming(seqA, seqB)
    
    def localSearch(self):
        if self.hammingLoss()==0.0:return
        target = self.getTarget()
        folded_design = self.getDesignedFold()
        hamming_distance = self.hamming_distance(folded_design, target)
        if 0 < hamming_distance <= self.mutation_threshold:
            _ , self.designed_seq = self.localImprovement()

    def localImprovement(self):
       target = self.getTarget()
       if float(self.hammingLoss()) == 0.0: return 0.0, list(self.getDesigned())
       differing_sites = self.getDiff()
       hamming_distances = {}
       for mutation in product("AGCU", repeat = len(differing_sites)):
           mutated = self.getMutated(mutation, differing_sites)
           folded_mutated, _ = RNA.fold(mutated)
           hamming_distance = self.hamming_distance(folded_mutated, target)
           hamming_distances[mutated] = hamming_distance
           if hamming_distance==0: 
               print(f"returning the correct fold from local search")
               return 0.0, list(mutated)
       res = sorted(hamming_distances.items(), key = lambda x:x[1], reverse = False)
       targetx, distance = res[0]
       return distance, list(targetx)
    
    def getTarget(self):return "".join(deepcopy(self.target_seq))
    
    def getDesigned(self): return "".join(deepcopy(self.designed_seq))
    
    def getDesignedFold(self): return RNA.fold(self.getDesigned())[0]

    def reset(self):#to use with gym interface as a standlone
        self.game = RNAEnv(random.choice(self.train))
        return self.coded_state_value[-1]
    
    def reward(self):
        if self.hammingLoss() == 0.0:return 1.0
        return -1.0    
        
    def render(self, mode='human'):pass #override for Gym interface standalone

    def close (self):pass #override for Gym interface standalone