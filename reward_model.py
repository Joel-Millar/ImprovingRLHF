import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import itertools
import tqdm
import copy
import scipy.stats as st
import os
import time
import math
import random

from scipy.stats import norm
from scipy import stats
from sklearn.metrics import silhouette_score as ss
from sklearn.metrics import davies_bouldin_score as db
from sklearn.metrics import calinski_harabasz_score as ch
device = 'cuda'

# Build a single neural network with given parameters which influence the layers used 
def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []

    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net


def gen_net2(in_size=1, out_size=1, H=128, n_layers=3, activation='sig'):
    net = []
    
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.Dropout(0.5))
        in_size = H
    net.append(nn.Linear(in_size, out_size))

    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

def gen_net3(in_size=1, out_size=1, H=128, n_layers=3, activation='sig'):
    net = []
    for i in range(n_layers*3):
        net.append(nn.Linear(in_size, H))
        in_size = H
    net.append(nn.Linear(in_size, out_size))

    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net


def KCenterGreedy(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index

def KCenterGreedyMax(obs, full_obs, num_new_sample):
    selected_index = []
    current_index = list(range(obs.shape[0]))
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_largest_dist(new_obs, new_full_obs)
        max_index = torch.argmax(dist)
        max_index = max_index.item()

        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]

        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs,
            obs[selected_index]],
            axis=0)
    return selected_index


def compute_largest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx+1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                max_dists = torch.torch.max(dists, dim=1).values
                total_dists.append(max_dists)
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)


def compute_smallest_dist(obs, full_obs):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)

class RewardModel:

    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0):
        
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        
        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
                
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin

    '''Ensemble learning is based on the principle that combining the predictions or decisions of
    multiple models can often lead to better results than relying on a single model'''
    
    # Calculates the Softmax Cross-Entropy Loss
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    # Change the batch by a percentage
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)

    # Change the batch to be a given value
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
    
    # Change the teacher threshold for skiping the comparison of 2 clips by a given value
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
    
    # Change the teacher threshold for giving input that both clips are equal
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
    
    # Builds multiple nn models using gen_net to produce an ensemble (group of models)
    # and add the different layers.
    # The ensemble list appends this neural network model
    # The paramlst then appends all values in the model.parameters structure
    # opt class variable is then assigned to be Adam optimizer
    # Adam computes adaptive learning rates for each parameter by estimating the first and second moments of gradients
    def construct_ensemble(self):
        #for i in range(self.de):
            #model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           #out_size=1, H=256, n_layers=3, 
                                           #activation=self.activation)).float().to(device)

        model = nn.Sequential(*gen_net(in_size=self.ds+self.da,
                                      out_size=1, H=256, n_layers=3, activation=self.activation)).float().to(device)

        self.ensemble.append(model)
        self.paramlst.extend(model.parameters())
        
        model = nn.Sequential(*gen_net2(in_size=self.ds+self.da, out_size=1, H=256, n_layers=3, activation=self.activation)).float().to(device)

        self.ensemble.append(model)
        self.paramlst.extend(model.parameters())

        model = nn.Sequential(*gen_net3(in_size=self.ds+self.da, out_size=1, H=256, n_layers=3, activation=self.activation)).float().to(device)

        self.ensemble.append(model)
        self.paramlst.extend(model.parameters())
    
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
    
    # Parameters : obvs = state, act = action taken, rew = reward recieved, done = true when episode finished
    # Creates an array containing the state-action pair (sa_t)
    # Reshapes the input to have dimensions 1x(the sum of action and state dimensions)
    # Reshapes target to be 1x1
    # Checks if self.inputs is empty, will append the input and target to the inputs and targets lists
    # If the episode is done then the current flat_target and flat_input are appended to the last element of the lists targets and inputs
    # If the length of inputs is greater than max_size then oldest data is removed
    # New empty lists inputs and targets are initialised for the next episode
    # If the episode is not done and the last element of inputs is empty:
    # last element is initialised with flat_input and flat_target
    # If the last element is not empty then it concatenates the current flat_input and flat_target to the last elements of inputs and targets
    # 
    # Function manages input-target pairs, storing state-action pairs(flat_input) and corresponding rewards(flat_target)
    def add_data(self, obs, act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])

    # obses is a list of observations made by the agent
    # Determines the number of environments from obses row count
    # For each environment the observation is added to inputs and the reward is added to targets
    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
    
    # Calculates if the probability of x_1 is greater than that of x_2 using the p_hat_member function for each member
    # Loops through ensemble and calculates the probability that x_1 is greater than x_2 and appends as a numpy array
    # Returns the mean probability of x_1 > x_2 and the standard deviation
    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    # Finds the probability that x_1 is greater than x_2 using the entropy of the probability distribution
    # Returns the mean probability and the standard deviation
    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    # Calculates the probability that x_1 is greater than x_2 for a specific member
    # Softmax function is used to convert the outputs of model predictions to probabilities
    # Predicts values for x_1 and x_2, sum these predictions along axis 1 - reducing multi-dimensional outputs to 1 dimension
    # Summed predictions are concatenated along the last dimension, forming r_hat which combines scores for x_1 and x_2
    # Softmax function is applied to r_hat which converts scores to probabilities
    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    # Computes the entropy of the probability distribution that x_1 is greater than x_2 for a specific member
    # Follows same steps as p_hat_member however then computes entropy and returns it
    def p_hat_entropy(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    # Computes the predicted values for a given x using a specific member of an ensemble of models
    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    # Computes the average prediction for a given input x using all members in an ensemble
    # Loops through each ensemble member and computes their prediction, then return the mean
    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    # For each member in an ensemble compute their prediction and then return mean along axis 0
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    # For each member in the ensemble save their state dictionary.
    # Params include where to save and the current epoch of the models
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
    
    # Load an ensemble from a given directory and a specific epoch to specify the stage of training
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    
    # Calculates the training accuracy of an ensemble over a dataset stored in a buffer
    # Initialises a zero filled numpy array with same length as ensemble
    # num_epochs is the number of epochs needed to cover a dataset of max_len samples using specified batch_size
    # Iterates for num_epochs and calculates the end index of the current batch, then retrives sa_t_1, sa_t_2 and labels from the current batch
    # Converts the labels to a tensor flattening and casting to type long
    # For each ensemble member computes predictions r_hat using r_hat_member, sums logits, concatenates and calculates accuracy
    # The accuracy is then computed and returned.
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    # Extracts batches of data from a stored dataset(inputs and targets)
    # Fills train_inputs and train_targets with the tail of inputs and targets
    # Randomly selects an indices twice and allocates it to sa_t_x and r_t_x where x is the batch index
    # Reshapes these values to be suitable for processing
    # Generate time indices for segementation
    # Segments sa_t_x and r_t_x based on generated time indices
    # Returns the segmented state-action pairs  
    def get_queries(self, mb_size=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
   
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2] # Batch x T x 1
        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1] # Batch x T x 1
                
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (Batch x T) x 1

        # Generate time index 
        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
                
        return sa_t_1, sa_t_2, r_t_1, r_t_2

    # Put batches of queries (state-action pairs) into a buffer
    # Determines the total number of samples
    # Calculates the next index in the buffer
    # When there is enough space, fills the buffer index
    # If there are samples left over after filling the buffer then they are copied into the beginning
    # When there is not enough space, state action pairs and labels are copied into the buffer from self.buffer_index to next_index 
    def put_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
    
    # Assigns state-action pairs with their reward
    # Calculates the total sum reward for each sample within the time step
    # Check the rewards dont exceed the max threshold
    # Identify the queries where the difference is summed rewards is less than teacher_thres_equal
    # Modify the rewards based on teacher_gamma
    # Apply the Bradley-Terry model is teacher_beta > 0 otherwise use previously computed rational_labels
    # Introduce random noise by randomly flipping labels to account for uncertainty
    # Return the processed state-action pairs, rewards and associated labels
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        if self.teacher_thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1 
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
    
    # Uses clustering to sample and label data points based off their representative nature compared to existing data
    # Gets the state-action pairs and their associated reward
    # Prepare the data for clustering by concatenating them and reshaping them
    # Preapre existing data for clustering by flattening and retrieving existing state-action pairs
    # Perform clustering using the K-Centre Greedy algorithm
    # Update the queries to retain only the selected indices
    # If the labels are non-empty then store the labeled queries and their labels
    #  Return the number of labeled queries 
    def kcenter_sampling(self):
        
        # get queries
        num_init = self.mb_size*self.large_batch
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1),  
                                  temp_sa_t_2.reshape(num_init, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)

        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    # Provides a more advanced sampling method than kcenter_sampling whilst still using clustering
    # Retreives the initial queries (state-action pairs and their associated rewards)
    # Evaluates the uncertainty - Identifies which queries the models predictions are uncertain or disagree
    # Selects the queries with the highest disagreement score
    # Retrieve and flatten existing state-action pairs from the buffer for comparison and clustering
    # Perform K-centre clustering on the selected queries
    # Update the state-action pairs and their associated reward based on selected_index
    # Assign labels to the state-action pairs based on their rewards
    # store the labeled queries and their labels
    # Return the number of labeled queries
    def kcenter_disagree_sampling(self):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    # Extends the kcentre_sampling by introducing entropy-based uncertainty sampling
    # Get initial queries
    # Compute entropy
    # Select top entropic queries
    # Prepare data for clustering 
    # Prepare exiting data for clustering
    # Perform K-Centre greedy clustering on selected queries
    # Select and updated selected queries
    # Get labels for selected queries
    # Store labeled queries
    # Return the number of labeled queries
    def kcenter_entropy_sampling(self):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=num_init)
        
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]
        
        temp_sa = np.concatenate([temp_sa_t_1.reshape(num_init_half, -1),  
                                  temp_sa_t_2.reshape(num_init_half, -1)], axis=1)
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        
        tot_sa_1 = self.buffer_seg1[:max_len, :, :self.ds]
        tot_sa_2 = self.buffer_seg2[:max_len, :, :self.ds]
        tot_sa = np.concatenate([tot_sa_1.reshape(max_len, -1),  
                                 tot_sa_2.reshape(max_len, -1)], axis=1)
        
        selected_index = KCenterGreedy(temp_sa, tot_sa, self.mb_size)
        
        r_t_1, sa_t_1 = r_t_1[selected_index], sa_t_1[selected_index]
        r_t_2, sa_t_2 = r_t_2[selected_index], sa_t_2[selected_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    # Performs sampling where queries are selected uniformly at random from the dataset
    # Retrieve queries(state-action pairs and associated reward)
    # Retrieve the labels for the retrieved queries
    # If there are > 0 labels then add the queries to the buffer
    # Return the number of labels
    def uniform_sampling(self):
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size)
            
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
            
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)
    
    # Performs sampling where the selected selected queries have a high uncertainty between their prediction
    def disagreement_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]        
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    # Performs sampling where queries are selected based on the uncertainty or entropy of predictions made by the model
    def entropy_sampling(self):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(    
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels)
    
    def custom_sampling(self):

        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size*self.large_batch)
        
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        
        top_k_index = (-disagree).argsort()[:self.mb_size]
        bottom_k_index = (disagree).argsort()[:self.mb_size]

        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[bottom_k_index], sa_t_2[bottom_k_index]

        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def custom_clustering_sampling(self):
        # KMeans cluster sampling
  
        # Include pre-processing of data
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(
            mb_size=self.mb_size*self.large_batch)

        n_clusters = 2

        sa_comb = np.concatenate([sa_t_1, sa_t_2], axis=0)
        rt_comb = np.concatenate([r_t_1, r_t_2], axis=0)
        
        # Apply PCA feature reduction
        orig_data = sa_comb.copy()
        sa_comb = self.preprocess_data(sa_comb)

        c1 = Cluster(sa_comb, n_clusters)

        attempt = 5

        sil_score1 = float('-inf')
        sil_score2 = -1000

        for i in range(attempt):
            d_p, d_c, o_p  = c1.kmeans(start=2)

            d_c = np.array(d_c)

            sil_avg = self.evaluate(sa_comb, d_c)
            
            if sil_avg > sil_score1:
                best_dp1 = d_p
                best_dc1 = d_c
                best_op1 = o_p
                sil_score1 = sil_avg

        samp1 = []
        true_in1 = []

        half_samples = self.mb_size

        #Minmax cluster sampling
        split = self.split_data(sa_comb, best_dc1, n_clusters)
        
        for i in range(n_clusters):
            cluster_samps = []
            true_in_samps = []
            while len(cluster_samps) < half_samples:
                minmax = self.minmax_selection(split[i], half_samples)
                cluster_samps.extend(minmax[0])
                true_in_samps.extend(minmax[1])

            samp1.append(cluster_samps)
            true_in1.append(true_in_samps)

        # Select sa values at true_in index
        # Select r_t values at true_in index
        
        orig_data = np.array(orig_data)
        sa_comb = np.array(sa_comb)
        rt_comb = np.array(rt_comb)

        if n_clusters == 2:
            sa_t_1 = orig_data[true_in1[0]]
            sa_t_2 = orig_data[true_in1[1]]
            r_t_1 = rt_comb[true_in1[0]]
            r_t_2 = rt_comb[true_in1[1]]
        else:
            # Assumes clusters have > samples_per_cluster datapoints
            samples_per_cluster = self.mb_size // (n_clusters // 2)

            for i in range(0, n_clusters, 2):
                sa_t_1.extend(orig_data[true_in1[i]][:samples_per_cluster])
                sa_t_2.extend(orig_data[true_in1[i+1]][:samples_per_cluster])
                r_t_1.extend(rt_comb[true_in1[i]][:samples_per_cluster])
                r_t_2.extend(rt_comb[true_in1[i+1]][:samples_per_cluster])

        sa_t_1 = np.array(sa_t_1)[:self.mb_size]
        sa_t_2 = np.array(sa_t_2)[:self.mb_size]
        r_t_1 = np.array(r_t_1)[:self.mb_size]
        r_t_2 = np.array(r_t_2)[:self.mb_size]
   
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)

        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)

        return len(labels)

    def minmax_selection(self, dic, samples):
        datapoints = dic[0]
        true_idx = dic[1]

        if samples >= len(datapoints):
            return datapoints, true_idx

        true_idx_out = []
        c = Cluster([], 0)

        picked = []
        out = []

        rand = random.randint(0, len(datapoints)-1)
        picked.append(rand)
        out.append(datapoints[rand])
        true_idx_out.append(true_idx[rand])

        while len(out) < samples:
            minmax_dist = -1
            minmax_idx = -1

            for i in range(len(datapoints)):
                if i not in picked:
                    min_dist = float('inf')
                for p in picked:
                    dist = c.calculate_distance(datapoints[i], datapoints[p])
                    min_dist = min(min_dist, dist)

                if min_dist > minmax_dist:
                    minmax_dist = min_dist
                    minmax_idx = i

            picked.append(minmax_idx)
            out.append(datapoints[minmax_idx])
            true_idx_out.append(true_idx[minmax_idx])

        return out, true_idx_out


    def split_data(self, datapoints, class_list, num_classes):
        dic = {}

        for i in range(num_classes):
            dic[i] = [[], []]
      
        for j in range(len(class_list)):
            class_label = class_list[j]
            if class_label in dic:
                dic[class_label][0].append(datapoints[j])
                dic[class_label][1].append(j)

        return dic
        
    def preprocess_data(self, data, thresh=0.95):
        data = np.array(data)

        reshaped = data.reshape(data.shape[0], -1)
        
        mean = np.mean(reshaped, axis=0)
        std = np.std(reshaped, axis=0)
        std[std==0] = 1
        stand = (reshaped - mean) / std

        cov = np.cov(stand, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        index = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:, index]

        ex_variance = eigenvalues / np.sum(eigenvalues)
        cum_variance = np.cumsum(ex_variance)

        features = np.argmax(cum_variance >= thresh) + 1

        selected = eigenvectors[:, :features]

        final = np.dot(stand, selected)

        return final

    def evaluate(self, data, labels):
        sil_avg = ss(data, labels)
        db_score = db(data, labels)
        ch_score = ch(data, labels)
        return sil_avg - db_score + (ch_score / 1000)

    
    # Initialise ensemble accuracy, loss lists, max len(based on buffer size) and total_batch index
    # For each model in the ensemble generates random batch indices - ensures each member samples independently during sampling
    # Training loop - calculates number of epochs needed to cover all data points
    # For each ensemble member, retrieve a batch of state-action pairs and labels from buffer
    # Compute the logits for both state-action pairs using the ensemble members model
    # Compute cross-entropy loss between the logits and the labels. Accumulates the loss
    # Computes accuracy
    # Backpropagates the accumulated loss through the network and performs a step in the optimizer to update model parameters
    # Calculate the ensemble accuracy
    # Return ensemble accuracy
    def train_reward(self, rand=False):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                if not(rand):
                    # get random batch
                    idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                    sa_t_1 = self.buffer_seg1[idxs]
                    sa_t_2 = self.buffer_seg2[idxs]
                    labels = self.buffer_label[idxs]
                    labels = torch.from_numpy(labels.flatten()).long().to(device)

                elif rand:
                    sa_t_1 = np.random.uniform(-1, 1, (self.train_batch_size, 50, 90))
                    sa_t_2 = np.random.uniform(-1, 1, (self.train_batch_size, 50, 90))

                    r_hat1 = self.r_hat_member(sa_t_1, member=member).detach().cpu().numpy()
                    r_hat2 = self.r_hat_member(sa_t_2, member=member).detach().cpu().numpy()

                    labels = self.get_label(sa_t_1, sa_t_2, r_hat1, r_hat2)
                    labels = torch.from_numpy(labels.flatten()).long().to(device)                

                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc
    
    # Similair to train_reward however introduces soft targets and potentially non-binary labels
    # Generates random batch indices for each ensemble member
    # Training loop where epochs is established
    # For each ensemble member they retrieve a batch of state-action pairs, their reward from the buffer
    # Prepres soft targets for the cross-entropy loss
    # Computes logits for both state-action pairs using the ensemble members model
    # Compute loss and backpropagation
    # Compute accuracy
    # Backward propagation and optimization
    # calculate ensemble accuracy
    # Return ensemble accuracy
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc

class Cluster:

    def __init__(self, _data, n_clus):
        self.n_clusters = n_clus
        self.data = _data
        self.centroids = []

    def reg_start(self):
        ip = np.random.choice(np.arange(0, len(self.data)), size=self.n_clusters, replace=False)
        f_sp = []
        for i in ip:
            f_sp.append(self.data[i])
        return f_sp

    def start_plus(self):
        f_sp = []
        ip = np.random.randint(0, len(self.data))
        f_sp.append(self.data[ip])

        for i in range(self.n_clusters - 1):
            dists = []

            for j in range(len(self.data)):
                nearest = float('inf')

                for k in f_sp:
                    distance = self.calculate_distance(self.data[j], k)
                    if distance < nearest:
                        nearest = distance

                dists.append(nearest)

            dists = np.array(dists)**2
            probs = dists / dists.sum()
            next_cent = np.random.choice(len(self.data), p=probs)
            f_sp.append(self.data[next_cent])

        return f_sp

    def minmax_start(self):
        ip = np.random.randint(0, len(self.data))
        start = self.data[ip]

        furth = 0
        furth_val = 0

        for idx, dp in enumerate(self.data):
            dist = self.calculate_distance(start, dp)
            if dist > furth:
                furth = dist
                furth_val = dp

        return [start, furth_val]
          

    def minmax_start_new(self):
        euc_data = []
        output = []

        for i in self.data:
            e = np.linalg.norm(i)
            euc_data.append(e)

        for i in range((self.n_clusters // 2)+1):
            max_val = np.argmax(euc_data)
            min_val = np.argmin(euc_data)

            output.append(self.data[max_val])
            output.append(self.data[min_val])

            euc_data[max_val] = -np.inf
            euc_data[min_val] = np.inf

        return output[:self.n_clusters]

    def calculate_distance(self, start, end):

        #return self.manhattan(start, end)
        #return self.cos_sim(start, end)
        return self.euclidean(start, end)

    def cos_sim(self, start, end):
        dot_product = np.dot(start, end)
        norm_start = np.linalg.norm(start)
        norm_end = np.linalg.norm(end)

        if norm_start == 0 or norm_end == 0:
            return 0

        return dot_product / (norm_start * norm_end)

    def manhattan(self, start, end):
        #if np.array(start).shape != np.array(end).shape:
            #return -1
        s = np.array(start)
        e = np.array(end)

        d = np.abs(s - e)
        return np.sum(d)
    
    def euclidean(self, start, end):
        start = np.array(start)
        end = np.array(end)
        return np.linalg.norm(start - end)

    def kmeans(self, start=0, max_iter=1000, crit=1e-5):
        
        # Select cluster initialisation algorithm
        if start == 0:
            # Random initialisation
            sp = self.reg_start()
        elif start == 1:
            # KMeans++
            sp = self.start_plus()
        elif start == 2:
            # Min max start - Pick furthest points 
            sp = self.minmax_start()

        self.centroids = sp
        data_copy = self.data.copy()

        # Continue until max iterations value reached
        # or the difference criteria between centroids is met
        for iters in range(max_iter):

            data_points = []
            data_clusters = []
            original_pos = []

            # Iterate through every datapoint and assign it to the closest centroid
            for j in range(len(data_copy)):
                point = data_copy[j]
                closest_d = float('inf')
                closest_cluster = 0
                
                # Find closest centroid for this point
                for i in range(len(sp)):
                    distance = self.calculate_distance(sp[i], point)
                    if distance < closest_d:
                        closest_d = distance
                        closest_cluster = i

                data_points.append(point)
                data_clusters.append(closest_cluster)
                original_pos.append(j)

            # Calculate new centroids
            # Calculate the mean values for each cluster and then replace original centroid with this
            new_centroids = []
            for k in range(self.n_clusters):
                clusters = []
                for i in range(len(data_points)):
                    if data_clusters[i] == k:
                        clusters.append(data_points[i])
                if len(clusters) > 0:
                    new_centroids.append(np.mean(clusters, axis=0))
                else:
                    new_centroids.append(sp[k])

            sp = np.array(sp)
            new_centroids = np.array(new_centroids)

            # Check criteria
            if np.linalg.norm(np.array(new_centroids) - np.array(sp)) < crit:
                break

            sp = new_centroids

        return data_points, data_clusters, original_pos


    def cluster_static_centroids(self, start=0):
        # Select cluster initialisation algorithm
        if start == 0:
            # Random initialisation
            sp = self.reg_start()
        elif start == 1:
            # KMeans++
            sp = self.start_plus()
        elif start == 2:
            # Min max start - Pick furthest points 
            sp = self.minmax_start()

        self.centroids = sp
        data_copy = self.data.copy()

        # Continue until max iterations value reached
        # or the difference criteria between centroids is met

        data_points = []
        data_clusters = []
        original_pos = []

        # Iterate through every datapoint and assign it to the closest centroid
        for j in range(len(data_copy)):
            point = data_copy[j]
            closest_d = float('inf')
            closest_cluster = 0
            
            # Find closest centroid for this point
            for i in range(len(sp)):
                distance = self.calculate_distance(sp[i], point)
                if distance < closest_d:
                    closest_d = distance
                    closest_cluster = i

            data_points.append(point)
            data_clusters.append(closest_cluster)
            original_pos.append(j)

        return data_points, data_clusters, original_pos
