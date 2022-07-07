import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import numpy as np
import collections

import gym

from policy_module import *
from value_module import *

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Obs_Wrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super(Obs_Wrapper, self).__init__(env)
  
  def observation(self, obs):
    obs=torch.FloatTensor(obs).to(device)
    return obs

Ep_Step=collections.namedtuple('EpStep', field_names=['obs', 'action','reward','obs_f','termin_signal'])

class Replay_Buffer(torch.utils.data.Dataset):
  def __init__(self):
    super(Replay_Buffer, self).__init__()
    self.ep_steps=[]
    self._max_length=100000
    self._length=0
  
  def sample_batch(self, batch_size):
    batch=[]
    batch_idx=np.random.choice(self._length, batch_size)
    for idx in batch_idx:
      batch.append(self.ep_steps[idx])
    return batch
  
  def add_item(self, ep_step):
    if self._length>=self._max_length:
      #remove earliest element
      self.ep_steps.pop(0)
      self._length=self._length-1
    #add element
    self.ep_steps.append(ep_step)
    self._length=self._length+1
    return
  
  def __getitem__(self, idx):
    return self.ep_steps[idx]
  
  def __len__(self):
    return len(self.ep_steps)

class TD3_Agent():
  def __init__(self, env, test_env, gamma, lambd):
    self.env=env
    self.test_env=test_env
    self.a_low=torch.FloatTensor(self.env.action_space.low).to(device)
    self.a_high=torch.FloatTensor(self.env.action_space.high).to(device)

    self.gamma=gamma
    self.lambd=lambd

    self.i_size=self.env.observation_space.shape[0]
    self.a_dim=self.env.action_space.shape[0]
    #policy module(actor)
    self.pm=Cont_Deterministic_Policy_Module(self.i_size, [400,300], self.a_dim)
    self.target_pm=Cont_Deterministic_Policy_Module(self.i_size, [400,300], self.a_dim)
    #use same paramters
    pm_pv=self.pm.vectorize_parameters()
    self.target_pm.inherit_parameters(pm_pv)
    #value modules(critic)
    self.vm1=TD3_Value_Module(self.i_size, self.a_dim, [400,300])
    self.target_vm1=TD3_Value_Module(self.i_size, self.a_dim, [400,300])
    vm1_pv=self.vm1.vectorize_parameters()
    self.target_vm1.inherit_parameters(vm1_pv)

    self.vm2=TD3_Value_Module(self.i_size, self.a_dim, [400,300])
    self.target_vm2=TD3_Value_Module(self.i_size, self.a_dim, [400,300])
    vm2_pv=self.vm2.vectorize_parameters()
    self.target_vm2.inherit_parameters(vm2_pv)
  
  def add_noise(self, action, std): 
    noise=torch.normal(torch.zeros(self.a_dim), torch.full([self.a_dim], std)).to(device)
    action=action+noise
    return action

  #used for training
  def get_action(self, obs, a_mode, std, train=True):
    if a_mode=='random':
      action=torch.FloatTensor(self.env.action_space.sample()).to(device)
    else:
      action=self.pm(obs)
      if train:
        action=self.add_noise(action, std)
    #clip action within action boundaries
    action=torch.clamp(action, self.a_low, self.a_high)
    return action

class TD3():
  def __init__(self, agent):
    self.agent=agent
    self.replay_buffer=Replay_Buffer()
  
  def check_performance(self):
    #w.r.t test env.: run 10 episodes
    len_eps=[]
    acc_rews=[]
    ep_datas=[]
    for _ in range(10):
      obs=self.agent.test_env.reset()
      len_ep=1
      acc_rew=0
      ep_data=[]
      while True:
        action=self.agent.pm(obs)
        action=torch.clamp(action, self.agent.a_low, self.agent.a_high).detach().cpu().numpy()
        obs_f, reward, termin_signal, _=self.agent.test_env.step(action)
        ep_step=Ep_Step(obs, action, reward, obs_f, termin_signal)
        ep_data.append(ep_step)
        acc_rew+=reward
        len_ep+=1
        obs=obs_f
        if termin_signal:
          break
      len_eps.append(len_ep)
      acc_rews.append(acc_rew)
      ep_datas.append(ep_data)
    avg_acc_rew=sum(acc_rews)/10
    avg_len_ep=sum(len_eps)/10
    return avg_acc_rew, avg_len_ep, ep_datas
  
  def get_policy_loss(self, batch_data):
    #update in a direction of maximizing vm1 estimates
    batch_size=len(batch_data)
    policy_loss=torch.FloatTensor([0]).to(device)

    for ep_step in batch_data:
      obs=ep_step.obs
      action=self.agent.pm(obs)
      #clamp within boundaries
      action=torch.clamp(action, self.agent.a_low, self.agent.a_high)
      Q=self.agent.vm1(obs, action)
      policy_loss=policy_loss-Q
    policy_loss=policy_loss/batch_size
    return policy_loss
  
  def get_value_loss(self, batch_data, target_action_std, target_noise_thresh, vm_idx):
    if vm_idx==1:
      vm=self.agent.vm1
    elif vm_idx==2:
      vm=self.agent.vm2

    batch_size=len(batch_data)
    value_loss=torch.FloatTensor([0]).to(device)
    for ep_step in batch_data:
      obs=ep_step.obs
      action=torch.FloatTensor(ep_step.action).to(device)
      reward=ep_step.reward
      obs_f=ep_step.obs_f
      termin_signal=ep_step.termin_signal

      #get target action
      target_action=self.agent.target_pm(obs_f)
      #generalize target action by adding clipped noise
      target_noise=torch.normal(torch.zeros(self.agent.a_dim), torch.full([self.agent.a_dim], target_action_std)).to(device)
      clipped_noise=torch.clamp(target_noise, -target_noise_thresh, target_noise_thresh)
      target_action=target_action+clipped_noise
      #clip target_action+noise into action boundaries
      target_action=torch.clamp(target_action, self.agent.a_low, self.agent.a_high)

      #create target
      q1_f=self.agent.target_vm1(obs_f, target_action)
      q2_f=self.agent.target_vm2(obs_f, target_action)
      min_q_f=min(q1_f,q2_f)
      #target doesn't require gradient
      target=(reward+self.agent.gamma*(1-termin_signal)*min_q_f).detach()
      Q=vm(obs, action)
      value_loss=value_loss+(target-Q)**2
    value_loss=value_loss/batch_size
    return value_loss
  
  def train(self, batch_size, n_epochs, steps_per_epoch, start_after, update_after, update_every, 
            update_actor_every, action_std, target_action_std, target_noise_thresh, p_lr, v_lr, polyak):
    #optimizers
    policy_optim=optim.Adam(self.agent.pm.parameters(), lr=p_lr)
    value_optim1=optim.Adam(self.agent.vm1.parameters(), lr=v_lr)
    value_optim2=optim.Adam(self.agent.vm2.parameters(), lr=v_lr)

    #train process
    obs=self.agent.env.reset()
    step=1
    a_mode='random'
    update=False
    for epoch in range(1, n_epochs+1):
      while True:
        if step>start_after:
          a_mode='policy'
        if step>update_after:
          update=True
        
        #action doesn't require grad for progression
        action=self.agent.get_action(obs, a_mode, action_std).detach().cpu().numpy()
        obs_f, reward, termin_signal, _=self.agent.env.step(action)
        if termin_signal:
          if obs_f[0]>0.45:
            #real termination of reaching goal
            termin_signal=1
          else:
            #just reaching horizon
            termin_signal=0
          obs_f=self.agent.env.reset()
        
        ep_step=Ep_Step(obs, action, reward, obs_f, termin_signal)
        self.replay_buffer.add_item(ep_step)

        if step%update_every==0 and update:
          for u_step in range(1, update_every+1):
            #"update actor every" -> rate of actor update
            batch_data=self.replay_buffer.sample_batch(batch_size)
            #update both critics w.r.t same target
            value_loss1=self.get_value_loss(batch_data, target_action_std, target_noise_thresh, vm_idx=1)
            value_optim1.zero_grad()
            value_loss1.backward()
            value_optim1.step()

            value_loss2=self.get_value_loss(batch_data, target_action_std, target_noise_thresh, vm_idx=2)
            value_optim2.zero_grad()
            value_loss2.backward()
            value_optim2.step()

            if u_step%update_actor_every==0:
              #update actor
              policy_loss=self.get_policy_loss(batch_data)
              
              policy_optim.zero_grad()
              policy_loss.backward()
              policy_optim.step()

              #update target networks
              pm_pv=self.agent.pm.vectorize_parameters()
              tpm_pv=self.agent.target_pm.vectorize_parameters()
              new_pm_pv=tpm_pv*polyak+(1-polyak)*pm_pv
              self.agent.target_pm.inherit_parameters(new_pm_pv)

              vm1_pv=self.agent.vm1.vectorize_parameters()
              tvm1_pv=self.agent.target_vm1.vectorize_parameters()
              new_vm1_pv=tvm1_pv*polyak+(1-polyak)*vm1_pv
              self.agent.target_vm1.inherit_parameters(new_vm1_pv)

              vm2_pv=self.agent.vm2.vectorize_parameters()
              tvm2_pv=self.agent.target_vm2.vectorize_parameters()
              new_vm2_pv=tvm2_pv*polyak+(1-polyak)*vm2_pv
              self.agent.target_vm2.inherit_parameters(new_vm2_pv)
        obs=obs_f
        step=step+1
        if step%steps_per_epoch==1:
          break
      #per epoch performance measure
      avg_acc_rew, avg_len_ep, ep_datas=self.check_performance()
      print("Epoch: {:d}, Avg_Return, {:.3f}, Avg_Ep_Length: {:.2f}".format(epoch, avg_acc_rew, avg_len_ep))
    return

cont_mc=Obs_Wrapper(gym.make('MountainCarContinuous-v0')) #Example: continuous mountain car task
cont_mc_t=Obs_Wrapper(gym.make('MountainCarContinuous-v0'))
agent=TD3_Agent(cont_mc, cont_mc_t, 0.99, 0.97)
td3=TD3(agent)
td3.train(batch_size=64, n_epochs=100, steps_per_epoch=4000, start_after=10000, update_after=1000, update_every=50, 
          update_actor_every=2, action_std=0.1, target_action_std=0.2, target_noise_thresh=0.5, p_lr=1e-3, v_lr=1e-3, polyak=0.995)