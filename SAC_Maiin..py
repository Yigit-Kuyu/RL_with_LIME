from gym import make
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import time
from collections import deque
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt 
import highway_env
highway_env.register_highway_envs()
import gymnasium as gym
import pandas as pd
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paralel_computing=0 # 1 on, 0 off


# Highway Enviroment'ta "done" ne demek:
# https://github.com/Farama-Foundation/HighwayEnv/issues/285


# Original paper
# https://arxiv.org/pdf/1801.01290.pdf

#  For entropy regularization
# https://docs.cleanrl.dev/rl-algorithms/sac/#explanation-of-the-logged-metrics



# Use Xavier initialization for the weights and initializes the biases to zero for linear layers.
# It sets the weights to values drawn from a Gaussian distribution with mean 0 and variance
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class ValueNetwork(nn.Module): # state-Value network
    def __init__(self, num_features_per_vehicle, num_of_vehicles, hidden_dim):
        super(ValueNetwork, self).__init__()

        num_inputs = num_of_vehicles * num_features_per_vehicle  # Flattened input size
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_) # Optional


    def forward(self, state,num_of_vehicles):
          # Assuming state shape is (batch_size, 1, num_of_vehicles, num_features_per_vehicle)
        # We first need to remove the singleton dimension and then flatten the remaining dimensions
        state = state.squeeze(1)  # Now shape is (batch_size, num_of_vehicles, num_features_per_vehicle)
        state = state.view(state.size(0), -1)  # Flatten to (batch_size, num_of_vehicles*num_features_per_vehicle)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# stocastic policy
class Actor(nn.Module):
    def __init__(self, state_size, num_of_vehicles, action_size, hidden_dim, high, low):
        super(Actor, self).__init__()
        
        
        state_size =  num_of_vehicles * state_size
        
        self.linear1 = nn.Linear(state_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_size)
        self.log_std = nn.Linear(hidden_dim, action_size)
        
        self.high = torch.tensor(high).to(device)
        self.low = torch.tensor(low).to(device)
        
        self.apply(weights_init_) # Optional
        
        # Action rescaling
        self.action_scale = torch.FloatTensor((high - low) / 2.).to(device)
        self.action_bias = torch.FloatTensor((high + low) / 2.).to(device)
    
    def forward(self, state):
        
        # Assuming state shape is (batch_size, 1, 7, 5)
        #  After "view", each of the 20 states in the batch is now represented by a single 
        #  35-dimensional vector that includes all features from all vehicles.
        state = state.view(state.size(0), -1)  
        
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        m = self.mean(x)
        s = self.log_std(x)
        
        # Clamping for stability
        log_std_min = -20
        log_std_max = 2
        s = torch.clamp(s, min=log_std_min, max=log_std_max)
        
        return m, s
    
    def sample(self, state):
        noise=1e-6
        m, s = self.forward(state) 
        std = s.exp()
        normal = Normal(m, std)
        
        
        ## Reparameterization (https://spinningup.openai.com/en/latest/algorithms/sac.html)
        # There are two sample functions in normal distributions one gives you normal sample ( .sample() ),
        # other one gives you a sample + some noise ( .rsample() )
        a = normal.rsample() # This is for the reparamitization
        tanh = torch.tanh(a)
        action = tanh * self.action_scale + self.action_bias
        
        logp = normal.log_prob(a)
        # Comes from the appendix C of the original paper for scaling of the action:
        logp =logp-torch.log(self.action_scale * (1 - tanh.pow(2)) + noise)
        logp = logp.sum(1, keepdim=True)
        
        return action, logp


# Action-Value
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_of_vehicles, hidden_dim):
        super(Critic, self).__init__()

        # Define the first part of the Critic network for processing the state with convolutional layers
        self.conv1 = nn.Conv1d(in_channels=state_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        
        # Calculate the output size after convolutional layers to know how many units to expect before concatenation with action
        conv_output_size = num_of_vehicles * hidden_dim
        
        # Critic-1: FC layers
        self.linear1 = nn.Linear(conv_output_size + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Critic-2: FC layers
        self.linear4 = nn.Linear(conv_output_size + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)


        self.apply(weights_init_) # Apply any desired initialization

    def forward(self, state, action):
        
        
        state = state.squeeze(1) # (batch_size, num_vehicles, num_features_per_vehicle)
        state = state.permute(0, 2, 1) # reorder the dimensions to (batch_size, num_features_per_vehicle, num_vehicles)
        state_features = F.relu(self.conv1(state))
        state_features = F.relu(self.conv2(state_features)) #  each of the 20 samples (or batch entries), there are 256 features for each of the 7 vehicles. 
        
       
        # Flatten state_features to shape (20, 256*7)
        state_features = state_features.view(state_features.size(0), -1)


        state_action = torch.cat([state_features, action], dim=1)
        
        # Process through the remaining fully connected layers for Critic-1
        q1 = F.relu(self.linear1(state_action))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)
        
        # Repeat for Critic-2
        q2 = F.relu(self.linear4(state_action))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)
        
        return q1, q2

    
# Buffer
class ReplayMemory:
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.memory_capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.memory_capacity

    def sample(self):
        return list(zip(*random.sample(self.memory, self.batch_size)))

    def __len__(self):
        return len(self.memory)


class Sac_agent:
    def __init__(self, state_size, action_size, hidden_dim, high, low, memory_capacity, batch_size,
                 gamma, tau,num_updates, policy_freq, alpha, num_of_vehicles):
        
         # Actor Network 
        self.actor = Actor(state_size, num_of_vehicles,action_size,hidden_dim, high, low).to(device)
        if torch.cuda.device_count() > 1 and paralel_computing==1:
            print(f"Using {torch.cuda.device_count()} GPUs for Actor.")
            self.actor = nn.DataParallel(self.actor)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        

        # Critic Network and Target Network
        self.critic = Critic(state_size, action_size, num_of_vehicles, hidden_dim).to(device)   
        if torch.cuda.device_count() > 1 and paralel_computing==1:
            print(f"Using {torch.cuda.device_count()} GPUs for Critic.")
            self.critic = nn.DataParallel(self.critic)


        self.critic_target = Critic(state_size, action_size, num_of_vehicles, hidden_dim).to(device)        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        
        # copy weights
        self.hard_update(self.critic_target, self.critic)
        
        # Value Network and Target Network
        self.value = ValueNetwork(state_size,num_of_vehicles, hidden_dim).to(device)
        self.value_optim =optim.Adam(self.value.parameters(), lr=1e-4)
        self.target_value = ValueNetwork(state_size,num_of_vehicles, hidden_dim).to(device)
        
        # Copy weights
        self.hard_update(self.target_value, self.value)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = ReplayMemory(memory_capacity, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.num_updates = num_updates
        self.iters = 0
        self.policy_freq=policy_freq
        
        ## For Dynamic Adjustment of the Parameter alpha (entropy coefficient) according to Gaussion policy (stochastic):
        self.target_entropy = -float(self.action_size) # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2, -11 for Reacher-v2)
        self.log_alpha = torch.zeros(1, requires_grad=True, device = device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.alpha = alpha # Entropy Coefficient
        
        
    def hard_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(param.data)
            
    def soft_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
            
    def learn(self, batch):
        for _ in range(self.num_updates):                
            state, action, reward, next_state, mask = batch

            state = torch.tensor(state).unsqueeze(1).to(device).float()
            next_state = torch.tensor(next_state).unsqueeze(1).to(device).float()
            reward = torch.tensor(reward).to(device).float()
            action = torch.tensor(action).to(device).float()
            mask = torch.tensor(mask).to(device).float()
                         
            #  The ValueNetwork would then output a single value per state, resulting in value_current having a shape of (20, 1), 
            #  where 20 corresponds to the batch size, and 1 corresponds to the value estimate for each state in the batch.
            value_current=self.value(state,num_of_vehicles) # state-value function
            value_next=self.target_value(next_state, num_of_vehicles)
            if isinstance(self.actor, nn.DataParallel) and paralel_computing==1:
                act_next, logp_next = self.actor.module.sample(next_state)  # Use .module to access the underlying Actor
            else:
                act_next, logp_next = self.actor.sample(next_state)
                
            ## Compute targets
            # Q values maps a state-action pair to a single value.
            # Q values should have a dimension that corresponds to the batch size, because for each experience in the batch, 
            # you would typically compute a single Q-value.
            # This means Q_target_main should ideally have a shape of (20,), 20 is batch size.
            Q_target_main = reward[0] + self.gamma*mask[0]*value_next # Eq.8 of the original paper

            ## Update Value Network
            Q_target1, Q_target2 = self.critic_target(next_state, act_next) 
            min_Q = torch.min(Q_target1, Q_target2)
            value_difference = min_Q - logp_next # substract min Q value from the policy's log probability of slelecting that action
            value_loss = 0.5 * F.mse_loss(value_current, value_difference) # Eq.5 from the paper
            # Gradient steps 
            self.value_optim.zero_grad()
            value_loss.backward(retain_graph=True)
            self.value_optim.step()
            
            ## Update Critic Network       
            critic_1, critic_2 = self.critic(state, action)
            critic_loss1 = 0.5*F.mse_loss(critic_1, Q_target_main) # Eq. 7 of the original paper
            critic_loss2 = 0.5* F.mse_loss(critic_2, Q_target_main) # Eq. 7 of the original paper
            total_critic_loss=critic_loss1+ critic_loss2 
            # Gradient steps
            self.critic_optimizer.zero_grad()
            total_critic_loss.backward() 
            self.critic_optimizer.step() 

            ## Update Actor Network with Entropy Regularized (look at the link for entropy regularization)
            if isinstance(self.actor, nn.DataParallel) and paralel_computing==1:
                act_pi, log_pi = self.actor.module.sample(state)  # Use .module to access the underlying Actor
            else:
                act_pi, log_pi = self.actor.sample(state) # Reparameterize sampling
            
            Q1_pi, Q2_pi = self.critic(state, act_pi)
            min_Q_pi = torch.min(Q1_pi, Q2_pi)
            actor_loss =-(min_Q_pi-self.alpha*log_pi ).mean() # For minimization
            # Gradient steps
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            ## Dynamic adjustment of the Entropy Parameter alpha (look at the link for entropy regularization)
            alpha_loss = (-self.log_alpha * (log_pi.detach()) - self.log_alpha* self.target_entropy).mean() # Maximize entropy of policy
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            
            ## Soft Update Target Networks using Polyak Averaging
            if (self.iters % self.policy_freq == 0):         
                self.soft_update(self.critic_target, self.critic)
                self.soft_update(self.target_value, self.value)
        
    def act(self, state):
        s1=state
        state =  torch.tensor(state).unsqueeze(0).to(device).float()
        if isinstance(self.actor, nn.DataParallel) and paralel_computing==1:
                action, logp = self.actor.module.sample(state)  # Use .module to access the underlying Actor
        else:
                action, logp = self.actor.sample(state)
        return action.cpu().data.numpy()[0]
    
    def step(self):
        self.learn(self.memory.sample())
        
    def save(self):
        torch.save(self.actor.state_dict(), "sac_actor.pkl")
        torch.save(self.critic.state_dict(), "sac_critic.pkl")
        

def custom_reward_function(next_state, reward, done): # (optional), if needed, modify or directly use it
    num_vehicles = next_state.shape[0]
    ego_vx = next_state[0][3].item()
    ego_vy = next_state[0][4].item()
    ego_x = next_state[0][1].item()
    ego_y = next_state[0][2].item()
    closest_front_vehicle_distance = float('inf')
    closest_front_vehicle_relative_speed = float('inf')

    # Loop through all vehicles to find the closest front vehicle
    for veh in range(1, num_vehicles):
        veh_x = next_state[veh][1].item()
        veh_y = next_state[veh][2].item()
        veh_vx = next_state[veh][3].item()
        veh_vy = next_state[veh][4].item()

        # Calculate distance to front vehicle
        distance_to_front_vehicle = abs(ego_y - veh_y)

        if veh_x > ego_x and distance_to_front_vehicle < closest_front_vehicle_distance:
            closest_front_vehicle_distance = distance_to_front_vehicle
            closest_front_vehicle_relative_speed = veh_vx - ego_vx

    # Reward for maintaining appropriate distance from the front vehicle
    if closest_front_vehicle_distance < 0.15 and closest_front_vehicle_distance > 0.09:
        reward += 0.2

    # Reward for maintaining relative speed to the front vehicle
    if closest_front_vehicle_relative_speed < 0.07:
        reward += 0.1

    # Penalize if the ego vehicle is getting too close to the front vehicle
    if closest_front_vehicle_distance < 0.075:
        reward -= 0.3

    # Reward for moving faster if there is no vehicle within the safe distance
    if closest_front_vehicle_distance == float('inf') and ego_vx > 0.28 and ego_vx < 0.31:
        reward += 0.4

    # Reward for moving with appropriate x-axis speed and not making a sharp y-axis movement
    if abs(ego_vy) < 0.05 and ego_vx > 0.24 and ego_vx < 0.31:
        reward += 0.4

    # Penalize for moving too slow but still above the threshold
    if ego_vx < 0.2:
        reward -= 0.4

    # Penalize for making a very quick movement in the y-axis
    if abs(ego_vy) > 0.2:
        reward -= 0.4
        #done = True

    return reward, done



def sac(episodes,num_of_termination_iterations):
    agent = Sac_agent(state_size = state_size, action_size = action_size, hidden_dim = hidden_dim, high = high, low = low, 
                  memory_capacity = memory_capacity, batch_size = batch_size, gamma = gamma, tau = tau, 
                  num_updates = num_updates, policy_freq =policy_freq, alpha = entropy_coef,num_of_vehicles=num_of_vehicles)

    reward_list = []
    avg_score_deque = deque(maxlen = 100)
    avg_scores_list = []
    episodes_done_list=[]
    episode_steps = 0
    for i in range(episodes):
        state = env.reset()
        state=state[0]
        total_reward = 0
        done = False
        truncated=False # for time limit
        episode_steps+=1 
        termination_iteration=0
        while (not done and not truncated) and termination_iteration<num_of_termination_iterations:
            termination_iteration+=1
            agent.iters=episode_steps
            if i < 100: # To increase exploration
                action = env.action_space.sample() # to sample the random actions by randomly
            else:
                action = agent.act(state) # to sample the actions by Gaussian 
            next_state, reward, done, truncated, info = env.step(action)
            
             # Ignore the "done" signal if it comes from hitting the time horizon.
            if episode_steps == episodes: # if the current episode has reached its maximum allowed steps
                mask = 1
            else:
                mask = float(not done)
            
            if (len(agent.memory) >= agent.memory.batch_size): 
                agent.step()
            
            
            print("Done ", done)
            print("truncated ", truncated)

            if info['crashed']==True:
               print("vehicle crashed in tranining episode ",i)
               #break
            
            state = next_state
            #reward, done=custom_reward_function(next_state,reward, done)
            total_reward += reward
            print(f"episode: {i+1}, steps:{episode_steps}, current reward: {total_reward}")
            agent.memory.push((state, action, reward, next_state, mask)) # Replay buffer
            # env.render() # If simulate, open it
        
        
        episodes_done_list.append(done)
        reward_list.append(total_reward)
        avg_score_deque.append(total_reward)
        mean = np.mean(avg_score_deque)
        avg_scores_list.append(mean)
        
    # Assuming episode_steps is supposed to contain the number of steps for each episode
    episode_steps = np.arange(len(reward_list)) + 1  # Create array from 1 to the length of reward_list
    episode__done_steps = np.arange(len(episodes_done_list)) + 1
    # Plotting
    plt.plot(episode_steps, reward_list)
    plt.xlabel('Episode Steps')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.show()

    # Plotting
    plt.plot(episode__done_steps, episodes_done_list)
    plt.xlabel('Episode Steps')
    plt.ylabel('Completed Episode')
    plt.title('Completed Episodes in Training')
    plt.show()


    agent.save()
    print(f"episode: {i+1}, steps:{episode_steps}, current reward: {total_reward}, max reward: {np.max(reward_list)}")
    
                
    return reward_list, avg_scores_list

# Load State Dict with `module.` Prefix Handling
def load_state_dict_no_module(model, state_dict):
    """Load a state dict removing `module.` prefix if it exists."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    return model

# Environment
env = gym.make('highway-v0', render_mode='rgb_array')
# Environment configuration
configuration={"observation": {
        "type": "Kinematics",
        "vehicles_count": 7, #rows of observation
        "features": ["presence", "x", "y", "vx", "vy"], # column of the observation
    },

        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True
        },
        "absolute": False,
        "lanes_count": 4,
        "reward_speed_range": [20, 60], #  [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
        "simulation_frequency": 15,
        "policy_frequency": 10,
        "initial_spacing": 5,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "duration": 20, # [s], The episode is truncated if the time limit is reached
        "collision_reward": -3, #  The reward received when colliding with a vehicle.
        "on_road_reward": 2, #  The reward received when driving on a road without collision.
        "action_reward": -0.3, # penalty
        "screen_width": 600,
        "screen_height": 300,
        "centering_position": [0.3, 0.5],
        "scaling": 7,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    }

env.configure(configuration)
state = env.reset()

action_size = env.action_space.shape[0]
print(f'size of each action = {action_size}') # acceleration and speed
num_of_vehicles= env.observation_space.shape[0]
print(f'num_of_vehice = {num_of_vehicles}')
num_of_features= env.observation_space.shape[1]
print(f'num_of_features= {num_of_features}')
low = env.action_space.low
high = env.action_space.high
print(f'low of each action = {low}')
print(f'high of each action = {high}')
state_size=num_of_features


batch_size=20 # size that will be sampled from the replay memory that has maximum of "memory_capacity"
memory_capacity = 50 # 2000, maximum size of the memory, replay memory size
gamma = 0.99    # discount factor          
tau = 0.005       # soft update parameter            
num_of_train_episodes = 5 # 1500
num_of_termination_iterations=20 # steps number, if whole steps in an episode is not "done", finish an episode when iteration reachs given number
num_updates = 1 # how many times you want to update the networks in each episode
policy_freq= 2 # lower value more probability to soft update,  policy frequency for soft update of the target network borrowed by TD3 algorithm
entropy_coef = 0.2 # For entropy regularization
num_of_test_episodes=5 # 200
hidden_dim=256
num_of_datacollection_episodes=20

############################### Traning agent
reward, avg_reward = sac(num_of_train_episodes,num_of_termination_iterations)



###################### Data Related Process

# Data collection structure
data = {
    'x': [],         # X positions of vehicles
    'y': [],         # Y positions of vehicles
    'vx': [],        # X velocities of vehicles
    'vy': [],        # Y velocities of vehicles
    'action_speed': [] ,   # Action: speed
    'action_steering_angle': []  # Action: speed
}


# Data Collection Process
new_env = gym.make('highway-v0', render_mode='rgb_array')
new_env.configure(configuration)
best_actor = Actor(state_size, num_of_vehicles, action_size, hidden_dim = hidden_dim, high = high, low = low)
state_dict = torch.load("sac_actor.pkl")
load_state_dict_no_module(best_actor, state_dict)        
best_actor.to(device) 


for i in range(num_of_datacollection_episodes):
    s = new_env.reset()
    state=s[0]
    local_reward = 0
    done = truncated=False
    
    termination_iteration=0
    episode_collided=False
    
    while (not done and not truncated):
        termination_iteration+=1
        state =  torch.tensor(state).unsqueeze(0).unsqueeze(0).to(device).float()
        action,logp = best_actor(state)        
        action = action.squeeze(0).cpu().detach().numpy()
        state, r, done, truncated, info = new_env.step(action)
        local_reward += r
        data['x'].append(state[0][1])
        data['y'].append(state[0][2])
        data['vx'].append(state[0][3])
        data['vy'].append(state[0][4])
        data['action_speed'].append(action[0])
        data['action_steering_angle'].append(action[1])    
        
env.close()
df = pd.DataFrame(data)
df.to_csv('Highway.csv')
print(df.info())



