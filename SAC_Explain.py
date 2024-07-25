import highway_env
highway_env.register_highway_envs()
import gymnasium as gym
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import lime
import lime.lime_tabular




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

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



env = gym.make('highway-v0', render_mode='rgb_array')
env.configure(configuration)
state = env.reset()
state=state[0]


df = pd.read_csv('Highway.csv')
print(df.head())
print('######################################')
df_droped= df.drop('Unnamed: 0', axis=1)
print(df_droped.head())
print('######################################')


action_size = env.action_space.shape[0]
num_of_vehicles= env.observation_space.shape[0]
num_of_features= env.observation_space.shape[1]
state_size=num_of_features
low = env.action_space.low
high = env.action_space.high
hidden_dim=256
num_of_episodes=10



########## Model Re-load Part
new_env = gym.make('highway-v0', render_mode='rgb_array')
new_env.configure(configuration)
best_actor = Actor(state_size, num_of_vehicles, action_size, hidden_dim = hidden_dim, high = high, low = low)

state_dict = torch.load("sac_actor.pkl")
load_state_dict_no_module(best_actor, state_dict)        
best_actor.to(device) 



###################### Explanation part
feature_names = ['x', 'y', 'vx', 'vy']
target_names = np.array(['action_speed', 'action_steering_angle'])

# LIME Interpretation Part
def actor_predict(state):
    # Handle both single instance and batched inputs
    if state.ndim == 1:
        state = state.reshape(1, -1)
    
    batch_size = state.shape[0]
    
    # Add "1" to the beginning of each state because we drop it but original network wanted it
    state = np.column_stack((np.ones(batch_size), state))
    
    # Create a dummy state for all vehicles
    full_state = np.zeros((batch_size, num_of_vehicles, 5))
    full_state[:, 0, :] = state  # Set the first vehicle (ego) to the actual state
    
    # Reshape to (batch_size, 1, num_of_vehicles, 5)
    s = torch.tensor(full_state).unsqueeze(1).to(device).float()
    
    action, _ = best_actor.sample(s)
    act = action.cpu().detach().numpy()
    
    return act  # (batch_size, 2) array with [speed, steering angle]


def actor_predict_action_speed(state):
    predicted_action = actor_predict(state)
    return predicted_action[:, 0]  # Return action_speed for all instances

def actor_predict_action_steering_angle(state):
    predicted_action = actor_predict(state)
    return predicted_action[:, 1]  # Return action_steering_angle for all instances

# Prepare the data for LIME
X = df_droped[['x', 'y', 'vx', 'vy']].values
y = df_droped[['action_speed', 'action_steering_angle']].values

print("X shape:", X.shape) # (4000, 4)
print("y shape:", y.shape)  # (4000,2)



# Create LIME explainers
explainer_speed = lime.lime_tabular.LimeTabularExplainer(
    X,
    feature_names=feature_names,
    class_names=['action_speed'],
    mode='regression'
)

explainer_steering = lime.lime_tabular.LimeTabularExplainer(
    X,
    feature_names=feature_names,
    class_names=['action_steering_angle'],
    mode='regression'
)


# Choose an instance to explain (e.g., the first one), you can also choose one from your test data
instance = X[0]  # (4,0)

# Get the explanation for action_speed
exp_speed = explainer_speed.explain_instance(
    instance,
    actor_predict_action_speed,
    num_features=len(feature_names),
)

# Get the explanation for action_steering_angle
exp_steering = explainer_steering.explain_instance(
    instance,
    actor_predict_action_steering_angle,
    num_features=len(feature_names)
)




# Display the explanation
print('######################################')
print('steering explanations')
print(exp_steering.as_list())
print('######################################')
print('speed explanations')
print(exp_speed.as_list())
print('######################################')



print('stop')