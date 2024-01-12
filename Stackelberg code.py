# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:12:41 2024

@author: kirvi
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import random
import matplotlib.pyplot as plt

#Environment
class StackelbergEnvironment:
    def __init__(self, state_dim, leader_action_space, follower_action_space):
        self.state_dim = state_dim
        self.leader_action_space = leader_action_space
        self.follower_action_space = follower_action_space
        self.state = None  # Current state
        self.time_step = 0

    def reset(self):
        # Initialize or reset the environment to an initial state
        self.state = np.zeros(self.state_dim)
        self.time_step = 0
        return self.state
    
    def calculate_rewards(self, leader_action, follower_action):
        leader_reward = 99 * leader_action - leader_action * leader_action - ((99 - leader_action)/4)* leader_action - 4*(leader_action * leader_action) - 16
        follower_reward = (99 - (leader_action + follower_action)) * follower_action - (follower_action * follower_action) - 9
        
        return leader_reward, follower_reward

    def step(self, leader_action, follower_action):
        # Update the environment based on the actions taken by the leader and follower
        # Return the next state, rewards, and whether the episode is done
        
        # Example dynamics: State update based on actions
        self.state[0] += leader_action
        self.state[1] += follower_action

        # Reward function: 
        leader_reward, follower_reward = self.calculate_rewards(leader_action, follower_action)

        # Example terminal condition: End episode after a certain number of time steps
        done = self.time_step == max_steps
        self.time_step += 1

        return self.state, (leader_reward, follower_reward), done

#Agents

# Agents
class JointAgent:
    def __init__(self, state_dim, leader_action_space, follower_action_space):
        self.state_dim = state_dim
        self.leader_action_space = leader_action_space
        self.follower_action_space = follower_action_space

        # DQN parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation trade-off
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.7

        # Build the joint Q-network
        self.model = self.build_model()

        # Experience replay buffer
        self.replay_buffer = []

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(32, input_dim=self.state_dim, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(len(self.leader_action_space) * len(self.follower_action_space), activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def choose_actions(self, state):
    # Exploration-exploitation trade-off using epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            leader_action = np.random.choice(self.leader_action_space)
            follower_action = np.random.choice(self.follower_action_space)
        else:
            q_values = self.model.predict(np.array([state]))

        # Choose the leader's action based on the highest Q-value
            leader_action_index = np.argmax(q_values)
            leader_action = self.leader_action_space[leader_action_index]

        # Follower considers the state and leader's action to choose its action
            follower_state = np.concatenate([state, [leader_action]])
            follower_q_values = self.model.predict(np.array([follower_state]))
            follower_action_index = np.argmax(follower_q_values)
            follower_action = self.follower_action_space[follower_action_index]

        return leader_action, follower_action


        return leader_action, follower_action

    def remember(self, state, leader_action, follower_action, leader_reward, follower_reward, next_state, done):
        # Store the joint experience in the replay buffer
        self.replay_buffer.append((state, (leader_action, follower_action), (leader_reward, follower_reward), next_state, done))

    def replay(self, batch_size):
    # Experience replay: Train the Q-network using random samples from the replay buffer
        if len(self.replay_buffer) < batch_size:
            return

        samples = random.sample(self.replay_buffer, batch_size)
        for state, actions, rewards, next_state, done in samples:
            leader_action, follower_action = actions
            leader_reward, follower_reward = rewards

            target = leader_reward + follower_reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
        target_f = self.model.predict(np.array([state]))

        # Convert the discrete actions to indices for updating the target_f
        leader_action_index = np.where(self.leader_action_space == leader_action)[0][0]
        follower_action_index = np.where(self.follower_action_space == follower_action)[0][0]

        # Calculate the flat index
        action_index = len(self.follower_action_space) * leader_action_index + follower_action_index

        # Use np.clip to ensure action_index stays within bounds
        action_index = np.clip(action_index, 0, len(target_f[0]) - 1)

        target_f[0, action_index] = target

        self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)



  

# Initialize environment and other parameters
state_dim = 2  # Change according to your problem
leader_action_space = np.arange(0, 10, dtype=int)
follower_action_space = np.arange(0, 30, dtype=int)
max_steps = 1  
num_episodes = 200  
batch_size = 32  
env = StackelbergEnvironment(state_dim, leader_action_space, follower_action_space)
joint_agent = JointAgent(state_dim, leader_action_space, follower_action_space)

# Lists to store metrics for visualization
episode_rewards = []
leader_rewards = []
follower_rewards = []


#  Inside your training loop
for episode in range(num_episodes):
    state = env.reset()
    total_leader_reward = 0
    total_follower_reward = 0

    for step in range(max_steps):
        # Choose joint actions using the epsilon-greedy policy
        leader_action, follower_action = joint_agent.choose_actions(state)

        # Perform joint actions and observe the next state and rewards
        next_state, rewards, done = env.step(leader_action, follower_action)
        leader_reward, follower_reward = rewards

        # Store joint experiences in the replay buffer
        joint_agent.remember(state, leader_action, follower_action, leader_reward, follower_reward, next_state, done)

        # Update the state for the next iteration
        state = next_state
        total_leader_reward += leader_reward
        total_follower_reward += follower_reward

        # Perform joint experience replay and train the Q-network
        joint_agent.replay(batch_size)

        if done:
            print(f"Episode {episode + 1} terminated after {step + 1} steps.")
            break
        
    # Store metrics for visualization
    episode_rewards.append(total_leader_reward + total_follower_reward)
    leader_rewards.append(total_leader_reward)
    follower_rewards.append(total_follower_reward)

    # Print the total rewards for the episode
    print(f"Episode: {episode + 1}, Total Leader Reward: {total_leader_reward}, Total Follower Reward: {total_follower_reward}")
    
    # Plotting the learning process
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label='Total Rewards')
plt.plot(leader_rewards, label='Leader Rewards')
plt.plot(follower_rewards, label='Follower Rewards')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Learning Process')
plt.legend()
plt.show()