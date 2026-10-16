import gymnasium as gym

env = gym.make("LunarLander-v3")

obs, info = env.reset(seed=0)

print("Observation:")
print(obs)

print("Observation space:")
print(env.observation_space)

print("Action space:")
print(env.action_space)

env.close()