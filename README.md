# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
To implement the Q-Learning reinforcement learning algorithm using the FrozenLake environment in Gymnasium and train an agent to learn the optimal policy for reaching the goal while avoiding holes. The experiment uses an epsilon-greedy strategy with epsilon decay to balance exploration and exploitation, and evaluates the learning performance using the Q-table, learned policy, state-value function, and reward curve.

## Software Requirements
Python 3.x – Programming language used for implementing Q-Learning.
Gymnasium – Used to create and simulate the FrozenLake environment.
NumPy – Used for Q-table creation and numerical calculations.
Matplotlib – Used to plot the learning/reward curve.
Jupyter Notebook / Google Colab / VS Code – Environment for writing and running the Python code.

## Environment Description
The experiment uses the FrozenLake-v1 environment from the Gymnasium library. It consists of a 4 × 4 grid with 16 states, where the agent starts from the Start (S) state and must reach the Goal (G) while avoiding Holes (H). The agent can move Left, Down, Right, or Up. A reward of 1 is given when the agent reaches the goal, while other movements provide a reward of 0. Falling into a hole terminates the episode. The environment is configured with is_slippery=False, making the movement deterministic. The Q-Learning algorithm learns the optimal actions for each state through repeated interactions with the environment.
## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
Initialize the FrozenLake environment and create a Q-table with all values set to zero.
Set the learning rate, discount factor, epsilon, epsilon decay, and number of episodes.
Start each episode from the initial state of the FrozenLake environment.
Select an action using the epsilon-greedy strategy, either exploring randomly or choosing the action with the highest Q-value.
Perform the action and observe the next state and reward.
Find the maximum Q-value of the next state.
Update the Q-table using the Q-Learning update formula.
Move to the next state and repeat the process until the episode ends.
Reduce epsilon after each episode to gradually decrease exploration.
Repeat the training process for the specified number of episodes.
Determine the learned policy by selecting the action with the highest Q-value for each state.
Evaluate the performance using the final Q-table, state-value function, learned policy, and average reward curve.


## Python Program

```
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
learning_rate = 0.8
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 5000

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------
Q = np.zeros((env.observation_space.n, env.action_space.n))

episode_rewards = []

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------
def choose_action(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
for episode in range(episodes):

    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:

        action = choose_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-Learning update
        best_next_action = np.max(Q[next_state])

        Q[state, action] = Q[state, action] + learning_rate * (
            reward
            + discount_factor * best_next_action * (not done)
            - Q[state, action]
        )

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)

    # Reduce exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------
def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)


# -------------------------------------------------
# Calculate State Values and Policy
# -------------------------------------------------
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

# -------------------------------------------------
# Output
# -------------------------------------------------
print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])

print("\nAverage reward over last 1000 episodes:", average_reward)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()
```
---

## Output

Final Q-table:
<img width="342" height="406" alt="image" src="https://github.com/user-attachments/assets/1d4a36aa-ea4a-4417-b6b5-8f99f81e3df3" />


Estimated State-Value Function:

<img width="457" height="152" alt="image" src="https://github.com/user-attachments/assets/79d9d089-b0b6-45ff-966f-b68bc46723f0" />

Learned Policy:
<img width="367" height="142" alt="image" src="https://github.com/user-attachments/assets/fa9aa71d-86d8-41b0-9fc4-01150dedc6b9" />

Average reward over last 1000 episodes: 

## Result


The Q-Learning agent successfully learned a policy to reach the goal in the FrozenLake environment while avoiding holes. The Q-table and reward curve showed improvement through repeated training.

## Inference

The experiment demonstrates that Q-Learning can effectively learn an optimal policy through trial and error. The epsilon-greedy strategy helped balance exploration and exploitation during learning.

---

