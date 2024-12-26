# AI Assignment for CSF407: Shallow Q-Network for Tic-Tac-Toe

## Problem Overview

The objective of this assignment is to train a shallow Q-network to play the game of Tic-Tac-Toe against a smart player, maximizing the number of wins. The network is considered "shallow" because it has a limited number of hidden layers. The problem is broken into three phases:

1. **Data Preparation Phase**: In this phase, an untrained neural network plays against the smart player, collecting data in the form of `(state, action, reward, next_state, done)` from the interaction.
2. **Training Phase**: The collected data is used to train the network by defining input and target batches. The model is then trained through backpropagation of the loss.
3. **Testing Phase**: The trained model is tested by playing against the smart player to assess its performance.

## Data Preparation Phase

In the data preparation phase, an untrained neural network plays against a smart player for several thousand episodes (ranging from 3,000 to 48,000). During this phase, mild exploration is allowed by setting the epsilon value to a number between 0 and 1. The data generated is stored in a buffer and consists of:

- `state`: The current state of the game.
- `action`: The chosen action by the agent.
- `reward`: The reward received after taking the action.
- `next_state`: The resulting state after the action.
- `done`: Boolean indicating whether the game has ended.

The generated data points vary in number, ranging from 8,000 to over 200,000 depending on the number of episodes.

## Training Phase

In the training phase, the data from the buffer is divided into batches. I used **PyTorch** to train the network, where the training data is converted into PyTorch tensors. A mini-batch of 32 samples is randomly selected, and for each sample, the Q-values for the `state` and `next_state` are calculated.

The target value is computed as:

\[
Q_{\text{target}} = \text{reward} + \gamma \cdot \max Q_{\text{next\_state}}
\]

The loss between the predicted Q-values and the target Q-values is calculated and backpropagated to update the model's parameters. The model is then saved to a file and loaded for the testing phase.

## Methodology

For this assignment, I trained **8 different models** based on variations in the neural network architecture, number of training episodes, and epsilon values. The models used different architectures and training configurations as described below:

### Neural Network Architectures

1. **NN1**:
   - Input: 9 vector input
   - Layer 1: 64 neurons with ReLU activation
   - Layer 2: 64 neurons with ReLU activation
   - Output: 9 vector output (linear)

2. **NN2**:
   - Input: 9 vector input
   - Layer 1: 32 neurons with ReLU activation
   - Layer 2: 32 neurons with ReLU activation
   - Output: 9 vector output (linear)

3. **NN3**:
   - Input: 9 vector input
   - Layer 1: 128 neurons with ReLU activation
   - Layer 2: 128 neurons with ReLU activation
   - Output: 9 vector output (linear)

### Model Descriptions

1. **NN1, 6000 episodes**
2. **NN1, 3000 episodes**
3. **NN1, 12000 episodes**
4. **NN1, 48000 episodes, decaying epsilon at 1% per episode**
5. **NN1, 24000 episodes**
6. **NN2, 12000 episodes**
7. **NN3, 12000 episodes**
8. **NN3, 24000 episodes**

These 8 models differ in the number of episodes used for training and the structure of the neural network.

## Testing Phase

After training the models, they were tested by playing against the smart player. The performance of each model is evaluated based on the number of wins, demonstrating the effectiveness of different architectures and training strategies.

## Conclusion

This assignment demonstrates the use of a shallow Q-network to play Tic-Tac-Toe, exploring different training setups and neural network architectures. The models were able to learn and improve their performance over time by interacting with the smart player.

