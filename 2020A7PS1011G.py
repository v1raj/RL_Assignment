import sys
import random
from TicTacToe import *
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow
import random
import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "2020A7PS1011G_MODEL.pth")
"""
You may import additional, commonly used libraries that are widely installed.
Please do not request the installation of new libraries to run your program.
"""

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NN1(nn.Module):
    def __init__(self):
        super(NN1, self).__init__()
        self.l1 = nn.Linear(9, 64)  
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 9)  

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x) 
        return x
class NN2(nn.Module):
    def __init__(self):
        super(NN2,self).__init__()
        self.l1 = nn.Linear(9,32)
        self.l2 = nn.Linear(32,32)
        self.l3 = nn.Linear(32,9)
    def forward(self,x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x
class NN3(nn.Module):
    def __init__(self):
        super(NN3,self).__init__()
        self.l1 = nn.Linear(9,128)
        self.l2 = nn.Linear(128,128)
        self.l3 = nn.Linear(128,9)
    def forward(self,x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x

class Model:
    def __init__(self):
        self.buffer = deque(maxlen=10000000)
        self.model = NN3()
        self.game = TicTacToe(0.8,self)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0015)
        self.criterion = nn.MSELoss()
    def prepare_data(self):
        #this performs initial data collection and slightly trains the model
        
        calculate = True
        for episode in range(24000):
            done = False
            print(f"Episode: {episode}")
            #resets the intial game board initially
            self.game.board = [0,0,0,0,0,0,0,0,0]
            cur_state= self.game.board
            #to ensure that there's no point in calculating huge exponentials after a point
            """
            if calculate and 0.99**episode<0.01:
                calculate = False
            #we want to decay but don't want exploration to go below 1%
            if calculate:
                epsilon = 1*(0.99**episode)
            else:
                epsilon = 0.01
            """
            epsilon = 0.01
            #generate the data    
            while not done:
                #exectutes while game board is not full
                if not self.game.is_full():
                    print(f"Current game state: {cur_state}")
                    self.game.player1_move()
                    print(f"After player1 moves: {self.game.board}")
                    cur_state = self.game.board
                    #model takes in tensors not lists
                    cur_state = torch.tensor(cur_state, dtype=torch.float32)
                    curr_predictions = self.model(cur_state).cpu().detach().numpy()
                    #iterate through the predictions to find max action value
                    best_curr = 0
                    best_curr_i = 0
                    for i in range(0,len(curr_predictions)):
                        action = curr_predictions[i]
                        #check if the action value is valid
                        if action>best_curr and self.game.board[i] ==0:
                            best_curr = action
                            best_curr_i = i
                    
                    #update the next state
                    #update has to be done manually cannot call the sqnfunction of the board
                    #implementing epsilon greedy strategy
                    if random.random() > epsilon:
                        self.game.board[best_curr_i] = 2
                    else:
                        action = random.randint(0,8)
                        if self.game.board[action] ==0:
                            self.game.board[action] = 2
                            best_curr_i = action
                        else:
                            self.game.board[best_curr_i] =2
                    #again predict the for the next
                    print(f"After player2 moves: {self.game.board}")
                    next_state = self.game.board
                    #model takes in tensors not lists
                    next_state = torch.tensor(next_state, dtype=torch.float32)
                    next_predictions = self.model(next_state).cpu().detach().numpy()
                    best_next = 0
                    #no need to select best next action index no use of  it just need the value
                    #which you get from best_next no use of best_next_i
                    for i in range(0,len(next_predictions)):
                        action = next_predictions[i]
                        #check if the action value is valid
                        if action>best_next and self.game.board[i] ==0:
                            best_next = action
                            
                    #update the target value and train the model
                    reward = self.game.get_reward()
                    self.buffer.append((cur_state,best_curr,reward,next_state,done))
                    cur_state = self.game.board
                    """
                    self.model.train()
                    self.optimizer.zero_grad()
                    model_pred = self.model(cur_state)[best_curr_i]
                    #got the q_target value for the action
                    q_target = reward + 0.99*best_next

                    q_target = torch.tensor(q_target, dtype=torch.float32).to(device)
                    q_target = q_target.view_as(model_pred)

                    loss = self.criterion(model_pred,q_target)
                    loss.backward()
                    self.optimizer.step()
                    print(f"Loss: {loss.item():.4f}")
                    #update the buffer for further training
                    """
                #executes when the game board is full resetting the game board
                else:
                    done = True
                
        #need to use the initial data collection to train the model further
        #but how will you get the target values that's what is also really important

        
               
                
                
                
                #minimize the values predicted
                #add this tuple to buffer

    def train(self):
        #you will also have to store for targets and see what comes up

        training_set = self.buffer
        print(f"Total number of training instances: {len(training_set)}\n")
        for epoch in range(0,30):
            training_loss = 0
            print(f"Epoch: {epoch}")
            for step in range(len(training_set)//32):
                minibatch = random.sample(training_set,32)
                states = np.zeros((32,9))
                targets = np.zeros((32,9))
                #prepares the target tensor to train the network
                for j,(state,action,reward,next_state,done) in enumerate(minibatch):

                    state_tensor = torch.tensor(state, dtype=torch.float32).clone().detach()
                    next_tensor = torch.tensor(next_state, dtype=torch.float32).clone().detach()
                    q_values = self.model(state_tensor).cpu().detach().numpy()  # Shape (1, 9)
                    q_next = self.model(next_tensor).cpu().detach().numpy()  # Shape (1, 9)

                    # Compute the Q-target using Bellman equation
                    if done:
                        q_target = reward
                    else:
                        best_next = 0
                        for i in range(0,len(q_next)):
                            action = q_next[i]
                            #check if the action value is valid
                            if action>best_next and next_state[i] == 0:
                                best_next = action
                        q_target = reward + 0.98* best_next

                    # Update the Q-value for the chosen action
                    q_values[int(action)] = q_target

                    # Store the updated values in the batch
                    states[j] = state
                    targets[j] = q_values

                states = torch.tensor(states, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)

                self.model.train()
                self.optimizer.zero_grad()
                predictions = self.model(states)
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()
                training_loss+=loss.item()
            print(f"Loss: {training_loss:.6f}")
        print("training is done")
        torch.save(self.model.state_dict(),model_path)

class PlayerSQN:
    def __init__(self):
        self.model = NN3()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
     
        pass
         
    def move(self, state):
        """
        Determines Player 2's move based on the current state of the game.

        Parameters:
        state (list): A list representing the current state of the TicTacToe board.

        Returns:
        int: The position (0-8) where Player 2 wants to make a move.
        
        """
        
        # In your final submission, PlayerSQN must be controlled by an SQN. Use an epsilon-greedy action selection policy.
        # In this implementation, PlayerSQN is controlled by terminal input.
        print(f"Current state: {state}")
        #action = int(input("Player 2 (You), enter your move (1-9): ")) - 1
        state = torch.tensor(state, dtype=torch.float32).clone().detach()
        predictions = self.model(state).cpu().detach().numpy()
        best_action = 0
        best_pos = 0
        for i in range(0,len(predictions)):
            
                        #check if the action value is valid
            if predictions[i]>best_action and state[i] ==0:
                best_action = predictions[i]
                best_pos = i
        #action will be represented as  position
        if random.random()>0.005:
            action = best_pos
        else:
            action = random.randint(0,8)
        return action
#have to write code about the performance as a function of smartplayer and no of games
def performance(smartplayer,playerSQN,n_games):
    loss = 0
    win = 0
    draw = 0
    for i in range(n_games):
        game = TicTacToe(smartplayer,playerSQN)
        game.play_game()
        reward = game.get_reward()
        if reward>0:
            win+=1
        elif reward<0:
            loss+=1
        else:
            draw+=1
    #return (smartplayer,n_games,win,loss,draw)
    print(f"Smartplayer value: {smartplayer} Total games: {n_games}, Wins: {win}, Losses: {loss}, Draws: {draw}")
    
models_list = ["2020A7PS1011G_MODEL.pth","2020A7PS1011G_MODEL_1.pth",
               "2020A7PS1011G_MODEL_2.pth","2020A7PS1011G_MODEL_3.pth",
               "2020A7PS1011G_MODEL_4.pth","2020A7PS1011G_MODEL_5.pth",
               "2020A7PS1011G_MODEL_6.pth","2020A7PS1011G_MODEL_7.pth"]
def generate_model_stats(models):
    result = []
    for i,model in enumerate(models):
        m_path = os.path.join(current_dir, model)
        print(f"Model: {model}\n")
        playerSQN = PlayerSQN()
        if i==5:
            playerSQN.model = NN2()
        if i==6 or i==7:
            playerSQN.model = NN3()
        playerSQN.model.load_state_dict(torch.load(m_path))
        playerSQN.model.eval()
        for value in [0,0.5,1]:
            result.append(performance(value,playerSQN,20))
    number=0
    for i in range(0,len(result)):
        
        if i%3==0:
            number+=1
            print(f"Model: {number}\n")
        print(f"Smartplayer value: {result[i][0]}, Total games: {result[i][1]}, Wins: {result[i][2]}, Losses: {result[i][3]}, Draws: {result[i][4]}\n")
import matplotlib.pyplot as plt
def main(smartMovePlayer1):
    """
    Simulates a TicTacToe game between Player 1 (random move player) and Player 2 (SQN-based player).

    Parameters:
    smartMovePlayer1: Probability that Player 1 will make a smart move at each time step.
                     During a smart move, Player 1 either tries to win the game or block the opponent.
    """
#    random.seed(42)
    #model = Model()
    #model.prepare_data()
    #model.train()
    playerSQN = PlayerSQN()
    #performance(0,playerSQN,20)
    game = TicTacToe(smartMovePlayer1,playerSQN)
    game.play_game()
    #generate_model_stats(models=models_list)
    # Get and print the reward at the end of the episode
    reward = game.get_reward()
    print(f"Reward for Player 2 (You): {reward}")
    
    # Data
    #training_episodes = [3000, 6000, 12000, 24000, 48000]
    #wins = [4, 6, 3, 6, 6]

    # Plotting
    #plt.figure(figsize=(8, 5))
    #plt.plot(training_episodes, wins, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)

    # Labels and Title
    #plt.xlabel('Training Episodes', fontsize=12)
    #plt.ylabel('Wins', fontsize=12)
    #plt.title('Wins vs Training Episodes', fontsize=14)
    #plt.grid(True)
    #plt.xticks(training_episodes)  # Setting x-ticks to match the training episodes

    # Display the plot
    #plt.show()

    
if __name__ == "__main__":
    try:
        smartMovePlayer1 = float(sys.argv[1])
        assert 0<=smartMovePlayer1<=1
    except:
        print("Usage: python YourBITSid.py <smartMovePlayer1Probability>")
        print("Example: python 2020A7PS0001.py 0.5")
        print("There is an error. Probability must lie between 0 and 1.")
        sys.exit(1)
    
    main(smartMovePlayer1)