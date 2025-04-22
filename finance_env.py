from gymnasium import Env
from gymnasium.spaces.box import Box
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

class FinanceEnv(Env):
  """
  Custom Gym Environment for finance.
  """
  def __init__(self, initial_cash:tuple = (1500, 20000), window_size:int = 60, timespan:["1m", "5m", "1h", "1d"] = "1m", data_dir:str = "./data", train_val_split= (0.8, 0.1), min_length_of_episode:int = 350):
    # Actions: Percentage of portfolio to invest in long (+) or short (-)
    self.action_space = Box(low=np.array([-1]), high = np.array([1]), dtype=np.float32)
    # Observations: Current portfolio (1) + previous action (1) + minutes since 00:00 (1) + time series window (window_size) 
    # next experiment: + timestamp window (window_size)
    self.observation_space = Box(low=np.array([0, -1, 0]+[0 for _ in range(window_size)]), 
                                 high = np.array([np.inf, 1, 1440]+[np.inf for _ in range(window_size)]), dtype=np.float32)
    self.path = data_dir+"/"+timespan
    self.files = os.listdir(self.path)
    self.trainfiles = random.sample(self.files, int(train_val_split[0]*len(self.files)))
    others = [x for x in self.files if x not in self.trainfiles]
    self.valfiles = random.sample(others, int(train_val_split[1]/(1-train_val_split[0])*len(others)) )
    self.testfiles = [x for x in self.files if x not in self.trainfiles+self.valfiles]
    self.timespan = timespan
    self.min_length_of_episode = min_length_of_episode
    if self.timespan != "1m":
        self.func = pd.read_excel
        self.idx = 0
    else:
        self.func = pd.read_csv
        self.idx = None
    len_uniques = self.new_ticker()
    while len_uniques < self.min_length_of_episode:
       print("New ticker too short", self.ticker, len_uniques)
       len_uniques = self.new_ticker()

    self.window_size = window_size
    self.timestep = self.window_size-1
    self.initial_cash_range = initial_cash
    self.initial_cash = random.randint(initial_cash[0], initial_cash[1])
    self.prev_money = self.initial_cash
    self.cash = self.initial_cash
    dttm = self.data["timestamp"].iloc[self.timestep]
    self.dttm_ref = datetime(year = dttm.year, month = dttm.month, day=dttm.day, tzinfo=dttm.tzinfo)
    self.state = np.array([self.cash, 0, (dttm -self.dttm_ref).total_seconds()/60] + 
                          self.data["close"].iloc[self.timestep - self.window_size+1:self.timestep+1].to_list())
    self.options_possessed = 0
    self.steps = [self.state]
    self.datetimes = self.data["timestamp"].iloc[:self.window_size].to_list()
    self.rewards = []


  def new_ticker(self):
      self.ticker = self.trainfiles[random.randint(0, len(self.trainfiles)-1)]
      self.data = self.func(f"{self.path}/{self.ticker}", index_col= self.idx)
      if self.timespan == "1m":
          self.data.drop("symbol", axis = 1, inplace = True)
          self.data["timestamp"] = pd.to_datetime(self.data["timestamp"]).dt.tz_convert("America/New_York")
      uniques = self.data["timestamp"].apply(lambda x: x.date()).unique()
      selected_date = uniques[random.randint(0, len(uniques)-1)]
      #print(selected_date)
      self.data = self.data[self.data["timestamp"].apply(lambda x: x.date()) == selected_date]
      #print(self.data)
      return len(self.data)
        

  def step(self, action):
    # Apply action
    action = action[0]
    current_price, prev_price = self.state[-1], self.state[-2]
    percent_change = (current_price - prev_price)/prev_price
    self.timestep = self.timestep + 1
    new_window = self.data["close"].iloc[self.timestep - self.window_size+1:self.timestep+1].to_list()
    self.cash = self.cash + action * percent_change * self.cash
    dttm = self.data["timestamp"].iloc[self.timestep]
    self.datetimes.append(dttm)
    self.state = np.array([self.cash, action, (dttm -self.dttm_ref).total_seconds()/60] + 
                          new_window)

    # Calculate reward
    """reward = (action * percent_change - percent_change)*10000
    self.prev_money = self.cash """
    reward = (action * percent_change)*10000
    self.prev_money = self.cash 

    # Check if experiment is done
    if self.timestep >= len(self.data["close"])-1:
      reward += ((self.cash - self.initial_cash)/self.initial_cash)*100
      truncated = True
    else: truncated = False
    # Set placeholder for info
    info = {"datetime":dttm,
            "percent_change": percent_change}
    terminated = False
    self.steps.append(self.state)
    self.rewards.append(reward)
    return self.state, reward, terminated, truncated, info


  def render(self, yes = "yes", normalize = False, seed = 0, show = False, hspace = 0.4):
    # create data
    x = self.datetimes
    y_price = list(self.steps[0][3:]) + [x[-1] for x in self.steps[1:]]
    y_money = [self.initial_cash]*self.window_size + [x[0] for x in self.steps[1:]]
    y_actions = [0]*self.window_size + [x[1] for x in self.steps[1:]]
    y_reward = self.rewards
    x_reward = list(range(0, len(self.rewards)))

    # plot lines
    fig = plt.figure(figsize=(20,10))
    plt.subplots_adjust(hspace=hspace)  # Increase vertical spacing
    ax = fig.add_subplot(221)
    #ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    #ax.tick_params(axis='y', colors='white')  #setting up Y-axis tick color to black
    if normalize: ax.set_ylim(bottom=8, top=23)
    #ax.tick_params(axis='x', rotation=90)
    plt.xlabel("Datetime") #, color='white')
    plt.ylabel("Option price ($)")
    plt.title(f"{self.ticker[:-4]} price evolution")
    plt.plot(x, y_price)

    ax = fig.add_subplot(222)
    plt.xlabel("Datetime") #, color='white')
    plt.ylabel("Savings ($, in cash and options)") #, color='white')
    plt.title(f"Savings evolution") #, color='white')
    plt.plot(x, y_money)#, label = "Airgap 2")

    ax = fig.add_subplot(223)
    colors = ['green' if y > 0 else 'red' for y in y_actions]
    plt.plot(x, y_actions)
    plt.xlabel("Datetime") 
    plt.ylabel("Action taken") 
    plt.title("Actions taken over time")

    ax = fig.add_subplot(224)
    plt.xlabel("Step") #, color='white')
    plt.ylabel("Reward") #, color='white')
    plt.title(f"Reward per step") #, color='white')
    plt.plot(x_reward, y_reward)#, label = "Airgap 2")
    #plt.legend()
    if not os.path.exists("images"):
       os.mkdir("images")
    plt.savefig(f"images/experiment{seed}-{self.ticker[:-4]}-{self.dttm_ref.date()}.png")
    if show: plt.show()

  def reset(self, seed = 0):
    len_uniques = self.new_ticker()
    while len_uniques < self.min_length_of_episode:
       print("New ticker too short", self.ticker, len_uniques) # Change the prints with logs!!!!
       len_uniques = self.new_ticker()
    self.timestep = self.window_size-1
    self.initial_cash = random.randint(self.initial_cash_range[0], self.initial_cash_range[1])
    self.prev_money = self.initial_cash
    self.cash = self.initial_cash
    dttm = self.data["timestamp"].iloc[self.timestep]
    self.dttm_ref = datetime(year = dttm.year, month = dttm.month, day=dttm.day, tzinfo=dttm.tzinfo)
    self.state = np.array([self.cash, 0, (dttm -self.dttm_ref).total_seconds()/60] + 
                          self.data["close"].iloc[self.timestep - self.window_size+1:self.timestep+1].to_list())
    self.options_possessed = 0
    self.steps = [self.state]
    self.datetimes = self.data["timestamp"].iloc[:self.window_size].to_list()
    self.rewards = []
    return self.state, seed