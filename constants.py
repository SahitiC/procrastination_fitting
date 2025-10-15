"""
define quantities that are held constant across scripts
"""

import numpy as np

# define some standard params:
# states of markov chain
STATES_NO = 20+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)

# actions = no. of units to complete in each state
# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 15  # no. of weeks for task
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.8  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_UNIT_DIFF_DISCOUNTS = 1  # reward per unit for diff-disc model
REWARD_SHIRK = 0.0
EFFORT_WORK = -0.3
REWARD_EXTRA = 0.0  # extra reward for completing all units
BETA = 5  # softmax beta
BETA_DIFF_DISCOUNTS = 10  # softmax beta for diff-disc model
THR = 14
N_TRIALS = 20  # no. of trajectories per dataset for recovery
# N = 1000  # no of params sets to recover

# params for no commitment model
STATES_NO_NO_COMMIT = (20+1) * 2
STATES_NO_COMMIT = np.arange((20+1) * 2)
ACTIONS_BASE = [np.arange(21-i) for i in range(21)]
INTEREST_STATES = np.array([0, 1])
P_STAY_LOW = 0.95
P_STAY_HIGH = 0.05
REWARD_INTEREST = 2.0

STATES_NO_FATIGUE = (20+1) * 2
STATES_FATIGUE = np.arange((20+1) * 2)
ACTIONS_BASE = [np.arange(21-i) for i in range(21)]
P_LOW = 0.2
P_HIGH = 0.3
EFFORT_LOW_FATIGUE = EFFORT_WORK
EFFICACY_FATIGUE = 0.5
