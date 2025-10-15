"""
Functions for constructing the reward functions and transition matrices for
Zhang and Ma (2023) NYU study
"""

import numpy as np
from scipy.special import comb


def reward_threshold(states, actions, reward_shirk, reward_thr,
                     reward_extra, thr, states_no):
    """
    construct reward function where units are rewarded immediately once
    threshold no. of units are hit (compensated at reward_thr per unit) & then
    reward_extra per every extra unit until max no. of states units
    (in Zhang and Ma data, thr=14 and max no of units = 22); reward for
    shirking is immediate

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        reward_shirk (float): reward for doing an alternate task (i.e., for
        each unit that is not used for work)
        reward_thr (float): reward for each unit of work completed until thr
        reward_extra (float): reward for each unit completed beyond thr
        thr (int): threshold number of units until which no reward is obtained
        states_no (int): max. no of units that can be completed

    returns:
        reward_func (list): rewards at each time point on taking each action at
        each state

    """

    reward_func = []
    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, state_current:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        # if less than thr credits have been completed, then thresholded reward
        # at thr units, flatter extra rewards till states_no and then flat
        if state_current < thr:

            for i, action in enumerate(
                    actions[state_current][thr-state_current:
                                           states_no-state_current]):

                reward_temp[action, thr:action+state_current+1] += (
                    thr*reward_thr
                    + np.arange(0, action+state_current+1-thr, step=1)
                    * reward_extra)

            for i, action in enumerate(actions[state_current]
                                       [states_no-state_current:]):

                reward_temp[action, thr:states_no] += np.arange(
                    thr*reward_thr,
                    thr*reward_thr + (states_no-thr)*reward_extra,
                    step=reward_extra)
                reward_temp[action, states_no:action+state_current+1] += (
                    thr*reward_thr + (states_no-1-thr)*reward_extra)

        # if more than 14 units completed, extra reward until 22 is reached
        # and then nothing
        elif state_current >= thr and state_current < states_no-1:

            for i, action in enumerate(actions[state_current]
                                       [:states_no-state_current]):

                reward_temp[action, state_current+1:
                            action+state_current+1] += (
                                np.arange(1, action+1)*reward_extra)

            # reward_temp[states_no-state_current:, :] = reward_temp[
            # states_no-1-state_current, :]
        reward_func.append(reward_temp)

    return reward_func


def reward_immediate(states, actions, reward_shirk, reward_unit):
    """
    construct reward function where units are rewarded immediately (compensated
    at reward_unit per unit); reward for shirking is also immediate

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        reward_shirk (float): reward for doing an alternate task (i.e., for
        each unit that is not used for work)
        reward_unit (float): reward for each unit of work completed

    returns:
        reward_func (list): rewards at each time point on taking each action at
        each state

    """

    reward_func = []

    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, state_current:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        # immediate rewards for units completed
        for action in range(len(actions[state_current])):

            reward_temp[action, state_current:state_current+action+1] += (
                np.arange(0, action+1) * reward_unit)

        reward_func.append(reward_temp)

    return reward_func


def reward_immediate_thr(states, actions, reward_shirk, reward_unit):
    """
    construct reward function where units are rewarded immediately (compensated
    at reward_unit per unit); reward for shirking is also immediate

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        reward_shirk (float): reward for doing an alternate task (i.e., for
        each unit that is not used for work)
        reward_unit (float): reward for each unit of work completed

    returns:
        reward_func (list): rewards at each time point on taking each action at
        each state

    """

    reward_func = []

    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, state_current:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        # immediate rewards for units completed
        for action in range(len(actions[state_current])):

            reward_temp[action, state_current:state_current+action+1] += (
                np.arange(0, action+1) * reward_unit
            )

        reward_func.append(reward_temp)

    return reward_func


def reward_no_immediate(states, actions, reward_shirk):
    """
    construct reward function where only reward for shirking is immediate

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        reward_shirk (float): reward for doing an alternate task (i.e., for
        each unit that is not used for work)

    returns:
        reward_func (list): rewards at each time point on taking each action at
        each state

    """

    reward_func = []
    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, state_current:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        reward_func.append(reward_temp)

    return reward_func


def reward_final_no_thr(states, reward_unit, states_no):

    total_reward_func_last = np.arange(0, states_no, 1)*reward_unit

    return total_reward_func_last


def reward_final(states, reward_thr, reward_extra, thr, states_no):
    """
    construct reward function where units are rewarded at end of task
    (compensated at reward_thr per unit)

    params:
        states (ndarray): states of an MDP
        reward_thr (float): reward for each unit of work completed until thr
        reward_extra (float): reward for each unit completed beyond thr
        thr (int): threshold number of units until which no reward is obtained
        states_no (int): max. no of units that can be completed

    returns:
        reward_func (list): rewards at each time point on taking each action at
        each state

    """
    total_reward_func_last = np.zeros(len(states))
    # np.zeros(len(states))
    # np.arange(0, states_no, 1)*reward_thr
    total_reward_func_last[thr:states_no] = (
        thr*reward_thr + np.arange(0, states_no-thr)*reward_extra)
    total_reward_func_last[states_no:] = (
        thr*reward_thr + (states_no-1-thr)*reward_extra)

    return total_reward_func_last


def effort(states, actions, effort_work):
    """
    construct effort function (effort is always immediate)

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        effort_work (float): cost for working per unit work

    returns:
        effort_func (list): effort at each time point on taking each action at
        each state
    """

    effort_func = []
    for state_current in range(len(states)):

        effort_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            effort_temp[action, state_current:state_current +
                        action+1] = action * effort_work

        effort_func.append(effort_temp)

    return effort_func


def effort_convex_concave(states, actions, effort_work, exponent):
    """
    construct effort function where cost per unit changes as an exponent in
    no. of units (e = effort_work * actions^exponent)

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        effort_work (float): cost for working per unit work
        exponent (float): >1, defines convexity of effort function

    returns:
        effort_func (list): effort at each time point on taking each action at
        each state
    """

    effort_func = []
    for state_current in range(len(states)):

        effort_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            effort_temp[action, state_current:state_current +
                        action+1] = (action**exponent) * effort_work

        effort_func.append(effort_temp)

    return effort_func


def T_uniform(states, actions):
    """
    transition function as a uniformly random process
    equal probability of next state for each action

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state

    returns:
        T (list): transition function
    """

    T = []
    for state_current in range(len(states)):

        T_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            T_temp[action, state_current:state_current+action+1] = (
                np.full((action+1,), 1/(action+1))
            )

        T.append(T_temp)

    return T


def binomial_pmf(n, p, k):
    """
    calculates binomial probability mass function

    params:
        n (int): number of trials
        p (float) = (0<=p<=1) probability of success
        k (int)= number of successes

    returns:
        binomial_prob: binomail probability given parameters
    """

    if not isinstance(n, (int, np.int32, np.int64)):
        raise TypeError("Input must be an integer.")

    binomial_prob = comb(n, k) * p**k * (1-p)**(n-k)

    return binomial_prob


def T_binomial(states, actions, efficacy):
    """
    transition function as binomial number of successes with
    probability=efficacy for number of units worked  (=action)

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each states
        efficacy (float): (0<=efficacy<=1) binomial probability of success on
                          doing some units of work (action)

    returns:
        T (list): transition matrix
    """

    T = []
    for state_current in range(len(states)):

        T_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            # T_temp[action, state_current:state_current+action+1] = (
            #     binom(action, efficacy).pmf(np.arange(action+1)))
            T_temp[action, state_current:state_current+action+1] = (
                binomial_pmf(action, efficacy, np.arange(action+1))
            )

        T.append(T_temp)

    return T


def deterministic_policy(a):
    """
    output determinsitic policy by choosing action with max q-value

    params:
        a (ndarray): q-values

    returns:
        (ndarray): larray where chosen action is marked with 1
    """
    p = np.where(a == np.max(a), 1, 0)
    return p / sum(p)


def softmax_policy(a, beta):
    """
    output softmax policy given q-values

    params:
        a (ndarray): q-values

    returns:
        p (ndarray): probabilities of choosing each action
    """
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p
