"""
module to calculate likelihood of data under each of the models and maximise
log likelihood (minimise negative log likelihood) to find best fitting params
"""

import task_structure
import mdp_algms
import numpy as np
import helper
import constants


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p


def log_likelihood(params, data):
    """Compute the log likelihood for a given model and data."""

    nllkhd = likelihood_basic_model_transformed(
        params, constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA,
        constants.REWARD_SHIRK, constants.BETA, constants.THR,
        constants.STATES_NO, data)

    return -nllkhd


def calculate_likelihood_single(data, Q_values, beta, T, actions):
    """
    calculate negative likelihood of data under model given optimal Q_values,
    beta, transitions and actions available
    """
    nllkhd = 0

    if isinstance(data[0], (int, np.integer)):
        data = [data]
    else:
        data = data

    for traj in data:

        for t in range(len(traj)-1):

            partial = 0
            # enumerate over all posible actions for the observed state
            for i_a, action in enumerate(actions[traj[t]]):

                partial += (
                    softmax_policy(Q_values[traj[t]]
                                   [:, t], beta)[action]
                    * T[traj[t]][action][traj[t+1]])

            nllkhd = nllkhd - np.log(partial)

    return nllkhd


def calculate_likelihood(data, Q_values, beta, T, actions):
    """
    calculate negative likelihood of data under model given optimal Q_values,
    beta, transitions and actions available
    """
    nllkhd = 0

    if isinstance(data[0], (int, np.integer)):
        data = [data]
    else:
        data = data

    for trajectories in data:

        for traj in trajectories:

            for t in range(len(traj)-1):

                partial = 0
                # enumerate over all posible actions for the observed state
                for i_a, action in enumerate(actions[traj[t]]):

                    partial += (
                        softmax_policy(Q_values[traj[t]]
                                       [:, t], beta)[action]
                        * T[traj[t]][action][traj[t+1]])

                nllkhd = nllkhd - np.log(partial)

    return nllkhd


def pad_Q_values(Q_values):
    n_states = len(Q_values)
    max_actions = max(Q.shape[0] for Q in Q_values)

    padded_Q = np.full(
        (n_states, max_actions, constants.HORIZON), np.nan, dtype=np.float64)

    for s in range(n_states):
        rows = Q_values[s].shape[0]
        padded_Q[s, :rows, :] = Q_values[s]

    return padded_Q


# def calculate_likelihood(data, q, beta, T, actions):
#     """
#     calculate negative likelihood of data under model given optimal Q_values,
#     beta, transitions and actions available
#     """
#     nllkhd = 0

#     if isinstance(data[0], (int, np.integer)):
#         data = [data]
#     else:
#         data = data

#     for trajectories in data:

#         for traj in trajectories:

#             for t in range(len(traj)-1):

#                 partial = 0
#                 # enumerate over all posible actions for the observed state
#                 for i_a, action in enumerate(actions[traj[t]]):

#                     s = traj[t]
#                     q_s = q[s, :, t]

#                     partial += (
#                         softmax_policy(q_s, beta)[action]
#                         * T[traj[t]][action][traj[t+1]])

#                 nllkhd = nllkhd - np.log(partial)

#     return nllkhd


def calculate_likelihood_interest_rewards(data, Q_values, beta, T, p_stay,
                                          actions, interest_states):
    """
    calculate likelihood of data under interest reward model given
    optimal Q_values, beta, transitions, probability of staying in low and
    high states, and actions available
    """
    nllkhd = 0

    for i_trial in range(len(data)):

        # marginal prob of interest rewards at very first time step
        p_interest = np.zeros(len(interest_states))
        for i_a, action in enumerate(actions[data[i_trial][0]]):

            p_interest += (p_stay[0, :]  # assume 1st interest state = 0 (low)
                           * softmax_policy(Q_values[0][data[i_trial][0]]
                                            [:, 0], beta)[action]
                           * T[data[i_trial][0]][action][data[i_trial][1]])

        # marginal prob for rest of time steps
        for i_time in range(1, len(data[i_trial])-1):

            partial = np.zeros(len(interest_states))

            # enumerate over all possible interest states
            for i_i, interest_state in enumerate(interest_states):

                # enumerate over all possible actions for the observed state
                for i_a, action in enumerate(actions[data[i_trial][i_time]]):

                    partial += (
                        p_stay[interest_state, :]
                        * softmax_policy(Q_values[interest_state]
                                         [data[i_trial][i_time]]
                                         [:, i_time], beta)[action]
                        * T[data[i_trial][i_time]][action][
                            data[i_trial][i_time+1]]
                        * p_interest[interest_state])

            # the above calculation results in a marginal prob over the (two)
            # possible interest states, which'll be added up in next iteration
            p_interest = partial

        # final prob is over the two interest states, so must be added up
        nllkhd = nllkhd - np.log(np.sum(p_interest))

    return nllkhd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def likelihood_rl_basic_model(params, data):

    alpha, beta = params

    actions = data["actions"]
    rewards = data["rewards"]

    n_actions = len(np.unique(actions))
    Q = np.zeros(n_actions)
    nll = 0.0

    for t in range(len(actions)):
        p = np.exp(beta * Q) / np.sum(np.exp(beta * Q))
        # avoid log(0)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        nll -= np.log(p[actions[t]])

        # RL update
        Q[actions[t]] += alpha * (rewards[t] - Q[actions[t]])

    return nll


def likelihood_basic_lite(x,
                          states, actions, horizon,
                          reward_thr, reward_extra, reward_shirk,
                          beta, efficacy, thr, states_no, data):
    """
    implement likelihood calculation for basic model
    """

    discount_factor = x[0]
    effort_work = x[1]

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood(data, Q_values, beta, T, actions)

    return nllkhd


def likelihood_basic_model(x,
                           states, actions, horizon,
                           reward_thr, reward_extra, reward_shirk,
                           beta, thr, states_no, data):
    """
    implement likelihood calculation for basic model
    """

    discount_factor = x[0]
    efficacy = x[1]
    effort_work = x[2]

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood(data, Q_values, beta, T, actions)

    return nllkhd


def likelihood_basic_model_transformed(
        x, states, actions, horizon, reward_thr, reward_extra, reward_shirk,
        beta, thr, states_no, data):
    """
    implement likelihood calculation for basic model
    """

    discount_factor_unbounded = x[0]
    efficacy_unbounded = x[1]
    effort_work_unbounded = x[2]

    [discount_factor, efficacy, effort_work] = helper.trans_to_bounded(
        [discount_factor_unbounded, efficacy_unbounded, effort_work_unbounded],
        [(0, 1), (0, 1), (None, 0)])

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood_single(data, Q_values, beta, T, actions)

    return nllkhd


def likelihood_efficacy_gap_model(x,
                                  states, actions, horizon,
                                  reward_thr, reward_extra, reward_shirk,
                                  beta, thr, states_no, data):
    """
    implement likelihood calculation for efficacy gap model
    """

    discount_factor = x[0]
    efficacy_assumed = x[1]
    efficacy_actual = x[2]
    effort_work = x[3]

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T_assumed = task_structure.T_binomial(states, actions, efficacy_assumed)

    # optimal policy under assumed efficacy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T_assumed)

    T_actual = task_structure.T_binomial(states, actions, efficacy_actual)

    nllkhd = calculate_likelihood(data, Q_values, beta, T_actual, actions)

    return nllkhd


def likelihood_convex_concave_model(x,
                                    states, actions, horizon,
                                    reward_thr, reward_extra, reward_shirk,
                                    beta, thr, states_no, data):
    """
    implement likelihood calculation for convex concave model
    """

    discount_factor = x[0]
    efficacy = x[1]
    effort_work = x[2]
    exponent = x[3]

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort_convex_concave(states, actions,
                                                       effort_work, exponent)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy under assumed efficacy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood(data, Q_values, beta, T, actions)

    return nllkhd


def likelihood_immediate_basic_model(x,
                                     states, actions, horizon,
                                     reward_thr, reward_extra, reward_shirk,
                                     beta, thr, states_no, data):
    """
    implement likelihood calculation for immediate basic model
    """

    discount_factor = x[0]
    efficacy = x[1]
    effort_work = x[2]
    exponent = x[3]

    # define task structure
    reward_func = task_structure.reward_threshold(
        states, actions, reward_shirk, reward_thr, reward_extra, thr,
        states_no)

    effort_func = task_structure.effort_convex_concave(states, actions,
                                                       effort_work, exponent)

    total_reward_func_last = np.zeros(len(states))

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy under assumed efficacy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood(data, Q_values, beta, T, actions)

    return nllkhd


def likelihood_diff_discounts_model(
        x, states, actions, horizon, reward_thr, reward_extra,
        reward_shirk, beta, thr, states_no, data):
    """
    implement likelihood calculation for diff discount model
    """

    discount_factor_reward = x[0]
    discount_factor_cost = x[1]
    efficacy = x[2]
    effort_work = x[3]
    # reward_shirk = x[2]

    reward_func = task_structure.reward_threshold(
        states, actions, reward_shirk, reward_thr, reward_extra, thr,
        states_no)

    effort_func = task_structure.effort(states, actions, effort_work)

    reward_func_last = np.zeros(len(states))
    effort_func_last = np.zeros(len(states))

    T = task_structure.T_binomial(states, actions, efficacy)

    V_opt_full, policy_opt_full, Q_values_full = (
        mdp_algms.find_optimal_policy_diff_discount_factors(
            states, actions, horizon, discount_factor_reward,
            discount_factor_cost, reward_func, effort_func, reward_func_last,
            effort_func_last, T))

    # effective Q_values for the agent
    effective_Q = []
    for i_s in range(len(states)):
        Q_s_temp = []
        for i in range(horizon):
            Q_s_temp.append(Q_values_full[horizon-1-i][i_s][:, i])
        effective_Q.append(np.array(Q_s_temp).T)

    nllkhd = calculate_likelihood(data, effective_Q, beta, T, actions)

    return nllkhd


def likelihood_no_commitment_model(
        x, states, interest_states, actions_base, horizon, p_stay_low,
        p_stay_high, reward_thr, reward_extra, reward_shirk,
        beta, thr, states_no, data):
    """
    implement likelihood calculation for no commit model
    """

    discount_factor = x[0]
    efficacy = x[1]
    effort_work = x[2]
    reward_interest = x[3]

    states_no = len(states)

    # reward for completion
    reward_func_base = task_structure.reward_threshold(
        states[:int(states_no/2)], actions_base, reward_shirk,
        reward_thr, reward_extra, thr, states_no)

    # immediate interest rewards
    reward_func_interest = task_structure.reward_immediate(
        states[:int(states_no/2)], actions_base, 0, reward_interest,
        reward_interest)

    # effort costs
    effort_func = task_structure.effort(states[:int(states_no/2)],
                                        actions_base, effort_work)

    # total reward for low reward state = reward_base + effort
    total_reward_func_low = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = reward_func_base[state_current] + effort_func[state_current]
        # replicate rewards for high reward states
        total_reward_func_low.append(np.block([temp, temp]))

    # total reward for high reward state = reward_base+interest rewards+effort
    total_reward_func_high = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = (reward_func_base[state_current]
                + reward_func_interest[state_current]
                + effort_func[state_current])
        total_reward_func_high.append(np.block([temp, temp]))

    total_reward_func = []
    total_reward_func.extend(total_reward_func_low)
    total_reward_func.extend(total_reward_func_high)

    total_reward_func_last = np.zeros(len(states))

    # tranistion matrix based on efficacy and stay-switch probabilities
    T_partial = task_structure.T_binomial(states[:int(states_no/2)],
                                          actions_base, efficacy)
    T_low = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = np.block([p_stay_low * T_partial[state_current],
                         (1 - p_stay_low) * T_partial[state_current]])
        # assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_low.append(temp)

    T_high = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = np.block([(1 - p_stay_high) * T_partial[state_current],
                         p_stay_high * T_partial[state_current]])
        # assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_high.append(temp)

    T = []
    T.extend(T_low)
    T.extend(T_high)

    # optimal policy based on task structure
    actions_all = actions_base.copy()
    # same actions available for low and high reward states: so repeat
    actions_all.extend(actions_base)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions_all, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    # adapt some quantities for llkhd calculating function
    p_stay = np.array([[p_stay_low, 1-p_stay_low],
                      [1-p_stay_high, p_stay_high]])

    Q_values_unstacked = np.array([Q_values[:int(states_no/2)],
                                  Q_values[int(states_no/2):]])

    nllkhd = calculate_likelihood_interest_rewards(
        data, Q_values_unstacked, beta, T_partial, p_stay, actions_base,
        interest_states)

    return nllkhd
