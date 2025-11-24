# %%
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import likelihoods
import constants
import gen_data

# %%


def sample_params():
    """Sample parameters to generate data."""

    discount_factor = np.random.uniform(0.2, 1)
    efficacy = np.random.uniform(0.35, 1)
    effort_work = -1 * np.random.exponential(0.5)
    pars = [discount_factor, efficacy, effort_work]

    return pars


def log_likelihood(discount, efficacy, effort, data):
    """Compute the log likelihood for a given model and data."""

    params = [discount, efficacy, effort]
    nllkhd = [likelihoods.likelihood_basic_model(
        params, constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA,
        constants.REWARD_SHIRK, constants.BETA, constants.THR,
        constants.STATES_NO, [datum]) for datum in data]
    # nllkhd = likelihoods.likelihood_basic_model(
    #     params, constants.STATES, constants.ACTIONS, constants.HORIZON,
    #     constants.REWARD_THR, constants.REWARD_EXTRA,
    #     constants.REWARD_SHIRK, constants.BETA, constants.THR,
    #     constants.STATES_NO, data)

    return -np.array(nllkhd)

# %%


class LogLike(Op):
    def __init__(self, data):
        self.data = data
        super().__init__()

    def make_node(self, discount, efficacy, effort) -> Apply:
        # Convert inputs to tensor variables
        discount = pt.as_tensor_variable(discount)
        efficacy = pt.as_tensor_variable(efficacy)
        effort = pt.as_tensor_variable(effort)
        # data = pt.as_tensor_variable(data, dtype="object")

        inputs = [discount, efficacy, effort]

        outputs = [pt.vector()]

        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # This is the method that compute numerical output
        # given numerical inputs. Everything here is numpy arrays
        discount, efficacy, effort = inputs
        # call our numpy log-likelihood function
        loglike_eval = log_likelihood(
            discount, efficacy, effort, self.data)

        outputs[0][0] = np.asarray(loglike_eval)


# %%
# gen data
np.random.seed(0)
n_participants = 5
n_trials = 1
input_params = []
data = []
for i in range(n_participants):
    [discount_factor, efficacy, effort_work] = sample_params()
    datum = gen_data.gen_data_basic(
        constants.STATES, constants.ACTIONS,  constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA,
        constants.REWARD_SHIRK, constants.BETA, discount_factor, efficacy,
        effort_work, n_trials, constants.THR, constants.STATES_NO)
    data.append(datum)
    input_params.append([discount_factor, efficacy, effort_work])

loglike_op = LogLike(data)
test_out = loglike_op(input_params[0][0],
                      input_params[0][1],
                      input_params[0][2])

pytensor.dprint(test_out, print_type=True)
print(test_out.eval())
log_likelihood(input_params[0][0],
               input_params[0][1],
               input_params[0][2], data)

# %%

data = np.array(data)


def custom_dist_loglike(obs, discount, efficacy, effort):
    return loglike_op(discount, efficacy, effort)


with pm.Model() as no_grad_model:

    # priors
    discount = pm.Uniform('discount', lower=0, upper=1, initval=0.6)
    efficacy = pm.Uniform('efficacy', lower=0, upper=1, initval=0.6)
    effort = pm.Uniform('effort', lower=-10, upper=0, initval=-1)

    likelihood = pm.CustomDist(
        "likelihood", discount, efficacy, effort, observed=np.zeros(1),
        logp=custom_dist_loglike)

# try compiling logp
ip = no_grad_model.initial_point()
print(ip)
print(no_grad_model.compile_logp(vars=[likelihood], sum=False)(ip))

# test that extracting gradients fails
try:
    no_grad_model.compile_dlogp()
except Exception as exc:
    print(type(exc))

# sample
with no_grad_model:
    idata_no_grad = pm.sample(1, tune=1)

# %%
