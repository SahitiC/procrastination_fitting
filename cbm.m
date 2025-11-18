cbm_path = 'C:\Users\schebolu\cbm-master\codes';
addpath(genpath(cbm_path));

n_trials = int32(1);
n_participants = int32(2);
py_params = py.helper.sample_params(n_participants);
py_data = py.gen_data.simulate(py_params, n_trials, n_participants);
bounds = py.list({py.tuple({0,1}), py.tuple({0,1}), py.tuple({py.None,0})});

py_param_unbounded = py.helper.trans_to_unbounded(py_params{1}, bounds);
loglik_wrapper(py_param_unbounded, py_data{1})

v = 6.25;
prior = struct('mean',zeros(3,1),'variance',v);
fname = 'lap_basic.mat';

cbm_lap([py_data], @loglik_wrapper, prior, fname);

function loglik = loglik_wrapper(params, py_data)
% wrapper for python log likelihood function
    py_params = py.numpy.array(params);
    
    py_loglik = py.likelihoods.log_likelihood(py_params, py_data);

    loglik = double(py_loglik);
end

