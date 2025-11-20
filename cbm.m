cbm_path = 'C:\Users\schebolu\cbm-master\codes';
addpath(genpath(cbm_path));

n_trials = int32(1);
n_participants = int32(50);
py_params = py.helper.sample_params(n_participants);
py_data = py.gen_data.simulate(py_params, n_trials, n_participants);
bounds = py.list({py.tuple({0,1}), py.tuple({0,1}), py.tuple({py.None,0})});

py_param_unbounded = py.helper.trans_to_unbounded(py_params{1}, bounds);
loglik_wrapper(py_param_unbounded, py_data{1})

%% cbm_lap for individual fits
prior = struct('mean', zeros(3,1),'variance', [6.25; 6.25; 1]);
fname = 'lap_basic.mat';

pconfig = struct();
pconfig.numinit = 15;
pconfig.numinit_med = 30;
cbm_lap(py_data, @loglik_wrapper, prior, fname, pconfig)

%% inspect cbm lap
fname = load('lap_basic.mat');
cbm = fname.cbm;
fitted = cbm.output.parameters;
bounded_fitted = zeros(n_participants, 3);
for i = 1:n_participants
    py_row = py.numpy.array(fitted(i, :));
    py_bounded = py.helper.trans_to_bounded(py_row, bounds);
    bounded_fitted(i, :) = double(py.array.array('d', py_bounded));
end

bounded_input = zeros(n_participants, 3);
for i = 1:n_participants
    py_row = py_params{i};
    bounded_input(i, :) = double(py.array.array('d', py_row));
end

for p = 1:3
    figure;
    scatter(bounded_input(:,p), bounded_fitted(:,p), 'filled');
    hold on;
end

%% do cbm_hbi
models = {@loglik_wrapper};
fcbm_maps = {'lap_basic.mat'};
fname_hbi = 'hbi.mat';

cbm_hbi(py_data, models, fcbm_maps, fname_hbi)

%% functions
function loglik = loglik_wrapper(params, py_data)
% wrapper for python log likelihood function
    py_params = py.numpy.array(params);
    
    py_loglik = py.likelihoods.log_likelihood(py_params, py_data);

    loglik = double(py_loglik);
end