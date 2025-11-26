cbm_path = 'C:\Users\schebolu\cbm-master\codes';
addpath(genpath(cbm_path));

folder = 'C:\Users\schebolu\procrastination\procrastination_fitting';
if count(py.sys.path, folder) == 0
    insert(py.sys.path, int32(0), folder);
end

%% gen data
n_trials = int32(1);
n_participants = int32(10);
py_params = py.helper.sample_params(n_participants);

py_data = py.gen_data.simulate(py_params, n_trials, n_participants);
bounds = py.list({py.tuple({0,1}), py.tuple({0,1}), py.tuple({py.None,0})});

py_param_unbounded = py.helper.trans_to_unbounded(py_params{1}, bounds);

loglik_wrapper(py_param_unbounded, py_data{1})

%% actual data

py_data = py.pickle.load(py.open('data.pkl', 'rb'));
n_participants = length(py_data);
bounds = py.list({py.tuple({0,1}), py.tuple({0,1}), py.tuple({py.None,0})});

%%
prior = struct('mean', zeros(3,1),'variance', [6.25; 6.25; 1]);
fname = 'lap_basic.mat';

pconfig = struct();
pconfig.numinit = 15;
pconfig.numinit_med = 15;
pconfig.numinit_up = 15;

%%
cbm_lap(py_data, @loglik_wrapper, prior, fname, pconfig)

%% inspect cbm lap
fname = load('lap_basic.mat');
cbm_lap = fname.cbm;
fitted = cbm_lap.output.parameters;
n_participants = length(py_data);
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
    axis square;
    hold on;
    plot(xlim,ylim,'-b');
    hold on;
    r = corr(bounded_input(:,p), bounded_fitted(:,p))
end

%% do cbm_hbi
models = {@loglik_wrapper};
fcbm_maps = {'lap_basic.mat'};
fname_hbi = 'hbi.mat';

config = struct();
config.maxiter = 20;
config.tolx = 0.05;

optimconfigs = struct();
optimconfigs.numinit = 15;
optimconfigs.numinit_med = 15;
optimconfigs.numinit_up = 15;

%%

cbm_hbi(py_data, models, fcbm_maps, fname_hbi, config, optimconfigs)

%% inspect hbi

fname_hbi = load('hbi.mat');
cbm_hbi = fname_hbi.cbm;
cbm_hbi.output

transform = {'sigmoid', 'sigmoid', 'exp'};
model_names = {'discounting'};
param_names = {'\gamma', '\eta', 'effort'};
cbm_hbi_plot(cbm_hbi, model_names, param_names, transform)

fitted_hbi = cbm_hbi.output.parameters;
fitted_hbi = fitted_hbi{1};

n_participants = length(py_data);
bounded_fitted_hbi = zeros(n_participants, 3);
for i = 1:n_participants
    py_row = py.numpy.array(fitted_hbi(i, :));
    py_bounded = py.helper.trans_to_bounded(py_row, bounds);
    bounded_fitted_hbi(i, :) = double(py.array.array('d', py_bounded));
end

for p = 1:3
    figure;
    scatter(bounded_input(:,p), bounded_fitted_hbi(:,p), 'filled');
    axis square;
    hold on;
    plot(xlim,ylim,'-b');
    hold on;
    xl = xlim;
    ylim(xl);
    r = corr(bounded_input(:,p), bounded_fitted_hbi(:,p))
end




