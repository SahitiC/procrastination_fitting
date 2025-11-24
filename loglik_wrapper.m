function loglik = loglik_wrapper(params, py_data)
% wrapper for python log likelihood function
    py_params = py.numpy.array(params);
    
    py_loglik = py.likelihoods.log_likelihood(py_params, py_data);

    loglik = double(py_loglik);
end
