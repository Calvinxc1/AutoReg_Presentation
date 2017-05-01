#%% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%% gradient descent linear regression function
def grad_descent(data_table, train_obs, valid_obs, test_obs, input_features, output_features, l2 = 0, constant = True, max_iters = 10000):

    def initialize_model(data_table, input_features, output_features, constant):
        
        def setup_array(table, features, constant = False):
            data_array = np.array(table[features].values, order = 'C')
            if constant:
                data_array = np.insert(data_array, 0, 1, axis = 1)
            return data_array
            
        input_array = setup_array(data_table, input_features, constant = constant)
        output_array = setup_array(data_table, output_features)
        betas = np.zeros(shape = (len(input_features) + constant, len(output_features)), order = 'C')
        return (input_array, output_array, betas)

    def calc_gradient(input_array, output_array, betas, l2):
        predict_array = np.dot(input_array, betas)
        error_array = output_array - predict_array
        error_grad = -2 * np.dot(input_array.T, error_array) / len(input_array)
        l2_grad = 2 * betas * l2 / len(betas)
        error_grad_two = 2 * np.expand_dims(np.sum(input_array ** 2, axis = 0), axis = 1) / len(input_array)
        l2_grad_two = 2 * l2 / len(betas)
        gradient = error_grad + l2_grad
        gradient_two = error_grad_two + l2_grad_two
        return (gradient, gradient_two)

    def update_betas(gradient, gradient_two, betas):
        stepsize = gradient / gradient_two
        new_betas = betas - (stepsize / len(betas))
        return new_betas

    def model_error(input_array, output_array, betas):
        predict_array = np.dot(input_array, betas)
        error_array = output_array - predict_array
        model_error = np.sqrt(np.mean(error_array ** 2))
        return model_error

    input_array, output_array, betas = initialize_model(data_table, input_features, output_features, constant = constant)
    
    error_paths = np.empty(shape = (0, 3))
    
    for iter_count in range(max_iters):
        gradient, gradient_two = calc_gradient(input_array[train_obs], output_array[train_obs], betas, l2 = l2)
        betas = update_betas(gradient, gradient_two, betas)
        train_error = model_error(input_array[train_obs], output_array[train_obs], betas)
        valid_error = model_error(input_array[valid_obs], output_array[valid_obs], betas)
        test_error = model_error(input_array[test_obs], output_array[test_obs], betas)
        error_paths = np.append(error_paths, [[train_error, valid_error, test_error]], axis = 0)
    
    return (betas, iter_count + 1, error_paths)
#%%
def grid_search(data_table, train_obs, valid_obs, test_obs, input_features, output_features, constant = True, max_iters = 10000):
    best_error = np.inf
    best_model = None
    for l2 in [0] + [10 ** i for i in range(-10, 11)]:
        current_model = grad_descent(data_table, train_obs, valid_obs, test_obs, input_features, output_features, l2 = l2, constant = constant, max_iters = max_iters)
        if current_model[2][-1][1] < best_error:
            best_error = current_model[2][-1][1]
            best_model = (current_model[0], current_model[1], current_model[2], l2)
    return best_model
#%% auto-regularization linear regression
def auto_reg(data_table, train_obs, valid_obs, test_obs, input_features, output_features, constant = True, max_iters = 10000):
    
    def initialize_model(data_table, input_features, output_features, constant):
        
        def setup_array(table, features, constant = False):
            data_array = np.array(table[features].values)
            if constant:
                data_array = np.insert(data_array, 0, 1, axis = 1)
            return data_array
            
        input_array = setup_array(data_table, input_features, constant = constant)
        output_array = setup_array(data_table, output_features)
        betas = np.zeros(shape = (len(input_features) + constant, len(output_features)))
        lambda_two = 0
        return (input_array, output_array, betas, lambda_two)
        
    def gradients(train_input, train_output, valid_input, valid_output, betas, lambda_two):
        train_obs = train_input.shape[0]
        valid_obs = valid_input.shape[0]
        feature_count = len(betas)
        
        grad_err_one = np.dot(train_input.transpose(), train_output - np.dot(train_input, betas))
        grad_err_two = np.expand_dims(np.sum(train_input ** 2, axis = 0), axis = 1)
        
        beta_step_top = (train_obs * np.exp(lambda_two) * betas) - (feature_count * grad_err_one)
        beta_step_bot = (train_obs * np.exp(lambda_two)) + (feature_count * grad_err_two)
        beta_step = beta_step_top / beta_step_bot
        new_beta = betas - np.nan_to_num(beta_step / feature_count)
        
        grad_lam_one = (betas * grad_err_two) + grad_err_one
        grad_lam_two = (train_obs * np.exp(lambda_two)) + (feature_count * grad_err_two)
        valid_error = valid_output - np.dot(valid_input, new_beta)
        
        lamb_step_top = ((2 * train_obs * np.exp(lambda_two)) / valid_obs) * np.dot(valid_error.transpose(), np.dot(valid_input, grad_lam_one / (grad_lam_two ** 2)))
        lamb_step_bot_one = feature_count * (np.dot(valid_input, grad_lam_one / (grad_lam_two ** 2)) ** 2)
        lamb_step_bot_two = 2 * valid_error * np.dot(valid_input, grad_lam_one / (grad_lam_two ** 3))
        lamb_step_bot = lamb_step_top + (((2 * (train_obs ** 2) * np.exp(2 * lambda_two)) / valid_obs) * np.sum(lamb_step_bot_one + lamb_step_bot_two))
        lamb_step = lamb_step_top / lamb_step_bot
        new_lamb = lambda_two - np.nan_to_num(lamb_step)
        
        grad_err_one = np.dot(train_input.transpose(), train_output - np.dot(train_input, betas))
        grad_err_two = np.expand_dims(np.sum(train_input ** 2, axis = 0), axis = 1)
        
        beta_step_top = (train_obs * np.exp(new_lamb) * betas) - (feature_count * grad_err_one)
        beta_step_bot = (train_obs * np.exp(new_lamb)) + (feature_count * grad_err_two)
        beta_step = beta_step_top / beta_step_bot
        new_beta = betas - np.nan_to_num(beta_step / feature_count)

        return (new_beta, new_lamb)
    
    def model_error(input_array, output_array, betas):
        predict_array = np.dot(input_array, betas)
        error_array = output_array - predict_array
        model_error = np.sqrt(np.mean(error_array ** 2))
        return model_error
        
    (input_array, output_array, betas, lambda_two) = initialize_model(data_table, input_features, output_features, constant)
    lambda_path = []
    error_paths = np.empty(shape = (0, 3))
    
    for iter_count in range(max_iters):
        new_vals = gradients(input_array[train_obs], output_array[train_obs], input_array[valid_obs], output_array[valid_obs], betas, lambda_two)
        betas = new_vals[0]
        lambda_two = new_vals[1]
        lambda_path.append(float(lambda_two))
        train_error = model_error(input_array[train_obs], output_array[train_obs], betas)
        valid_error = model_error(input_array[valid_obs], output_array[valid_obs], betas)
        test_error = model_error(input_array[test_obs], output_array[test_obs], betas)
        error_paths = np.append(error_paths, [[train_error, valid_error, test_error]], axis = 0)
    
    return (betas, iter_count + 1, error_paths, lambda_path)
#%% Data cluster setup
data_table = pd.read_csv('kc_house_data.csv')
input_features = ['sqft_living']
output_features = ['price']
for i in range(2, 16):
    new_column = 'sqft_living_' + str(i)
    data_table[new_column] = data_table['sqft_living'] ** float(i)
    input_features.append(new_column)
data_table[input_features + output_features] = (data_table[input_features + output_features] - data_table[input_features + output_features].mean()) / data_table[input_features + output_features].std()
#%%
np.random.seed(1701)
data_shuffles = 100
data_iterations = []
for j in range(data_shuffles):
    train_obs, valid_obs, test_obs = np.split(np.random.permutation(len(data_table)), [int(len(data_table) * 0.5), int(len(data_table) * 0.75)])
    data_iterations.append([list(train_obs), list(valid_obs), list(test_obs)])
#%%
%timeit grad_descent(data_table, train_obs, valid_obs, test_obs, input_features, output_features, max_iters = 1)
%timeit grid_search(data_table, train_obs, valid_obs, test_obs, input_features, output_features, max_iters = 1)
%timeit auto_reg(data_table, train_obs, valid_obs, test_obs, input_features, output_features, max_iters = 1)
#%%
model_iters = 1000
grad_errors = []
grad_models = []
reg_errors = []
grid_lambdas = []
reg_models = []
auto_reg_errors = []
lambda_paths = []
auto_reg_models = []
for data_iter in data_iterations:
    grad_model = grad_descent(data_table, data_iter[0], data_iter[1], data_iter[2], input_features, output_features, max_iters = model_iters)
    grad_errors.append(grad_model[2])
    grad_models.append(grad_model)
    reg_model = grid_search(data_table, data_iter[0], data_iter[1], data_iter[2], input_features, output_features, max_iters = model_iters)
    reg_errors.append(reg_model[2])
    grid_lambdas.append(reg_model[3])
    reg_models.append(reg_model)
    auto_reg_model = auto_reg(data_table, data_iter[0], data_iter[1], data_iter[2], input_features, output_features, max_iters = model_iters)
    auto_reg_errors.append(auto_reg_model[2])
    lambda_paths.append(auto_reg_model[3])
    auto_reg_models.append(auto_reg_model)
#%%
grad_improve = []
reg_improve = []
valid_grad_improve = []
valid_reg_improve = []
for k in range(data_shuffles):
    grad_improve.append(grad_errors[k][-1][2] - auto_reg_errors[k][-1][2])
    reg_improve.append(reg_errors[k][-1][2] - auto_reg_errors[k][-1][2])
    valid_grad_improve.append(grad_errors[k][-1][1] - auto_reg_errors[k][-1][1])
    valid_reg_improve.append(reg_errors[k][-1][1] - auto_reg_errors[k][-1][1])
#%%
plt.plot(valid_grad_improve, label = 'grad')
plt.plot(valid_reg_improve, label = 'reg')
plt.legend()
#%%
plt.plot(grad_improve, label = 'grad')
plt.plot(reg_improve, label = 'reg')
plt.legend()
#%%
pd.DataFrame(np.array([grad_improve, reg_improve]).T, columns = ['gradient', 'manual reg']).to_csv('export_results.csv')
for i in range(data_shuffles):
    data_export = np.append(np.append(np.append(grad_errors[i], reg_errors[i], axis = 1), auto_reg_errors[i], axis = 1), np.insert(np.array([lambda_paths[i]]).T, 0, grid_lambdas[i], axis = 1), axis = 1)
    data_columns = ['Train-Grad', 'Valid-Grad', 'Test-Grad', 'Train-Reg', 'Valid-Reg', 'Test-Reg', 'Train-Auto', 'Valid-Auto', 'Test-Auto', 'Lambda-Grid', 'Lambda-Auto']
    pd.DataFrame(data_export, columns = data_columns).to_csv('export_results_detail_' + str(i) + '.csv')
#%%
look_index = 0
#%%
plt.plot(grad_errors[look_index][10:, 0], label = 'train')
plt.plot(grad_errors[look_index][10:, 1], label = 'valid')
plt.plot(grad_errors[look_index][10:, 2], label = 'test')
plt.legend()
#%%
plt.plot(reg_errors[look_index][10:, 0], label = 'train')
plt.plot(reg_errors[look_index][10:, 1], label = 'valid')
plt.plot(reg_errors[look_index][10:, 2], label = 'test')
plt.legend()
#%%
plt.plot(auto_reg_errors[look_index][10:, 0], label = 'train')
plt.plot(auto_reg_errors[look_index][10:, 1], label = 'valid')
plt.plot(auto_reg_errors[look_index][10:, 2], label = 'test')
plt.legend()
#%%
plt.plot(np.exp(lambda_paths[look_index]), label = 'lambda')