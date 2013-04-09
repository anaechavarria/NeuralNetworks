% Create and train a neural network using the backpropagation algorithm
% * num_inputs = number of input variables
% * num_outputs = number of output variables
% * num_hidden_layers = number of hidden layers of the network
% * neurons_on_hidden_layer = a [number_hidden_layers x 1] array with the number
%                             of neurons on each hidden layer
% * function_on_layer = a [num_hidden_layers + 2 x 1] array of objects of the
%                       class ActivationFunction with the activation function of
%                       each layer
% * inputs = a [p x num_inputs] array with the input variables for the p test
%            cases
% * outputs = a [p x num_outputs] array with the output variables for the p test
%             cases
% * max_iterations = the maximum number of iteratios for the backpropagation
%                    algorithm
% * learning_rate = the learning rate of the network when changing its weights
% * desired_error = the desired error for the output of the training data

function network = backpropagation(num_inputs,num_outputs, num_hidden_layers, ...
                                   neurons_on_hidden_layer, function_on_layer, ...
                                   inputs, outputs, max_iterations,  ...
                                   learning_rate, desired_error)

num_layers = num_hidden_layers + 2;

% Assertions for input
size_hidden = size(neurons_on_hidden_layer);
assert(size_hidden(1) == num_hidden_layers, 'Size of array of neurons on hidden layer must be [number of hidden layers x 1]');

size_input = size(inputs);
assert(size_input(2) == num_inputs, 'Size of inputs array must be [p x num_inputs] where p are the different cases to evaluate');

size_output = size(outputs);
assert(size_output(2) == num_outputs, 'Size of outputs array must be [p x num_outputs] where p are the different cases to evaluate');

assert(size_input(1) == size_output(1), 'Number of rows of input must match number of rows of output');

size_functions = size(function_on_layer);
assert(min(class(function_on_layer) == 'ActivationFunction') == 1, 'The functions on each layer must be an array of type ActivationFunction')
assert(size_functions(1) == num_layers, 'Size of function array must be of size [num_hidden_layers + 2, 1]');


% Create vector of neurons on layer
neurons_on_layer = vertcat(num_inputs, neurons_on_hidden_layer, num_outputs);
assert(length(neurons_on_layer) ==  num_layers);
max_neurons = max(neurons_on_layer);

% Create network
network = Network(num_inputs,num_outputs, num_hidden_layers, neurons_on_layer, function_on_layer);

% Chose trainig data form input
num_cases = size_input(1);
num_training_cases = ceil(0.6 * num_cases);
num_validation_cases = ceil(0.2 * num_cases);
training_input_data = zeros(num_training_cases, num_inputs);
training_output_data = zeros(num_training_cases, num_outputs);
validation_input_data = zeros(num_validation_cases, num_inputs);
validation_output_data = zeros(num_validation_cases, num_outputs);

permutation = randperm(num_cases)';

for i = 1 : num_training_cases
    chosen_index = permutation(i);
    ... fprintf('Element %d chosen for training\n', chosen_index);
    training_input_data(i, :) = inputs(chosen_index, :);
    training_output_data(i, :) = outputs(chosen_index, :); 
end

for i = 1 : num_validation_cases
    chosen_index = permutation(num_training_cases + i);
    ... fprintf('Element %d chosen for validation\n', chosen_index);
    validation_input_data(i, :) = inputs(chosen_index, :);
    validation_output_data(i, :) = outputs(chosen_index, :); 
end

% Perform backpropagation with training data
average_error = 100 * ones(max_iterations, 1);
iterations_made = 0;
for n = 1 : max_iterations    
    [test_output, out_layer, in_layer] = feed(network, training_input_data);
    size_out_layer = size(out_layer);
    size_in_layer = size(in_layer);
    assert(size_out_layer(1) == num_training_cases && size_out_layer(3) == num_layers);
    assert(size_in_layer(1)  == num_training_cases && size_in_layer(3)  == num_layers);
    
    % local_error_on_case(p, i) = error of output neuron i when fed with input case p
    % error_on_case(p) = square sum of local error when fed with input case p
    % average_error = average sum of error on each case;
    local_error_on_case = zeros(num_training_cases, num_outputs);
    error_on_case = zeros(num_training_cases);
    average_error(n) = 0;
    for p = 1 : num_training_cases
        sq_sum = 0;
        for i = 1 : num_outputs
            local_error_on_case(p, i) = training_output_data(p, i) - test_output(p, i);
            sq_sum = sq_sum + (local_error_on_case(p, i)) ^ 2;
        end
        error_on_case(p) = 0.5 * sq_sum;
        average_error(n) = average_error(n) + error_on_case(p);
    end
    average_error(n) = average_error(n) / num_training_cases;
    ... fprintf('On iteration %d average error is %e\n', n, average_error(n));
        
    iterations_made = n;
    if (average_error(n) <= desired_error), break; end
        
    % delta(p, i, k) = local gradient of neuron i of layer k when fed with input
    % case p
    delta = zeros(num_training_cases, max_neurons, num_layers);
    for p = 1 : num_training_cases
        for i = 1 : num_outputs
            delta(p, i, num_layers) = local_error_on_case(p, i) * function_on_layer(num_layers).df(in_layer(p, i, num_layers));
        end
        
        for k = num_layers - 1 : -1 : 1
            for i = 1 : neurons_on_layer(k)
                delta(p, i, k) = 0;
                for j = 1 : neurons_on_layer(k+1)
                    delta(p, i, k) = delta(p, i, k) + delta(p, j, k+1) * network.weights(i, j);
                end
                delta(p, i, k) = delta(p, i, k) * function_on_layer(k).df(in_layer(p, i, k));
            end
        end
    end
    
    % delta_w(i, j, k) change in weight between neuron i of layer k and neuron j
    % of layer k + 1
    delta_w = zeros(max_neurons, max_neurons, num_layers);
    for k = 1 : num_layers - 1
       for i = 1 : neurons_on_layer(k)
           for j = 1 : neurons_on_layer(k+1)
               delta_w(i, j, k) = 0;
               for p = 1 : num_training_cases
                   delta_w(i, j, k) = delta_w(i, j, k) + delta(p, j, k + 1) * out_layer(p, i, k);
               end
               delta_w(i, j, k) = delta_w(i, j, k) * (learning_rate  / num_training_cases);
           end
       end
    end
    
    update_weights(network, delta_w);    
end
fprintf('Average error on training data afeter last iteration is %e\n', average_error(iterations_made));

% Feed network with validation data
validation_output = feed(network, validation_input_data);
val_average_error = 0;
for p = 1 : num_validation_cases
    sq_sum = 0;
    for i = 1 : num_outputs
        sq_sum = sq_sum + (validation_output_data(p, i) - validation_output(p, i)) ^ 2;
    end
    val_average_error = val_average_error + 0.5 * sq_sum;
end
val_average_error = val_average_error / num_validation_cases;
fprintf('Average error on validation data is %e\n', val_average_error);

% Feed network with complete data
total_output = feed(network, inputs);
total_average_error = 0;
for p = 1 : num_cases
    sq_sum = 0;
    for i = 1 : num_outputs
        sq_sum = sq_sum + (outputs(p, i) - total_output(p, i)) ^ 2;
    end
    total_average_error = total_average_error + 0.5 * sq_sum;
end
total_average_error =total_average_error / num_cases;
fprintf('Average error on total data is %e\n', total_average_error);


% Plot average error
subplot(4, num_outputs, 1 : num_outputs);
plot(average_error(1 : iterations_made));
xlabel('Iteration');
ylabel('Error');
title('Average Error');

% Plot training results
for j = 1 : num_outputs
    subplot(4, num_outputs, num_outputs + j);

    X = (permutation(1 : num_training_cases))';
    scatter(X, test_output(:, j), 5, 'r');
    hold 'on'
    scatter(X, training_output_data(:, j) , 5, 'b');
    hold 'off'
    xlabel('Input');
    ylabel(sprintf('Output %d', j));
    title('Results on training data');
end

% Plot validation results
for j = 1 : num_outputs
    subplot(4, num_outputs, 2 * num_outputs + j);

    X = (permutation(num_training_cases + 1: num_training_cases + num_validation_cases))';
    scatter(X, validation_output(:, j), 5, 'r');
    hold 'on'
    scatter(X, validation_output_data(:, j) , 5, 'b');
    hold 'off'
    xlabel('Input');
    ylabel(sprintf('Output %d', j));
    title('Results on validation data');
end

% Plot total results
for j = 1 : num_outputs
    subplot(4, num_outputs, 3 * num_outputs + j);

    X = (1: num_cases)';
    scatter(X, total_output(:, j), 5, 'r');
    hold 'on'
    scatter(X, outputs(:, j) , 5, 'b');
    hold 'off'
    xlabel('Input');
    ylabel(sprintf('Output %d', j));
    title('Results on total data');
end

end