% Create and train a neural network using the backpropagation algorithm

function network = backpropagation(num_inputs,num_outputs, num_hidden_layers, ...
                                   neurons_on_hidden_layer, function_on_layer, ...
                                   inputs, outputs, max_iterations, learning_rate)

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
permutation = randperm(num_cases)';
training_input_data = zeros(num_training_cases, num_inputs);
training_output_data = zeros(num_training_cases, num_outputs);

for i = 1 : num_training_cases
    chosen_index = permutation(i);
    ... fprintf('Element %d chosen for training\n', chosen_index);
    training_input_data(i, :) = inputs(chosen_index, :);
    training_output_data(i, :) = outputs(chosen_index, :);
end

average_error = 100 * ones(max_iterations, 1);
iterations_made = 0;
for n = 1 : max_iterations
    if (average_error(n) < 1e-3), break; end
    
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
            local_error_on_case(p, i) = outputs(p, i) - test_output(p, i);
            sq_sum = sq_sum + (local_error_on_case(p, i)) ^ 2;
        end
        error_on_case(p) = 0.5 * sq_sum;
        average_error(n) = average_error(n) + error_on_case(p);
    end
    average_error(n) = average_error(n) / num_training_cases;
    fprintf('On iteration %d average error is %d\n', n, average_error(n));
        
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
               delta_w(i, j, k) = delta_w(i, j, k) * (learning_rate / num_training_cases);
           end
       end
    end
    
    update_weights(network, delta_w);    
    
    iterations_made = n;
end

plot(average_error(1 : iterations_made));

end