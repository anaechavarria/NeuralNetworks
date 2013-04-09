% Runs a test for the backpropagation algorithm with the data stored in
% file_name. The data must contain 2 matrices named inputs and outputs of the
% same number of rows (number of test cases) and a column for each input/output
% variable

%% Read data
file_name = 'TestData.mat';
Data = load('-mat', file_name);

vars = fieldnames(Data);
for i = 1:length(vars)
    assignin('base', vars{i}, Data.(vars{i}));
end
clear file_name i Data vars 

% Check matrix inputs and matrix outputs and find number of input and output
% variables
input_size = size(inputs);
num_inputs = input_size(2);
output_size = size(outputs);
num_outputs = output_size(2);
assert(input_size(1) == output_size(1), 'The number of cases in the input does not match the number of cases in the output');
num_cases = input_size(1);

% Input details of hidden layers 
num_hidden_layers = input('Input the number of hidden layers: ');
neurons_on_hidden_layer = input('Input a column array with the number of neurons on each hidden layer: ');


% Input activation functions
for k = 1 : num_hidden_layers + 2
    function_on_layer(k, 1) = ActivationFunction(k);
end
clear k

% Input details of backpropagation algorithm
max_iterations = input('Input the maximum number of iterations for the backpropagation algorithm: ');
learning_rate = input('Input the learning rate for the backpropagation algorithm: ');
desired_error = input('Input the desired error for the output of the network during training: ');

%% Normalize data
normalized_inputs = inputs;
normalized_outputs = outputs;


min_input = zeros(num_inputs, 1);
max_input = zeros(num_inputs, 1);
for i = 1 : num_inputs
    min_input(i) = inputs(1, i);
    max_input(i) = inputs(1, i);
    for p = 1 : num_cases
       min_input(i) = min(min_input(i), inputs(p, i)); 
       max_input(i) = max(max_input(i), inputs(p, i)); 
    end
end
min_output = zeros(num_outputs, 1);
max_output = zeros(num_outputs, 1);
for i = 1 : num_outputs
    min_output(i) = outputs(1, i);
    max_output(i) = outputs(1, i);
    for p = 1 : num_cases
       min_output(i) = min(min_output(i), outputs(p, i)); 
       max_output(i) = max(max_output(i), outputs(p, i)); 
    end
end

assert(length(min_input) == num_inputs && length(max_input) == num_inputs);
for p = 1 : num_cases
    for i = 1 : num_inputs
        normalized_inputs(p, i) = (inputs(p, i) - min_input(i)) / (max_input(i) - min_input(i));
    end
    for i = 1 : num_outputs
        normalized_outputs(p, i) = (outputs(p, i) - min_output(i)) / (max_output(i) - min_output(i));
    end
end
clear p min_input max_input min_output max_output input_size output_size

%% Run backpropagation

network = backpropagation(num_inputs,num_outputs, num_hidden_layers, neurons_on_hidden_layer, function_on_layer, normalized_inputs, normalized_outputs, max_iterations, learning_rate, desired_error)
