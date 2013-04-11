% Runs a test for the backpropagation algorithm with the data stored in
% parcial.mat. Additionally there can be a file named ActiFunction.mat with
% three activation functions [linear(1, 0); tan-sigmoid(1, 1); linear(1, 0). 
% If you want to change the activation functions or the number of layers
% uncomment code on lines 47-50

% Copyrights Ana Echavarria (anaechavarriau@gmail.com)

%% Read data
clear

file_name = 'parcial.mat';
Data = load('-mat', file_name);

vars = fieldnames(Data);
for i = 1:length(vars)
    assignin('base', vars{i}, Data.(vars{i}));
end
clear file_name i Data vars 

MAXN = 10000;
inputs = [Pm(1:MAXN), omega(1:MAXN), malfa(1:MAXN), lambda(1:MAXN), m_Psi(1:MAXN)]; 
outputs = [efi_conver(1:MAXN), eficiencia_vol_omega_Pm(1:MAXN)];

% Check matrix inputs and matrix outputs and find number of input and output
% variables
input_size = size(inputs);
num_inputs = input_size(2);
output_size = size(outputs);
num_outputs = output_size(2);
assert(input_size(1) == output_size(1), 'The number of cases in the input does not match the number of cases in the output');
num_cases = input_size(1);

% Input details of hidden layers 
num_hidden_layers = 1;
neurons_on_hidden_layer = [10];

% Input activation functions

file_name = 'ActiFunction.mat';
Data = load('-mat', file_name);

vars = fieldnames(Data);
for i = 1:length(vars)
    assignin('base', vars{i}, Data.(vars{i}));
end
clear file_name i Data vars 

% Optional if we want to change activation functions

% for k = 1 : num_hidden_layers + 2
%     function_on_layer(k, 1) = ActivationFunction(k);
% end
% clear k

% Input details of backpropagation algorithm
max_iterations = 200;
learning_rate = 0.2;
desired_error = 1e-4;

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

network = backpropagation(num_inputs,num_outputs, num_hidden_layers, neurons_on_hidden_layer, function_on_layer, normalized_inputs, normalized_outputs, max_iterations, learning_rate, desired_error);

fprintf('Output variable 1 is efi_conver\n');
fprintf('Output variable 2 is eficiencia_vol_omega_Pm\n');