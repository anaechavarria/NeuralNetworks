%Utils

%assert(length(neurons_on_hidden_layer) == num_hidden_layers);
%network.neurons_on_layer = vertcat(num_inputs, neurons_on_hidden_layer, num_outputs);

% Input activation functions
for k = 1 : 1 + 2
    function_on_layer(k, 1) = ActivationFunction(k);
end