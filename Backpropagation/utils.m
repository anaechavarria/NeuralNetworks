%Utils

%assert(length(neurons_on_hidden_layer) == num_hidden_layers);
%network.neurons_on_layer = vertcat(num_inputs, neurons_on_hidden_layer, num_outputs);

for k = 1 : 1 + 2
    function_on_layer2(k, 1) = ActivationFunction(k);
end