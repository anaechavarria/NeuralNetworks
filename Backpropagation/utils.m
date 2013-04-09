%Utils

%assert(length(neurons_on_hidden_layer) == num_hidden_layers);
%network.neurons_on_layer = vertcat(num_inputs, neurons_on_hidden_layer, num_outputs);

