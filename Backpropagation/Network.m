classdef Network < handle
    % Network: creates and operates on neural networks
    %   Methods in thid function are:
    %   * Network = class constructor
    %   * feed = output when feeding network with a given set of inputs
    %   * uptade_weights = update network weights given a weight chage matrix
    %   * reset_weights = reset network weights to random numbers
    %   * print_weights = print network weights in a detailed manner
    
    % Copyrights Ana Echavarria (anaechavarriau@gmail.com)
    
    properties (SetAccess = private)
        % Number of inputs of the network
        num_inputs;
        % Number of outputs of the network
        num_outputs;
        % Number of layers of network
        num_layers;
        % A (num_layers x 1) array with the number of neurons on each layer
        neurons_on_layer;
        % A (num_layers x 1) array of objects of class the ActivationFunction
        % with the activation function of each layer
        function_on_layer;
        % weights(i, j, k) weight of connection from neuron i of layer k and
        % neuron j of layer k+1 -> k = 1 to num_layers - 1
        weights;
    end    
    
    methods
        
        % Class constructor: receives que number of inputs, the number of
        % outputs, the number of hidden layers (layers exluding the input and
        % output layers), the number of neurons on each layer (including input
        % and output layers) and the activation function of each layer
        % (including input and output layers) which is an array of objects of
        % class the ActivationFunction.
        % Returns the built network with random weights.
        function network = Network(num_inputs,num_outputs, num_hidden_layers, ...
                                   neurons_on_layer, function_on_layer)
            % Assertions for input
            assert(length(neurons_on_layer) == num_hidden_layers + 2);
            assert(neurons_on_layer(1) == num_inputs);
            assert(neurons_on_layer(num_hidden_layers + 2) == num_outputs);
            
            assert(min(class(function_on_layer) == 'ActivationFunction') == 1);            
            assert(length(function_on_layer) == num_hidden_layers + 2);
            
            % Variable assignment
            network.num_inputs = num_inputs;
            network.num_outputs = num_outputs;
            network.num_layers = num_hidden_layers + 2;
            
            network.neurons_on_layer = neurons_on_layer;
            network.function_on_layer = function_on_layer;
            
            % Weight assignment
            max_neurons = max(network.neurons_on_layer);
            network.weights = rand(max_neurons, max_neurons, network.num_layers);
        end
        
        % Feed network with input data. Input is a (p x num_inputs) matrix where
        % p are the number of different inputs cases to evaluate.        
        % Output is a (p x num_outputs) matrix with the results of the network,
        % out_layer is a (p x max_neurons x num_layers) matrix with the output
        % given by each neuron when fed with each input case and in_layer is a
        % (p x max_neurons x num_layers) matrix with the input of the neuron
        % when fed with each input case
        function [output, out_layer, in_layer] = feed(network, input)
            % Assertion on input
            input_size = size(input);
            assert(input_size(2) == network.num_inputs);
            %jo que solo se esta haciendo para un patron
            
            max_neurons = max(network.neurons_on_layer);
            
            % print_weights(network);
                            
            % in_layer(p, i, k) Input of nueron i of layer k when fed with input p
            in_layer  = zeros(input_size(1), max_neurons, network.num_layers);
            % out_layer(p, i, k) Output of nueron i of layer k when fed with input p
            out_layer = zeros(input_size(1), max_neurons, network.num_layers);
            
            for p = 1 : input_size(1)                
                ... fprintf('Feeding input case %d\n', p);

                % Compute output for input layer (layer 1)
                for i = 1 : network.neurons_on_layer(1)                    
                    in_layer(p, i, 1) = input(p, i);
                    out_layer(p, i, 1) = network.function_on_layer(1).f(in_layer(p, i, 1));
                    ... fprintf('Input  for neuron %d in layer 1 is = %g\n', i, in_layer(p, i, 1));
                    ... fprintf('Output for neuron %d in layer 1 is = %g\n', i, out_layer(p, i, 1));
                end                
                ... fprintf('\n');

                % Compute outpur for remaining layers
                for k = 2 : network.num_layers
                    for j = 1 : network.neurons_on_layer(k)
                        % Find the input of neuron j of layer k
                        in_layer(p, j, k) = 0;
                        for i = 1 : network.neurons_on_layer(k-1)
                            in_layer(p, j, k) = in_layer(p, j, k) + network.weights(i, j, k-1) * out_layer(p, i, k-1);
                        end
                        % Evaluate activation function of neuron j of layer k                    
                        out_layer(p, j, k) = network.function_on_layer(k).f(in_layer(p, j, k));

                        ... fprintf('Input  for neuron %d in layer %d is = %g\n', j, k, in_layer(p, j, k));
                        ... fprintf('Output for neuron %d in layer %d is = %g\n', j, k, out_layer(p, j, k));
                    end                    
                    ... fprintf('\n');
                end 
            end
            
            output = out_layer(1 : input_size(1) , 1 : network.num_outputs , network.num_layers);
            assert(min(size(output) == [input_size(1) network.num_outputs]) == 1);
        end        
        
        % Update network weights according to data on delta_w 
        function update_weights(network, delta_w)
            assert(max(size(delta_w) == size(network.weights)) == 1);
            % print_weights(network);
            network.weights = network.weights + delta_w;
            % print_weights(network);
        end
        
        % Reset network weights to random
        function reset_weights(network)
            network.weights = rand(size(network.weights));
        end
        
        % Print network wieghts
        function print_weights(network)            
            fprintf('\n');
            for k = 1 : network.num_layers - 1
                fprintf('Weights connecting layer %d and layer %d:\n', k, k+1);
                for i = 1 : network.neurons_on_layer(k)
                    for j = 1 : network.neurons_on_layer(k+1)
                        fprintf('%-*g ', 9, network.weights(i, j, k));
                    end
                    fprintf('\n');
                end
            end
            fprintf('\n');            
        end
        
    end   
end

