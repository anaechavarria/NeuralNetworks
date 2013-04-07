classdef ActivationFunction
    % ActivationFunction: stores function and its derivative in inline
    % format
    %   The constructur ask the user to input the type of function (1 = linear,
    %   2 = log-sigmoid, 3 = tan-sigmoid) and according to the function type
    %   asks for the functions parameters. The result is the function and its
    %   derivative in the inline format.
    
    properties
        f;
        df;
    end
    
    methods
        function fun = ActivationFunction(layer)
            msg = sprintf('Insert function type for layer %d.\n(1 = linear, 2 = log-sigmoid, 3 = tan-sigmoid) ', layer);
            type = input(msg);
            while (type ~= 1 && type ~= 2 && type ~= 3)
                fprintf('\nInvalid function type! Please try again. ');
                type = input(msg);
            end
            
            switch type
                case 1
                    fprintf('\nYou have chosen a linear function\n\n');
                    a = input('Insert the value of the slope: ');
                    b = input('Insert the value of the y-intercept: ');
                    
                    % In case any parameter is empty
                    if isempty(a), a = 1; end
                    if isempty(b), b = 0; end
                    
                    y = sprintf('%g * x + %g', a, b);
                    fun.f = inline(y);
                    dy = sprintf('%g', a);
                    fun.df = inline(dy);
                    
                case 2
                    fprintf('\nYou have chosen a log-sigmoid function\n\n');
                    a = input('Insert the value of the slope: ');
                    
                    % In case parameter is empty
                    if isempty(a), a = 1; end
                    
                    y = sprintf('1 / (1 + exp(-%g * x))', a);
                    fun.f = inline(y);
                    dy = sprintf('%g * (%s) * (1 - (%s))', a, y, y);
                    fun.df = inline(dy);
                    
                case 3
                    fprintf('\nYou have chosen a tan-sigmoid function\n\n');
                    a = input('Insert the value of the scale: ');
                    b = input('Insert the value of the slope: ');
                    
                    % In case any parameter is empty
                    if isempty(a), a = 1; end
                    if isempty(b), b = 1; end
                    
                    y = sprintf('%g * tanh(%g * x)', a, b);
                    fun.f = inline(y);
                    dy = sprintf('(%g / %g) * (%g^2 - (%s)^2)', b, a, a, y);
                    fun.df = inline(dy);
                otherwise
                    assert(false);
            end
        end
    end
end

