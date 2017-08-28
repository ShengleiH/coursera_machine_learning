function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    
    % implement in matrix representation
    h = X * theta;
    theta = theta - alpha/m * (X' * (h - y));
    
%     % implement in conventional way
%     num_feature = length(theta);
%
%     % matlab list and matrix start from "1"
%     for j = 1:num_feature
%         sum_gradient = 0;
%         for i = 1:m
%             sum_gradient = sum_gradient + (h(i) - y(i)) * X(i,j);
%         end
%         theta(j) = theta(j) - alpha/m * sum_gradient;
%     end
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
