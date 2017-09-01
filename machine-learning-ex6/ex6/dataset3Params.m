function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vector = [0.01, 0.03, 0.1, 0.3, 1.3, 10, 30];
sigma_vector = [0.01, 0.03, 0.1, 0.3, 1.3, 10, 30];

num_C = length(C_vector);
num_sigma = length(sigma_vector);

error_matrix = zeros(num_C, num_sigma);

for i = 1: num_C
    for j = 1: num_sigma
        temp_C = C_vector(i);
        temp_sigma = sigma_vector(j);
        
        model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
        predictions = svmPredict(model, Xval);
        
        error_matrix(i, j) = mean(double(predictions ~= yval));
    end
end

[temp_errors, temp_indices] = min(error_matrix, [], 1);
[~, j] = min(temp_errors, [], 2);
i = temp_indices(j);

C = C_vector(i);
sigma = sigma_vector(j);

fprintf('------------------optimal C = %f and optimal sigma = %f------------------\n', C, sigma);

% =========================================================================

end
