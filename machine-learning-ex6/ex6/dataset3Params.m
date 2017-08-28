function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma = [0.01 0.03 0.1 0.3 1 3 10 30];
x1 = [1 2 1]; x2 = [0 4 -1];
f_score = 0;
c_max = 0;
sigma_max = 0;
num_params = length(C);
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
%UNCOMMENT THIS TO TRAIN
%{
  for i = 1:num_params
  for j = 1:num_params
    % train model given C and sigma
    model = svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
    % make predicions on the cross validation set 
    predictions = svmPredict(model, Xval);
    % calculate precision, recall and f1score
    predicted_positive = sum(predictions == 1); 
    true_positive = sum(yval == 1 & predictions == 1);   
    false_negative = sum(yval == 1 & predictions == 0);
    precision = true_positive / predicted_positive;
    recall = true_positive / (true_positive + false_negative);
    display(precision);
    display(recall);
    f_score_temp =  (2 * precision * recall) / (precision + recall);
    % if it's the highest f1score we had, we save it with i and j
    if f_score_temp > f_score
      f_score = f_score_temp;
      c_max = i;
      sigma_max = j;
    end
  end
end

C = C(c_max);
sigma = sigma(sigma_max); 
%}

# COMMENT THIS TO TRAIN
C = 1;
sigma = 0.1;
% =========================================================================

end
