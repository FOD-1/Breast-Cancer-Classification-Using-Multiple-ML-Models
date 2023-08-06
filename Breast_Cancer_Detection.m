clear,clc;
% Breast Cancer Classification using Machine Learning Classifiers
% Author: F. O. DJIBRILLAH

% Load the dataset
data = readtable('breast-cancer.csv');

% Separate features and labels
X = data{:, 2:end}; % Assuming features start from the second column
y = data{:, 1};    % Assuming labels are in the first column (0 for Benign, 1 for Malignant)

% Convert 'y' from cell array to a numeric array
y(strcmp(y, 'M')) = {1};
y(strcmp(y, 'B')) = {0};
ynew = cell2mat(y);
y = ynew;

%% 
% Split the dataset into training and testing sets
rng(42); % For reproducibility
cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

%%
% Support Vector Machine (SVM)
svm_model = fitcsvm(X_train, y_train, "KernelFunction","linear");
svm_pred = predict(svm_model, X_test);

% K-Nearest Neighbors (KNN)
knn_model = fitcknn(X_train, y_train);
knn_pred = predict(knn_model, X_test);

% Logistic Regression
logreg_model = fitglm(X_train, y_train, 'Distribution', 'binomial');
logreg_pred = round(predict(logreg_model, X_test));

% Decision Tree
tree_model = fitctree(X_train, y_train);
tree_pred = predict(tree_model, X_test);

% Naive Bayes
nb_model = fitcnb(X_train, y_train);
nb_pred = predict(nb_model, X_test);

% Neural Networks using Neural Network Toolbox
net = patternnet(10); % 10 hidden units
net = train(net, X_train', y_train');
nn_pred = round(net(X_test'));

%%
% Evaluate the classifiers
accuracy = @(y_true, y_pred) sum(y_true == y_pred) / numel(y_true);
precision = @(y_true, y_pred) sum(y_true & y_pred) / sum(y_pred);
recall = @(y_true, y_pred) sum(y_true & y_pred) / sum(y_true);
f1_score = @(y_true, y_pred) 2.* precision(y_true, y_pred).* recall(y_true, y_pred) / (precision(y_true, y_pred) + recall(y_true, y_pred));

svm_accuracy = accuracy(y_test, svm_pred);
knn_accuracy = accuracy(y_test, knn_pred);
logreg_accuracy = accuracy(y_test, logreg_pred);
tree_accuracy = accuracy(y_test, tree_pred);
nb_accuracy = accuracy(y_test, nb_pred);
nn_accuracy = accuracy(y_test, nn_pred);

svm_f1 = f1_score(y_test, svm_pred);
knn_f1 = f1_score(y_test, knn_pred);
logreg_f1 = f1_score(y_test, logreg_pred);
tree_f1 = f1_score(y_test, tree_pred);
nb_f1 = f1_score(y_test, nb_pred);
nn_f1 = f1_score(y_test, nn_pred);

% Display the results
disp('Classifier Accuracy:');
disp(['SVM: ', num2str(svm_accuracy)]);
disp(['K-Nearest Neighbors: ', num2str(knn_accuracy)]);
disp(['Logistic Regression: ', num2str(logreg_accuracy)]);
disp(['Decision Tree: ', num2str(tree_accuracy)]);
disp(['Naive Bayes: ', num2str(nb_accuracy)]);
disp(['Neural Networks: ', num2str(nn_accuracy)]);

disp('Classifier F1-score:');
disp(['SVM: ', num2str(svm_f1)]);
disp(['K-Nearest Neighbors: ', num2str(knn_f1)]);
disp(['Logistic Regression: ', num2str(logreg_f1)]);
disp(['Decision Tree: ', num2str(tree_f1)]);
disp(['Naive Bayes: ', num2str(nb_f1)]);
disp(['Neural Networks: ', num2str(nn_f1)]);