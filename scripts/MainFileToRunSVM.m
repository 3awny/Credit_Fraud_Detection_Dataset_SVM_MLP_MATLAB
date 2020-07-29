%% Initialising variables
% ************************
X = readmatrix('X_Final_CreditFraud_Balanced4.csv');
Y = readmatrix('Y_Final_CreditFraud_Balanced4.csv');

rng(1)
cv = cvpartition(size(X,1),'HoldOut',0.25);
idx = cv.test;

X_train = X(~idx,:);
X_test  = X(idx,:);

cv2 = cvpartition(Y,'HoldOut',0.25);
idx2 = cv2.test;

y_train = Y(~idx2,:);
y_test  = Y(idx2,:);
icol = size(X_train,2);     


final_predictions = [];
final_scores = [];

%% RUN FINAL MODEL SVM
% *********************

% THROUGH A HYPERPARAMETER GRID-SEARCH THE RBF KERNEL FUNCTION WAS FOUND
% TO BE THE BEST ALONG WITH A BOX-CONSTRAINT OF 100 AND A GAMMA AT 10

y_target = [];

% Train Model
rng(1)
val_model = fitcsvm(X_train, y_train, 'KernelFunction','rbf', 'KernelScale', 10,...
    'BoxConstraint', 100, 'Standardize', true);

[predicted_labels, scores] = predict(val_model, X_test);
final_predictions = [final_predictions; predicted_labels];
final_scores = [final_scores; scores];

y_target = [y_target; y_test];

val_model_cm = confusionmat(y_test, predicted_labels);
% Confusion Matrix
disp(val_model_cm)

val_model_acc = 100*sum(diag(val_model_cm))./sum(val_model_cm(:));
disp(val_model_acc)

[accuracy, fscore, fp_rate, recall, precision] = performance(val_model_cm);

[x_fp_rbf, y_tp_rbf, t_cv_rbf, auc_rbf] = perfcurve(y_target, scores(:,2),'1');

% ROC Curve
figure
plot(x_fp_rbf,y_tp_rbf)
title('SVM RBF Final Model ROC Curve')
xlabel('False-positive rate')
ylabel('True-positive rate')

% Performances
sprintf('SVM: Final Model on the unseen Test Data')
sprintf('SVM: The main performance metric AUC-ROC = %f',auc_rbf*100)
sprintf('SVM: The Recall (i.e. True-Positive rate) = %f', recall*100)
sprintf('SVM: The False-Positive rate = %f', fp_rate*100)
sprintf('SVM: The Accuracy = %f', accuracy*100)
sprintf('SVM: The F1-score = %f', fscore*100)
sprintf('SVM: The Precision = %f', precision*100)