%% Initialising variables
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

cv = cvpartition(size(y_train,1), 'Holdout', 1/3); 

%% RUN FINAL MODEL Multilayer Perceptron

% THROUGH A HYPERPARAMETER GRID-SEARCH (AUC-ROC), THE MLP WITH A DEPTH OF 2, NUMBER OF
% NEURONS FOR HIDDEN LAYERS OF 15, TRAINING FUNCTION - TRAINOSS AND
% TRANSFER FUNCTION (ACTIVATION FUNCTION) - POSLIN ... WAS FOUND TO BE THE
% BEST PERFORMING MODEL

y_target = [];

numberOfHiddenLayers = 15 * ones(1, 2); % Vector of Hidden Layer Size (Network Architecture)
net = patternnet(numberOfHiddenLayers, char('trainoss')); % Build Network
net.trainParam.epochs = 500; % Number of epochs
net.trainParam.max_fail = 6; % Early stopping after six consecutive increases of Validation Performance

if any(strcmp({'traingda', 'traingdm', 'traingdx'} , char('trainoss')))
    net.trainParam.lr = 0.01; % Update Learning Rate
    if ~strcmp('traingda', char('trainoss'))
        net.trainParam.mc = 0.95; % Update Momentum
    end
end

for j=1:2 % Update Activation Function of Hidden-Layers
    net.layers{j}.transferFcn = char('poslin'); 
end

% Train Network
rng(1);
[net, tr] = train(net, X_train', y_train');

y_prd = net(X_test');
y_prd2 = round(y_prd);

val_model_cm = confusionmat(y_test, y_prd2);
% Confusion Matrix
disp(val_model_cm)

[accuracy, fscore, fp_rate, recall, precision] = performance(val_model_cm);
val_model_acc = 100*sum(diag(val_model_cm))./sum(val_model_cm(:));

[x_fp_mlp, y_tp_mlp, t_cv_mlp, auc_mlp] = perfcurve(y_test, y_prd(1,:),'1');

% ROC Curve
figure
plot(x_fp_mlp,y_tp_mlp)
title('MLP Final Model ROC Curve')
xlabel('False-positive rate')
ylabel('True-positive rate')

% Performances
sprintf('Final Model (MLP) on the unseen Test Data')
sprintf('MLP: The main performance metric AUC-ROC = %f',auc_mlp*100)
sprintf('MLP: The Recall (i.e. True-Positive rate) = %f', recall*100)
sprintf('MLP: The False-Positive rate = %f', fp_rate*100)
sprintf('MLP: The Accuracy = %f', accuracy*100)
sprintf('MLP: The F1-score = %f', fscore*100)
sprintf('MLP: The Precision = %f', precision*100)