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

cv = cvpartition(size(y_train,1), 'Holdout', 1/3);

%% Hyper-Parameters and Storage Variables
% ****************************************

depth = [1, 2, 3];
neurons = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
trainingFunction = ["traingda", "traingdm", "traingdx", "trainscg", "trainoss"];
transferFunction = ["logsig", "poslin", "tansig", "purelin"];

auc_check = 0;
fscore_check = 0;
auc_all = [];
fscore_all = [];
folds = 5;
partScheme = cvpartition(y_train, 'KFold', folds);

acc_all = [];
acc_check=0;
recall_all = [];
recall_check = 0;
fp_all = [];
fp_check = 1000;

%% Optimize Hyperparameters (Grid-Search)
% ****************************************
mlp_time = tic;
for d=depth
    for n=neurons
        for trngfunc=trainingFunction
            for trnsfunc=transferFunction
                indx = crossvalind('kfold', y_train, folds);
                final_predictions = [];
                final_scores = [];                

                tic;
                for i=1:folds
                    y_target = [];
                    sprintf('MLP Iteration d=%d | n=%d | trng=%d | trns=%d | k-%d \n',...
                        find(depth==d), find(neurons==n), find(trngfunc==trainingFunction),...
                        find(trnsfunc==transferFunction), i) % to keep track of progress
                    
                    % Cross-Validation Splitting
                    xTrain = X_train(training(partScheme, i),:);
                    tTrain = y_train(training(partScheme, i),:);
                    xVal = X_train(test(partScheme,i),:);
                    tVal = y_train(test(partScheme,i),:);
                    
                    numberOfHiddenLayers = n * ones(1, d); % Vector of Hidden Layer Size (Network Architecture)
                    net = patternnet(numberOfHiddenLayers, char(trngfunc)); % Build Network
                    net.trainParam.epochs = 500; % Number of epochs
                    net.trainParam.max_fail = 6; % Early stopping after six consecutive increases of Validation Performance

                    if any(strcmp({'traingda', 'traingdm', 'traingdx'} , char(trngfunc)))
                        net.trainParam.lr = 0.01; % Update Learning Rate
                        if ~strcmp('traingda', char(trngfunc))
                            net.trainParam.mc = 0.95; % Update Momentum
                        end
                    end
                    
                    for j=1:d-1 % Update Activation Function of Hidden-Layers
                        net.layers{j}.transferFcn = char(trnsfunc);
                    end
                    
                    % Train Network
                    rng(1);
                    [net, tr] = train(net, xTrain', tTrain');
                    
                    tVal_t = tVal';
                    y_prd = net(xVal');
                    y_prd2 = round(y_prd);
                    y_prd_train = net(xTrain');
                    y_prd_train2 = round(y_prd_train);
                    
                    val_model_cm = confusionmat(tVal_t, y_prd2);
                    train_model_cm = confusionmat(tTrain', y_prd_train2);
                    [accuracy, fscore, fp_rate, recall] = performance(val_model_cm);
                    [accuracy_tr, fscore_tr, fp_rate_rt, recall_tr] = performance(train_model_cm);
                    
                    % AUC-ROC
                    [x_fp_mlp, y_tp_mlp, t_cv_mlp, auc_mlp] = perfcurve(tVal, y_prd(1,:),'1');
                    
                    val_model_acc = 100*sum(diag(val_model_cm))./sum(val_model_cm(:));

                    fscore_all(i) = fscore;
                    recall_all(i) = recall;
                    fp_all(i) = fp_rate;
                    acc_all(i) = accuracy;
                    auc_all(i) = auc_mlp;
                end
                
                recall_mean = mean(recall_all);
                fp_rate_mean = mean(fp_all);
                fscore_mean = mean(fscore_all);
                acc_mean = mean(acc_all);
                auc_mean = mean(auc_all);
                
                % to print last best performing hyperparameter combination
                
                if auc_mean>auc_check
                    d1=d;
                    n1=n;
                    trngfunc1=trngfunc;
                    trnsfunc1=trnsfunc;
                    auc_check=auc_mean;
                end
                fprintf('MLP Params: d=%d | n=%d | trng=%s | trns=%s | AUC-ROC=%d \n', d1, n1, trngfunc1, trnsfunc1, auc_check*100)

                if acc_mean>acc_check
                    d2 = d;
                    n2 = n;
                    trngfunc2=trngfunc;
                    trnsfunc2=trnsfunc;
                    acc_check = acc_mean;
                end
                fprintf('MLP Params: d=%d | n=%d | trng=%s | trns=%s | Accuracy=%d \n', d2, n2, trngfunc2, trnsfunc2, acc_check*100)
                
                if fscore_mean>fscore_check
                    d3 = d;
                    n3 = n;
                    trngfunc3=trngfunc;
                    trnsfunc3=trnsfunc;
                    fscore_check=fscore_mean;
                end
                fprintf('MLP Params: d=%d | n=%d | trng=%s | trns=%s | F1-score=%d \n', d3, n3, trngfunc3, trnsfunc3, fscore_check*100)
                
                if recall_mean>recall_check
                    d4 = d;
                    n4 = n;
                    trngfunc4=trngfunc;
                    trnsfunc4=trnsfunc;
                    recall_check=recall_mean;
                end
                fprintf('MLP Params: d=%d | n=%d | trng=%s | trns=%s | Recall=%d \n', d4, n4, trngfunc4, trnsfunc4, recall_check*100)

                if fp_rate_mean<fp_check
                    d5 = d;
                    n5 = n;
                    trngfunc5=trngfunc;
                    trnsfunc5=trnsfunc;
                    fp_check=fp_rate_mean;
                end
                fprintf('MLP Params: d=%d | n=%d | trng=%s | trns=%s | FP-Rate=%d \n', d5, n5, trngfunc5, trnsfunc5, fp_check*100)
            end
        end
    end
end
mlp_time_final = toc(mlp_time)/60;
