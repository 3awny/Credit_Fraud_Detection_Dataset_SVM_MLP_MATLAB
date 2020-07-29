%% Reading and Processing Data, Initialising Storage Variables and Hyperparameters
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

box_constraint = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000];

kernel_scale = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300];


poly_order = [2, 3, 4, 5];

kernel_function = ['rbf', 'linear', 'polynomial'];

f1_crossval_grid = zeros(length(kernel_scale),length(box_constraint));
acc_crossval_grid = zeros(length(kernel_scale), length(box_constraint));
training_error_grid = zeros(length(kernel_scale), length(box_constraint));
validation_error_grid = zeros(length(kernel_scale), length(box_constraint));
time_taken_grid = zeros(length(kernel_scale), length(box_constraint));
fp_crossval_grid = zeros(length(kernel_scale), length(box_constraint));
recall_crossval_grid = zeros(length(kernel_scale), length(box_constraint));
auc_crossval_grid = zeros(length(kernel_scale), length(box_constraint));

f1_crossval_grid_poly = zeros(length(poly_order),length(box_constraint));
acc_crossval_grid_poly = zeros(length(poly_order), length(box_constraint));
training_error_grid_poly = zeros(length(poly_order), length(box_constraint));
validation_error_grid_poly = zeros(length(poly_order), length(box_constraint));
time_taken_grid_poly = zeros(length(poly_order), length(box_constraint));
fp_crossval_grid_poly = zeros(length(poly_order), length(box_constraint));
recall_crossval_grid_poly = zeros(length(poly_order), length(box_constraint));
auc_crossval_grid_rbf = zeros(length(poly_order), length(box_constraint));

f1_crossval_grid_linear = zeros(length(box_constraint));
acc_crossval_grid_linear = zeros(length(box_constraint));
training_error_grid_linear = zeros(length(box_constraint));
validation_error_grid_linear = zeros(length(box_constraint));
time_taken_grid_linear = zeros(length(box_constraint));
fp_crossval_grid_linear = zeros(length(box_constraint));
recall_crossval_grid_linear = zeros(length(box_constraint));
auc_crossval_grid_linear = zeros(length(box_constraint));


% Stored Results
parameters = [];
parameters_poly = [];
parameters_linear = [];

final_predictions = [];
final_predictions_poly = [];
final_predictions_linear = [];

final_scores = [];
final_scores_poly = [];
final_scores_linear = [];

%rbf
accuracy_rbf = [];
f1_score_rbf = [];
fp_rate_rbf = [];
recall_rbf = [];
training_loss_rbf = [];
validation_loss_rbf = [];
time_taken_rbf = [];
auc_rbf_all = [];

% Polynomial
accuracy_poly = [];
f1_score_poly = [];
fpratePoly = [];
recallPoly = [];
training_loss_poly = [];
validation_loss_poly = [];
time_taken_poly = [];
auc_poly_all = [];

% Linear
accuracy_linear = [];
f1_score_linear = [];
fprateLinear = [];
recallLinear = [];
training_loss_linear = [];
validation_loss_linear = [];
time_taken_linear = [];
auc_linear_all = [];

%% Optimising SVM

svm_time=tic;
for c=box_constraint
    for g=kernel_scale
        folds_rbf=5;
        indx = crossvalind('kfold', y_train, folds_rbf);
        final_predictions = [];
        final_scores = [];
        parameters = [parameters; c; g];
    
        
        
        for i=1:folds_rbf
            rbfTic = tic;
            y_target = [];
            
            sprintf('RBF Iteration c=%d | g=%d | k-%d \n',...
                find(box_constraint==c), find(kernel_scale==g), i)
            
            x2_fold = X_train(indx==i,:);
            x1_fold = X_train(indx~=i,:);
            y1_fold = y_train(indx~=i,:);
            y2_fold = y_train(indx==i,:);
            idx_test=(indx==i);
            
            rng(1)
            val_model = fitcsvm(x1_fold, y1_fold, 'KernelFunction','rbf', 'KernelScale', g,...
                'BoxConstraint', c, 'Standardize', true);
            
            [predicted_labels, scores] = predict(val_model, x2_fold);
            [pred_tr, scr_tr] = predict(val_model, x1_fold);
            final_predictions = [final_predictions; predicted_labels];
            final_scores = [final_scores; scores];
            
            y_target = [y_target; y_train(idx_test)];
            y2_fold2 = num2cell(num2str(y2_fold));

            val_model_cm = confusionmat(y2_fold, predicted_labels);
            train_model_cm = confusionmat(y1_fold, pred_tr);
            val_model_acc = 100*sum(diag(val_model_cm))./sum(val_model_cm(:));
            disp(val_model_acc)
            accuracy_rbf(i) = val_model_acc;
            
            [tr_acc, tr_fscore, tr_fprate, tr_recall] = performance(train_model_cm);
            [accuracy, fscore, fp_rate, recall] = performance(val_model_cm);
            [x_fp_rbf, y_tp_rbf, t_cv_rbf, auc_rbf] = perfcurve(y_target, scores(:,2),'1');
            
            auc_rbf_all(i) = auc_rbf;
            
            disp(fp_rate)
            disp(recall)
            disp(fscore)
            disp(auc_rbf)
            %validation_err = sum(y2_fold ~= predicted_labels)/y2_fold;
            %fprintf("The Validation Accuracy is: %.2f%%\n", ((1-validation_err)*100))

            if ~isnan(fscore)
                f1_score_rbf(i) = fscore;
            end
            fp_rate_rbf(i) = fp_rate;
            recall_rbf(i) = recall;
            
            time_taken_rbf(i) = toc(rbfTic)/60;
        end
        
        % GRIDS FOR RBF SVM HYPER-PARAMETER STORAGE 
        f1_crossval_grid(find(box_constraint==c),find(kernel_scale==g))=mean(f1_score_rbf);
        acc_crossval_grid(find(box_constraint==c), find(kernel_scale==g)) = mean(accuracy_rbf);
        training_error_grid(find(box_constraint==c), find(kernel_scale==g)) = mean(training_loss_rbf);
        validation_error_grid(find(box_constraint==c), find(kernel_scale==g)) = mean(validation_loss_rbf);
        time_taken_grid(find(box_constraint==c), find(kernel_scale==g)) = mean(time_taken_rbf);
        fp_crossval_grid(find(box_constraint==c), find(kernel_scale==g)) = mean(fp_rate_rbf);
        recall_crossval_grid(find(box_constraint==c), find(kernel_scale==g)) = mean(recall_rbf);
        auc_crossval_grid(find(box_constraint==c), find(kernel_scale==g)) = mean(auc_rbf_all);
    end
end

for c=box_constraint
    for po=poly_order
        folds_poly=5;
        indx_poly = crossvalind('kfold', y_train, folds_poly);
        final_predictions_poly = [];
        final_scores_poly = [];
        
        parameters_poly = [parameters_poly; c; po];
        
        for i=1:folds_poly
            polyTic = tic;
            y_target_poly = [];
            
            sprintf('Polynomial Iteration c=%d | po=%d | k-%d \n',...
                find(box_constraint==c), find(poly_order==po), i)
            
            x2_fold_poly = X_train(indx_poly==i,:);
            x1_fold_poly = X_train(indx_poly~=i,:);
            y1_fold_poly = y_train(indx_poly~=i,:);
            y2_fold_poly = y_train(indx_poly==i,:);
            idx_test_poly=(indx_poly==i);
            
            rng(1)
            val_model_poly = fitcsvm(x1_fold_poly, y1_fold_poly, 'KernelFunction','polynomial', 'PolynomialOrder', po,...
                'BoxConstraint', c);
            
            [predicted_labels_poly, scores_poly] = predict(val_model_poly, x2_fold_poly);
            final_predictions_poly = [final_predictions_poly; predicted_labels_poly];
            final_scores_poly = [final_scores_poly; scores_poly];
            
            y_target_poly = [y_target_poly; y_train(idx_test_poly)];
            y2_fold2_poly = num2cell(num2str(y2_fold_poly));
            
            val_model_cm_poly = confusionmat(y2_fold_poly, predicted_labels_poly);
            val_model_acc_poly = 100*sum(diag(val_model_cm_poly))./sum(val_model_cm_poly(:));
            disp(val_model_acc_poly)
            accuracy_poly(i) = val_model_acc_poly;
            
            [accuracy_poly, fscore_poly, fp_rate_poly, recall_poly] = performance(val_model_cm_poly);
            [x_fp_poly, y_tp_poly, t_cv_poly, auc_poly] = perfcurve(y_target_poly,scores_poly(:,2),'1');
            
            auc_poly_all(i) = auc_poly;
            
            disp(fp_rate_poly)
            disp(recall_poly)
            disp(fscore)
            disp(auc_poly)
            
            if ~isnan(fscore_poly)
                f1_score_poly(i) = fscore_poly;
            end
            
            fpratePoly(i) = fp_rate_poly;
            recallPoly(i) = recall_poly;
            
            training_loss_poly(i) = mean(loss(val_model_poly, x1_fold_poly, y1_fold_poly));
            validation_loss_poly(i) = mean(loss(val_model_poly, x2_fold_poly, y2_fold_poly));
            
            time_taken_poly(i) = toc(polyTic)/60;
        end
        
        % GRIDS FOR POLYNOMIAL SVM HYPERPARAMETER STORAGE
        f1_crossval_grid_poly(find(box_constraint==c),find(poly_order==po))=mean(f1_score_poly);
        acc_crossval_grid_poly(find(box_constraint==c), find(poly_order==po)) = mean(accuracy_poly);
        training_error_grid_poly(find(box_constraint==c), find(poly_order==po)) = mean(training_loss_poly);
        validation_error_grid_poly(find(box_constraint==c), find(poly_order==po)) = mean(validation_loss_poly);
        time_taken_grid_poly(find(box_constraint==c), find(poly_order==po)) = mean(time_taken_poly);
        fp_crossval_grid_poly(find(box_constraint==c), find(poly_order==po)) = mean(fpratePoly);
        recall_crossval_grid_poly(find(box_constraint==c), find(poly_order==po)) = mean(recallPoly);
        auc_crossval_grid_poly(find(box_constraint==c), find(poly_order==po)) = mean(auc_poly_all);
    end
end

for c=box_constraint
    folds_linear=5;
    indx_linear = crossvalind('kfold', y_train, folds_linear);
    final_predictions_linear = [];
    final_scores_linear = [];
    parameters_linear = [parameters_linear; c]; 
    
    
    for i=1:folds_linear
        linearTic = tic;
        
        y_target_linear = [];
        
        sprintf('Linear Iteration c=%d | k-%d \n',...
                find(box_constraint==c), i)
            
        x2_fold_linear = X_train(indx_linear==i,:);
        x1_fold_linear = X_train(indx_linear~=i,:);
        y1_fold_linear = y_train(indx_linear~=i,:);
        y2_fold_linear = y_train(indx_linear==i,:);
        idx_test_linear=(indx_linear==i);

        rng(1)
        val_model_linear = fitcsvm(x1_fold_linear, y1_fold_linear, 'KernelFunction','linear',...
            'BoxConstraint', c);

        [predicted_labels_linear, scores_linear] = predict(val_model_linear, x2_fold_linear);
        final_predictions_linear = [final_predictions_linear; predicted_labels_linear];
        final_scores_linear = [final_scores_linear; scores_linear];

        y_target_linear = [y_target_linear; y_train(idx_test_linear)];
        y2_fold2_linear = num2cell(num2str(y2_fold_linear));

        val_model_cm_linear = confusionmat(y2_fold_linear, predicted_labels_linear);
        val_model_acc_linear = 100*sum(diag(val_model_cm_linear))./sum(val_model_cm_linear(:));
        disp(val_model_acc_linear)
        accuracy_linear(i) = val_model_acc_linear;

        [accuracy_linear, fscore_linear, fp_rate_linear, recall_linear] = performance(val_model_cm_linear);
        
        [x_fp_linear, y_tp_linear, t_cv_linear, auc_linear] = perfcurve(y_target_linear,scores_linear(:,2),'1');
        
        auc_linear_all(i) = auc_linear;

        if ~isnan(fscore_linear)
            f1_score_linear(i) = fscore_linear;
        end
        fprateLinear(i) = fp_rate_linear;
        recallLinear(i) = recall_linear;

        training_loss_linear(i) = mean(loss(val_model_linear, x1_fold_linear, y1_fold_linear));
        validation_loss_linear(i) = mean(loss(val_model_linear, x2_fold_linear, y2_fold_linear));

        time_taken_linear(i) = toc(linearTic)/60;
    end
    f1_crossval_grid_linear(find(box_constraint==c))=mean(f1_score_linear);
    acc_crossval_grid_linear(find(box_constraint==c)) = mean(accuracy_linear);
    training_error_grid_linear(find(box_constraint==c)) = mean(training_loss_linear);
    validation_error_grid_linear(find(box_constraint==c)) = mean(validation_loss_linear);
    time_taken_grid_linear(find(box_constraint==c)) = mean(time_taken_linear);
    fp_crossval_grid_linear(find(box_constraint==c)) = mean(fprateLinear);
    recall_crossval_grid_linear(find(box_constraint==c)) = mean(recallLinear);
    auc_crossval_grid_linear(find(box_constraint==c)) = mean(auc_linear_all);
end
svm_time_final = toc(svm_time)/60;

%% Making plots

[C2, gamma2] = ndgrid(box_constraint,kernel_scale);
[C2_poly, poly_order2] = ndgrid(box_constraint,poly_order);

%% Area Under the ROC Curve

% Plotting the crossvalidation AU ROC Hyperparameter Grid SVM RBF
figure
surf(C2,gamma2,auc_crossval_grid,'FaceColor', 'y', 'FaceAlpha', 0.5)
title('SVM Hyperparameter Optimisation (Area Under ROC Curve)')
xlabel('Box Constraint')
ylabel('Gamma')
zlabel('Score')
legend('AU ROC Curve')
hold off
%savefig('RBF_Accuracy_PLOT.fig');

% Plotting the crossvalidation AU ROC Hyperparameter Grid SVM Poly
figure
surf(C2_poly,poly_order2,auc_crossval_grid_poly,'FaceColor', 'y', 'FaceAlpha', 0.5)
title('SVM Polynomial Hyperparameter Optimisation (Area Under ROC Curve)')
xlabel('Box Constraint')
ylabel('Polynomial Order')
zlabel('Score')
legend('AU ROC Curve')
hold off
%savefig('Poly_Accuracy_PLOT.fig');

% Plotting the crossvalidation AU ROC Hyperparameter Tuning Linear SVM 
figure
p = plot(box_constraint, auc_crossval_grid_linear, 'LineWidth', 2);
title('SVM Linear - Area Under ROC Curve')
xlabel('Box Constraint')
ylabel('AU ROC Curve')
%savefig('Linear_Accuracy_PLOT.fig');




%% Training and Validation Loss

% RBF SVM Hyperparameter Optimisation with Training and Validation Error
figure
surf(C2,gamma2,training_error_grid','FaceColor','r','FaceAlpha',0.5);
hold on
surf(C2,gamma2,validation_error_grid,'FaceColor','b','FaceAlpha',0.5);
title('SVM RBF - Hyperparameter Graph (Training & Validation Accuracy)')
xlabel('Box Constraint')
ylabel('Gamma')
zlabel('Accuracy Score')
legend('Training Acurracy', 'Validation Accuracy')
hold off
%savefig('RBF_TV_ERR_PLOT.fig');

% Polynomial SVM Hyperparameter Optimisation with Training and Validation Error
figure
surf(C2_poly,poly_order2,training_error_grid_poly,'FaceColor','r','FaceAlpha',0.5);
hold on
surf(C2_poly,poly_order2,validation_error_grid_poly,'FaceColor','b','FaceAlpha',0.5);
title('SVM Polynomial - Training Optimisation')
xlabel('Box Constraint')
ylabel('Polynomial Order')
zlabel('Mean Error')
legend('Training Loss', 'Validation Loss')
hold off
%savefig('Poly_TV_ERR_PLOT.fig');

% Linear SVM Hyperparameter Optimisation with Training and Validation Error
figure
p = plot(box_constraint,training_error_grid_linear,box_constraint,validation_error_grid_linear);
title('SVM Linear - Training and Validation Error')
xlabel('Box Constraint')
ylabel('Error Score')
p(1).LineWidth = 2;
p(2).Marker = '*';
%savefig('Linear_TV_ERR_PLOT.fig');

%% Accuracy

% Plotting the crossvalidation Accuracy Hyperparameter Grid SVM RBF
figure
surf(C2,gamma2,acc_crossval_grid,'FaceColor', 'y', 'FaceAlpha', 0.5)
title('SVM RBF Hyperparameter Optimisation (Accuracy)')
xlabel('Box Constraint')
ylabel('Gamma')
zlabel('Accuracy Score')
hold off
%savefig('RBF_Accuracy_PLOT.fig');

% Plotting the crossvalidation Accuracy Hyperparameter Grid SVM Poly
figure
surf(C2_poly,poly_order2,acc_crossval_grid_poly,'FaceColor', 'y', 'FaceAlpha', 0.5)
title('SVM Polynomial Hyperparameter Optimisation (Accuracy)')
xlabel('Box Constraint')
ylabel('Polynomial Order')
zlabel('Score')
legend('Accuracy')
hold off
%savefig('Poly_Accuracy_PLOT.fig');

% Plotting the crossvalidation Accuracy Hyperparameter Tuning Linear SVM 
figure
p = plot(box_constraint, acc_crossval_grid_linear, 'LineWidth', 2);
title('SVM Linear - Accuracy')
xlabel('Box Constraint')
ylabel('Accuracy Score')
%savefig('Linear_Accuracy_PLOT.fig');


%% F1-score
% Plotting the crossvalidation F1-score Hyperparameter Grid RBF
figure
surf(C2,gamma2,f1_crossval_grid,'FaceColor','b','FaceAlpha',0.5)
title('SVM RBF Hyperparameter Optimisation (F1-Score)')
xlabel('Box Constraint')
ylabel('Gamma')
zlabel('Score')
legend('F1-score')
hold off
savefig('RBF_F1score_PLOT.fig');

% Plotting the crossvalidation F1-score Hyperparameter Grid Poly
figure
surf(C2_poly,poly_order2,f1_crossval_grid_poly,'FaceColor','b','FaceAlpha',0.5)
title('SVM Poly Hyperparameter Optimisation (F1-Score)')
xlabel('Box Constraint')
ylabel('Polynomial Order')
zlabel('Score')
legend('F1-score')
hold off
savefig('Poly_F1score_PLOT.fig');

% Plotting the crossvalidation F1-Score Hyperparameter Tuning Linear SVM 
figure
p = plot(box_constraint, f1_crossval_grid_linear, 'LineWidth', 2);
title('SVM Linear - F1-Score')
xlabel('Box Constraint')
ylabel('F1-Score')
savefig('Linear_F1score_PLOT.fig');

%% Recall
% Plotting the crossvalidation Recall Hyperparameter Grid RBF
figure
surf(C2,gamma2,recall_crossval_grid,'FaceColor','b','FaceAlpha',0.5)
title('SVM RBF Hyperparameter Optimisation (Recall)')
xlabel('Box Constraint')
ylabel('Gamma')
zlabel('Score')
legend('Recall')
hold off
savefig('RBF_Recall_PLOT.fig');

% Plotting the crossvalidation Recall Hyperparameter Grid Poly
figure
surf(C2_poly,poly_order2,recall_crossval_grid_poly,'FaceColor','b','FaceAlpha',0.5)
title('SVM Poly Hyperparameter Optimisation (Recall)')
xlabel('Box Constraint')
ylabel('Polynomial Order')
zlabel('Score')
legend('Recall')
hold off
savefig('Poly_Recall_PLOT.fig');


% Plotting the crossvalidation Recall Hyperparameter Tuning Linear SVM 
figure
p = plot(box_constraint, recall_crossval_grid_linear, 'LineWidth', 2);
title('SVM Linear - Recall')
xlabel('Box Constraint')
ylabel('Recall')
savefig('Linear_Recall_PLOT.fig');

%% FP_Rate

% Plotting the crossvalidation FP Rate Hyperparameter Grid RBF
figure
surf(C2,gamma2,fp_crossval_grid,'FaceColor','b','FaceAlpha',0.5)
title('SVM RBF Hyperparameter Optimisation (FP Rate)')
xlabel('Box Constraint')
ylabel('Gamma')
zlabel('Score')
legend('FP_Rate')
hold off
savefig('RBF_FPrate_PLOT.fig');

% Plotting the crossvalidation FPrate Hyperparameter Grid Poly
figure
surf(C2_poly,poly_order2,fp_crossval_grid_poly,'FaceColor','b','FaceAlpha',0.5)
title('SVM Poly Hyperparameter Optimisation (FP Rate)')
xlabel('Box Constraint')
ylabel('Polynomial Order')
zlabel('Score')
legend('FP Rate')
hold off
savefig('Poly_FPrate_PLOT.fig');

% Plotting the crossvalidation FP Rate Hyperparameter Tuning Linear SVM 
figure
p = plot(box_constraint, fp_crossval_grid_linear, 'LineWidth', 2);
title('SVM Linear - FP Rate')
xlabel('Box Constraint')
ylabel('FP Rate')
savefig('Linear_FPrate_PLOT.fig');


%% Time Complexity

% Time Complexity RBF SVM
figure
surf(C2,gamma2,time_taken_grid,'FaceColor','c','FaceAlpha',0.5)
title('SVM RBF - TIME')
xlabel('Box Constraint')
ylabel('Gamma')
zlabel('Time (mins)')
legend('Time (mins')
savefig('RBF_TimeComplexity_PLOT.fig');

% Time Complexity Poly SVM
figure
surf(C2_poly,poly_order2,time_taken_grid_poly,'FaceColor','c','FaceAlpha',0.5)
title('SVM - TIME')
xlabel('Box Constraint')
ylabel('Gamma')
zlabel('Time (mins)')
legend('Time (mins')
savefig('Poly_TimeComplexity_PLOT.fig');

% Time Complexity Linear SVM
figure
p = plot(box_constraint, time_taken_grid_linear, 'LineWidth', 2);
title('SVM Linear - F1-Score')
xlabel('Box Constraint')
ylabel('F1-Score')
savefig('Linear_TimeComplexity_PLOT.fig');
