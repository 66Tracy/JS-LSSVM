function [temp_accu,temp_t] = JSLSSVM_2c(train_data, train_label, params, test_data, test_label)
% JS-LSSVM model for binary classification tasks
%   
%    [accuracy, running time] = JSLSSVM_2c(train_data, train_label, params, test_data, test_label)
%
%    Input:
%        train_data: the training data matrix. Each column is a sample vector.
%        train_label: the actual label of training data.
%        test_data: the testing data matrix. Each column is a sample vector.
%        test_label: the actual label of testing data.
%        params: struct contains related parameters, including:
%                   dim (dimension): the projection dimensionality;
%                   theta : parameter in our model ;
%                   zeta : parameter in our model ;
%                   gam (gamma) : parameter of LS-SVM;
%                   sig2: parameter of LS-SVM.
%
%    Output:
%        accuracy : recognition rate.
%        time : running time.
%


% train, train_label, params, test, test_label
[features,~]=size(train_data);

tic;
% initialization:
P = rand(features, params.dim);
% D = eye(features);
D = eye(features);
for i = 1:features
    D(i, i) = 1 / (norm(P(i, :)) + params.zeta);
end

obj = inf;
if features < params.dim
    error('features less than dimension!');
end

type='c';
% get first LS-SVM model.
p_train = P' * train_data;
model=trainlssvm({p_train',train_label',type,params.gam,params.sig2,'lin_kernel','original'});

% Tsum corresponds to the (Z*alpha) in paper
tmpv = train_data * (model.alpha .* model.ytrain);
TSum = tmpv * tmpv';

for lp = 1:10
    
    for times = 1:25
        % A equals to (-Z*alpha*Z'*alpha'+theta*D)
        A = -TSum + params.theta * D;
        % for computing stability
        A = (A + A') / 2;
        [V, Lambda] = eig(A);
        [seq, index] = sort(diag(Lambda), 'ascend');
        
        P = V(:, index(1:params.dim));
        P = (abs(P) > params.zeta) .* P;
        
        % compute D
        for i = 1:features
            D(i, i) = 1 / (norm(P(i, :)) + params.zeta);
        end
        
        obj_old = obj;
        obj = sum(seq(1:params.dim)) - norm(model.alpha .* model.ytrain, 1);
        
        if abs(obj - obj_old) <= 1e-2
            break;
        end
        
    end
    
    p_train = P' * train_data;
    model=trainlssvm({p_train',train_label',type,params.gam,params.sig2,'lin_kernel','original'});
    
    % update TSum.
    tmpv = train_data * (model.alpha .* model.ytrain);
    TSum = tmpv * tmpv';
    
    A = -TSum + params.beta * D;
    A = (A + A') / 2;
    obj_old = obj;
    obj = trace(P' * A * P) - norm(model.alpha .* model.ytrain, 1);
    
    if abs(obj - obj_old)/abs(obj_old) <= params.conv_factor
        break;
    end 
end

p_train = P' * train_data;
p_test = P' * test_data;
model=trainlssvm({p_train',train_label',type,params.gam,params.sig2,'lin_kernel','original'});
Predict_Y=simlssvm({p_train',train_label',type,params.gam,params.sig2,'lin_kernel','original'},{model.alpha,model.b},p_test');
temp_accu=100*sum(Predict_Y==test_label')/length(test_label);
temp_t=toc;



