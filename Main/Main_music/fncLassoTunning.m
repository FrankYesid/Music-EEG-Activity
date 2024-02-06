function [acc,ks,u] = fncLassoTunning(Xf,y,tr_ind,ts_ind,param) 
%% Lasso regression
% [acc,ks,u] = fncLassoTunning(Xf,y,tr_ind,ts_ind,param) 
% acc - Accuracy.
% Ks  - Confusion Matrix.
% u   - Lasso regression.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2019 Signal Processing and Recognition Group.
% Universidad Nacional de Colombia.
% L.F. Velasquez-Martinez
% F.Y. Zapata-Castaño.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Regression target
target = mapminmax(y(tr_ind)')';
% Lasso regression
u = lasso(Xf(tr_ind,:),target,'Lambda',param);
acc = nan(1,numel(param));
ks = nan(1,numel(param));
for l=1:numel(param)
    % Feature selection
    selected = abs(u(:,l))>eps;
    if sum(selected)==0
        continue
    end
    mdl = fitcdiscr(Xf(tr_ind,selected),y(tr_ind));
    acc(1,l) = mean(mdl.predict(Xf(ts_ind,selected))==y(ts_ind(:)));
    %Kappa computation
    tar_pred = mdl.predict(Xf(ts_ind,selected));
    tar_true = reshape(y(ts_ind(:)),[sum(ts_ind(:)) 1]);
    %Confusion Matrix
    conM = confusionmat(tar_true,tar_pred);
    ks(1,l) = kappa(conM);
end  % end lasso
                            