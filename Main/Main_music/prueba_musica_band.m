%% limpiar datos
clear; close all; clc
%% Direccion de la base de datos
SUBJECTS_DIR = 'F:\Brain-Rhythms-Multiplexing-master\Brain-Rhythms-Multiplexing-master\Data 1';
%% Direccion del fold de las funciones
addpath(genpath('D:\Dropbox\ERD\Codes\TP\Matlab_wang\csp\CSP_fun\functions'));

%% DataBase
% % BCIIII_4a_
% % BCICIV_2a_
% % GIGASCIENCE_
% % P_
% COHORT = 'P_';
% SUBJECTS = dir([SUBJECTS_DIR filesep '*' COHORT '*']);
% SUBJECTS = struct2cell(SUBJECTS);
% SUBJECTS = SUBJECTS(1,:)';

%%  Nombre del archivo para almacenar resultados
experiment_name = mfilename;

%% sujetos
SS =[1:1:17];% [37,15,7,1:6]; %6,14 [18:41]
% if strcmp(COHORT,'GIGASCIENCE_')
%     SubInd = [50,14];
%     SS(SubInd) = [];
% end

%% paramaters definition
seg_start = 1000;
seg_end = 8500;
% definir parametros de filter bank
f_low  = 4;
f_high = 40;
Window = 4;
Ovrlap = 2;
filter_bank = [f_low:Ovrlap:f_high-Window;...
    f_low+Window:Ovrlap:f_high]';
N_bands = size(filter_bank,1);
orden_filter = 5;
labels = [1 2];
load('cv_.mat')
%%
for s = SS
    %     clearvars -except s SS rho experiment_name COHORT param SUBJECTS SUBJECTS_DIR Acc table PPval Rep tstart tend
    %
    %     reporte = ['D:\Luisa\Dropbox\ERD\results_ERDfc_subjects\Codigo corriendo' SUBJECTS{s} '.txt'];
    %     diary('on')
    %     diary(reporte)
    fprintf(['Sujeto...' num2str(s)])
    path = [SUBJECTS_DIR filesep 'P_' num2str(s) filesep 'P' num2str(s) '_BCMI_frontHN_2017.mat'];
    [X,y,fs] = organizar(path);
    
    y = y(:);
    ind = ismember(y,labels);
    y = y(ind);
    X = X(ind);
    X = cellfun(@(x) double(x)/1000000 ,X,'UniformOutput',false);
    %     X = cellfun(@(x) downsample(x,2) ,X,'UniformOutput',false); fs = fs/2;
    tic
    Xa = cell(size(filter_bank,1),1);
    for b = 1:size(filter_bank,1)%Precompute all filters and trim
        Xa{b} = fcnfiltband(X, fs, filter_bank(b,:), 5);
        Xa{b} = cellfun(@(x) x(seg_start:seg_end,:),Xa{b},'UniformOutput',false);
    end
    %definitions
    acc=nan(5,1);
    ks=nan(5,1);
    Xcp = cell(5,1);
    %     sfeats = cell(5,numel(threshold));
    for fold = 1:5
        %         tic;
        tr_ind   = cv{s}.training(fold); tr_ind = tr_ind(ind);
        ts_ind   = cv{s}.test(fold); ts_ind = ts_ind(ind);
        Xc_ = cell(N_bands,1);
        for b = 1:N_bands
            C = cell2mat(reshape(cellfun(@(x)(cov(x)/trace(cov(x))),Xa{b},'UniformOutput',false),[1 1 numel(Xa{b})]));
            W = csp_feats(C(:,:,tr_ind),y(tr_ind),'train','Q',3);%floor(numel(chan)/2)
            Xc_{b} = csp_feats(C,W,'test');
        end
        Xc = cell2mat(Xc_);
        Xcp{fold} = Xc;
        clear C W
        % Lasso
        target = mapminmax(y(tr_ind)')';
        %             B = lasso(Xc(tr_ind,:),target,'Lambda',param);
        %             selected_feats = abs(B)>eps;
        %             sfeats{fold,u} = selected_feats;
        %
        %             for l=1:numel(param)
        %                 Xcc = Xc(:,selected_feats(:,l));
        %                 if size(Xcc,2)<2
        %                     continue
        %                 end
        mdl = fitcdiscr(Xc(tr_ind,:),y(tr_ind)); %LDA
        acc(fold) = mean(mdl.predict(Xc(ts_ind,:))==reshape(y(ts_ind),[sum(ts_ind) 1]));
        %Confusion Matrix
        tar_pred = mdl.predict(Xc(ts_ind,:)); %tar_pred(tar_pred==1)=0; tar_pred(tar_pred==2)=1;
        tar_true = reshape(y(ts_ind),[sum(ts_ind) 1]); %tar_true(tar_true==1)=0; tar_true(tar_true==2)=1;
        conM = confusionmat(tar_true,tar_pred);
        ks(fold) = kappa(conM);
        %plotconfusion(tar_true',tar_pred');
    end % folds
    act = squeeze(mean(acc,1));
    actstd = squeeze(std(acc));
    %     actstd = squeeze(std(acc,1)); actstd = actstd(:); actstd = actstd(indp);
    %     [u_opt,l_opt]=ind2sub(size(act),indp);
    %     table(1,:) = [threshold(u_opt),param(l_opt),dato*100,actstd*100];
    
    %% Guardar resultados
    %     save([SUBJECTS_DIR filesep SUBJECTS{s} filesep 'results\' experiment_name 'acc.mat'],'acc','table');
    %     save(['D:\BCI' ...
    %         filesep SUBJECTS{s} filesep 'results\' experiment_name 'Results_2ene.mat'],'acc','table','Xcp','sfeats','ks');
    %
    %     save([SUBJECTS_DIR2 ...
    %         filesep SUBJECTS{s} filesep experiment_name 'Results_2ene.mat'],'acc','table','ks');
    fprintf([' ...acc: ' num2str(act*100,'%02.1f') ' std: ' num2str(actstd*100,'%02.1f')...
        ' ...time: ' num2str(toc) '\n']);
    %     diary(reporte)
    %     diary('off')
    %     clear acc table Xcp sfeats ks
end


