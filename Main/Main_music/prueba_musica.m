%% limpiar datos
clear; close all; clc
%% Direccion de la base de datos
SUBJECTS_DIR = 'G:\Brain-Rhythms-Multiplexing-master\Brain-Rhythms-Multiplexing-master\Data 1';
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
SS =1:21;% [37,15,7,1:6]; %6,14 [18:41]
% if strcmp(COHORT,'GIGASCIENCE_')
%     SubInd = [50,14];
%     SS(SubInd) = [];
% end

%% paramaters definition
tstart = 0;
tend = 9.5;
%% grilla de busqueda
param = linspace(0,0.9,100);
% definir parametros de filter bank
f_low  = 4;
f_high = 40;
Window = 4;
Ovrlap = 2;
filter_bank = [f_low:Ovrlap:f_high-Window;...
    f_low+Window:Ovrlap:f_high]';
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
    [X,y,fs] = organizar2(path,s);
    
    y = y(:);
    ind = ismember(y,labels);
    y = y(ind);
    X = X(ind);
    X = cellfun(@(x) double(x)/1000000 ,X,'UniformOutput',false);
    %     X = cellfun(@(x) downsample(x,2) ,X,'UniformOutput',false); fs = fs/2;
    tic
    
    %definitions
    acc =nan(5,numel(param));
    ks  = nan(5,numel(param));
    Xcp = cell(5,numel(param));
    %     sfeats = cell(5,numel(threshold));
    sfeats= cell(5,1);
    for fold = 1:5
        %         tic;
        tr_ind   = cv{s}.training(fold); tr_ind = tr_ind(ind);
        ts_ind   = cv{s}.test(fold); ts_ind = ts_ind(ind);
        %         for u = 1:numel(threshold)
        %             tic
        %             mask = reshape(rho,[size(filter_bank,1) size(X{1},2)]) <= threshold(u);
        %             mask = reshape(p,[33,22])<= threshold(u);
        %             SelBand = sum(mask,2);
        %             valQ = floor(SelBand/2);
        %             selected = SelBand>=6;%6 seleccuionde canales en esa frecuencia
        %             band = find(selected==1);
        %             if sum(selected) == 0
        %                 continue
        %             end
        %         Xc = cell(1,:);
        %             for b = 1:numel(band)
        %                 [~,chan]=find(mask(band(b),:)>0);
        %                 if numel(chan) < 6%1
        %                     continue
        %                 end
        C = cell2mat(reshape(cellfun(@(x)(cov(x)/trace(cov(x))),X,'UniformOutput',false),[1 1 numel(X)]));
        W = csp_feats(C(:,:,tr_ind),y(tr_ind),'train','Q',3);%floor(numel(chan)/2)
        Xc = csp_feats(C,W,'test');
        %             end
        %         Xc = cell2mat(Xc);
        Xcp{fold} = Xc;
        clear C W
        % Lasso
        target = mapminmax(y(tr_ind)')';
        B = lasso(real(Xc(tr_ind,:)),target,'Lambda',param);
        selected_feats = abs(B)>eps;
        sfeats{fold} = selected_feats;
        %
        for l=1:numel(param)
            Xcc = Xc(:,selected_feats(:,l));
            if size(Xcc,2)<2
                continue
            end
            mdl = fitcdiscr(Xcc(tr_ind,:),y(tr_ind)); %LDA
            acc(fold,l) = mean(mdl.predict(Xcc(ts_ind,:))==reshape(y(ts_ind),[sum(ts_ind) 1]));
            %Confusion Matrix
            tar_pred = mdl.predict(Xcc(ts_ind,:)); %tar_pred(tar_pred==1)=0; tar_pred(tar_pred==2)=1;
            tar_true = reshape(y(ts_ind),[sum(ts_ind) 1]); %tar_true(tar_true==1)=0; tar_true(tar_true==2)=1;
            conM = confusionmat(tar_true,tar_pred);
            ks(fold,l) = kappa(conM);
            %plotconfusion(tar_true',tar_pred');
        end %lambda
        %         fprintf(['Threshold...' num2str(u) '...' num2str(toc) '\n'])
        %             [fold,u]
        % end % signficance
        %         fprintf(['Sujeto: ' SUBJECTS{s} ' Fold...' num2str(fold) '\n'])
        %         toc
        %         diary(reporte)
    end % folds
    
    act_ = squeeze(mean(acc,1));
    [act,pos_] = max(act_);
    actstd_ = squeeze(std(acc,1));
    actstd  = actstd_(pos_);
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
    aacc(s,:) = [act*100,actstd*100]; 
    %     diary(reporte)
    %     diary('off')
    %     clear acc table Xcp sfeats ks
end


