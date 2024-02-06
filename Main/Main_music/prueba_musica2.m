%%
clear all; close all; clc
% 1

% % Direccion de la base de datos
% SUBJECTS_DIR = 'D:\BCI';
SUBJECTS_DIR = 'D:\BCI';
SUBJECTS_DIR2 = 'D:\Luisa\Dropbox\ERD\results_ERDfc_subjects\BCI';
% % Direccion del fold de las funciones
% addpath(genpath('C:\Users\lfvelasquezm\Dropbox\ERD\Codes\TP\Matlab_wang\csp\CSP_fun\functions'))
% addpath(genpath('C:\Users\lfvelasquezm\Desktop\frank\functions'))
% addpath(genpath('C:\Users\frany\Dropbox\Event-related\Codes\TP\Matlab_wang\csp\CSP_fun\functions'));
addpath(genpath('D:\Luisa\Dropbox\ERD\Codes\TP\Matlab_wang\csp\CSP_fun\functions'))
%


% %% DataBase
% % BCIIII_4a_
% % BCICIV_2a_
%  GIGASCIENCE_
%
COHORT = 'BCICIV_2a_';
SUBJECTS = dir([SUBJECTS_DIR filesep '*' COHORT '*']);
SUBJECTS = struct2cell(SUBJECTS);
SUBJECTS = SUBJECTS(1,:)';
%
%
% %% grilla de busqueda
param = linspace(0,0.9,100);

experiment_name = mfilename;

% SS = [37 32 12 18 42 34 3 7 35 33 21 2 4 39 29 43 28]; % UNO BUENO Y UNO MALO%%%%INDEXACDOS DE ACIERDO A CSP
% SS = [37 32 12 18 42 34 3 7];
SS = [1:9]; %6,14 [18:41]
% if strcmp(COHORT,'GIGASCIENCE_')
%     SubInd = [50,14];
%     SS(SubInd) = [];
% end

%% paramaters definition
Rep = 10000;  % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% numero de repeticiones
PPval = cell(numel(SS),1);
rho = [];
tstart = 0;
tend = 2;
% definir parametros de filter bank
f_low  = 4;
f_high = 40;
Window = 4;
Ovrlap = 2;
filter_bank = [f_low:Ovrlap:f_high-Window;...
    f_low+Window:Ovrlap:f_high]';
orden_filter = 5;
labels = [1 2];

%%

for s = SS
    %     clearvars -except s SS rho experiment_name COHORT param SUBJECTS SUBJECTS_DIR Acc table PPval Rep tstart tend
    %
    reporte = ['D:\Luisa\Dropbox\ERD\results_ERDfc_subjects\Codigo corriendo' SUBJECTS{s} '.txt'];
    diary('on')
    diary(reporte)
    load([SUBJECTS_DIR filesep SUBJECTS{s} filesep 'eeg' filesep 'raw.mat'])
    %     seg_start = seg_star;
    y = y(:);
    ind = ismember(y,labels);
    y = y(ind);
    X = X(ind);
    X = cellfun(@(x) double(x) ,X,'UniformOutput',false);
    %     X = cellfun(@(x) downsample(x,2) ,X,'UniformOutput',false); fs = fs/2;
    tic
    %     %% Power bands estimation
    %     Xf = cell(1,size(filter_bank,1));
    %     for b = 1:size(filter_bank,1)
    %         Xf{b} = fcnfiltband(X, fs, filter_bank(b,:), 5);
    %     end
    %     P = nan(size(filter_bank,1),size(X{1},1),size(X{1},2),size(X,1));
    %     for n=1:numel(X) % trials
    %         for b = 1:size(filter_bank,1)
    %             Xtemp = Xf{b}{n};
    %             for ch =1 :size(X{1},2)
    %                 xtemp = Xtemp(:,ch);
    %                 [upper,~] = envelope(xtemp,10,'peak');
    %                [P(b,:,ch,n)] = upper.^2;
    % %                 [P(b,:,ch,n)] = xtemp.^2;
    %             end
    %         end
    %     end
    %     fprintf(['Potencia: Sujeto ' num2str(s) '... Done.\n'])
    %     %Normalización de escala de cada canal y cada trial
    %     Z = sum(sum(P,1),2);
    %     P = bsxfun(@times,P,1./Z);
    %     t = (1:size(X{1},1))/fs;
    %     fprintf(['P-valor: Sujeto ' num2str(s) '\n'])
    %
    %     PP = cell(1,size(P,3));
    %     for ch=1:size(P,3); PP{ch} = squeeze(P(:,:,ch,:)); end
    %     diary(reporte)
    %     tt = cell(1,size(P,3));
    %     tt = cellfun(@(x) t,tt,'UniformOutput', false);
    %     %     yy = cell(1,size(P,3)); yy = cellfun(@(x) y,yy,'UniformOutput', false);
    %     ttstart = cell(1,size(P,3)); ttstart = cellfun(@(x) tstart,ttstart,'UniformOutput', false);
    %     ttend = cell(1,size(P,3)); ttend = cellfun(@(x) tend,ttend,'UniformOutput', false);
    %     c1 = cell(1,size(P,3)); c1 = cellfun(@(x) 1,c1,'UniformOutput', false);
    %     c2 = cell(1,size(P,3)); c2 = cellfun(@(x) 2,c2,'UniformOutput', false);
    %     Pval = nan(size(P,1),size(P,3),Rep); %F x Ch
    %     ERD1 = cell(1,size(P,3)); ERD2 = cell(1,size(P,3));
    %     %     ERD1_ = cell(1,size(P,3)); ERD2_ = cell(1,size(P,3));
    %     tic
    % %     clear P Z Xf
    %     %     tic;
    %     %     tmp = cell(Rep,1);
    %     indy = cell(1,Rep);
    %     for r = 1 :Rep
    % %                 tic;
    %         Cv = cvpartition(y,'HoldOut',0.3);
    %         indr(:,1) = ismember(y,1) & Cv.training; indr(:,2)= ismember(y,2) & Cv.training;
    %         tmpr  = cell(1,numel(PP));
    %         indy{r} = cellfun(@(x) indr,tmpr,'UniformOutput', false);
    % %                 toc
    %     end
    %     for r = 1:100%Rep
    %         tic;
    %         Cv = cvpartition(y,'HoldOut',0.3);
    %         indr(:,1) = ismember(y,1) & Cv.training; indr(:,2)= ismember(y,2) & Cv.training;
    %         indy  = cell(1,numel(PP)); indy = cellfun(@(x) indr,indy,'UniformOutput', false);
    %         ERD = cellfun(@compute_ERD2,PP,indy,tt,ttstart,ttend,'UniformOutput', false);
    %         %         ERD2 = cellfun(@compute_ERD2,PP,indy,c2,tt,ttstart,ttend,'UniformOutput', false);
    %         ERD1 = cellfun(@(x) x(1:17,t>=seg_start/fs & t<=seg_end/fs,:), ERD, 'UniformOutput', false);
    %         ERD2 = cellfun(@(x) x(18:end,t>=seg_start/fs & t<=seg_end/fs,:), ERD, 'UniformOutput', false);
    %         %mean
    %         mtemp = cellfun(@(x) mean(x,2), ERD1, 'UniformOutput', false);
    %         mtemp = cellfun(@(x) x', mtemp, 'UniformOutput', false);
    %         mERD1(r,:) = mtemp;
    %         %var
    %         vtemp = cellfun(@(x) var(x'), ERD1, 'UniformOutput', false);
    %         vERD1(r,:) = vtemp;
    %         %mean
    %         mtemp = cellfun(@(x) mean(x,2), ERD2, 'UniformOutput', false);
    %         mtemp = cellfun(@(x) x', mtemp, 'UniformOutput', false);
    %         mERD2(r,:) = mtemp;
    %         %var
    %         vtemp = cellfun(@(x) var(x'), ERD2, 'UniformOutput', false);
    %         vERD2(r,:) = vtemp;
    % %         tmp = cellfun(@compute_pairedtest, ERD1, ERD2,'UniformOutput', false);
    % %         Pval(:,:,r) = cell2mat(tmp);
    % %                 r
    %         toc
    %     end % 25 min
    %
    %     for c =1 :22
    %         for f=1:17
    %             m1 = cell2mat(squeeze(mERD1(:,c)));
    %             m2 = cell2mat(squeeze(mERD2(:,c)));
    %             v1 = cell2mat(squeeze(vERD1(:,c)));
    %             v2 = cell2mat(squeeze(vERD2(:,c)));
    %             [d(f,c),p(f,c)] = manova1([[m1(:,f);m2(:,f)],[v1(:,f);v2(:,f)]],[zeros(1,100),ones(1,100)])
    % %             [h(f,c),p(f,c)]=ttest2(m1(:,f),m2(:,f));
    %         end
    %     end
    %
    %     toc
    %     clear tmp PP yy c1 c2 tt ttstart ttend indy indr
    %     fprintf(['Repet done. ' num2str(s) '\n'])
    %     fprintf(['Rho: Sujeto ' num2str(s) '\n'])
    %     diary(reporte)
    %     tmp = reshape(Pval,[size(Pval,1)*size(Pval,2) size(Pval,3)]);
    %     rho = nan(1,size(tmp,1));
    %     for i = 1:size(tmp,1)
    %         try
    %             [FDR, Q, rho(1,i)] = mafdr(tmp(i,:)');
    %             %             [FDR, rho_s(1,i)] = mafdr(tmp(i,:)');
    %         catch
    %             rho(1,i) = nan;
    %         end
    %     end
    %     % Guardar resultados
    %     fprintf(['Rho estimation ' ' ...time: ' num2str(toc) '\n'])
    %     save([SUBJECTS_DIR filesep SUBJECTS{s} filesep 'results\' experiment_name 'rho_vLF.mat'],'Pval','rho');
    %% Cargar mascara Ex_Giga_vLFVM_mod_vNovrho_vLF
    load([SUBJECTS_DIR filesep SUBJECTS{s} filesep 'results\' 'ERDCSP_rho.mat']); %ERDCSP_rho
    % %         load([SUBJECTS_DIR filesep SUBJECTS{s} filesep 'results\' experiment_name 'rho.mat'],'rho_s');
    %     load(['C:\Users\lfvelasquezm\Dropbox\ERD\results_ERDfc_subjects\BCI' filesep SUBJECTS{s} filesep 'results\Ex_Giga_vLFVM' 'rho.mat'],'rho_s')
    threshold = zeros(1,100);
    threshold(1:end-1) = linspace(min(rho),max(rho),99);
    threshold(end)=1;
    if sum(isnan(rho))>1; rho(isnan(rho)==1)=1;  end %para los nan detectados
    
    Xa = cell(size(filter_bank,1),1);
    diary(reporte)
    for b = 1:size(filter_bank,1)%Precompute all filters and trim
        Xa{b} = fcnfiltband(X, fs, filter_bank(b,:), 5);
        Xa{b} = cellfun(@(x) x(seg_start:seg_end,:),Xa{b},'UniformOutput',false);
    end
    %definitions
    acc=nan(5,numel(threshold),numel(param));
    ks=nan(5,numel(threshold),numel(param));
    Xcp = cell(5,numel(threshold));
    sfeats = cell(5,numel(threshold));
    for fold = 1:5
        %         tic;
        tr_ind   = cv.training(fold); tr_ind = tr_ind(ind);
        ts_ind   = cv.test(fold); ts_ind = ts_ind(ind);
        for u = 1: numel(threshold)
            %             tic
            mask = reshape(rho,[size(filter_bank,1) size(X{1},2)]) <= threshold(u);
            %             mask = reshape(p,[33,22])<= threshold(u);
            SelBand = sum(mask,2);
            valQ = floor(SelBand/2);
            selected = SelBand>=6;%2
            band = find(selected==1);
            if sum(selected) == 0
                continue
            end
            Xc = cell(1,numel(band));
            for b = 1:numel(band)
                [~,chan]=find(mask(band(b),:)>0);
                if numel(chan) < 6%1
                    continue
                end
                C = cell2mat(reshape(cellfun(@(x)(cov(x(:,chan))/trace(cov(x(:,chan)))),Xa{band(b)},'UniformOutput',false),[1 1 numel(Xa{b})]));
                W = csp_feats(C(:,:,tr_ind),y(tr_ind),'train','Q',3);%floor(numel(chan)/2)
                Xc{b} = csp_feats(C,W,'test');
            end
            Xc = cell2mat(Xc);
            Xcp{fold,u} = Xc;
            clear C W
            % Lasso
            target = mapminmax(y(tr_ind)')';
            B = lasso(Xc(tr_ind,:),target,'Lambda',param);
            selected_feats = abs(B)>eps;
            sfeats{fold,u} = selected_feats;
            %
            for l=1:numel(param)
                Xcc = Xc(:,selected_feats(:,l));
                if size(Xcc,2)<2
                    continue
                end
                mdl = fitcdiscr(Xcc(tr_ind,:),y(tr_ind)); %LDA
                acc(fold,u,l) = mean(mdl.predict(Xcc(ts_ind,:))==reshape(y(ts_ind),[sum(ts_ind) 1]));
                %Confusion Matrix
                tar_pred = mdl.predict(Xcc(ts_ind,:)); %tar_pred(tar_pred==1)=0; tar_pred(tar_pred==2)=1;
                tar_true = reshape(y(ts_ind),[sum(ts_ind) 1]); %tar_true(tar_true==1)=0; tar_true(tar_true==2)=1;
                conM = confusionmat(tar_true,tar_pred);
                ks(fold,u,l) = kappa(conM);
                %plotconfusion(tar_true',tar_pred');
            end %lambda
            fprintf(['Threshold...' num2str(u) '...' num2str(toc) '\n'])
            %             [fold,u]
        end % signficance
        fprintf(['Sujeto: ' SUBJECTS{s} ' Fold...' num2str(fold) '\n'])
        %         toc
        diary(reporte)
    end % folds
    act = squeeze(mean(acc,1));
    [dato,indp] = max(act(:));
    actstd = squeeze(std(acc,1)); actstd = actstd(:); actstd = actstd(indp);
    [u_opt,l_opt]=ind2sub(size(act),indp);
    table(1,:) = [threshold(u_opt),param(l_opt),dato*100,actstd*100];
    
    toc
    %% Guardar resultados
    %     save([SUBJECTS_DIR filesep SUBJECTS{s} filesep 'results\' experiment_name 'acc.mat'],'acc','table');
    %     save(['D:\BCI' ...
    %         filesep SUBJECTS{s} filesep 'results\' experiment_name 'Results_vLF.mat'],'acc','table','Xcp','sfeats','ks');
    %
    %     save([SUBJECTS_DIR2 ...
    %         filesep SUBJECTS{s} filesep experiment_name 'Results_vLF.mat'],'acc','table','ks');
    %     fprintf([' ...acc: ' num2str(dato*100,'%02.1f') ' std: ' num2str(actstd*100,'%02.1f')...
    %         ' ...time: ' num2str(toc) '\n']);
    %     diary(reporte)
    %     diary('off')
    %     clear acc table Xcp sfeats ks
end