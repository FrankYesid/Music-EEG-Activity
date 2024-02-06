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

%% sujetos
SS =2:21;% [37,15,7,1:6]; %6,14 [18:41]
% if strcmp(COHORT,'GIGASCIENCE_')
%     SubInd = [50,14];
%     SS(SubInd) = [];
% end

%% paramaters definition
tstart = 0;
tend = 9.5;

load('cv_.mat')

%% Paramaters definition
% Lasso parameters
param = linspace(0,0.9,100);
experiment_name = mfilename;

%% Filter bank
f_low  = 0; f_high = 30; %40
Window = 4; Ovrlap = 2;
filter_bank = [f_low:Ovrlap:f_high-Window;f_low+Window:Ovrlap:f_high]';
filter_bank(1,1) = 1;
filter_bank = filter_bank(5,:);
orden_filter = 5;
labels = [1 2];
% definitions
nfold = 5;
Xa    = cell(size(filter_bank,1),1);
Wfolds= cell(1, nfold);
j     = cell(1,9);
poverlapp = 0.9;
tfin  = 9.5;
w     = 1; % size of windows in Rayleight.

%% Rayleight.
for w_tao = w
    for s = SS
        fprintf(['Sujeto...' num2str(s) '\n'])
        path = [SUBJECTS_DIR filesep 'P_' num2str(s) filesep 'P' num2str(s) '_BCMI_frontHN_2017.mat'];
        [X,y,fs] = organizar(path,s);
        y = y(:);
        ind = ismember(y,labels);
        y = y(ind); X = X(ind);
        X = cellfun(@(x) double(x),X,'UniformOutput',false);
        tic
        twin = w_tao*fs; %------------ time segment
        ovlpt = round(poverlapp*twin);
        tseg = 1:twin-ovlpt:(tfin*fs)-twin;
        jfold = zeros(size(filter_bank,1),numel(tseg),nfold);
        facc = zeros(numel(tseg),numel(param),size(filter_bank,1),nfold);
        fks  = zeros(numel(tseg),numel(param),size(filter_bank,1),nfold);
        u_fold = cell(1,nfold);
        for fold = 1:nfold
            tr_ind   = cv{s}.training(fold); tr_ind = tr_ind(ind);
            ts_ind   = cv{s}.test(fold); ts_ind = ts_ind(ind);
            [F,T] = ndgrid(filter_bank(:,1),tseg);
            jtmp = zeros(size(F));
            u_v = cell(1,numel(tseg));
            acc = nan(numel(tseg),numel(param),size(filter_bank,1));
            ks = nan(numel(tseg),numel(param),size(filter_bank,1));
            Wall = cell(size(F));
            %% Rayleight computation
            y_ = y(tr_ind);
            F_ = numel(F);
            parfor ii = 1 :numel(F)
                Xa = fncFilbank(X,[F(ii),F(ii)+4],T(ii),T(ii)+twin-1,fs);
                fprintf(['Rayleight. Sub:' num2str(s) ' - w: ' num2str(ii) ' of ' num2str(F_) '\n '])
                C = cell2mat(reshape(cellfun(@(x)(cov(x)/...
                    trace(cov(x))),Xa{1},'UniformOutput',false),[1 1 numel(Xa{1})]));
                
                W = csp_feats(C(:,:,tr_ind),y_,'train','Q',3);
                Wall{ii} = W; W = W(1:6,:);
                ind1 = y==1; ind1 = ind1 & tr_ind;
                ind2 = y==2; ind2 = ind2 & tr_ind;
                C1 = mean(C(:,:,ind1),3);
                C2 = mean(C(:,:,ind2),3);
                %--- Rayleight quotient
                nc = (diag(W*C1*W')/trace(W*C1*W'));
                dc = (diag(W*(C1+C2)*W')/trace(W*(C1+C2)*W'));
                jtmp(ii) = (mean(nc(1:3)))/(mean(dc(1:3)));
            end
            jfold(:,:,fold) = jtmp;
            fprintf(['Fold ' num2str(fold) '... Sub:' num2str(s)  '\n '])
            clear jtmp C
            %% Accuracy by window
            t_seg = numel(tseg);
            parfor v = 1 :numel(tseg)
                %% FilterBank
                acc_ = zeros(numel(param),size(filter_bank,1));
                ks_ = zeros(numel(param),size(filter_bank,1));
                u_v_ = cell(size(filter_bank,1));
                C = cell(1,size(filter_bank,1));
                for b = 1:size(filter_bank,1)
                    temp = fcnfiltband(X, fs, filter_bank(b,:), 3);
                    temp = cellfun(@(x) cov(x(tseg(v):tseg(v)+twin-1,:)),temp,'UniformOutput',false);
                    temp = cellfun(@(x)(x/trace(x)),temp,'UniformOutput',false);
                    C{b} = cell2mat(reshape(temp,[1 1 numel(temp)]));
%                 end
                %% CSP - LASSO
                %                 Xc = cell(1,size(filter_bank,1));
%                 parfor b=1:size(filter_bank,1)
                    W = csp_feats(C{b}(:,:,tr_ind),y_,'train','Q',3);
                    Xc = csp_feats(C{b},W,'test');
                    %                 end
                    %                 for b=1:size(filter_bank,1)
                    Xf = Xc;
                    [acc_(:,b),ks_(:,b),u_v_{b}] = fncLassoTunning(Xf,y,tr_ind,ts_ind,param);
                    fprintf(['Acc window ' num2str(v) '... of ' num2str(t_seg)  '\n '])
                end
                acc(v,:,:) = acc_;
                ks(v,:,:) = ks_;
                u_v{v} = u_v_;
            end % window
            facc(:,:,:,fold) = acc;
            fks(:,:,:,fold) = ks;
            u_fold{fold} = u_v;
            Wfolds{fold} = Wall;
            fprintf(['Fold ' num2str(fold) '... of ' num2str(nfold)   '\n '])
        end % end fold
        toc
        fprintf(['Fold ' num2str(fold) '... Done \n '])
%         macc = squeeze(mean(facc,3));    % mean acc
%         mstd = squeeze(std(facc,[],3));  % std acc
%         mmacc = zeros(2,numel(tseg));    % acc by each window
%         indAcc = zeros(1,numel(tseg));
%         for v = 1:numel(tseg)
%             [mmacc(1,v),indAcc(v)] = max(macc(v,:));
%             mmacc(2,v) = mstd(v,indAcc(v));
%         end
%         mks = squeeze(mean(fks,3));      % mean acc
%         mksstd = squeeze(std(fks,[],3)); % std acc
%         mmks = zeros(2,numel(tseg));     % acc by each window
%         indKs = zeros(1,numel(tseg));
%         for v = 1:numel(tseg)
%             [mmks(1,v),indKs(v)] = max(mks(v,:));
%             mmks(2,v) = mksstd(v,indKs(v));
%         end
        
        %% J computation
        j{s} = jfold;
        %Normalization
        jmean = mean(jfold,3);
        m = 1/(max(max(jmean))-min(min(jmean)));
        tmp = jmean.*m;
        j_mean = tmp+(1-(1/(max(max(jmean))-min(min(jmean))))*(max(max(jmean))));
        %         % figura de J
%         figure(1)
%         imagesc(j_mean)
%         axis xy
%         colorbar()
%         title(['Sujeto ' num2str(s)])
%         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
%             'XTickLabelRotation',90,'YTick',1:17,...
%             'YTickLabel',mean(filter_bank,2),'TickLabelInterpreter','latex')
%         xlabel('Ventanas de Tiempo','Interpreter','latex')
%         ylabel('Bandas de Frecuencia','Interpreter','latex')
        %         saveas(gca,[SUBJECTS_DIR  filesep 'P_' num2str(s) filesep 'Sujeto '...
        %             num2str(s) '_jmean_' experiment_name '_w' num2str((twin/fs)*1000) ...
        %             'msec'],'png')
%         figure(2)
%         plot(j_mean)
%         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
%             'XTickLabelRotation',90)
        %         % figura contorno primeras bandas.
        %         figure(2)
        %         plot(sum(j_mean(1:6,:),1))
        %         hold on
        %         plot(sum(j_mean,1),'--r')
        %         title(['Sujeto ' num2str(s)])
        %         ylim([0 10])
        %         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
        %             'XTickLabelRotation',90)
        %         xlabel('Ventanas de Tiempo','Interpreter','latex')
        %         saveas(gca,[SUBJECTS_DIR  filesep 'P_' num2str(s) filesep 'Sujeto '...
        %             num2str(s) '_contorno_' experiment_name '_w' num2str((twin/fs)*1000) ...
        %             'msec'],'png')
        %
        %         % figura de acc
%         figure(3)
%         imagesc(mean(squeeze(facc),3)')
%         %         imagesc(macc')
%         axis xy
%         colorbar()
%         title(['Sujeto ' num2str(s)])
%         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
%             'XTickLabelRotation',90,'YTick',1:5:length(param),...
%             'YTickLabel',round(param(1:5:end),2),'TickLabelInterpreter','latex')
%         xlabel('Ventanas de Tiempo','Interpreter','latex')
%         ylabel('$\lambda$ lasso','Interpreter','latex')
        %         saveas(gca,[SUBJECTS_DIR  filesep 'P_' num2str(s) filesep 'Sujeto '...
        %             num2str(s) '_acc_' experiment_name '_w' num2str((twin/fs)*1000) ...
        %             'msec'],'png')
%         figure(4)
%         plot(max(mean(squeeze(facc),3)')); hold on
%         valores = mean(squeeze(facc),3)';
%         valores(isnan(valores)) = 0;
%         [val,pos_]= sort(valores,'descend'); val = mean(val(1:30,:),1);
%         plot(val)
%         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
%             'XTickLabelRotation',90)
%         legend('Max-Acc','Mean-Acc')
%         close all
        %% Saving results
        save([SUBJECTS_DIR filesep 'P_' num2str(s) filesep 'results' filesep ...
            experiment_name '_w' num2str((twin/fs)*1000) 'msec'],...
            'jfold','j_mean','u_fold','Wfolds','facc','fks')
    end
end

%% load data
SS = 1:21;
facc_ = zeros([numel(SS),numel(tseg)]);
for s = SS
    load([SUBJECTS_DIR filesep 'P_' num2str(s) filesep 'results' filesep ...
        experiment_name '_w' num2str((twin/fs)*1000) 'msec'])
    valores = mean(squeeze(facc),3)';
    valores(isnan(valores)) = 0;
    [val,pos_]= sort(valores,'descend'); val = mean(val(1:30,:),1);
    facc_(s,:) = val;
end
save([pwd filename 'Main_music.mat'],'facc_')
figure(100)
plot(facc_')
set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1), 'XTickLabelRotation',90)
legend('s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14',...
    's15','s16','s17','s18','s19','s20','s21')