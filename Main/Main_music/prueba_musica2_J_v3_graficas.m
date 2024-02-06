%% limpiar datos
clear; close all; clc
%% Direccion de la base de datos
SUBJECTS_DIR = 'Data 2';
%% Direccion del fold de las funciones
addpath(genpath('functions'));

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
SS =1:20;% [37,15,7,1:6]; %6,14 [18:41]
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
experiment_name = 'prueba_musica2_J_v3';
% experiment_name2= 'prueba_musica_J';
%% Filter bank
f_low  = 0; f_high = 30; %40
Window = 4; Ovrlap = 2;
filter_bank = [f_low:Ovrlap:f_high-Window;f_low+Window:Ovrlap:f_high]';
filter_bank(1,1) = 1;
% filter_bank = [[1,4];[4,8];[8,12];[12,30];[30,48]];
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
bad_tr_rest = [2,8,1,7,0,9,3,2,4,1,8,0,0,7,9,1,7,9,1,7,9,2,10,3]+12;
bad_tr_musi = [7,2,8,7,1,0,9,2,2,7,0,0,0,0,0,0,7,5,7,7,6];
bad_chan    = [19,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19];
%% Rayleight.
for w_tao = w
    for s = SS
        fprintf(['Sujeto...' num2str(s) '\n'])
        if s < 10
            path     = [SUBJECTS_DIR filesep 'music_listening_experiment_s0' num2str(s) '.mat'];
        else
            path     = [SUBJECTS_DIR filesep 'music_listening_experiment_s' num2str(s) '.mat'];
        end
        [X,y,fs] = organizar_2(path,s);
        %         y = y(:);
        %         ind  = ismember(y,labels);
        %         y    = y(ind); X = X(ind);
        %         X    = cellfun(@(x) double(x),X,'UniformOutput',false);
        %         bads = [bad_tr_musi(s),bad_tr_rest(s)];
        %         bads = bads(~bads==0);
        %         X(bads) = [];
        %         y(bads) = [];
        %         tic
        twin = w_tao*fs; %------------ time segment
        ovlpt = round(poverlapp*twin);
        tseg = 1:twin-ovlpt:(tfin*fs)-twin; [F,T] = ndgrid(filter_bank(:,1),tseg);
        %         load([SUBJECTS_DIR filesep 'P_' num2str(s) filesep experiment_name ...
        %             '_w' num2str((twin/fs)*1000) 'msec.mat'])
        %         %         jfold = zeros(size(filter_bank,1),numel(tseg),nfold);
        %         %         facc = zeros(numel(tseg),numel(param),nfold);
        %         %         fks = zeros(numel(tseg),numel(param),nfold);
        %         %         u_fold = cell(1,nfold);
        %         [F,T] = ndgrid(filter_bank(:,1),tseg);
        %         %         for fold = 1:nfold
        %         %             tr_ind   = cv{s}.training(fold); tr_ind = tr_ind(ind);
        %         %             tr_ind(bads) = [];
        %         %             ts_ind   = cv{s}.test(fold); ts_ind = ts_ind(ind);
        %         %             ts_ind(bads) = [];
        %         %             [F,T] = ndgrid(filter_bank(:,1),tseg);
        %         %             jtmp = zeros(size(F));
        %         %             u_v = cell(1,numel(tseg));
        %         %             acc = nan(numel(tseg),numel(param));
        %         %             ks = nan(numel(tseg),numel(param));
        %         %             Wall = cell(size(F));
        %         %             %% Rayleight computation
        %         %             y_ = y(tr_ind);
        %         %             F_ = numel(F);
        %         %             parfor ii = 1 :numel(F)
        %         %                 if F(ii) == filter_bank(1,1)
        %         %                     F2 = 4;
        %         %                 elseif F(ii) == filter_bank(2,1)
        %         %                     F2 = 8;
        %         %                 elseif F(ii) == filter_bank(3,1)
        %         %                     F2 = 12;
        %         %                 elseif F(ii) == filter_bank(4,1)
        %         %                     F2 = 30;
        %         %                 elseif F(ii) == filter_bank(5,1)
        %         %                     F2 = 48;
        %         %                 end
        %         %                 Xa = fncFilbank(X,[F(ii),F2],T(ii),T(ii)+twin-1,fs);
        %         %                 fprintf(['Rayleight. Sub:' num2str(s) ' - w: ' num2str(ii) ' of ' num2str(F_) '\n '])
        %         %                 C = cell2mat(reshape(cellfun(@(x)(cov(x)/...
        %         %                     trace(cov(x))),Xa{1},'UniformOutput',false),[1 1 numel(Xa{1})]));
        %         %                 W = csp_feats(C(:,:,tr_ind),y_,'train','Q',3);
        %         %                 Wall{ii} = W; W = W(1:6,:);
        %         %                 ind1 = y==1; ind1 = ind1 & tr_ind;
        %         %                 ind2 = y==2; ind2 = ind2 & tr_ind;
        %         %                 C1 = mean(C(:,:,ind1),3);
        %         %                 C2 = mean(C(:,:,ind2),3);
        %         %                 %--- Rayleight quotient
        %         %                 nc = (diag(W*C1*W')/trace(W*C1*W'));
        %         %                 dc = (diag(W*(C1+C2)*W')/trace(W*(C1+C2)*W'));
        %         %                 jtmp(ii) = (mean(nc(1:3)))/(mean(dc(1:3)));
        %         %             end
        %         %             jfold(:,:,fold) = jtmp;
        %         %             fprintf(['Fold ' num2str(fold) '... Sub:' num2str(s)  '\n '])
        %         %             clear jtmp C
        %         %             %% Accuracy by window
        %         %             t_seg = numel(tseg);
        %         %             parfor v = 1 :numel(tseg)
        %         %                 %% FilterBank
        %         %                 C = cell(1,size(filter_bank,1));
        %         %                 for b = 1:size(filter_bank,1)
        %         %                     temp = fcnfiltband(X, fs, filter_bank(b,:), 3);
        %         %                     temp = cellfun(@(x) cov(x(tseg(v):tseg(v)+twin-1,:)),temp,'UniformOutput',false);
        %         %                     temp = cellfun(@(x)(x/trace(x)),temp,'UniformOutput',false);
        %         %                     C{b} = cell2mat(reshape(temp,[1 1 numel(temp)]));
        %         %                 end
        %         %                 %% CSP - LASSO
        %         %                 Xc = cell(1,size(filter_bank,1));
        %         %                 for b=1:size(filter_bank,1)
        %         %                     W = csp_feats(C{b}(:,:,tr_ind),y_,'train','Q',3);
        %         %                     Xc{b} = csp_feats(C{b},W,'test');
        %         %                 end
        %         %                 Xf = cell2mat(Xc);
        %         %                 [acc(v,:),ks(v,:),u_v{v}] = fncLassoTunning(Xf,y,tr_ind,ts_ind,param);
        %         %                 fprintf(['Acc window ' num2str(v) '... of ' num2str(t_seg)  '\n '])
        %         %             end % window
        %         %             facc(:,:,fold) = acc;
        %         %             fks(:,:,fold) = ks;
        %         %             u_fold{fold} = u_v;
        %         %             Wfolds{fold} = Wall;
        %         %             fprintf(['Fold ' num2str(fold) '... of ' num2str(nfold)   '\n '])
        %         %         end % end fold
        %         toc
        %         %         fprintf(['Fold ' num2str(fold) '... Done \n '])
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
        %
        %         %% J computation
        %         j{s} = jfold;
        %         %Normalization
        %         jmean = mean(jfold,3);
        %         m = 1/(max(max(jmean))-min(min(jmean)));
        %         tmp = jmean.*m;
        %         j_mean = tmp+(1-(1/(max(max(jmean))-min(min(jmean))))*(max(max(jmean))));
        %         % figura de J
        %         figure
        %         imagesc(j_mean)
        %         axis xy
        %         colorbar()
        %         title(['Sujeto ' num2str(s)])
        %         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
        %             'XTickLabelRotation',90,'YTick',1:17,...
        %             'YTickLabel',mean(filter_bank,2),'TickLabelInterpreter','latex')
        %         xlabel('Ventanas de Tiempo','Interpreter','latex')
        %         ylabel('Bandas de Frecuencia','Interpreter','latex')
        %         %         saveas(gca,[SUBJECTS_DIR  filesep 'P_' num2str(s) filesep 'Sujeto '...
        %         %             num2str(s) '_jmean_' experiment_name '_w' num2str((twin/fs)*1000) ...
        %         %             'msec'],'png')
        %         %
        %         %         % figura contorno primeras bandas.
        %         %         figure
        %         %         plot(sum(j_mean(1:6,:),1))
        %         %         hold on
        %         %         plot(sum(j_mean,1),'--r')
        %         %         title(['Sujeto ' num2str(s)])
        %         %         ylim([0 10])
        %         %         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
        %         %             'XTickLabelRotation',90)
        %         %         xlabel('Ventanas de Tiempo','Interpreter','latex')
        %         %         saveas(gca,[SUBJECTS_DIR  filesep 'P_' num2str(s) filesep 'Sujeto '...
        %         %             num2str(s) '_contorno_' experiment_name '_w' num2str((twin/fs)*1000) ...
        %         %             'msec'],'png')
        %         %
        %         %         % figura de acc
        %         figure
        %         imagesc(macc')
        %         axis xy
        %         colorbar()
        %         title(['Sujeto ' num2str(s)])
        %         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
        %             'XTickLabelRotation',90,'YTick',1:5:length(param),...
        %             'YTickLabel',round(param(1:5:end),2),'TickLabelInterpreter','latex')
        %         xlabel('Ventanas de Tiempo','Interpreter','latex')
        %         ylabel('$\lambda$ lasso','Interpreter','latex')
        %         %         saveas(gca,[SUBJECTS_DIR  filesep 'P_' num2str(s) filesep 'Sujeto '...
        %         %             num2str(s) '_acc_' experiment_name '_w' num2str((twin/fs)*1000) ...
        %         %             'msec'],'png')
        %
        %         %Figura de J antes y después
        %         figure
        %         errorbar(mean(j_mean,1),std(j_mean,1))
        %         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
        %             'XTickLabelRotation',90)
        %         load([SUBJECTS_DIR filesep 'P_' num2str(s) filesep experiment_name2 ...
        %             '_w' num2str((twin/fs)*1000) 'msec.mat'])
        % %         figure
        % % hold on
        % %         errorbar(mean(j_mean,1),std(j_mean,1))
        %         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
        %             'XTickLabelRotation',90)
        %         %% Saving results
        %         %         save([SUBJECTS_DIR filesep 'P_' num2str(s) filesep ...
        %         %             experiment_name '_w' num2str((twin/fs)*1000) 'msec'],...
        %         %             'jfold','j_mean','mmacc','mmks','u_fold','Wfolds','facc','fks')
    end
end

%%
SS = [1,2,3,4,5,11,12,13,14,15];
facc_ = zeros([numel(SS),numel(tseg)]);
for s = SS
%     load([SUBJECTS_DIR filesep 's_' num2str(s)  ...
%         'prueba_musica2_J_v3_w1000msec.mat'])
    load(['C:\Users\Luisa F\Desktop\Frank\Main_musica\Data 2\'...'
    's_' num2str(s) 'prueba_musica2_J_v3_w1000msec.mat'])
    valores = mean(squeeze(facc),3)';
    valores(isnan(valores)) = 0;
    [val,pos_]= sort(valores,'descend'); val = mean(val(1:30,:),1);
    facc_(s,:) = val;
end
% save([pwd filesep 'Main2_music2.mat'],'facc_')
figure(100)
plot(facc_')
set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1), 'XTickLabelRotation',90)
legend('s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14',...
    's15','s16','s17','s18','s19','s20','s21')