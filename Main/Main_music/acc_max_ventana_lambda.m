%% limpiar datos
clear; close all; clc
%% Direccion de la base de datos
% SUBJECTS_DIR = 'G:\Brain-Rhythms-Multiplexing-master\Brain-Rhythms-Multiplexing-master\Data 1';
SUBJECTS_DIR = 'E:\';
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
SS = 1:21;% [37,15,7,1:6]; %6*--,14 [18:41]
% if strcmp(COHORT,'GIGASCIENCE_')
%     SubInd = [50,14];
%     SS(SubInd) = [];
% end

%% paramaters definition
tstart = 0;
tend   = 9.5;

load('cv_.mat')

%% Paramaters definition
% Lasso parameters
param = linspace(0,0.9,100);
experiment_name = 'prueba_musica_J';

%% Filter bank
f_low  = 0; f_high = 30; %40
Window = 4; Ovrlap = 2;
filter_bank = [f_low:Ovrlap:f_high-Window;f_low+Window:Ovrlap:f_high]';
filter_bank(1,1) = 1;
orden_filter = 5;
labels = [1 2];
% definitions
nfold = 5;
Xa    = cell(size(filter_bank,1),1);
Wfolds= cell(1, nfold);
j     = cell(1,9);
poverlapp = 0.9;
tfin  = 9.5;
w     = [1]; % size of windows in Rayleight.

%% Rayleight.
for w_tao = w
    figure
    for s = SS
        fs = 1000;
        twin = w_tao*fs; %------------ time segment
        ovlpt = round(poverlapp*twin);
        tseg = 1:twin-ovlpt:(tfin*fs)-twin;
        [F,T] = ndgrid(filter_bank(:,1),tseg);
        fprintf(['Sujeto...' num2str(s) '\n'])
%                 load([SUBJECTS_DIR filesep 'P_' num2str(s) filesep 'results' filesep ...
        load([SUBJECTS_DIR filesep 'P_' num2str(s) filesep ...
            experiment_name '_w' num2str((twin/fs)*1000) 'msec.mat'])
        macc = squeeze(mean(facc,3));    % mean acc
        
        macc_ = macc;
        val = zeros(size(macc_));
        for ven = 1:size(macc_,1)
            for l = 1:size(macc_,2)
                if ven~=1 && l~=1 && ven~=size(macc_,1) && l~=size(macc_,2)
                    val(ven,l)= macc_(ven,l)+macc_(ven+1,l)+macc_(ven-1,l)+macc_(ven,l+1)+macc_(ven,l-1)+macc_(ven+1,l+1)+macc_(ven+1,l-1)+macc_(ven-1,l+1)+macc_(ven-1,l-1);
                elseif ven == 1 && l == 1
                    val(ven,l)= macc_(ven,l)+macc_(ven+1,l)+macc_(ven,l+1)+macc_(ven+1,l+1);
                elseif ven==size(macc_,1) && l == 1
                    val(ven,l)= macc_(ven,l)+macc_(ven-1,l)+macc_(ven,l+1)+macc_(ven-1,l+1);
                elseif ven== 1  && l > 1 && l < size(macc_,2)
                    val(ven,l)= macc_(ven,l)+macc_(ven+1,l)+macc_(ven,l+1)+macc_(ven,l-1)+macc_(ven+1,+l)+macc_(ven+1,l-1);
                elseif ven > 1 && ven < size(macc_,1) && l == 1
                    val(ven,l)= macc_(ven,l)+macc_(ven+1,l)+macc_(ven-1,l)+macc_(ven,l+1)+macc_(ven+1,l+1)+macc_(ven-1,l+1);
                elseif ven == size(macc_,1) && l > 1 && l < size(macc_,2)
                    val(ven,l)= macc_(ven,l)+macc_(ven,l+1)+macc_(ven,l-1)+macc_(ven-1,l)+macc_(ven-1,l-1)+macc_(ven-1,l+1);
                end
            end
        end
        %         figure;imagesc(val'); axis xy
        %         colorbar()
        %
        %         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
        %             'XTickLabelRotation',90,'YTick',1:5:length(param),...
        %             'YTickLabel',round(param(1:5:end),2),'TickLabelInterpreter','latex')
        %         xlabel('Ventanas de Tiempo','Interpreter','latex')
        %         ylabel('$\lambda$ lasso','Interpreter','latex')
        [valor_maximo,pos] = max(val(:));
        [u_opt,l_opt]=ind2sub(size(val),pos);
        %         hold on
        %         scatter(u_opt,l_opt,'r','o','filled')
        
        [val_s,pos_] = sort(val','descend');
        val_ = macc_(pos_);
        val_ord = val_(~isnan(val_s(:,u_opt)),u_opt);
        
        Acc_m(s) = nanmean([macc_(u_opt,l_opt),macc_(u_opt+1,l_opt),macc_(u_opt-1,l_opt),...
            macc_(u_opt+1,l_opt+1),macc_(u_opt-1,l_opt-1),macc_(u_opt,l_opt+1),...
            macc_(u_opt,l_opt-1),macc_(u_opt-1,l_opt+1),macc_(u_opt-1,l_opt-1),val_ord(1:30)']);
        %         title(['Sujeto ' num2str(s) ' Acc medio ' num2str(Acc_m(s))])
        %         val_ord_all = cell(2,1);
        %         for c = 1:3
        %             if c == 1
        %                 for u = 1:size(val_s,2)
        %                     temo = mean(val_(~isnan(val_s(:,u)),u));
        %                     val_ord_all{c}(u,1) = temo;
        %                 end
        %             elseif c == 2
        for u = 1:size(val_s,2)
            temo = val_(~isnan(val_s(:,u)),u);
            val_ord_all(s,u,:) = temo(1:30);
        end
        %             else
        %                 for u = 1:size(val_s,2)
        %                     temo = val_(~isnan(val_s(:,u)),u);
        %                     val_ord_all{c}(u,1) = mean(temo(1:75),1);
        %                 end
        %             end
        %     end
        %         figure
        %         plot(val_ord_all{1},'b')
        %         hold on
        %         plot(val_ord_all{2})
        %         plot(val_ord_all{3},'--g')
        %         legend('todos','30 mejores','75 mejores')
        %         ylim([0.5 1])
        %         plot(mean(j_mean(:,:)))
        %         title(['Sujeto ' num2str(s)])
        
        j_mean_s(s,:,:) = j_mean;
    end
    %     leg = num2cell(SS);
    %     for a = 1:numel(leg)
    %         leg_{a} =num2str(leg{a}(1));
    %     end
    %     legend(leg_)
end
save('resultado_j_1seg.mat','val_ord_all','j_mean_s','Acc_m')