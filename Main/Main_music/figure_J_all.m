%% limpiar datos
% clear; close all; clc
%% Direccion de la base de datos
SUBJECTS_DIR = 'G:\Brain-Rhythms-Multiplexing-master\Brain-Rhythms-Multiplexing-master\Data 1';
% SUBJECTS_DIR = 'E:\';
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
SS =1:21;% [37,15,7,1:6]; %6,14 [18:41]
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
experiment_name = 'prueba_musica_J_v3';

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
    for s = SS
        fs = 1000;
        twin = w_tao*fs; %------------ time segment
        ovlpt = round(poverlapp*twin);
        tseg = 1:twin-ovlpt:(tfin*fs)-twin;
        [F,T] = ndgrid(filter_bank(:,1),tseg);
        fprintf(['Sujeto...' num2str(s) '\n'])
        load([SUBJECTS_DIR filesep 'P_' num2str(s) filesep 'results' filesep ...
            experiment_name '_w' num2str((twin/fs)*1000) 'msec.mat'])
%         load([SUBJECTS_DIR filesep 'P_' num2str(s) filesep ...
            
        
        % figura de J
                figure
%                 subplot(4,5,s)
                imagesc(j_mean)
                axis xy
                colorbar()
                title(['Sujeto ' num2str(s) 'ventana ' w])
                set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
                    'XTickLabelRotation',90,'YTick',1:17,...
                    'YTickLabel',mean(filter_bank,2),'TickLabelInterpreter','latex')
                xlabel('Ventanas de Tiempo','Interpreter','latex')
                ylabel('Bandas de Frecuencia','Interpreter','latex')
                saveas(gca,[SUBJECTS_DIR  filesep 'P_' num2str(s) filesep 'Sujeto '...
                    num2str(s) '_jmean_' experiment_name '_w' num2str((twin/fs)*1000) ...
                    'msec'],'png')
        
        % figura contorno primeras bandas.
%                 figure
% %                 subplot(4,5,s)
%                 plot(sum(j_mean(1:6,:),1))
%                 hold on
%                 plot(sum(j_mean,1),'--r')
%                 title(['Sujeto ' num2str(s)])
%                 ylim([0 10])
%                 set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
%                     'XTickLabelRotation',90)
%                 xlabel('Ventanas de Tiempo','Interpreter','latex')
%                 legend('1-16 Hz','1-30 Hz')
%         %         saveas(gca,[SUBJECTS_DIR  filesep 'P_' num2str(s) filesep 'Sujeto '...
%         %             num2str(s) '_contorno_' experiment_name '_w' num2str((twin/fs)*1000) ...
%         %             'msec'],'png')
        
        macc = squeeze(mean(facc,3));    % mean acc
        mstd = squeeze(std(facc,[],3));  % std acc
        mmacc = zeros(2,numel(tseg));    % acc by each window
        indAcc = zeros(1,numel(tseg));
        for v = 1:numel(tseg)
            [mmacc(1,v),indAcc(v)] = max(macc(v,:));
            mmacc(2,v) = mstd(v,indAcc(v));
        end
        mks = squeeze(mean(fks,3));      % mean acc
        mksstd = squeeze(std(fks,[],3)); % std acc
        mmks = zeros(2,numel(tseg));     % acc by each window
        indKs = zeros(1,numel(tseg));
        for v = 1:numel(tseg)
            [mmks(1,v),indKs(v)] = max(mks(v,:));
            mmks(2,v) = mksstd(v,indKs(v));
        end
        % figura de acc
        figure
        imagesc(macc')
        axis xy
        colorbar()
        caxis([0 1])
        title(['Sujeto ' num2str(s)])
        set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
            'XTickLabelRotation',90,'YTick',1:5:length(param),...
            'YTickLabel',round(param(1:5:end),2),'TickLabelInterpreter','latex')
        xlabel('Ventanas de Tiempo','Interpreter','latex')
        ylabel('$\lambda$ lasso','Interpreter','latex')
                saveas(gca,[SUBJECTS_DIR  filesep 'P_' num2str(s) filesep 'Sujeto '...
                            num2str(s) '_acc_' experiment_name '_w' num2str((twin/fs)*1000) ...
                            'msec'],'png')
        %
        %         figure(10)
        %         subplot(5,5,s)
        %         plot(mmacc(1,:))
        %         title(['Sujeto ' num2str(s)])
        %         ylim([0 1])
        %         set(gca,'XTick',1:2:size(T,2),'XTickLabel',round(T(1,1:2:end)'/fs,1),...
        %             'XTickLabelRotation',90)
        %         dats(s) = mean(mmacc(1,:));
    end
    %     figure(11)
    %     stem(dats)
    %     xlim([0 22])
    %     ylim([0 1])
    %     figure(12)
    %     stem(dats_)
    %     xlim([0 22])
    %     ylim([0 1])
end
close all
