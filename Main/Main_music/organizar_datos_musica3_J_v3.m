%% limpiar datos
clear; close all; clc
%% Direccion de la base de datos
SUBJECTS_DIR = 'I:\Brain-Rhythms-Multiplexing-master\Brain-Rhythms-Multiplexing-master\Data 3';
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
SS =1:31;% [37,15,7,1:6]; %6,14 [18:41]
% if strcmp(COHORT,'GIGASCIENCE_')
%     SubInd = [50,14];
%     SS(SubInd) = [];
% end

%% paramaters definition
tstart = 0;
tend = 9.5;
load('cv_music3.mat')

%% Paramaters definition
% Lasso parameters
param = linspace(0,0.9,100);
experiment_name = mfilename;

%% Filter bank
f_low  = 0; f_high = 48; %40
Window = 4; Ovrlap = 2;
filter_bank = [f_low:Ovrlap:f_high-Window;f_low+Window:Ovrlap:f_high]';
filter_bank(1,1) = 1;
% filter_bank = [[1,4];[4,8];[8,12];[12,30];[30,48]];
orden_filter = 5;
labels= [1 2];
% definitions
nfold = 5;
Xa    = cell(size(filter_bank,1),1);
Wfolds= cell(1, nfold);
j     = cell(1,9);
poverlapp = 0.9;
tfin  = 10;
w     = 1; % size of windows in Rayleight.
% bad_tr_rest = [2,8,1,7,0,9,3,2,4,1,8,0,0,7,9,1,7,9,1,7,9,2,10,3]+12;
% bad_tr_musi = [7,2,8,7,1,0,9,2,2,7,0,0,0,0,0,0,7,5,7,7,6];
% bad_chan    = [19,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,19];

[X_,fs] = organizar_all(SUBJECTS_DIR,SS);
    

