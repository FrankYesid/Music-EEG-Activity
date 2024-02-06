function [pvt_sub,sel,M1,t_e,database,tex,flag,cur,ticks,GS] = fncload_relevances(data,method)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load compouted relevances 
% [pvt_sub,sel,M1,t_e,database,type,tex,flag,cur,ticks,GS] = ...
%     fncload_relevances(data,method)
%% Input:
% data          ... database selection DI or DII.
% Method        ... estimator. (J)   Rayleight, 
%                              (ERD) Event-related Desynchronization. 
%                              (Cx)  Connectivity
%% Output:
% pvt_sub       ... reevence of the method.
% sel           ... Channels selection.
% M1            ... Structure of channels.
% database      ... database select.
% tex           ... ability to plot labels of channels 1,0.
% cur           ... boolean enables contour lines 1,0.
% ticks         ... ability to plot Boolean labels 1,0.
% Gs            ... Resolution of griddata in topoplot.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2019 Signal Processing and Recognition Group.
% Universidad Nacional de Colombia.
% L.F. Velasquez-Martinez
% F.Y. Zapata-Castaño.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cur = []; ticks =[]; GS = []; Jfilt_ = {};
X1 = {};  X2 = {};   elec_pos =[];
dir = 'demo'; 
if strcmp(data,'DI')
    %% Load relevances 
    if strcmp(method,'J')
        load([dir,method,filesep,data,filesep,'Rel_based_J_1sec.mat','pvt_sub']); % BCI dataset2a Rayleight-based relevances
    elseif strcmp(method,'ERD')
        load([dir,method,filesep,data,filesep,'X_erds_class.mat'])  % BCI dataset2a ERD-based relevances
        pvt_sub   = {X1 X2};
    elseif strcmp(method,'Cx')
        %% Load data
        Cx_1 = {};
        for i = 1:9
            load([dir,method,filesep,data,filesep,'Cx_1_sub' num2str(i) '.mat'])             % BCI dataset2a Cx-based relevances
            for c =1:2
                tmp = Cx_1{1}{c};
                for ch = 1:size(tmp,2) 
                    pvt_sub{c}{i}(:,ch,:) = tmp{ch};
                end
            end
        end
    end
    
    %% Parameters for topography 
    load(['data',filesep,data,filesep,'electrodesBCICIV2a.mat'])% Electrode positions
    database = 2;
    tex = 0;
    t_e = 10;                                                   % Electrodes size.
    M1.xy = elec_pos(:,1:2);                                    % positions channels.
    M1.lab = Channels;                                          % names of channels.
    sel = 1:22; 
    
elseif strcmp(data,'DII')
    %pvt_sub = cell(1,52);
    %% Load relevances
    name_dir = 'demo\';
    %% Load relevances
    if strcmp(method,'J')
        addpath(genpath([name_dir method]));
        listing = dir([name_dir method]);
        for i = 3:length(listing)
            subj = str2double(listing(i).name(12:13));
            load([listing(i).name],'Jfilt_'); 
            for ch = 1:size(Jfilt_{1}{1},2)
                tmp(ch,:,:) = Jfilt_{1}{1}{ch}';
            end
            pvt_sub{subj} = tmp;
        end
    elseif strcmp(method,'ERD')
        %% DII ERD-based relevances
        dir1 = 'demo\';
        load([dir1 method filesep 'X_erds_class_giga_b.mat'])
        pvt_sub = {X1 X2}; clear X1 X2         
        load([dir1 '\X_erds_class_giga_m1.mat'])
        for sub = 1:52
            if isempty(X1{sub}); else pvt_sub{1}{sub} = X1{sub}; end 
            if isempty(X2{sub}); else pvt_sub{2}{sub} = X2{sub}; end
            if isempty(X1{sub}) && isempty(X2{sub})
            end
        end
        clear X1 X2        
        load([dir1 'X_erds_class_giga_m2.mat'])
        for sub = 1:length(X1)
            if isempty(X1{sub}); else pvt_sub{1}{sub} = X1{sub}; end 
            if isempty(X2{sub}); else pvt_sub{2}{sub} = X2{sub}; end
            if isempty(X1{sub}) && isempty(X2{sub})
            end
        end
    elseif strcmp(method,'Cx')
        name_dir = 'demo\';
        addpath(genpath([name_dir method]));
        listing = 1:52; listing(29) = 0; listing(34) = 0;
        for i = 1:numel(listing)
            subj = listing(i);
            if subj ~=0
                load(['Cx_1_sub' num2str(subj)]); 
                tmp1 = Cx_1{1}{1}; tmp2 = Cx_1{1}{2};
                for ch = 1:numel(tmp1)
                    pvt_sub{1}{subj}(:,ch,:) = tmp1{ch};
                    pvt_sub{2}{subj}(:,ch,:) = tmp2{ch};
                end
            end
        end
    end    
    %% Parameters for topography 64 channels
    load('pos_struct_giga.mat');
    database = [];
    sel    = 1:64;
    t_e    = 20;
    cur    = 0;
    ticks  = 0;
    GS    = 500;
    tex = [];  
end


