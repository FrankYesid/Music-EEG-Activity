clear,close all
% load single_subject.mat % load a single-subject data (average-rereferenced data)
for sub = 7:20%:21
%     load(['Data 1\P',num2str(sub),'_BCMI_frontHN_2017.mat'])
    load(['Data 2\music_listening_experiment_s01.mat'])
    rest_eeg = double(EEG_Rest');
    music_eeg= double(permute(EEG_Songs,[3,2,1]));
    clear re_eeg_
    % a music-listening and a resting-state dataset and
    % along with the sensors' names and coordinates
    % sensor_coordinates
    name_channels = sensor_info.labels;
%     name_channels = [{'Fp1'},{'Fp2'},{'F7'},{'F3'},{'Fz'},{'F4'},{'F8'},{'T3'},{'C3'}...
%         ,{'Cz'},{'C4'},{'T4'},{'T5'},{'P3'},{'Pz'},{'P4'},{'T6'},{'O1'},{'O2'}];
    FBANDS = [1 4; 4 8; 8 12; 12 20; 20 30; 30 45]; % Define 7 Frequency bands
    Fs = 128; % Sampling Frequency
    
    %%%STEP-1
    %%%%%%%% Building Connectivity-Patterns; for each Brain Rhythm (frequency band) independently %%%%%%%%
    %%_______   Calculate and plot PSD
    % Calculate PSD
    % Trial_PSD=[]; for i_song=1:size(subject_trials,1) % loop over songs
    %     singleSong_EEG = squeeze(subject_trials(i_song,:,:)); %single trial
    %     [Px,Faxis] = pwelch(singleSong_EEG',128,100,256,Fs,'onesided'); %Welch method
    %     Px = Px'; Px = Px(:,3:91); Faxis = Faxis(3:91); Song_PSD =Px;  % Isolate frequency content in the [1-45]Hz range
    %     Trial_PSD(i_song,:,:) = Song_PSD; end
    % %Calculate the average PSD profile for the high and low rating trials
    % high_rating_mean_PSD = squeeze(mean(Trial_PSD(subject_ratings==5,:,:),1));
    % low_rating_mean_PSD = squeeze(mean(Trial_PSD(subject_ratings==1,:,:),1));
    
    %% % Rest
    %%_______   Preprocesing : rereferencing
    %Common Reference
    Nchannels = 14;
    for i_song=1 % loop over songs
        re_eeg_(1,:,:)=rest_eeg';
        singleSong_EEG = squeeze(re_eeg_(i_song,:,:)); %single trial
        singleSong_EEG = singleSong_EEG-repmat(mean(singleSong_EEG),Nchannels,1); % common-rereferencing
        re_eeg_ = singleSong_EEG;
    end
    rest_eeg = re_eeg_';
    
    % Calculate PSD
    REST_BAND_CPs=[];  % [(29x28/2)=406 pairs x 7 Bands]
    for i_band=1:size(FBANDS,1)
        [b,a]=butter(3,[FBANDS(i_band,1),FBANDS(i_band,2)]/(Fs/2));
        EEG=rest_eeg';
        filtered_EEG=filtfilt(b,a,EEG')';
        [Px,Faxis]   = pwelch(filtered_EEG',[],[],[1:100],Fs); %Welch method,128,100,256,
        Px = Px'; Px = Px(:,1:95); Faxis = Faxis(1:95); PSD_resting =Px; % Isolate frequency content in the [1-95]Hz range
        PLV_CP=[];[PLV_CP,IDX]  = PLV_multichannel_signal(filtered_EEG); %1 column vector (vectorized WA-matrix)
        REST_BAND_CPs(:,i_band) = PLV_CP;
    end
    
    %% Music listening
    %%_______   Preprocesing : rereferencing
    %Common Reference
    for i_song=1:size(music_eeg,3) % loop over songs
        music_eeg_=permute(music_eeg,[3,2,1]);
        singleSong_EEG = squeeze(music_eeg_(i_song,:,:)); %single trial
        singleSong_EEG = singleSong_EEG-repmat(mean(singleSong_EEG),Nchannels,1); % common-rereferencing
        music_eeg_(i_song,:,:) = singleSong_EEG;
    end
    music_eeg = permute(music_eeg_,[3,2,1]);
    
    for trial = 1:size(music_eeg,3)
        MUSIC_BAND_CPs=[];  % [(29x28/2)=406 pairs x 7 Bands]
        for i_band=1:size(FBANDS,1)
            [b,a]=butter(3,[FBANDS(i_band,1),FBANDS(i_band,2)]/(Fs/2));
            EEG=squeeze(music_eeg(:,:,trial))';
            filtered_EEG=filtfilt(b,a,EEG')';
            [Px,Faxis]   = pwelch(filtered_EEG',[],[],[1:100],Fs); %Welch method,128,100,256,
            Px = Px'; Px = Px(:,1:95); Faxis = Faxis(1:95); Trial_PSD(trial,:,:) =Px; % Isolate frequency content in the [1-95]Hz range
            PLV_CP = [];[PLV_CP,IDX]=PLV_multichannel_signal(filtered_EEG); %1 column vector (vectorized WA-matrix)
            MUSIC_BAND_CPs(:,i_band)=PLV_CP;
        end
    end
    
    PSD_music = squeeze(mean(Trial_PSD,1));
    
    for i_band=1:size(FBANDS,1)
        [~,F1] = min(abs(Faxis-FBANDS(i_band,1)));
        [~,F2] = min(abs(Faxis-FBANDS(i_band,2)));
        aa = zeros(size(PSD_music));
        Freqs{i_band} = [F1,F2];
        for Psd = F1:F2
            aa(:,Psd)= (PSD_music(:,Psd)-PSD_resting(:,Psd))./PSD_resting(:,Psd);
        end
        PSD_ryt{i_band} = aa;
    end
    
    FRR_names=['delta '; 'theta ';'alpha ';'betaL ';'betaH ';'gammaL';'gammaH'];
    fig = figure(1);
    for i_band=1:size(FBANDS,1)
        subplot(1,7,i_band)
        imagesc(Faxis([Freqs{i_band}(1):Freqs{i_band}(2)]),1:Nchannels,PSD_ryt{i_band}(:,[Freqs{i_band}(1):Freqs{i_band}(2)])) %#ok<NBRAK>
        if i_band == 1
            ylabel('Channel','Interpreter','latex')
        end
        xlabel('Frequency (f)','Interpreter','latex')
        title(FRR_names(i_band,:))
        set(gca,'YTick',1:Nchannels,'YTickLabel',name_channels,'TickLabelInterpreter','latex')
    end
    axis tight;
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]); %EXPANDING FIGURE ON SCREEN
    hold off;
    saveas(fig,['D:\Dropbox\Paper BrainScore_BCMI\figures\2sub',num2str(sub),'_figure1'],'png')
    close
    
    %% realizar test
    FRR_names_ = [{'$\delta$'},{'$\theta$'},{'$\alpha$'},{'$\beta_{L}$'},...
        {'$\beta_{H}$'},{'$\gamma_{L}$'},{'$\gamma_{H}$'}];
    ALPHA = 0.05;
    for i_band=1:size(FBANDS,1)
        freq = Freqs{i_band};
        a = 1;
        for Psd = freq(1):freq(2)
            X(:,a) = PSD_music(:,Psd);
            Y(:,a) = PSD_resting(:,Psd);
            a = a+1;
        end
        for ch = 1:size(X,1)
            [P(i_band,ch),H(i_band,ch)] = ranksum(X(ch,:),Y(ch,:),'alpha',ALPHA);
        end
    end
        figure(2)
        imagesc(P)
        title('P-value Music-Rest')
        set(gca,'XTick',1:19,'XTickLabel',name_channels,'YTick',1:7,...
            'YTickLabel',FRR_names_,'TickLabelInterpreter','latex')
        axis tight;
        set(gcf, 'units','normalized','outerposition',[0 0 1 1]); %EXPANDING FIGURE ON SCREEN
        hold off;
        saveas(gca,['D:\Dropbox\Paper BrainScore_BCMI\figures\2sub',num2str(sub),'_figure2'],'png')
        close
    %% realizar prueba de corrección
    fdr = zeros(size(P));
    for ch = 1:size(X,1)
        fdr(:,ch) = mafdr(P(:,ch),'Lambda',ALPHA);
    end
        figure(3)
        imagesc(fdr)
        title('FDR Music-Rest')
        set(gca,'XTick',1:19,'XTickLabel',name_channels,'YTick',1:7,...
            'YTickLabel',FRR_names_,'TickLabelInterpreter','latex')
        axis tight;
        set(gcf, 'units','normalized','outerposition',[0 0 1 1]); %EXPANDING FIGURE ON SCREEN
        hold off;
        saveas(gca,['D:\Dropbox\Paper BrainScore_BCMI\figures\2sub',num2str(sub),'_figure3'],'png')
        close
    
    %% %STEP-2
    %%%%%%%% NetWork analysis - Defining Modular-organization based on Dominant Sets Algorithm %%%%%%%%
    
    % The Dominant Sets algorithm ('iterative_dominant_set_extraction.m') follows an iterative,
    % subtractive-clustering approach that includes a random initialization.
    % Thus, the overall result may change from exacution to execution.
    % The ''official'' implementation is realized via 'recurrent_dominant_sets.m',
    % where after a number of repetition the ''best'' graph-partitioning result
    % is finally outputted.
    
    % Important Notice: For this demonstration, we will use a rng(seed) approach to minimize the execution time.
    % For actual cases, edit the 'recurrent_dominant_sets.m' file by replacing the rng(seed) with the rng('shuffle').
    % Then use a large ITERATION number below (e.g. 500) to guarantee optimal performance.
    
    ITERATIONS=1;
    MUSIC_DSgroups=[]; MUSIC_DSscores=[]; REST_DSgroups=[]; REST_DSscores=[];
    MUSIC_DSscores=zeros(7,7);REST_DSscores=zeros(7,7);
    
    tic
    for i_band=1:size(FBANDS,1)
        CP=REST_BAND_CPs(:,i_band);
        WA=[];for i1=1:size(IDX,1); WA(IDX(i1,1),IDX(i1,2))=CP(i1);WA(IDX(i1,2),IDX(i1,1))=CP(i1); end
        clear sorted_score sorted_cost_function;
        [sorted_groups,sorted_cost_function]=recurrent_dominant_sets(WA,ITERATIONS);
        REST_DSgroups(i_band,:)=sorted_groups;REST_DSscores(i_band,1:numel(sorted_cost_function))=sorted_cost_function;
    end
    
    for i_band=1:size(FBANDS,1)
        CP=MUSIC_BAND_CPs(:,i_band);
        WA=[];
        for i1=1:size(IDX,1)
            WA(IDX(i1,1),IDX(i1,2))=CP(i1);WA(IDX(i1,2),IDX(i1,1))=CP(i1); 
        end
        clear sorted_score sorted_cost_function
        [sorted_groups,sorted_cost_function]=recurrent_dominant_sets(WA,ITERATIONS);
        MUSIC_DSgroups(i_band,:)=sorted_groups;MUSIC_DSscores(i_band,1:numel(sorted_cost_function))=sorted_cost_function;
    end
    toc
    lim1=0; lim2=max(max([REST_DSscores MUSIC_DSscores]));
    
    %% Modular organization as a function of brain rhythm and recording condition.
    % Nodes belonging to the same functional community share color and size. Size indicates the level of within-group cohesiveness.
    % Presentation-format is similar to Figure 4 of the paper.
    % (i.e. upper line: Rest, lower line: Music-listening;  the brain-rhythms are presented in order of frequency,from left-to-right  ).
        figure(4)
        nch = Nchannels; %take care of this variable (nch must be according to matrix size you want to plot it)
        p = 0.03;   %proportion of weigthed links to keep for.
        for i = 1:size(MUSIC_BAND_CPs,2)
            subplot(2,7,i)
    
            cx_r{i} = squareform(REST_BAND_CPs(:,i));
            imagesc(cx_r{i})
            title(FRR_names(i,:))
            set(gca,'XTick',1:19,'XTickLabel',name_channels,'YTick',1:19,...
                'YTickLabel',name_channels,'TickLabelInterpreter','latex',...
                'XTickLabelRotation',90)
            axis square
            if i == 1
                ylabel('Rest')
            end
    
            subplot(2,7,7+i)
            cx{i} = squareform(MUSIC_BAND_CPs(:,i));
            imagesc(cx{i})
            title(FRR_names(i,:))
            set(gca,'XTick',1:19,'XTickLabel',name_channels,'YTick',1:19,...
                'YTickLabel',name_channels,'TickLabelInterpreter','latex',...
                'XTickLabelRotation',90)
            axis square
            if i == 1
                ylabel('Music')
            end
        end
        axis tight;
        set(gcf, 'units','normalized','outerposition',[0 0 1 1]); %EXPANDING FIGURE ON SCREEN
        hold off;
        saveas(gca,['D:\Dropbox\Paper BrainScore_BCMI\figures\2sub',num2str(sub),'_figure4'],'png')
        close
    
    %% topoplot
        figure(5)
        p = 0.05;
        for i = 1:size(MUSIC_BAND_CPs,2)
            subplot(4,7,i)
            aij = threshold_proportional(cx_r{i}, p); %thresholding networks due to proportion p
            ijw = adj2edgeL(triu(aij));             %passing from matrix form to edge list form
            f_PlotEEG_BrainNetwork(nch, ijw, 'w_wn2wx');
            title(FRR_names(i,:))
            if i == 1
                ylabel('Rest')
            end
    
            subplot(4,7,7+i)
    %         aij = threshold_proportional(, p); %thresholding networks due to proportion p
            ijw = adj2edgeL(triu(cx_r{i}));             %passing from matrix form to edge list form
            f_PlotEEG_BrainNetwork(nch, ijw, 'w_wn2wx',MUSIC_DSgroups(i,:));
            title(FRR_names(i,:))
    
            subplot(4,7,14+i)
            aij = threshold_proportional(cx{i}, p); %thresholding networks due to proportion p
            ijw = adj2edgeL(triu(aij));             %passing from matrix form to edge list form
            f_PlotEEG_BrainNetwork(nch, ijw, 'w_wn2wx');
            title(FRR_names(i,:))
            if i == 1
                ylabel('Music')
            end
    
            subplot(4,7,21+i)
    %         aij = threshold_proportional(, p); %thresholding networks due to proportion p
            ijw = adj2edgeL(triu(cx{i}));             %passing from matrix form to edge list form
            f_PlotEEG_BrainNetwork(nch, ijw, 'w_wn2wx',REST_DSgroups(i,:));
            title(FRR_names(i,:))
        end
        saveas(gca,['D:\Dropbox\Paper BrainScore_BCMI\figures\2sub',num2str(sub),'_figure5'],'png')
        close
    
    figure(1),clf,set(gcf,'color','w');clrs=colormap(lines(7));clrs = [0 0 0; clrs];
%     set(gcf,'units'],'points'],'position',[500,500,1600,400])
%     for j=1:7
%         subplot(2,7,j),hold
%         groups=REST_DSgroups(j,:);
%         spX=sensor_coordinates(:,1);spY=sensor_coordinates(:,2);
%         for i=1:29,
%             if(groups(i)),size_index=max(1,round(((REST_DSscores(j,groups(i))-lim1)/(lim2-lim1))*ceil(lim2*10)));
%                 plot(spX(i),spY(i),'bo','markersize',10+1.5*size_index,'MarkerFaceColor',clrs(groups(i)+1,:)),axis([-200 200 -200 200]),
%             end;
%         end;
%     
%                 for ii=1:29; text(spX(ii)+5,spY(ii),sensors_names(ii,:)), end
%         for ii=1:29; text(spX(ii)-0.025*range(spX),spY(ii)+0.001*range(spY),num2str(ii),'fontsize',8), end, axis off;
%         subplot(2,7,7+j),hold
%         groups=MUSIC_DSgroups(j,:);
%         for i=1:29,if(groups(i)),size_index=max(1,round(((MUSIC_DSscores(j,groups(i))-lim1)/(lim2-lim1))*ceil(lim2*10)));plot(spX(i),spY(i),'bo','markersize',10+1.5*size_index,'MarkerFaceColor',clrs(groups(i)+1,:)),axis([-200 200 -200 200]),end;end;
%         for ii=1:29; text(spX(ii)+5,spY(ii),sensors_names(ii,:)), end
%         for ii=1:29; text(spX(ii)-0.025*range(spX),spY(ii)+0.001*range(spY),num2str(ii),'fontsize',8), end, axis off;
%     end
    
    %% %STEP-3
    %%%%%%%%%  Exploring the change in modular organization of a brain rhythm by means of VI distance.
    FRR_names=['delta '; 'theta ';'alpha ';'betaL ';'betaH ';'gammaL';'gammaH'];
    VI_DM=zeros(7,48);VI_DM=zeros(size(FBANDS,1),1); %#ok<PREALL>
    for i_band=1:size(FBANDS,1)
        VI_DM(i_band)=partition_distance(REST_DSgroups(i_band,:),MUSIC_DSgroups(i_band,:));
    end
        figure(6),clf,bar(VI_DM,'k'); set(gca,'XTick',1:length(FRR_names),'XTickLabel',FRR_names);grid
        title('VI distance of MusicListening modular pattern from Rest')
        axis tight;
        set(gcf, 'units','normalized','outerposition',[0 0 1 1]); %EXPANDING FIGURE ON SCREEN
        hold off;
        saveas(gca,['D:\Dropbox\Paper BrainScore_BCMI\figures\2sub',num2str(sub),'_figure6'],'png')
        close
    %
    %% %STEP-4
    %%%%%  Estimating Phase-Amplitude Coupling (PAC) for DELTA->BETA_high interaction
    %
    STEP=1; WINDOW=8960-1;%size(music_eeg,1)-1;% PAC Window = 8s
    Pf1=1;Pf2=4; % Delta band
    Af1=20;Af2=30; % Beta-high band
    
    tic
    %% REST PAC
for I=1:length(Pf1)
    for ELECTRODE=1:size(rest_eeg,2)
        clear REST_curve;REST_curve=squeeze(rest_eeg(1:8960,ELECTRODE))';
        [REST_pac(I,ELECTRODE,:),EEG_REST_Times]=moving_multitrial_pac2_sur(0,REST_curve,Fs,Pf1,Pf2,Af1,Af2,WINDOW,STEP) ;
    end
end

%% MUSIC PAC
for I=1:length(Pf1)
    for ELECTRODE=1:size(music_eeg,2)
        clear MUSIC_curve;MUSIC_curve=squeeze(mean(music_eeg(1:8960,ELECTRODE,trial),3))';
        [MUSIC_pac(I,ELECTRODE,:),EEG_MUSIC_Times]=moving_multitrial_pac2_sur(0,MUSIC_curve,Fs,Pf1,Pf2,Af1,Af2,WINDOW,STEP) ;
    end
end
    % REST PAC
    %     para un par de bandas
    %     for I=1:length(Pf1)
    % para todas las bandas
%     for i1=1:size(FBANDS,1)-1
%         for i2=i1+1:size(FBANDS,1)
%             for ELECTRODE=1:size(rest_eeg,2)
%                 clear REST_curve;
%                 REST_curve=squeeze(rest_eeg(1:8960,ELECTRODE))';
%                 Pf1=FBANDS(i1,1);  Pf2=FBANDS(i1,2);
%                 Af1=FBANDS(i2,1);  Af2=FBANDS(i2,2);
%                 [rest_p,EEG_REST_Times]=moving_multitrial_pac2_sur(0,REST_curve,Fs,Pf1,Pf2,Af1,Af2,WINDOW,STEP) ;
%                 REST_pac(i1,i2,ELECTRODE,:) = rest_p;
%                 REST_pac(i2,i1,ELECTRODE,:) = rest_p;
%             end
%         end
%     end
%     
%     % MUSIC PAC
%     %     para un par de bandas
%     %     for I=1:length(Pf1)
%     % para todas las bandas
%     for i1=1:size(FBANDS,1)-1
%         for i2=i1+1:size(FBANDS,1)
%             for ELECTRODE=1:size(music_eeg,2)
%                 for trial = 1:size(music_eeg,3)
%                     clear MUSIC_curve;
%                     MUSIC_curve=squeeze(music_eeg(1:8960,ELECTRODE,trial))';
%                     Pf1=FBANDS(i1,1);  Pf2=FBANDS(i1,2);
%                     Af1=FBANDS(i2,1);  Af2=FBANDS(i2,2);
%                     [Music_p,EEG_MUSIC_Times]=moving_multitrial_pac2_sur(0,MUSIC_curve,Fs,Pf1,Pf2,Af1,Af2,WINDOW,STEP);
%                     MUSIC_pac(trial,i1,i2,ELECTRODE,:) = Music_p;
%                     MUSIC_pac(trial,i2,i1,ELECTRODE,:) = Music_p;
%                 end
%             end
%         end
%     end
    toc
        figure(7),clf;set(gcf,'color','w');
        bar(squeeze(MUSIC_pac-REST_pac),'k');title('Increase of PAC (MusicListening - Rest)')
        set(gca, 'XTick', 1:Nchannels,'XTickLabel',name_channels);grid,
        saveas(gca,['D:\Dropbox\Paper BrainScore_BCMI\figures\2sub',num2str(sub),'_figure7'],'png')
        close
%     
% %     y = music.tempo;
% %     y(ismember(y,50)) = 1;
% %     y(ismember(y,100)) = 1;
% %     y(ismember(y,150)) = 2;
% %     y(ismember(y,200)) = 2;
% %     FVs=reshape(MUSIC_pac,size(MUSIC_pac,1),size(MUSIC_pac,2)*size(MUSIC_pac,3)*size(MUSIC_pac,4));
%     
%     %___SVM classifier ________
%     %%%%  10-fold cross-validation scheme
%     nrFolds = 5;
% %     cvFolds = crossvalind('Kfold', y, nrFolds);
% %     for i=1:nrFolds
%         %     rlist=randperm(40);
% %         testIdx = (cvFolds == i);                % indices of test instances
% %         trainIdx = ~testIdx;                     % indices training instances
% %         Xtrain=FVs(trainIdx,:); train_labels=y(trainIdx);
% %         Xtest=FVs(testIdx,:); test_labels=y(testIdx);
%         %      net = svm(size(Xtrain, 2), 'linear', [], 10);
% %         svmStruct = fitcsvm(FVs,y','crossval','on','Leaveout','on','KernelFunction','gaussian','Standardize',true);
%         %      Pred_labels = svmclassify(svmStruct,Xtest);
%         
% %         [labelPred,scores] =  predict(svmStruct, Xtest);
% %         eq = sum(labelPred'==test_labels);
% %         accuracy_folds(i) = eq/numel(test_labels);
% %         Error(i)=sum(abs(labelPred'-test_labels))/numel(test_labels);
% %     end
% %     nanmean(Error) % ~6.40
% %     nanmean(accuracy_folds)
%     
%     
%     % %     %%%%  or Leave-one-out subject-validation for the SVM Classifier
% %     Error=[];
% %     for i=1:numel(y)   %label
% %         list=setdiff([1:numel(y)],i);
% %         Xtest=FVs(i,:); test_label=y(i);
% %         Xtrain=FVs(list,:); train_labels= y(list);
% %         %     svmStruct = svmtrain(Xtrain,train_labels,'KernelFunction','linear'); %#ok<SVMTRAIN>
% %         %     Pred_label = svmclassify(svmStruct,Xtest);
% %         svmStruct = fitcsvm(Xtrain,train_labels,'KernelFunction','gaussian','Standardize',true);
% %         [labelPred,scores] =  predict(svmStruct, Xtest);
% %         %     error =  crossval('mse',svmStruct,'Leaveout','on');
% %         eq = sum(labelPred==test_labels);
% %         accuracy_loo(i) = eq/numel(test_labels);
% %         Error(i)=labelPred-test_label;
% %     end
% %         mean(abs(Error)) %~2.5
%     
%     
%     %%_______   Standard Feature Screening with RELIEFF and Regression SVM based on PSD-representation
%     %This part is indicative about the potential to model the likeness from the data
% %     Predictors=reshape(Trial_PSD,size(Trial_PSD,1),19*95);  %Reshape to create a feature vector (PSD) for each trial
% %     [RANKS,Scores] = relieff(Predictors,y',5,'method','regression'); %RELIEFF to rank features
% %     sel_list=find(Scores>0);selDATA=Predictors(:,sel_list); %keep only features with a positive RELIEFF score
% %     mdl = fitrsvm(selDATA,y,'crossval','on','KFold',nrFolds,'standardize','on','KernelFunction','gaussian'); %Train and validate a simple SVM (5-fold cross validation)
% %     MSE = (kfoldLoss(mdl)); % calcuate Mean Squared Error
% %     MAE = sqrt(MSE); % calcuate Mean Absolute Error
% %     disp(['MAE, MSE of the example simple SVM model based on PSD: ', num2str(MAE), ', ', num2str(MSE)])
%     
% %     FVs=Predictors;%reshape(MUSIC_pac,size(MUSIC_pac,1),size(MUSIC_pac,2)*size(MUSIC_pac,3)*size(MUSIC_pac,4));
% %     
% %     %___SVM classifier ________
% %     %%%%  10-fold cross-validation scheme
% %     nrFolds = 5;
% %     cvFolds = crossvalind('Kfold', y, nrFolds);
% %     for i=1:nrFolds
% %         %     rlist=randperm(40);
% %         testIdx = (cvFolds == i);                % indices of test instances
% %         trainIdx = ~testIdx;                     % indices training instances
% %         Xtrain=FVs(trainIdx,:); train_labels=y(trainIdx);
% %         Xtest=FVs(testIdx,:); test_labels=y(testIdx);
% %         %      net = svm(size(Xtrain, 2), 'linear', [], 10);
% %         svmStruct = fitcsvm(Xtrain,train_labels','KernelFunction','gaussian','Standardize',true);
% %         %      Pred_labels = svmclassify(svmStruct,Xtest);
% %         
% %         [labelPred,scores] =  predict(svmStruct, Xtest);
% %         eq = sum(labelPred'==test_labels);
% %         accuracy_folds2(i) = eq/numel(test_labels);
% %         Error(i)=sum(abs(labelPred'-test_labels))/numel(test_labels);
% %     end
% %     nanmean(Error) % ~6.40
% %     nanmean(accuracy_folds)
%     
% %     save(['Resultado_sub',num2str(sub),'.mat'],'REST_pac','EEG_REST_Times','MUSIC_pac','EEG_MUSIC_Times','MAE','MSE','accuracy_folds','accuracy_folds2','accuracy_loo')
end