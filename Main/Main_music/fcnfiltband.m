function [Xfreq] = fcnfiltband(X,fs,freq,n)
%% FUNCTION rhythms
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% Xf = A_filterButter(Xcell,fs)
% X{trial}(samples x chan): Data to analysis
% 
% fs: sampled rate
% n : filter order
% freq : vector con las bandas
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs:
% Xf: Filtered data 
% Xf{trial}(samples x chan)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2017 Signal Processing and Recognition Group.
% Universidad Nacional de Colombia.
% L.F. Velasquez-Martinez 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  filter order
if nargin < 4  
   n = 5;   % by default 
end
% Filter desing
wn_freq = [freq(1)/(0.5*fs) freq(2)/(0.5*fs)]; 
% Filter
[b_freq,a_freq] = butter(n,wn_freq); 

% show filter response
tr = length(X);
Xfreq = cell(1,tr);

for k = 1: tr % Trials
    [s,c] = size(X{k});
    if c>s X{k} = X{k}'; [s,c] = size(X{k}); end    %#ok<ASGLU>
    for ch = 1 : c
        x = X{k}(:,ch);
        Xfreq{k}(:,ch) = filtfilt(b_freq,a_freq,x); %filter
    end
end


