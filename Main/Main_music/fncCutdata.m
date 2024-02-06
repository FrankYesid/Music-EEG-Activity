function Xdr = fncCutdata(Xd_,tini,tfin,fs) 
%% Function of record signal.
% 
%% Input:
% X2tr   - data
% tini   - starting point
% tfin   - ending point.
% fs     - sampled rate.
%% Output:
% Xdr    - Filter signal and record.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2019 Signal Processing and Recognition Group.
% Universidad Nacional de Colombia.
% L.F. Velasquez-Martinez
% F.Y. Zapata-Castaño.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% record 
ini = tini*fs;
fin = tfin*fs;
Xdr = cell(1,length(Xd_));
for i = 1:length(Xd_)
    Xf_ = Xd_{i};
    for k = 1:length(Xf_)
        [samples,~] = size(Xf_{k});
        lags = 1:samples;  
        pos = find(lags<fin+1 & lags>ini);
        Xdr{i}{k} = Xf_{k}([pos],:);  
    end
end