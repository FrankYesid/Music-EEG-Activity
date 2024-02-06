function Xa = fncFilbank(X2tr,filter_bank,seg_start,seg_end,fs)
%% Function of filter-bank
%% input:
% X2tr        - cell that containt the EEG trials.
% filter_bank - specified filter.
% seg_start   - starting point
% seg_end     - ending point.
% fs          - sampled rate.
%% Output:
% Xa          - Filter signal and record.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2019 Signal Processing and Recognition Group.
% Universidad Nacional de Colombia.
% L.F. Velasquez-Martinez
% F.Y. Zapata-Castaño.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Xa = cell(size(filter_bank,1),1);
for b = 1:size(filter_bank,1)    %Precompute all filters and trim
    Xa{b} = fcnfiltband(X2tr, fs, filter_bank(b,:), 3);
    Xa{b} = cellfun(@(x) x(seg_start:seg_end,:),Xa{b},'UniformOutput',false);
end
end