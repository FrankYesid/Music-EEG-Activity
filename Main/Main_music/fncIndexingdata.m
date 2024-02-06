function [tr_ind, ts_ind, cl] = fncIndexingdata(cv,y,fold)
%%
% [tr_ind, ts_ind, cl] = fncIndexingdata(cv,y,fold)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2019 Signal Processing and Recognition Group.
% Universidad Nacional de Colombia.
% L.F. Velasquez-Martinez
% F.Y. Zapata-Castaño.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tr_ind   = cv.training(fold);
ts_ind   = cv.test(fold);

indx = zeros(numel(y),1); cl = zeros(numel(y),1);
al = 1;
for ttr = 1:numel(tr_ind)
    if tr_ind(ttr) == 1
        indx(ttr,1) = tr_ind(al);
        cl(al,1) = y(al)*tr_ind(al);
        al = al+1;
    end
end
end