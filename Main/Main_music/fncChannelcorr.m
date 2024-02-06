function [RP] = fncChannelcorr(X,Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2019 Signal Processing and Recognition Group.
% Universidad Nacional de Colombia.
% L.F. Velasquez-Martinez
% F.Y. Zapata-Castaño.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

R = nan(size(X,1),1);
P = nan(size(X,1),1);
for i = 1:size(X,1)
     [tmpR,tmpP] = corrcoef(X(i,:),Y(i,:));
     R(i) = tmpR(1,2);
     P(i) = tmpP(1,2);    %entre más pequeño más diferentes
end
RP{1} = R;
RP{2} = P;