function [W,covcla] = fncCSP(Xtrain,ltr,nsf)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input:
% Xtrain: cell that containt the EEG trials
% nsf: number of space filters
% if nsf is [] the function will return all the eigen vectors. 
% 
%% Output:
% W: the learnt CSP filters (a [Nc*Nc] matrix with the filters as rows)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2018 Signal Processing and Recognition Group.
% Universidad Nacional de Colombia.
% SPRG - UNAL 
% L.F. Velasquez-Martinez
% F.Y. Zapata-Castaño.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 3
    flag = 1;
else
    flag = 0;
end
%%
for j = 1:length(Xtrain) 
    X.x(:,:,j) = Xtrain{j};
end
X.y = ltr;

nchan = size(X.x,2);
ntr = size(X.x,3);
classLabels = unique(X.y);
ncla = length(classLabels);
if ncla ~= 2
    disp('ERROR! CSP can only be used for two classes');
    return;
end
covcla = cell(ncla,1); %the covariance matrices for each class
covtr = zeros(nchan,nchan,ntr);

for tr = 1:ntr %covariance matrices for each trial
    E = X.x(:,:,tr)';
    covtr(:,:,tr) = cov(E');
end

for c = 1:ncla                    % covariance matrix for each class    
    covcla{c} = mean(covtr(:,:,X.y == classLabels(c)),3);  
end

%% CSP problem by Fabien Lotte
covTotal = covcla{1} + covcla{2}; % the total covariance matrix

%whitening transform of total covariance matrix
[Ut,Dt] = eig(covTotal);          % caution: the eigenvalues are initially in increasing order
eigenvalues = diag(Dt);
[eigenvalues,egIndex] = sort(eigenvalues, 'descend');
Ut = Ut(:,egIndex);
P = diag(sqrt(1./eigenvalues)) * Ut';

%transforming covariance matrix of first class using P
transformedCov1 =  P * covcla{1} * P';

%EVD of the transformed covariance matrix
[U1,D1] = eig(transformedCov1);
eigenvalues = diag(D1);
[~,egIndex] = sort(eigenvalues, 'descend');
% U = U1;
if flag == 1
    U1 = U1(:, egIndex);
elseif flag == 0
    U1 = U1(:, egIndex([1:nsf,end-(nsf-1):end]));
end
W = U1' * P;