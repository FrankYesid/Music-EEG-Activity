function fnc_MyTopo22(rel,sel,pos,label,lims,ticks,t_e,database,HeadModel,cur,tex,Color)
%% Function to graph the different topoplots. %%
% fnc_MyTopo22(rel,sel,pos,label,lims,ticks,t_e,database,HeadModel,cur,tex,Color)
% rel       -   Vector of weights 1xN.
% pos       -   Positions vector Nx2.
% sel       -   Channels selection.
% label     -   tag cell Nx1.
% lims      -   vector colorbar limits [min max].
% cur       -   boolean enables contour lines 1,0.
% ticks     -   ability to plot Boolean labels 1,0.
% t_e       -   size of the electrodes.
% database  -   selected database.
% HeadModel -   model of the human head.
% tex       -   ability to plot labels of channels 1,0.
% Example:
%       database = 2; % BCI_competition_IV_dataset_2a 
%       tex = 0;
%       load('electrodesBCICIV2a.mat') 
%       load('HeadModel.mat')          % model of the head.
%       t_e = 15;                      % size electrodes.
%       M1.xy = elec_pos;              % position of channels.
%       M1.lab = Channels;             % names of channels.
%       sel = 1:22;                    % channels selection.
%       rel = ones(1,22);              % relevance of channels.
%       fnc_MyTopo22(rel,sel,pos,label,lims,ticks,t_e,database,HeadModel,cur,tex,Color)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2019 Signal Processing and Recognition Group.
% Universidad Nacional de Colombia.
% L.F. Velasquez-Martinez
% F.Y. Zapata-Castaño.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:2
    pos(:,i) = 0.9.*((pos(:,i)-min(pos(:,i)))/range(pos(:,i))-0.5);
end
xc = HeadModel(1,:);
yc = HeadModel(2,:);
hold on
%% Topoplot
x = pos(:,1);
y = pos(:,2);
tmp = [x,y,x*0]*rotz(2);
x = tmp(:,1);
y = tmp(:,2);
pos(:,1) = x;
pos(:,2) = y;
GS = 500;
xi = linspace(min(x)-0.05, max(x)+0.05, GS);       % x-axis for interpolation (row vector)
yi = linspace(min(y)-0.05, max(y)+0.05, GS);       % y-axis for interpolation (row vector)
[Xi, Yi, Zi] = griddata([x' xc], [y' yc], [rel(:);zeros(numel(xc),1)], xi', yi,'v4'); % interpolate the topographic data
%%
[~,R] = cart2pol(Xi,Yi);
Zi(R>0.5) = NaN;
deltax = xi(2)-xi(1); % length of grid entry
deltay = yi(2)-yi(1); % length of grid entry
%% surf
surf(Xi-deltax/2, Yi-deltay/2+0.004, zeros(size(Zi)), Zi,'EdgeColor', 'none', 'FaceColor', 'flat');hold on
if cur == 1
    caxis([lims(1) lims(2)])
end
colormap(Color)
plot(0.999.*xc,0.999.*yc,'Color',[130,130,130]/255,'LineWidth',2) % cambié el tamaño del contorno
view([0,90])
axis('square')
if ticks == 1
    scatter(0.999.*x(sel),0.999.*y(sel),t_e,'b','filled')
end
if tex == 1
    text(pos(sel,1)-0.02,pos(sel,2)-0.04,label(sel),'Interpreter',...
        'latex','ButtonDownFcn',{@lineCallback,database},'Color','black',...
        'FontSize',10);
end
hold off
axis off