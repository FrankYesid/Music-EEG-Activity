function fnc_MyTopo_giga2(rel,sel,M1,lims,cur,ticks,t_e,HeadModel,ELECTRODE_HEIGHT,tex,Color,GS)
%% Function to graph the different topoplots of the database. %%
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
%               vector de los pesos.
%               tex = 0;
%               load('electrodesBCICIV2a.mat') 
%               load('HeadModel.mat')          % model of the head.
%               t_e = 15;                      % size electrodes.
%               M1.xy = elec_pos;              % position of channels.
%               M1.lab = Channels;             % names of channels.
%               sel = 1:22;                    % channels selection.
%               rel = ones(1,64);              % relevance of channels.
%               fnc_MyTopo_giga2(rel,sel,M1,lims,cur,ticks,t_e,HeadModel,type,tex,Color,GS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2019 Signal Processing and Recognition Group.
% Universidad Nacional de Colombia.
% L.F. Velasquez-Martinez
% F.Y. Zapata-Castaño.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% database - Database of GigaScience. - CHO2017
% Modificated 02-2020 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure 
[handle,Zi,grid,Xi,Yi] = topoplot(rel,M1,'maplimits','absmax','style','map','electrodes' ,'labels','headrad' ,0.65, 'plotrad',0.65,'gridscale',GS);
close;

% topo
xc = HeadModel(1,:);
yc = HeadModel(2,:);

set(gcf,'position',[667   528   404   420])
x1 = 0.02;     y1 = 0.06;     w = 0.95;      h = 0.92;
subplot('position',[x1,y1,w,h]);
surf(Xi, Yi, zeros(size(Zi)), Zi,'EdgeColor', 'none', 'FaceColor', 'flat');hold on
view([-90 90])
plot(0.999.*xc'+0.0045,0.999'.*yc+0.005,'Color',[130,130,130]/255,'LineWidth',3) % cambié el tamaño del contorno
view([0 90])
axis square off
[~,labels,~,~,~] = readlocs(M1);
for ch = 1:64; if ch == 28; xx(ch)=M1(ch).X-0.15; else; xx(ch)=M1(ch).X; end; yy(ch)=M1(ch).Y;end;
colormap(Color)
scal = 0.46;
if ticks == 1
    plot3(scal.*yy,scal.*xx,ones(size(xx))*ELECTRODE_HEIGHT,...
        '.','Color','b','markersize',t_e,'linewidth',1);
end
if cur == 1
    caxis([lims(1) lims(2)])
end

if tex == 1
    for i = 1:size(labels',1)
        text(scal.*yy(i)+0.05,scal.*xx(i),...
            ELECTRODE_HEIGHT,labels(i,:),'HorizontalAlignment','center',...
            'VerticalAlignment','middle','Color',[0,0,0],...
            'FontSize',0.5.*t_e)
    end
elseif tex == 2
    for i = 1:size(labels,1)
        text(scal.*yy(i)+0.05,scal.*xx(i),...
            ELECTRODE_HEIGHT,int2str(sel(i)),'HorizontalAlignment','center',...
            'VerticalAlignment','middle','Color',[0,0,0],...
            'FontSize',9)
    end
end
end