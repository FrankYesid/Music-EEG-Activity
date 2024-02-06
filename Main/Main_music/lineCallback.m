function lineCallback(snr,database)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2018 Signal Processing and Recognition Group.
% Universidad Nacional de Colombia.
% L.F. Velasquez-Martinez
% F.Y. Zapata-Castaño.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if database == 2
    load('data\DI\electrodesBCICIV2a.mat')
    electrodes = Channels;
    pos = elec_pos;
elseif database == 3
    load('data\DII\posiciones.mat')
    electrodes = M1.lab;
    pos = M1.xy;
end
if snr.Color == [0 0 0]
    snr.Color = 'red';
    b = strfind(electrodes,snr.String,'ForceCellOutput',true);
    for i = 1:length(electrodes)
        if b{i} == 1
            a = i;
        end
    end
    snr.String = num2str(a);
else
    snr.Color = 'black';
    snr.String = electrodes{str2num(snr.String)};
end