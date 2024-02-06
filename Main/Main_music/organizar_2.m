function [X,y,fs] = organizar_2(path,s)
% cargo la información de la base de datos
datos = load(path);
fs    = datos.Fs;
% cargo los datos de resting y parto en 20 trials de 9.5 segundos.
% rest = datos.EEG_Rest';
% rest = rest(10*fs:end-10*fs,:);
% Ntr_rest = 1;
% resting = cell(Ntr_rest,1);
% pos = [1:8959:Ntr_rest*8959];
% for tr = 1:Ntr_rest
%     resting{tr} = rest(pos(tr):pos(tr)+8959-1,:);
% end
% cargo los datos de musica que ya están organizados en
% trial,canales,tiempos
music = datos.EEG_Songs;
pos = [1:9.5*fs:size(music,3)-9.5*fs];
music_= cell(size(music,1)*numel(pos),1);
Ntr_music = size(music,1);
cont = 1;
y = zeros(size(music,1)*numel(pos),1);
for tr = 1:Ntr_music
    for tr1 = 1:numel(pos)
        music_{cont} = squeeze(music(tr,:,pos(tr1):pos(tr1)+9.5*fs-1))';
        if datos.song_ratings(tr) == 1
            y(cont) = 1;
        elseif datos.song_ratings(tr) == 3
            y(cont) = 2;
        else
            y(cont) = 3;
        end
        cont = cont+1;        
    end    
end
% load(['C:\Users\frany\Downloads\Resultado2_music_rest_Sujeto' num2str(s) '.mat'])
% for tr = 1:32
%     dats_{tr} = squeeze(X_ica(tr,:,:))';
% end
% X = dats_; 
X = music_;
% y = datos.song_ratings;%ones(Ntr_music+Ntr_rest,1);
% y(30:end) = 2;
