function [X,y,fs] = organizar_2_all(path,s)
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
music_= cell(size(music,1),1);
Ntr_music = size(music,1);
for tr = 1:Ntr_music
    music_{tr} = squeeze(music(tr,:,:))';
end

% load(['C:\Users\frany\Downloads\Resultado2_music_rest_Sujeto' num2str(s) '.mat'])
% for tr = 1:32
%     dats_{tr} = squeeze(X_ica(tr,:,:))';
% end
% X = dats_; 
X = music_;
y = datos.song_ratings;%ones(Ntr_music+Ntr_rest,1);
y(y==3) =2;
y(y==5) =3;
% y(30:end) = 2;
