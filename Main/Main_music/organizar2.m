function [X,y,fs] = organizar2(path,s)
% cargo la información de la base de datos
datos = load(path);
fs    = 1000;
% cargo los datos de resting y parto en 20 trials de 9.5 segundos.
rest = datos.base;
rest = rest(10*fs:end-10*fs,:);
Ntr_rest = 20;
resting = cell(Ntr_rest,1);
pos = [1:9500:Ntr_rest*9500];
for tr = 1:Ntr_rest
    resting{tr} = rest(pos(tr):pos(tr)+9500-1,:);
end
% cargo los datos de musica que ya están organizados en
% trial,canales,tiempos
music = datos.music.data;
music_= cell(size(music,3),1);
Ntr_music = size(music,3);
for tr = 1:Ntr_music
   music_{tr} = music(:,:,tr);
end
load(['C:\Users\frany\Downloads\Resultado2_music_rest_Sujeto' num2str(s) '.mat'])
for tr = 1:32
    dats_{tr} = squeeze(X_ica(tr,:,:))';
end
X = dats_; 
% X = [music_;resting];
y = ones(Ntr_music+Ntr_rest,1);
y(13:end) = 2;