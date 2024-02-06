function [X,y,fs] = organizar_(path,s)
% cargo la información de la base de datos
datos = load(path);
fs    = 1000;
% cargo los datos de resting y parto en 20 trials de 9.5 segundos.
rest = datos.rest;
rest = rest(:,10*fs:end-10*fs)';
pos = [1:10*fs*0.7: size(rest,1)-10*fs]; %#ok<NBRAK>
Ntr_rest = size(pos,2);
resting = cell(Ntr_rest,1);
for tr = 1:Ntr_rest
    resting{tr} = rest(pos(tr):pos(tr)+10*fs-1,:);
end
% cargo los datos de musica que ya están organizados en
% trial,canales,tiempos
music = datos.music;
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
X = [music_;resting];
y = ones(Ntr_music+Ntr_rest,1);
y(Ntr_music+1:end) = 2;