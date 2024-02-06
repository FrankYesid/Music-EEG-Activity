y = ones(24,1);
y(13:end) = 2;
for sub = 1:21
   cv{sub} = cvpartition(y,'KFold',10);
end

save('cv_2.mat','cv')

%%
%% Direccion de la base de datos
SUBJECTS_DIR = 'G:\Brain-Rhythms-Multiplexing-master\Brain-Rhythms-Multiplexing-master\Data 2';

SS = 1:20;
for s = SS
   fprintf(['Sujeto...' num2str(s) '\n'])
        if s < 10
            path     = [SUBJECTS_DIR filesep 'music_listening_experiment_s0' num2str(s) '.mat'];
        else
            path     = [SUBJECTS_DIR filesep 'music_listening_experiment_s' num2str(s) '.mat'];
        end
        [X,y,fs] = organizar_2_all(path,s);
        cv{s} = cvpartition(y,'KFold',10);
end
save('cv_music_all.mat','cv')

%%
y = ones(80,1);
y(41:end) = 2;
for sub = 1:31
   cv{sub} = cvpartition(y,'KFold',10);
end

save('cv_music3.mat','cv')


%%
load('H:\copia disco D\databases\esti.mat')
load('H:\copia disco D\databases\Indices_clases.mat')
es = double(estim);
indx = double(Indices);
y_ = indx(es);
y = [];
for sub = 1:31
   y = [y,y_(esti(sub,:)==1)]; %#ok<AGROW>
end
y(end+1) = 4;
for sub = 1
   cv{sub} = cvpartition(y,'KFold',10);
end

save('cv_music3_all.mat','cv','y')