figure(1)
imagesc(j_mean)
axis xy
colorbar()
title(['Sujeto ' num2str(s)])
set(gca,'XTick',1:38,'XTickLabel',round(T(1,:)'/fs,1),...
    'XTickLabelRotation',90,'YTick',1:17,...
    'YTickLabel',mean(filter_bank,2))