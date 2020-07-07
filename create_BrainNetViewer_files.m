load('pvaluemat_edge.mat');
mat = 2*nanmean(cat(3,squeeze(pvaluemat{4}(1,:,:)),squeeze(pvaluemat{4}(1,:,:))'),3);
mat(mat>0.05) = 1;
mat(mat<0.001) = 4;
mat(mat<0.01) = 3;
mat(mat<0.05) = 2;

dlmwrite('pvalues_TULIA_tot_re_both.edge',mat,...
    'Delimiter',' ','precision','%.6f')
