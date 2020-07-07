%% load and prepare data
FS = readtable('freesurfer_numbers_neu.xlsx');
FS.ROI = FS.Var4;
FS_old = readtable('freesurfer_numbers.xlsx');
FS_old.Properties.VariableNames{2} = 'ROI_FS';
FS_old.le = cellfun(@(x) ['le_' x(8:end), '_' upper(x(5))],FS_old.ROI_FS, 'UniformOutput',false);


TBL = readtable('Gesture_schizophrenia_2016_april_4_FINSLER_withoutdropouts_both_ohnep021.xlsx');
TBL(82:end,:) = [];
TBL2 = readtable('ConnMat_Finsler_Final_output3.csv','Delimiter',';');
G = 'pc';
TBL.caselist = arrayfun(@(g,x) sprintf('%s%03d',G(g),x),TBL.Group,TBL.Probandennummer,'UniformOutput',false);
TBL =  join(TBL, TBL2(:,{'caselist' 'ifgiplleft' 'ifgiplright' }));
TBL.Probandennummer(TBL.Group==2) = TBL.Probandennummer(TBL.Group==2) +100;

TBL.diagnose = cellfun(@(x) contains(x, ' Disorder'), TBL.Diagnose_DSM_V);

TBL.TULIA_tot_re = cellfun(@(x) str2double(strrep(x,',','.')), TBL.TULIA_tot_re);
TBL.TULIA_imit_re = cellfun(@(x) str2double(strrep(x,',','.')), TBL.TULIA_imit_re);
TBL.TULIA_panto_re = cellfun(@(x) str2double(strrep(x,',','.')), TBL.TULIA_panto_re);
TBL.TIV = cellfun(@(x) str2double(strrep(x,',','.')), TBL.TIV);

% remove row 14
TBL(14,:) = [];

TBL.GlobalEfficiency = cellfun(@(x) str2double(strrep(x,',','.')), TBL.GlobalEfficiency);
TBL.CRF_AUSBILDUNGSDAUER_JAHRE = cellfun(@(x) str2double(strrep(x,',','.')), TBL.CRF_AUSBILDUNGSDAUER_JAHRE);
% get variables of local efficiency
le=TBL.Properties.VariableNames(cellfun(@(x) ~isempty(regexp(x,'^le_')),TBL.Properties.VariableNames));
variablesOfInterest=cat(2,le,{'ccifg' 'ccipl' 'ifgiplleft' 'ifgiplright' 'ufwmql1left' 'ufwmql1right' 'GlobalEfficiency' ...
   'ufwmql2left' 'ufwmql2right' 'slf2left' 'slf2right' 'slf3left' 'slf3right' 'afleft' 'afright' 'aslantleft' 'aslantright' 'frontoparietalleft' 'frontoparietalright' 'frontotemporalleft' 'frontotemporalright' });
for f=variablesOfInterest
    try
    TBL.(char(f)) = cellfun(@(x) str2double(strrep(x,',','.')), TBL.(char(f)));
    catch
    end
end

%% Check Data Integrity
plot(TBL.GlobalEfficiency, global_efficiency','+');
xlabel('Gesture_schizophrenia_2016_april_4_FINSLER_withoutdropouts_both_ohnep021','interpreter','none');
ylabel('conn matrices/connmat_Finsler_graph_global.mat','interpreter','none');
title('Global efficiency');
axis('equal');

%% correct age and sex
lm = fitlm(TBL, 'TULIA_tot_re ~ 1 + Age + Gender + Age:Group + Gender:Group');
TBL.TULIA_res = TBL.TULIA_tot_re -  predict(lm,TBL);
gscatter(TBL.Age,TBL.TULIA_res,TBL.Group)


%% Node-level Analysis
cohort = {'all_patients', 'schizophrenia_only'};
nodes_template = readtable('Desikan-Killiany68.node', 'FileType', 'text','HeaderLines',0);
models = { '1+Age+Gender+TIV' };
scores = { 'PKT_tot_30' 'TULIA_panto_re' 'TULIA_imit_re' 'TULIA_tot_re' };
cohorts = {true(size(TBL.diagnose )), ~TBL.diagnose};

p=[];
fid = fopen('results_revision.csv','w');
clear pvalues_nodes mean_nodes mean_nodes;
for m=models
for s=scores
    pvalues_nodes.(char(s)) = table;
for f=variablesOfInterest
    for c=1:2
    %cat(2,TBL.Properties.VariableNames(cellfun(@(x) ~isempty(regexp(x,'^le_')),...
    %    TBL.Properties.VariableNames)), {'GlobalEfficiency' 'ccifg' ...
    %'ufwmql1left' 'ufwmql1right'})
    %[h,p(end+1)]=ttest2((TBL.(char(f))((TBL.Group==1)&(TBL.Age>37))), ...
    %    (TBL.(char(f))((TBL.Group==2)&(TBL.Age>37))));
    % fit linear model and correct for Age, Gender, and CRF_AUSBILDUNGSDAUER_JAHRE
    % using only controls
    lm = fitlm(TBL(TBL.Group==2, :), sprintf('%s ~ %s', char(f), char(m)));
    % compute residual
    TBL.([char(f) '_corr']) = TBL.(char(f)) -  predict(lm, TBL);
    % caompute correlation
    %[c,p]=corr(TBL(TBL.Group==1,:).([char(f) '_corr']),TBL(TBL.Group==1,:).TULIA_tot_re);
    %gscatter(TBL.([char(f) '_corr']),TBL.TULIA_tot_re,TBL.Group);
    lm1 = fitlm(TBL(cohorts{c}, :), sprintf('%s ~ 1 + %s',char(s),[char(f) '_corr']));
    lm2 = fitlm(TBL((TBL.Group==1) & cohorts{c}, :), sprintf('%s ~ 1 + %s', char(s), [char(f) '_corr']));
    lm3 = fitlm(TBL(cohorts{c}, :), sprintf('Group ~ 1 + %s',[char(f) '_corr'])); % group difference
    lm4 = fitlm(TBL((TBL.Group==2) & cohorts{c}, :), sprintf('%s ~ 1 + %s', char(s), [char(f) '_corr']));
    for stream=[1 fid]
    fprintf(stream,'%s\t%s\t%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\n',char(m),char(s),char(f),cohort{c}, ...
        table2array(lm1.Coefficients(2,'pValue')),...
        table2array(lm2.Coefficients(2,'pValue')),...
        table2array(lm3.Coefficients(2,'pValue')),...
        table2array(lm4.Coefficients(2,'pValue')));
    end
    pvalues_nodes.(char(s)).([char(f) '_lm1_' cohort{c}]) = table2array(lm1.Coefficients(2,'pValue'));
    pvalues_nodes.(char(s)).([char(f) '_lm2_' cohort{c}]) = table2array(lm2.Coefficients(2,'pValue'));
    pvalues_nodes.(char(s)).([char(f) '_lm3_' cohort{c}]) = table2array(lm3.Coefficients(2,'pValue'));
    pvalues_nodes.(char(s)).([char(f) '_lm4_' cohort{c}]) = table2array(lm4.Coefficients(2,'pValue'));
    mean_nodes.(char(s)).(char(f)) = mean(TBL.(char(f))(TBL.Group==2));
    %gscatter(TBL.([char(f) '_corr']),TBL.(char(s)),TBL.Group)
    %xlabel(char([f '_corr ' char(m)]));
    %ylabel(char(s));
    %pause
end
end
end
%writetable(TBL(:,cat(2,TBL.Properties.VariableNames(cellfun(@(x) ~isempty(regexp(x,'^le_')),TBL.Properties.VariableNames)), ...
%    {'GlobalEfficiency'},{'Age'},{'Gender'},{'Probandennummer'})),...
%    sprintf('coor_%s.csv',strrep(char(m),' ','_')),'delimiter','\t')
end
fclose(fid);

% global efficiency
predictor = 'GlobalEfficiency';
fprintf([predictor '\n']);
s=4;
fprintf('%.3f\t %.3f\t%.3f\n', ...
    pvalues_nodes.(scores{s}).([predictor '_lm1_all_patients']), ...
    pvalues_nodes.(scores{s}).([predictor '_lm2_all_patients']), ...
    pvalues_nodes.(scores{s}).([predictor '_lm4_all_patients']));
fprintf('%.3f\t %.3f\t%.3f', ...
    pvalues_nodes.(scores{s}).([predictor '_lm1_schizophrenia_only']), ...
    pvalues_nodes.(scores{s}).([predictor '_lm2_schizophrenia_only']), ...
    pvalues_nodes.(scores{s}).([predictor '_lm4_schizophrenia_only']));

%
clear BN;
le_ordered = FS_old.le(cellfun(@(x) find(contains(FS_old.Var4,x)), FS.ROI));
le_ordering = cellfun(@(y) find(contains(le,y)), FS_old.le(cellfun(@(x) find(contains(FS_old.Var4,x)), FS.ROI)));
for s=scores
    BN.(char(s)) = nodes_template;
    BN.(char(s)).Var4 = cellfun(@(x) pvalues_nodes.(char(s)).([char(x) '_lm1_all_patients']), le_ordered);
    BN.(char(s)).Var4(BN.(char(s)).Var4 > 0.05) = 1;
    BN.(char(s)).Var4(BN.(char(s)).Var4 < 0.001) = 4;
    BN.(char(s)).Var4(BN.(char(s)).Var4 < 0.01) = 3;
    BN.(char(s)).Var4(BN.(char(s)).Var4 < 0.05) = 2;
    BN.(char(s)).Var5 = cellfun(@(x) mean_nodes.(char(s)).(char(x)), le_ordered);
    node_names = BN.(char(s)).Var6;
    %BN.(char(s)).Var6([15 4 17 16 23]) = {''}; 
    writetable(BN.(char(s)), ['BrainNetNodes_' char(s) '_lm1_all_patients.node'], 'FileType', 'text','WriteVariableNames', false, 'Delimiter', ' ');
  
    BN.(char(s)).Var6 = cellfun(@(x) strrep(strrep(x,'r.',''),'l.',''), node_names,'UniformOutput',false);
    writetable(BN.(char(s)), ['BrainNetNodes_' char(s) '_lm1_all_patients_no_hemi_label.node'], 'FileType', 'text','WriteVariableNames', false, 'Delimiter', ' ');
    
    BN.(char(s)).Var6(:) = {''};
    writetable(BN.(char(s)), ['BrainNetNodes_' char(s) '_lm1_all_patients_no_labels.node'], 'FileType', 'text','WriteVariableNames', false, 'Delimiter', ' ');
    writetable(BN.(char(s)), ['BrainNetNodes_' char(s) '_lm1_all_patients_no_hemi_label_no_labels.node'], 'FileType', 'text','WriteVariableNames', false, 'Delimiter', ' ');
end

BN.(char(s)).Var5


save('pvalues_nodes.mat', 'pvalues_nodes', 'mean_nodes');

% original publication
BrainNet_MapCfg('BrainNet-Viewer/Data/SurfTemplate/BrainMesh_ICBM152.nv', 'BrainNetNodes_TULIA_tot_re_lm1_all_patients.node',   'pvalues_TULIA_tot_re_both.edge',   'BNV_config_sagittal.mat');
view([-97 4]); colorbar('off'); print('Figure_1a_sagittal_revision.png','-dpng','-r300');

% revision
for labels={'', '_no_labels'}
BrainNet_MapCfg('BrainNet-Viewer/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv', ['BrainNetNodes_TULIA_tot_re_lm1_all_patients' char(labels) '.node'],   'edge_pvalues\1+Age+Gender+TIV - TULIA_tot_re - all_patients - beideGruppen.edge',   'BNV_config_sagittal.mat');
view([-97 4]); colorbar('off'); print(['Figure_1a_sagittal_revision' char(labels) '.png'],'-dpng','-r300');
close all
BrainNet_MapCfg('BrainNet-Viewer/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv', ['BrainNetNodes_TULIA_tot_re_lm1_all_patients_no_hemi_label' char(labels) '.node'],   'edge_pvalues\1+Age+Gender+TIV - TULIA_tot_re - all_patients - beideGruppen.edge',   'BNV_config_axial.mat');
view([0 90]); print(['Figure_1a_axial_revision' char(labels) '.png'],'-dpng','-r300');
close all
BrainNet_MapCfg('BrainNet-Viewer/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv', ['BrainNetNodes_TULIA_panto_re_lm1_all_patients' char(labels) '.node'], 'edge_pvalues\1+Age+Gender+TIV - TULIA_panto_re - all_patients - beideGruppen.edge', 'BNV_config_sagittal.mat');
view([-97 4]); colorbar('off'); print(['Figure_1b_sagittal_revision' char(labels) '.png'],'-dpng','-r300');
close all
BrainNet_MapCfg('BrainNet-Viewer/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv', ['BrainNetNodes_TULIA_panto_re_lm1_all_patients_no_hemi_label' char(labels) '.node'], 'edge_pvalues\1+Age+Gender+TIV - TULIA_panto_re - all_patients - beideGruppen.edge', 'BNV_config_axial.mat');
view([0 90]); print(['Figure_1b_axial_revision' char(labels) '.png'],'-dpng','-r300');
close all
BrainNet_MapCfg('BrainNet-Viewer/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv', ['BrainNetNodes_TULIA_imit_re_lm1_all_patients' char(labels) '.node'],  'edge_pvalues\1+Age+Gender+TIV - TULIA_imit_re - all_patients - beideGruppen.edge',  'BNV_config_sagittal.mat');
view([-97 4]); colorbar('off'); print(['Figure_1c_sagittal_revision' char(labels) '.png'],'-dpng','-r300');
close all
BrainNet_MapCfg('BrainNet-Viewer/Data/SurfTemplate/BrainMesh_ICBM152_smoothed.nv', ['BrainNetNodes_TULIA_imit_re_lm1_all_patients_no_hemi_label' char(labels) '.node'],  'edge_pvalues\1+Age+Gender+TIV - TULIA_imit_re - all_patients - beideGruppen.edge',  'BNV_config_axial.mat');
view([0 90]); print(['Figure_1c_axial_revision' char(labels) '.png'],'-dpng','-r300');
end


BrainNet_MapCfg('BrainMesh_ICBM152.nv', 'BrainNetNodes_TULIA_tot_re_lm1_all_patients.node',   'edge_pvalues\1+Age+Gender+TIV - TULIA_tot_re - schizophrenia_only - beideGruppen.edge',   'BNV_config_sagittal.mat');
view([-101 13]); print('Figure_1d_revision.png','-dpng','-r300');
BrainNet_MapCfg('BrainMesh_ICBM152.nv', 'BrainNetNodes_TULIA_panto_re_lm1_all_patients.node', 'edge_pvalues\1+Age+Gender+TIV - TULIA_panto_re - schizophrenia_only - beideGruppen.edge', 'BNV_config_sagittal.mat');
view([-101 13]); print('Figure_1e_revision.png','-dpng','-r300');
BrainNet_MapCfg('BrainMesh_ICBM152.nv', 'BrainNetNodes_TULIA_imit_re_lm1_all_patients.node',  'edge_pvalues\1+Age+Gender+TIV - TULIA_imit_re - schizophrenia_only - beideGruppen.edge',  'BNV_config_sagittal.mat');
view([-101 13]); print('Figure_1f_revision.png','-dpng','-r300');


%% Edge Level Analysis
pvaluemat = repmat({NaN(4,26,26)},numel(scores),numel(models),numel(cohorts));
TBL.conMatrices = cellfun(@(c) load(fullfile('conn matrices', ...
    sprintf('%s_connmat_Finsler.txt',c))), TBL.caselist,'UniformOutput',false);
O = tril(ones(26),-1)>0;
ROI_indices = [arrayfun(@(i) find(cellfun(@(x) contains(x, le{i}(4:end-2)), FS.ROI(1:34))), 1:13) ...
               arrayfun(@(i) find(cellfun(@(x) contains(x, le{i}(4:end-2)), FS.ROI(35:68))), 14:26)+34];
for m=1:numel(models)
for s=2:numel(scores)
for c=1:numel(cohorts)-1
pvalmat = NaN(4,26,26);
%parfor i=1:26
pvalmat = NaN(4,26,26);
parfor k=1:(26^2)
    if ~O(k), continue, end
    [i,j] = ind2sub([26 26], k);
    %for j=1:(i-1)
    %tic
    edge = strrep(strrep(strrep([FS.ROI{ROI_indices(i)} '_to_' FS.ROI{ROI_indices(j)}],'-','_'),'ctx_',''),'.','_');
    a = TBL(cohorts{c},:);
    a.(edge) = cellfun(@(mat) mat(i,j), a.conMatrices);
    lm = fitlm(a(a.Group==2,:), sprintf('%s ~ %s',edge,char(models{m})));
    % compute residual
    a.([edge '_corr']) = a.(edge) -  predict(lm,a);
    % caompute correlation
    %[c,p]=corr(TBL(TBL.Group==1,:).([char(f) '_corr']),TBL(TBL.Group==1,:).TULIA_tot_re);
    %gscatter(TBL.([char(f) '_corr']),TBL.TULIA_tot_re,TBL.Group);
    lm1 = fitlm(a, sprintf('%s ~ 1 + %s',char(scores{s}),[edge '_corr']));
    lm2 = fitlm(a(a.Group==1,:), sprintf('%s ~ 1 + %s',char(scores{s}),[edge '_corr']));
    lm3 = fitlm(a, sprintf('Group ~ 1 + %s',[edge '_corr']));
    lm4 = fitlm(a(a.Group==2,:), sprintf('%s ~ 1 + %s',char(scores{s}),[edge '_corr']));
    fprintf('%s %s %s %.4f %.4f %.4f %.4f\n',char(models{m}),char(scores{s}),edge,table2array(lm1.Coefficients(2,'pValue')),...
        table2array(lm2.Coefficients(2,'pValue')),...
        table2array(lm3.Coefficients(2,'pValue')),...
        table2array(lm4.Coefficients(2,'pValue')));
    pvalma = NaN(4,1);
    pvalma(1) = table2array(lm1.Coefficients(2,'pValue'));
    pvalma(2) = table2array(lm2.Coefficients(2,'pValue'));
    pvalma(3) = table2array(lm3.Coefficients(2,'pValue'));
    pvalma(4) = table2array(lm4.Coefficients(2,'pValue'));
    %toc

    pvalmat(:,k) = pvalma;
end

pvaluemat{s,m,c} = pvalmat;
save(['pvaluemat_edge_revision.mat'],'pvaluemat');
end
end
end

%% Export to BrainNetViewer
load(fullfile('conn matrices','connmat_Finsler_graph_local.mat'));
load(fullfile('conn matrices','connmat_Finsler_graph_global.mat'));
for m = 1:1
    for s = 1:4
        for t = 1:4
            [~,p1] = ttest2(TBL.GlobalEfficiency((TBL.Group==1) & cohorts{1}), ...
                TBL.GlobalEfficiency((TBL.Group==2) & cohorts{1}));
            [~,p2] = ttest2(TBL.GlobalEfficiency((TBL.Group==1) & cohorts{2}), ...
                TBL.GlobalEfficiency((TBL.Group==2) & cohorts{2}));
            fprintf('%s\t%s\t%s\t%.3f %.3f\n',models{m},scores{s},statistics{t},p1,p2);
        end
    end
end
nodes


%% Display Edge Level p-Values
figure(1)
a=load('pvaluemat_edge_revision.mat');
statistics = {'beideGruppen' 'nurPatienten' 'Gruppenunterschied' 'nurKontrollen'};

% Options
m = 1; % models = { '1+Age+Gender+TIV' };
s = 2; % scores = { 'PKT_tot_30' 'TULIA_panto_re' 'TULIA_imit_re' 'TULIA_tot_re' };
t = 1; % lm1, lm2, lm3, lm4
c = 2; % {'all_patients', 'schizophrenia_only'};
for m = 1:1
    for s = 2:4
        for c = 1:2
            for t = 1:4
                pthres = 0.05; % Schwellwert p-Wert
                
                TL = tril(ones(26),-1);
                p = squeeze(a.pvaluemat{s,m,c}(t,:,:));
                % correct p-value
                tl = tril(ones(26,26),-1)>0;
                q = p(tl);
                %[~,~,~,q] = fdr_bh(q);
                pu=p(tl);
                p(tl)= q;

                subplot(2,5,5)
                nbins = 100;
                hist(p(TL>0),nbins)
                hold on
                line([0 1],[1 1].* sum(TL(:))/nbins,'LineStyle','-.','Color','k')
                xlabel('p-value');
                ylabel('frequency');
                grid on
                title('histogram p-values')
                hold off
                p = nanmean(cat(3,max(p,nan) ,max(p',nan)),3);
                p_no_thres = p;
                p = squeeze(min(p,pthres));
    
                
                %p = p(TL>0);
                
                %pFDR=pthres*ones(68);
                %pFDR(TL)=mafdr(p);
                subplot(1,5,1:4)
                imagesc(p);
                set(gca,'YTick',1:26);
                
                set(gca,'YTickLabels',FS.ROI)
                set(gca,'XTick',1:26);
                set(gca,'XTickLabels',FS.ROI)
                set(gca,'XTickLabelRotation',90)
                axis image
                colorbar
                cmap = flipud(hot);
                % make high p-values not appear black
                cmap(end,:) = [116 123 112]/255;
                colormap(cmap)
                title(strrep(sprintf('%s - %s - %s (p_{FDR}<%f)',models{m},scores{s},statistics{t},pthres),'_','\_'));
                print(strrep(sprintf('.\\edge_pvalues\\%s - %s - %s - %s (p<%f).png',models{m},scores{s},cohort{c},statistics{t},pthres),'<','_'),'-dpng','-r600')
                
                mat = nanmean(cat(3,max(p,nan) ,max(p',nan)),3);
                p= p_no_thres;
                p(p>0.05) = 1;
                p(p<0.0005) = 4;
                p(p<0.01) = 3;
                p(p<0.05) = 2;
                dlmwrite(sprintf('.\\edge_pvalues\\%s - %s - %s - %s.edge',models{m},scores{s},cohort{c},statistics{t}),p,...
                          'Delimiter',' ','precision','%.6f')
                dlmwrite(sprintf('.\\edge_pvalues\\p%s - %s - %s - %s.edge',models{m},scores{s},cohort{c},statistics{t}),1-p_no_thres,...
                          'Delimiter',' ','precision','%.6f')
                fprintf('.\\edge_pvalues\\%s - %s - %s - %s.edge\t%d\n',models{m},scores{s},cohort{c},statistics{t},sum(p(:)==4))
            end
        end
    end
end


%% 
p1 = cat(3,squeeze(a.pvaluemat{2,1,1}(2,:,:)),squeeze(pvaluemat{1,1,1}(2,:,:))');
p2 = cat(3,squeeze(a.pvaluemat{2,1,2}(2,:,:)),squeeze(pvaluemat{1,1,2}(2,:,:))');

plot(p1(:),p2(:),'.')

a.pvaluemat{1,1,2}(2,:)
