function plot_reward(varargin)
varargin
numAgents = 100;
legendLoc = 'SouthEast';
if length(varargin)==1
    numAgents = varargin;
elseif length(varargin)==2
    numAgents = varargin{1};
    legendLoc = varargin{2};
end

% If plotPerformance = 1, plot performance
% Else plot numLearning
plotPerformance=0;

figure;
epochs = 3000;
nights = 10;
capacity = 10;
numTrials = 100;

explorations = {'0.100000','0.200000','1.000000','2.000000','5.000000'};
legendStr = {'10','5','1','0.5','0.2'};
legendStr = arrayfun(@(x) strcat('$Tau = ', x,'$'), legendStr);

% On Desktop
paths = arrayfun(@(x) strcat('../Results/reward_D_150agents/tau_', x, '/grace_0'),explorations);

dataDict = containers.Map();

if plotPerformance
    csvFname = '/performance.csv';
else
    csvFname = '/numLearning.csv';
end

for i = 1:size(paths,2)
    exploreRate = explorations{i};

    path = paths{i};    
    
    trialFolders = arrayfun(@(x) strcat('/run_',num2str(x)), 0:numTrials-1, 'UniformOutput', false);
    file = strcat(path, '/run_0', csvFname)
    trial0 = csvread(file);
    data = zeros(size(trial0, 1), numTrials);
    
    for j = 1:numTrials
       file = strcat(path, trialFolders{j}, csvFname)
       trialData =  csvread(file);
       data(:,j) = trialData(:,2);
    end
    
    meanAndStd = zeros(size(trial0, 1), 3);
    meanAndStd(:,1) = trial0(:,1);
    meanAndStd(:,2) = mean(data, 2);
    meanAndStd(:,3) = std(data,0, 2)./sqrt(numTrials);
    
    dataDict(exploreRate) = meanAndStd;
    
end




markers = ['o'; 'v'; 's'; '^'; 'd'; 'p';'x'];
linestyles = {'-.'; '-'; '--'};
colors = get(gca, 'colororder');

set(gcf, 'Position', [1000, 800, 560, 420])
set(gca, 'FontName', 'Times New Roman');
lw = 1;
fs = 14;

increment = 20;
increment1 = 200;
maxEpoch = 3000;
dict_keys = explorations;


plotHandles = zeros(length(dict_keys),1);
errHandles = zeros(length(dict_keys),1);
sampleHandles = zeros(length(dict_keys),1);

for i = 1:length(dict_keys)
    key = dict_keys{i}
    value = dataDict(key);
    epochs = value(1:increment:maxEpoch,1);
    means = value(1:increment:maxEpoch,2);
    stderr = value(1:increment:maxEpoch,3);
    
    x_axis = value(1:increment1:maxEpoch,1);
    y_axis = value(1:increment1:maxEpoch,2);
    errors = value(1:increment1:maxEpoch,3);
%     e = errorbar(x_axis, y_axis, errors, ...
%         'Marker', markers(mod(i,length(markers))), ...
%         'Linestyle', linestyles{1+mod(i, length(linestyles))} ...
%         );
%     hold on

    % Plot line
    ls = linestyles{1 + mod(i, length(linestyles))};
    c = colors(i,:);
    mkr = markers(mod(i,length(markers)));
    plotHandles(i) = plot(epochs, means, 'LineStyle', ls, 'LineWidth', lw, 'Color', c);
    hold on
    errHandles(i) = errorbar(x_axis, y_axis, errors, ...
        'LineStyle', 'None', 'Marker', mkr , 'MarkerFaceColor', c, 'Color', c);
    sampleHandles(i) = errorbar(x_axis(1), y_axis(1), errors(1), ...
        'LineStyle', ls, 'Marker', mkr, 'MarkerFaceColor', c, 'Color', c,'LineWidth', lw);
end
% 
% title(strcat('Performance vs Number of Epochs for ', num2str(nights), ...
%     ' Nights of ', num2str(capacity), ' Capacity with ', num2str(numAgents), ' Adaptive ', 'Agents - Using Function of Epoch Number (Fixed)'));

if plotPerformance
    title(strcat('Performance vs Number of Epochs'));
else
    title(strcat('Number Agents Learning vs Number of Epochs'));
end

set(gca,'fontname','Times New Roman','FontSize',fs)
grid on 

xlabel('Epoch', 'FontSize', fs, 'Interpreter', 'latex');

if plotPerformance
    % ylabel('Performance (max 100)', 'FontSize', fs, 'Interpreter', 'latex');
    ylabel('Performance (max 90)', 'FontSize', fs, 'Interpreter', 'latex');
else
    % ylabel('Number Agents Learning (max 100)', 'FontSize', fs, 'Interpreter', 'latex');
    ylabel('Number Agents Learning (max 150)', 'FontSize', fs, 'Interpreter', 'latex');
end
% for i= 1:size(legendStr,2)
%     legendStr{i} = legendStr{i}{1};
% end

legend(sampleHandles, legendStr, 'Location', legendLoc, 'Interpreter', 'latex', 'FontSize', fs);
% legendStr(sampleHandles, legendStr, 'Location', legendLoc, 'Interpreter', 'latex', 'FontSize', fs);
if plotPerformance
    ylim([10,100]);
else
    ylim([10,160]);
end

if plotPerformance
    savefig('D_perf_150.fig')
    export_fig(gcf, 'D_perf_150.pdf', '-trans');
else
    savefig('D_numLearn_150.fig')
    export_fig(gcf, 'D_numLearn_150.pdf', '-trans');
end

% if numAgents == 100
%     ylabel('Performance (max 100)', 'FontSize', fs, 'Interpreter', 'latex');
%     savefig('bar_explore_100.fig')
%     export_fig(gcf, 'bar_prob_100agents.pdf', '-trans');
% 
% elseif numAgents == 150
%     ylabel('Performance (max 90)', 'FontSize', fs, 'Interpreter', 'latex');
%     savefig('bar_explore_150.fig')
%     export_fig(gcf, 'bar_prob_150agents.pdf', '-trans');
% 
% elseif numAgents == 200
%     ylabel('Performance (max 90)', 'FontSize', fs, 'Interpreter', 'latex');
%     savefig('bar_explore_200.fig')
%     export_fig(gcf, 'bar_prob_200agents.pdf', '-trans');

% else
% 'invalid number of agents, cant export_fig'
% end
end
