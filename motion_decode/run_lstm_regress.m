function i1 = run_lstm_regress(R, vel, nFolds, nNode)
% Run LSTM regression with cross-validation
%
% Inputs:
% R - A cell array of matrices, each representing sequential data for an observation (features x time).
% vel - A vector containing the target variable for each observation.
% nFolds - Number of folds to use in K-fold cross-validation.
% nNode - Number of nodes in the LSTM layer, also the size of the output from the LSTM layer.
%
% Output:
% i1 - A matrix summarizing the performance across all pairs of observations for different times.

%% Prepare for Regression
% Generate indices for K-fold cross-validation
ind = crossvalind('Kfold', length(vel), nFolds);
% Determine the number of time steps from the first observation
nrTimes = size(R{1},2);
% Total number of observations
nrImgs = size(R,2); % number of images or videos
% Input size determined by the number of features in the first observation
inputSize = size(R{1},1);
% Output size of LSTM layer
outputSize = nNode;
% Define the number of outputs from the network
numOutputs = 1;

% Define the layers of the LSTM network
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(outputSize, 'OutputMode', 'sequence')
    fullyConnectedLayer(numOutputs)
    regressionLayer];

% Training options for the network
maxEpochs = 120;
miniBatchSize = 27;
options = trainingOptions('sgdm', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize,...
    'InitialLearnRate', 1e-4,...
    'ExecutionEnvironment', 'cpu',...
    'Verbose',0);

disp('Starting with training the models')

% Train one model per fold
parfor i = 1:nFolds
    % Select the target variable for the current fold
    target_vel = vel(ind~=i);
    % Prepare the target variable for training
    vel_target = cell(1,length(target_vel));
    for x = 1:length(target_vel)
        vel_target{x} = repmat(target_vel(x),1,nrTimes);
    end

    % Select the training data for the current fold
    train = R(ind~=i);
    % Train the network
    net{i} = trainNetwork(train, vel_target, layers, options);
end

disp('Done with training the models')

%% Generate predictions
predictions = nan(nrImgs, nrTimes);

% Iterate over each fold to generate predictions for the test set
for i = 1:nFolds 
    test = R(ind==i);
    disp(i);

    indloc = find(ind==i);
    for j = 1:numel(indloc)
        mnet = net{i};
        for t = 1:nrTimes
             % Predict the target variable for each time step up to 19
             tmp = predict(mnet, test{j}(:,1:t));
             predictions(indloc(j), t) = tmp(end);
        end
    end
end
disp('Done getting the predictions')

%% Evaluate performance
% Initialize the performance matrix
perf = nan(nrImgs,nrImgs,nrTimes);
for i = 1:nrImgs
    for j = 1:nrImgs
        for k = 1:nrTimes
            % Ensure i is not equal to j to avoid self-comparison
            if(i~=j)
                % Compare predictions to determine if the prediction matches the expected trend
                if(vel(i)>vel(j))
                    if(pred(i,k)>pred(j,k))
                        perf(i,j,k)=1;
                    else
                        perf(i,j,k)=0;
                    end
                else
                    if(pred(i,k)>pred(j,k))
                        perf(i,j,k)=0;
                    else
                        perf(i,j,k)=1;
                    end
                end
            end
        end
    end
end

% Average the performance across all comparisons
i1 = squeeze(nanmean(perf,2));

disp('Completed')

end
