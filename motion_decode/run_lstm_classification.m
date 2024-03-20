function i1_m = run_lstm_classification(R, dir, nFolds, nNode)
% run_lstm Trains an LSTM network for classification on sequential data.
%
% Inputs:
% R - A cell array of matrices, each matrix representing sequential data for one observation (features x time).
% dir - A vector indicating the class or direction for each observation, used as the target variable for classification.
% nFolds - Number of folds for K-fold cross-validation.
% nNode - Number of nodes in the LSTM layer.

% Initialize cross-validation indices
ind = crossvalind('Kfold', length(dir),nFolds);
% Number of time steps (assumed uniform across all sequences)
nrTimes = size(R{1},2);
% Total number of sequences
nrImgs = size(R,2);
% Number of unique classes
nrObjs = length(unique(dir));
% Number of features per time step
inputSize = size(R{1},1);
% Number of LSTM output nodes, configurable for model complexity
outputSize = nNode;
% LSTM output mode set to return sequences
outputMode = 'sequence';
% Total number of unique classes
numClasses = nrObjs;

% Define the architecture of the LSTM network
layers = [ ...
    sequenceInputLayer(inputSize) % Input layer for sequences
    bilstmLayer(outputSize,'OutputMode',outputMode) % Bidirectional LSTM layer
    fullyConnectedLayer(numClasses) % Fully connected layer for class scores
    softmaxLayer % Softmax layer for probability distribution
    classificationLayer]; % Final classification layer

% Training options
maxEpochs = 120;
miniBatchSize = 27;
options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'ExecutionEnvironment', 'cpu',...
    'Verbose',0); % Use CPU for training, with specified epochs and batch size

% Convert class labels to categorical
objNr = categorical(dir);
% Repetition factor for each class label, matching the number of time steps
repTimes = nrTimes;

% Train one model per fold in cross-validation
parfor i = 1:nFolds
    % Generate labels for each time step in the sequence
    trainLabel = num2cell(repmat(objNr(ind~=i),1,repTimes),2);
    % Select training data for the current fold
    train = R(ind~=i);
    % Train the network
    net{i} = trainNetwork(train,trainLabel,layers,options);
end
disp('Done with model training')

% Initialize storage for the test results
score = nan(nrTimes,nrObjs,nrImgs);

% Run tests for each fold
for i = 1:nFolds
    test = R(ind==i); % Select test data for the current fold

    indloc = find(ind==i);
    for j = 1:numel(indloc)
        mnet = net{i}; % Select the model trained for the current fold

        for time = 1:nrTimes
            % Update LSTM state with each time step and store classification scores
            [mnet,~,score(time,:,indloc(j))] = classifyAndUpdateState(mnet,test{j}(:,time));
        end
    end
end

disp('Running tests')

% Initialize arrays for performance metrics
dp = nan(nrImgs, nrTimes);
i1_m = nan(nrImgs, nrTimes);

% Calculate metrics for each time step
for i = 1:nrTimes
    posterior = permute(score(i,:,:),[3 2 1]); % Adjust dimensions for processing
    % Function to calculate a metric based on posterior probabilities
    pr_c = @(a,i) (repmat(a(1,i),1,nrObjs-1)./(a(1,~ismember(1:nrObjs,i))+ repmat(a(1,i),1,nrObjs-1)));
    i2 = nan(sum(testLoc),nrObjs);
    for j = 1:numel(obj)
        i2(j,~ismember(1:nrObjs,obj(j)))= pr_c(posterior(j,:),obj(j));
    end
    i1_m(:,i) = nanmean(i2,2); % Average metric across classes for each time step
end
end
