env = DRL_HYBRID();
obsInfo = env.getObservationInfo;
actInfo = env.getActionInfo;

numObs  = obsInfo.Dimension(1);
numAct  = numel(actInfo.Elements);    
sequenceLength = 20; % Number of time steps in the sequence 

layers = [
    sequenceInputLayer([numObs 1 1], 'Normalization', 'none', 'Name', 'observations')
    lstmLayer(50, 'OutputMode', 'sequence', 'Name', 'LSTM') 
    fullyConnectedLayer(256, 'Name', 'fc1') 
    reluLayer('Name', 'relu1') 
    fullyConnectedLayer(128, 'Name', 'fc2') 
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(64, 'Name', 'fc3') 
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(numAct, 'Name', 'numActions')
];

% critic options
criticOpts = rlRepresentationOptions('LearnRate', 1e-3, 'GradientThreshold', 1, 'Optimizer', "rmsprop", 'UseDevice', "gpu");

% create critic function
critic = rlQValueRepresentation(layers, obsInfo, actInfo, 'Observation', {'observations'}, 'Action', {'numActions'}, criticOpts);

% agent options
agentOpts = rlDQNAgentOptions(...
    'ExperienceBufferLength', 1e6, ... % Size of the replay memory buffer
    'TargetSmoothFactor', 1e-2, ... % Smoothing factor for target network updates
    'SequenceLength', 70, ...
    'MiniBatchSize', 32 ... % Batch size for each training iteration
);

agentOpts.EpsilonGreedyExploration.EpsilonDecay = 1e-3;
agentOpts.DiscountFactor = 0.95;
agentOpts.TargetUpdateFrequency = 1;

% create agent
agent = rlDQNAgent(critic, agentOpts);

% Rest of your code

%% Training options
maxepisodes = 1000;
maxsteps = 250;
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes', maxepisodes, ...
    'MaxStepsPerEpisode', maxsteps, ...
    'StopOnError', 'on', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', Inf, ...
    'ScoreAveragingWindowLength', 10 ...
);

logger = rlDataLogger();
logger.EpisodeFinishedFcn = @localEpisodeFinishedFcn;
logger.LoggingOptions.LoggingDirectory = "SimulatedPendulumDataSet";

%% Train agent
trainingStats = train(agent, env, trainingOptions);
save('Agent', 'agent');
