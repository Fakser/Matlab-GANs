
addpath('C:\Program Files\MATLAB\R2020b\examples\nnet\main')
% downloading the data
%url = 'http://download.tensorflow.org/example_images/flower_photos.tgz';
%filename = fullfile(downloadFolder,'flower_dataset.tgz');
downloadFolder = pwd;
dataset_name = 'anime_faces';
imageFolder = fullfile(downloadFolder, dataset_name);
% if ~exist(imageFolder,'dir')
%     disp('Downloading Flowers data set (218 MB)...')
%     websave(filename,url);
%     untar(filename,downloadFolder)
% end

datasetFolder = fullfile(imageFolder);

imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders',true);


disp('data downloaded and loaded')

% data augmentation


augmenter = imageDataAugmenter('RandXReflection',true);
augimds = augmentedImageDatastore([64 64],imds,'DataAugmentation',augmenter);    

disp('data augmented')
% define generator network

filterSize = 5;
numFilters = 64;
numLatentInputs = 100;

projectionSize = [4 4 512];

layersGenerator = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    projectAndReshapeLayer(projectionSize,numLatentInputs,'proj');
    transposedConv2dLayer(filterSize,4*numFilters,'Name','tconv1')
    batchNormalizationLayer('Name','bnorm1')
    reluLayer('Name','relu1')
    transposedConv2dLayer(filterSize,2*numFilters,'Stride',2,'Cropping','same','Name','tconv2')
    batchNormalizationLayer('Name','bnorm2')
    reluLayer('Name','relu2')
    transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping','same','Name','tconv3')
    batchNormalizationLayer('Name','bnorm3')
    reluLayer('Name','relu3')
    transposedConv2dLayer(filterSize,3,'Stride',2,'Cropping','same','Name','tconv4')
    tanhLayer('Name','tanh')];

lgraphGenerator = layerGraph(layersGenerator);

% define discriminator network

dropoutProb = 0.5;
numFilters = 64;
scale = 0.2;

inputSize = [64 64 3];
filterSize = 5;

layersDiscriminator = [
    imageInputLayer(inputSize,'Normalization','none','Name','in')
    dropoutLayer(0.5,'Name','dropout')
    convolution2dLayer(filterSize,numFilters,'Stride',2,'Padding','same','Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    convolution2dLayer(filterSize,2*numFilters,'Stride',2,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    leakyReluLayer(scale,'Name','lrelu2')
    convolution2dLayer(filterSize,4*numFilters,'Stride',2,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    leakyReluLayer(scale,'Name','lrelu3')
    convolution2dLayer(filterSize,8*numFilters,'Stride',2,'Padding','same','Name','conv4')
    batchNormalizationLayer('Name','bn4')
    leakyReluLayer(scale,'Name','lrelu4')
    convolution2dLayer(4,1,'Name','conv5')];

lgraphDiscriminator = layerGraph(layersDiscriminator);

disp('neural networks defined')
% As we need to define our custom training loop, we have to transform
% layergraph to dlnetwork object

dlnetGenerator = dlnetwork(lgraphGenerator);
dlnetDiscriminator = dlnetwork(lgraphDiscriminator);

save(sprintf('%s_generator.mat', dataset_name), 'dlnetGenerator');
save(sprintf('%s_discriminator.mat', dataset_name), 'dlnetDiscriminator');

disp('test load of models')

load(sprintf('%s_generator.mat', dataset_name), 'dlnetGenerator');
load(sprintf('%s_discriminator.mat', dataset_name), 'dlnetDiscriminator');

% Training loop specification

numEpochs = 500;
miniBatchSize = 128;

% adam optimizer options 

learnRate = 0.0002;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;

% If the discriminator learns to discriminate between real and generated 
% images too quickly, then the generator may fail to train. To better 
% balance the learning of the discriminator and the generator, add noise to
% the real data by randomly flipping the labels.

% We will specify flip to 30% of the real labels. This means that 15% of the total 
% number of labels are flipped during training. Note that this does not 
% impair the generator as all the generated images are still labelled correctly.

flipFactor = 0.3;

% display validation images every 100 iterations

validationFrequency = 100;

%%% TRAINING THE MODEL %%%

% We will Use the minibatchqueue to process and manage the minibatches of
% images.

augimds.MiniBatchSize = miniBatchSize;

executionEnvironment = "gpu";

mbq = minibatchqueue(augimds,...
    'MiniBatchSize',miniBatchSize,...
    'PartialMiniBatch','discard',...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat','SSCB',...
    'OutputEnvironment',executionEnvironment);

% For monitoring the training process we will save some generated images
% each iteration. Also we will save the scores and plot them.

trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];

% To monitor training progress, we will display a batch of generated images 
% using a held-out batch of fixed arrays of random values fed into the 
% generator and plot the network scores.

numValidationImages = 25;
ZValidation = randn(1,1,numLatentInputs,numValidationImages,'single');
dlZValidation = dlarray(ZValidation,'SSCB');

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    disp("gpu is being used")
    dlZValidation = gpuArray(dlZValidation);
end

% We will initialize training process plots, and subplots for generated
% images and network scores

f = figure;
f.Position(3) = 2*f.Position(3);
imageAxes = subplot(1,2,1);
scoreAxes = subplot(1,2,2);

% Initialization of animated lines for the scores plot

lineScoreGenerator = animatedline(scoreAxes,'Color',[0 0.447 0.741]);
lineScoreDiscriminator = animatedline(scoreAxes, 'Color', [0.85 0.325 0.098]);
legend('Generator','Discriminator');
ylim([0 1])
xlabel("Iteration")
ylabel("Score")
grid on

iteration = 0;
start = tic;

disp('starting the training procedure')

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Reset and shuffle datastore.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        dlX = next(mbq);
        
        % Generate latent inputs for the generator network. Convert to
        % dlarray and specify the dimension labels 'SSCB' (spatial,
        % spatial, channel, batch). If training on a GPU, then convert
        % latent inputs to gpuArray.
        Z = randn(1,1,numLatentInputs,size(dlX,4),'single');
        dlZ = dlarray(Z,'SSCB');        
        
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlZ = gpuArray(dlZ);
        end
        
        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor);
        dlnetGenerator.State = stateGenerator;
        
        % Update the discriminator network parameters.
        [dlnetDiscriminator,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [dlnetGenerator,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Every validationFrequency iterations, display batch of generated images using the
        % held-out generator input
        if mod(iteration,validationFrequency) == 0 || iteration == 1
            % Generate images using the held-out generator input.
            dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);
            
            % Tile and rescale the images in the range [0 1].
            I = imtile(extractdata(dlXGeneratedValidation));
            I = rescale(I);
            
            % Display the images.
            subplot(1,2,1);
            image(imageAxes,I)
            xticklabels([]);
            yticklabels([]);
            title("Generated Images");
            saveas(f, sprintf('iterations_images/%s_%d.png', dataset_name, iteration))
        end
        
        % Update the scores plot
        subplot(1,2,2)
        addpoints(lineScoreGenerator,iteration,...
            double(gather(extractdata(scoreGenerator))));
        
        addpoints(lineScoreDiscriminator,iteration,...
            double(gather(extractdata(scoreDiscriminator))));
        
        % Update the title with training progress information.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title(...
            "Epoch: " + epoch + ", " + ...
            "Iteration: " + iteration + ", " + ...
            "Elapsed: " + string(D))
        
        drawnow
    end
    
end
% after the training save the models to .onnx and test them
save(sprintf('%s_generator.mat', dataset_name), 'dlnetGenerator');
save(sprintf('%s_discriminator.mat', dataset_name), 'dlnetDiscriminator');

ZNew = randn(1,1,numLatentInputs,25,'single');
dlZNew = dlarray(ZNew,'SSCB');
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZNew = gpuArray(dlZNew);
end
dlXGeneratedNew = predict(dlnetGenerator,dlZNew);

% display generated images

I = imtile(extractdata(dlXGeneratedNew));
I = rescale(I);
figure
image(I)
axis off
title("Generated Images")



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%           %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%           %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
    modelGradients(dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor)

    % Calculate the predictions for real data with the discriminator network.
    dlYPred = forward(dlnetDiscriminator, dlX);

    % Calculate the predictions for generated data with the discriminator network.
    [dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
    dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated);

    % Convert the discriminator outputs to probabilities.
    probGenerated = sigmoid(dlYPredGenerated);
    probReal = sigmoid(dlYPred);

    % Calculate the score of the discriminator.
    scoreDiscriminator = ((mean(probReal)+mean(1-probGenerated))/2);

    % Calculate the score of the generator.
    scoreGenerator = mean(probGenerated);

    % Randomly flip a fraction of the labels of the real images.
    numObservations = size(probReal,4);
    idx = randperm(numObservations,floor(flipFactor * numObservations));

    % Flip the labels
    probReal(:,:,:,idx) = 1-probReal(:,:,:,idx);

    % Calculate the GAN loss.
    [lossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated);

    % For each network, calculate the gradients with respect to the loss.
    gradientsGenerator = dlgradient(lossGenerator, dlnetGenerator.Learnables,'RetainData',true);
    gradientsDiscriminator = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);

end

function [lossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated)

    % Calculate the loss for the discriminator network.
    lossDiscriminator =  -mean(log(probReal)) -mean(log(1-probGenerated));

    % Calculate the loss for the generator network.
    lossGenerator = -mean(log(probGenerated));

end

function X = preprocessMiniBatch(data)
    % Concatenate mini-batch
    X = cat(4,data{:});
    
    % Rescale the images in the range [-1 1].
    X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
end