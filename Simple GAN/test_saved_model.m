% after the training save the models to .onnx and test them


dataset_name = 'anime_faces';


addpath('C:\Program Files\MATLAB\R2020b\examples\nnet\main')
downloadFolder = pwd;
imageFolder = fullfile(downloadFolder, dataset_name);
load(sprintf('models/%s_generator.mat', dataset_name), 'dlnetGenerator');
load(sprintf('models/%s_discriminator.mat', dataset_name), 'dlnetDiscriminator');
filterSize = 5;
numFilters = 64;
numLatentInputs = 100;

ZNew = randn(1,1,numLatentInputs,25,'single');
dlZNew = dlarray(ZNew,'SSCB');
executionEnvironment = 'gpu';
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