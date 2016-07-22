function AP = Wrapper( prms )
%WRAPPER Calls all the necessary functions to perform Active Learning (AL)
%   Gets as input all the parameters (see demo.m) and calls all the
%   functions for AL (i.e. trains a baseline classifier, and for N
%   iterations calls the active selector, selects a sets of samples from
%   the pool, retrains the classifier and evaluates all the classifiers).
%   In the end, returns the evaluator


dbstop if error

%% Init parameters
% dataset params
prms.TrainSet = 'ImageClef_train';
prms.Pool = 'mirflickr';
prms.TestSet = 'ImageClef_test';
prms.dataset_folder = './data/datasets/';

% paths where the images and tags are stored. By default they are
% expected to be in the data/datasets/ corresponding folders.
prms.train_img_folder = './data/datasets/imageclef/images/'; % folder where the training images are
prms.pool_img_folder = './data/datasets/mirflickr/images/'; % folder where the pool images are
prms.test_img_folder = './data/datasets/imageclef/images/'; % folder where the test images are
prms.train_tag_folder = './data/datasets/imageclef/train_metadata/tags/'; % folder where the training tags are
prms.pool_tag_folder = './data/datasets/mirflickr/tags/'; % folder where the pool tags are


% visual features params
prms.featpipe_folder = './'; % the folder where the /+featpipem/ and /models/ are (extracted from deepeval tool)
prms.CNN_pretrained_model = 'CNN_M_128'; % the pretrained model to use

% textual features params
prms.text_data_dir = './data/'; % the folder where the Vocabulary and PCA coeffs for the text analysis are
prms.NofDims = 7000; % # of dimensions to keep after PCA (COEFF var in PCAcoeffs file must have at least NofDims coefficients) 

% AL params
prms.N = 50; % #iterations
prms.pos = 50; % #positive instances selected in each iteration
prms.neg = 50; % #negative instances selected in each iteration



%% Load Datasets & Extract Features
VFE = ActiveLearner.util.VisualFeatureExtractor(); % initialize VFE 
VFE.model_dir = fullfile(prms.featpipe_folder, '/models/',prms.CNN_pretrained_model); % directory where the utilized CNN model is 
VFE.param_file = sprintf('%s/param.prototxt', VFE.model_dir); % file with parameters for the CNNs
VFE.model_file = sprintf('%s/model', VFE.model_dir); % the full path of the CNN model
VFE.average_image = fullfile(prms.featpipe_folder, '/models/mean.mat'); % file with mean image

TFE = ActiveLearner.util.TextualFeatureExtractor(); % initialize TFE
TFE.VocabFileName = fullfile(prms.text_data_dir, 'Vocabulary.mat'); % mat file with the word vocabulary (has a Vocabulary var)
TFE.PCAFileName = fullfile(prms.text_data_dir, 'PCA_coeffs.mat'); % mat file with the PCA coefficients (has a COEFF var)
TFE.NofDims = prms.NofDims;

% Training Dataset - Consists of images, tags and labels
load(fullfile(prms.dataset_folder,prms.TrainSet,'img_files.mat')); % load the image_files
VFE.img_files = img_files; % the filenames of the training images without extension (as a cell array)
VFE.infolder = prms.train_img_folder;
VFE.outfolder = fullfile(prms.dataset_folder,prms.TrainSet); % folder where the visual features will be saved

load(fullfile(prms.dataset_folder,prms.TrainSet,'tag_files.mat')); % load the tag_files
TFE.tag_files = tag_files; % the filenames of the tag files (text files with the list of tags) for training images (as a cell array)
TFE.infolder = prms.train_tag_folder;
TFE.outfolder = fullfile(prms.dataset_folder,prms.TrainSet); % folder where the textual features will be saved

trainDataset = ActiveLearner.util.Dataset(prms.TrainSet); % training set
trainDataset.bow = TFE.ExtractTextFeatures();
trainDataset.feats = VFE.ExtractCNNfeats();
load(fullfile(prms.dataset_folder,prms.TrainSet,'anns.mat')); % load the annotations
trainDataset.anns = anns; % multi-label annotations of training set
trainDataset.training_subset_file = fullfile(prms.dataset_folder,prms.TrainSet, 'image_ids.mat');
%trainDataset.training_subset_file = '/disk1/Results/IterativeBootstrapping/ImageClef2012/It_0/imageIDs.mat';
trainDataset.GetTrainingSubset();

% Pool of candidates dataset - no labels for this dataset
poolDataset = ActiveLearner.util.Dataset(prms.Pool); % pool of candidates

% change the filenames of the extractors, showing to the Pool folder
load(fullfile(prms.dataset_folder,prms.Pool,'img_files.mat')); % load the image_files
VFE.img_files = img_files;
VFE.infolder = prms.pool_img_folder;
VFE.outfolder= fullfile(prms.dataset_folder,prms.Pool); 
load(fullfile(prms.dataset_folder,prms.Pool,'tag_files.mat')); % load the tag_files
TFE.infolder = prms.pool_tag_folder;
TFE.tag_files = tag_files;
TFE.outfolder= fullfile(prms.dataset_folder,prms.Pool);

poolDataset.bow = TFE.ExtractTextFeatures();
poolDataset.feats = VFE.ExtractCNNfeats();

% Test set - no tag info for this dataset is required
testDataset = ActiveLearner.util.Dataset(prms.TestSet); % test set

% change the filenames of the extractors, showing to the Pool folder
load(fullfile(prms.dataset_folder,prms.TestSet,'img_files.mat')); % load the image_files
VFE.img_files = img_files; 
VFE.infolder = prms.test_img_folder;
VFE.outfolder= fullfile(prms.dataset_folder,prms.TestSet); 

testDataset.feats = VFE.ExtractCNNfeats(); 
load(fullfile(prms.dataset_folder,prms.TestSet,'anns.mat')); % load the annotations
testDataset.anns = anns; % multi-label annotations of testset

%% For every concept do AL
Nconcepts = size(testDataset.anns,2);
TextCls = ActiveLearner.Classifier.LIBSVMClassifier();
TextCls.b = 1;
% Set the selector and the fuser
IS = ActiveLearner.InstanceSelector.MaxSelector();
if strcmp(prms.Pool,'mirflickr') && strcmp(prms.TrainSet, 'ImageClef_train') % remove the 25k first images from pool that belong to Imageclef
    IS.oldIDs = (1:25000)';
end
Fuser = ActiveLearner.FusionStrategy.ProbabilisticFusion();
Eval = ActiveLearner.util.Evaluator();

for ci=1:Nconcepts
    %% Get the labels from multilabel annotations
    trainDataset.getLabelsFromAnns(ci);
    testDataset.getLabelsFromAnns(ci);

    %% Train Text based classifier
    disp(['Training Text based classifier for concept ' mat2str(ci) '...']);
    TextCls.train(trainDataset.bow,trainDataset.labels); % train text based classifier
    w(ci,:) = TextCls.w;
    b(ci,1) = TextCls.bias;
    models{ci,1} = TextCls.models;
    Oracle = ActiveLearner.Oracle.BoWOracle(TextCls);
    pst_pos = Oracle.GetOracleConfidence(poolDataset.bow); % get P(S|T) when dk = +1 (for positive examples)
    Oracle.class = -1;
    pst_neg = Oracle.GetOracleConfidence(poolDataset.bow); % get P(S|T) when dk = -1 (for negative examples)

    %% Get the subset for this concept
    subset = ActiveLearner.util.Dataset('subset');
    subset.feats = [trainDataset.feats(:,trainDataset.posIDs{ci}) trainDataset.feats(:,trainDataset.negIDs{ci})];
    subset.labels = [ones(numel(trainDataset.posIDs{ci}),1);-ones(numel(trainDataset.negIDs{ci}),1)];

    %% Train Baseline Classifier
    disp(['Training the initial classifier for concept ' mat2str(ci) '...']);
    VisCls{1} = ActiveLearner.Classifier.LIBSVMClassifier();
    VisCls{1}.cost = 10; % set the cost for CNN based features to 10
    VisCls{1} = VisCls{1}.train(subset.feats,subset.labels); % train
    [~,scores] = VisCls{1}.predict(testDataset.feats); % test
    Eval.results(ci,1) = Eval.evaluate(testDataset.labels,scores);


    %% Actively select new samples and retrain
    % Loop over iterations
    for i=2:prms.N
        % the inf criterion is initialized here since the classifier changes in
        % every iteration.
        disp(['Retraining the classifier for concept ' mat2str(ci) ' iteration ' mat2str(i) '...']);
        Inf = ActiveLearner.InformativenessCriterion.ActiveLearning(VisCls{i-1});
        psv = Inf.GetInformativeness(poolDataset.feats);
        Fuser.psv = psv;

        % Select positive samples
        Fuser.pst = pst_pos;
        IS.probs = Fuser.Fuse();
        selected_ids = IS.SelectSamples(); % This also adds the selected instances to the oldIDs so that they are not candidates again
        feats = poolDataset.getfeatsWithIndices(selected_ids);
        labs = ones(numel(selected_ids),1);
        subset = subset.addInstances(feats,labs);
        

        % Select negative samples
        Fuser.pst = pst_neg;
        IS.probs = Fuser.Fuse();
        selected_ids = IS.SelectSamples(); % This also adds the selected instances to the oldIDs so that they are not candidates again
        feats = poolDataset.getfeatsWithIndices(selected_ids);
        labs = -ones(numel(selected_ids),1);
        subset = subset.addInstances(feats,labs);

        % Retrain the model
        VisCls{i} = ActiveLearner.Classifier.LIBSVMClassifier();
        VisCls{i}.cost = 10; % set the cost for CNN based features to 10
        VisCls{i} = VisCls{i}.train(subset.feats,subset.labels); % train
        [~,scores] = VisCls{i}.predict(testDataset.feats); % test
        Eval.results(ci,i) = Eval.evaluate(testDataset.labels,scores);
    end
end

%% Show results
AP = Eval.results;
plot(mean(AP)','-+','LineWidth',3,'MarkerSize',8)


end

