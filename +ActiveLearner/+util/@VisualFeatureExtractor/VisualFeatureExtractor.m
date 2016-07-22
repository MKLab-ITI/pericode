classdef VisualFeatureExtractor < handle
    %VISUALFEATUREEXTRACTOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % Parameters for the folders and files of the images/features
        img_files; % the images to extract features from
        outfolder; % folder where the visual features will be saved
        infolder; % folder where the img_files exist (default './')
        extension; % extension of the images (default '.jpg')
        
        % Parameters for the CNN extractor 
        param_file; % file with parameters for the CNNs
        model_file; % file with the full path of the CNN model
        model_dir; % directory where the utilized CNN model is
        average_image; % file with mean image
    end
    
    methods
        function obj = VisualFeatureExtractor(img_files, outfolder)
            
            % Set default prms
            obj.model_dir = './models/CNN_M_128';
            obj.param_file = sprintf('%s/param.prototxt', obj.model_dir);
            obj.model_file = sprintf('%s/model', obj.model_dir);
            obj.average_image = './models/mean.mat';
            obj.infolder = './';
            obj.extension = '.jpg';
            if nargin > 0
                obj.img_files = img_files;
            end
            if nargin > 1
                obj.outfolder = outfolder;
            end
        end
        
        function [feats errImages] = ExtractCNNfeats(obj)
            featFile = fullfile(obj.outfolder,'Vis_feats.mat');
            if exist(featFile,'file')
                disp([featFile ' already exists... Loading features...']);
                load(featFile);
            else
                % initialize an instance of the ConvNet feature encoder class;
                encoder = featpipem.directencode.ConvNetEncoder(obj.param_file, obj.model_file, ...
                    obj.average_image, 'output_blob_name', 'fc7');
                errImages = [];
                for i=1:numel(obj.img_files)
                    imgName = [obj.img_files{i} obj.extension];
                    disp(['Extracting features for image ' imgName ' (' ...
                        mat2str(i) ' of ' mat2str(numel(obj.img_files)) ')']);
                    try
                        im = imread(fullfile(obj.infolder,imgName));
                        im = featpipem.utility.standardizeImage(im);
                        code = encoder.encode(im);
                        feats(:,i) = code;
                    catch err
                        errImages = [errImages;i];
                    end
                end
                save(featFile,'feats','errImages');
            end
        end
    end
    
end

