classdef TextualFeatureExtractor < handle
    %TEXTUALFEATUREEXTRACTOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        tag_files;
        Vocabulary;
        COEFF;
        NofDims;
        method;
        infolder; % folder where the tag_files exist (default './')
        outfolder; % folder where the textual features will be saved
        VocabFileName; % file with the word vocabulary
        PCAFileName; % file with the PCA coefficients
        extension; % the extension to add at the end of the tag files, default .txt
        q; % set to 1 for minimum printing
    end
    
    methods
        function obj = TextualFeatureExtractor(tag_files, outfolder)
            % Set default prms
            obj.VocabFileName = fullfile('./data/Vocabulary.mat');
            obj.PCAFileName = fullfile('./data/PCA_coeffs.mat');
            obj.NofDims = 7000;
            obj.infolder = './';
            obj.extension = '.txt'; % by default txt files are expected

            if nargin > 0
                obj.tag_files = tag_files;
            end
            if nargin > 1
                obj.outfolder= outfolder;
            end
        end
        function feats = ExtractTextFeatures(obj)
            if ~exist(obj.outfolder,'dir')
                mkdir(obj.outfolder);
            end
            featFile = fullfile(obj.outfolder,'Text_feats.mat');
            if ~exist(featFile,'file') % if feats have not been calculated
                load(obj.VocabFileName); % load the Vocabulary
                obj.Vocabulary = Vocabulary;
                load(obj.PCAFileName); % load the PCA coefficients
                obj.COEFF = COEFF(:,1:obj.NofDims);
                feats = single(zeros(obj.NofDims,numel(obj.tag_files)));
                parfor i=1:numel(obj.tag_files)
                    disp(['Extracting textual features for image ' mat2str(i) ...
                        ' out of ' mat2str(numel(obj.tag_files))]);
                    path_name = fullfile(obj.infolder,[obj.tag_files{i} obj.extension]);
                    [~, ~, ext] = fileparts(path_name);
                    switch ext
                        case '.mat'
                            a = load(path_name); % load the tags from mat file
                            Tags = a.Tags;
                        case '.txt'
                            Tags = textread(path_name,'%s'); % load the tags from txt file
                        otherwise
                            disp('Only txt and mat files are supported for tags');
                    end
                    if isempty(Tags)
                        feats(:,i) = zeros(obj.NofDims,1);
                    else
                        ids = ismember(obj.Vocabulary,Tags);
                        feats(:,i) = single(ids'*obj.COEFF)';
                        x = norm(feats(:,i)) ;
                        if x~=0
                            feats(:,i)=feats(:,i)/x;
                        end
                    end
                end
                save(featFile,'feats','-v7.3');
            else
                disp([featFile ' already exists... Loading features...']);
                load(featFile);
            end
        end
        
    end
end
