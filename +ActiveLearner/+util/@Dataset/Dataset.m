classdef Dataset < handle
    %DATASET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Dataset_name; % name of the Dataset
        feats; % DxM matrix, where D is the feature dimensionality
        % (M D-dimensional columns of features)
        labels; % a vector of Mx1 labels (+1 or -1)
        bow; % textual bow features, DxM matrix, where D is the feature
        % dimensionality (M D-dimensional columns of features)
        num_concepts; %the number of concepts
        concepts; % a cell matrix num_conceptsx1, where in each cell there
        % is a string corresponding to each concept
        anns; % a Mxnum_concepts matrix, which at (i,j) has the value of +1
        % if the image i is positive for the concept j and -1 otherwise
        posIDs; % the ids of the positive images for training the initial classifier (cell array with a vector for each concept)
        negIDs; % the ids of the negative images for training the initial classifier (cell array with a vector for each concept)
        training_subset_file; % the mat file containing the posIDs and negIDs
    end
    
    methods
        function obj = Dataset(Dataset_name,feats,labels)
            if nargin == 0
                error('Dataset cannot be initialized without a Dataset name')
            end
            obj.Dataset_name = Dataset_name;
            if nargin == 2
                obj.feats = feats;
            elseif nargin == 3
                obj.labels = labels;
            end
        end
        
        function feats = getfeatsWithIndices(obj, idx)
            % get instances of specific indices
            feats = obj.feats(:,idx);
        end
        function bow = getbowWithIndices(obj, idx)
            % get instances of specific indices
            bow = obj.bow(:,idx);
        end
        function labels = getlabelsWithIndices(obj, idx)
            % get instances of specific indices
            labels = obj.labels(:,idx);
        end
        
        function obj = getLabelsFromAnns(obj,ci)
            obj.labels = obj.anns(:,ci);
        end
        function obj = getLabels4Concept(obj,ci)
            obj.labels = obj.anns(:,ci);
        end
        
        function [numLabels numInstances numFeatures] = getInfo(obj)
            % get the number of labels
            numLabels = numel(unique(obj.labels));
            % get the number of instances and features
            [numFeatures numInstances] = size(obj.feats);
            if numFeatures==0 || numInstances==0 % if the visual features have not been set yet
                [numFeatures numInstances] = size(obj.bow);
            end
            if numFeatures==0 || numInstances==0 % if neither textual features have been set yet
                fprintf('Warning: Dataset %s does not have visual or textual features yet',obj.Dataset_name);
            end
        end
        
        function obj = removeInstancesWithIndices(obj, idx)
            % remove instances with specific indices. A new Dataset
            % object is returned by this functioned without the specified
            % instances
            obj.feats(:,idx) = [];
            obj.bow(:,idx) = [];
            obj.labels(idx) = [];
            obj.anns(idx,:)=[];
        end
        function obj = addInstances(obj,feats,labs)
            obj.feats = [obj.feats feats];
            obj.labels = [obj.labels;labs];
        end
        function obj = GetTrainingSubset(obj)
            if exist(obj.training_subset_file,'file')
                disp([obj.training_subset_file ' already exists... Loading features...']);
                load(obj.training_subset_file);
                obj.posIDs = posIDs;
                obj.negIDs = negIDs;
            else
                for ci=1:size(obj.anns,2)
                    pos = find(obj.anns(:,ci) == 1);
                    avail_images = min(numel(pos),100);
                    obj.posIDs{ci} = pos(randperm(numel(pos),avail_images));
                    avail_images = min(numel(pos),100);
                    neg = find(obj.anns(:,ci) == -1);
                    obj.negIDs{ci} = neg(randperm(numel(neg),avail_images));
                end
                posIDs = obj.posIDs;
                negIDs = obj.negIDs;
                save(obj.training_subset_file,'posIDs','negIDs');
            end
        end
        
    end
    
end

