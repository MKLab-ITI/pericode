classdef MaxSelector < ActiveLearner.InstanceSelector.BaseSelector
    %MAXSELECTOR This is a wrapper selector for the methods implemented in
    %[1]. Implements the abstract class BaseSelector.
    %   This class wraps the selection process of new samples based on the
    %   probability scores probs by returning the indexes of the max probs.
    %
    %   # oldIDs, errIDs and zeroIDs are ids of the Dataset that should be
    %   neglected.
    %   *NOTE*: CANNOT be both a logic vector of 0's and 1's
    %   MUST be a vector of indices.
    %
    %
    %   *NOTE* In order to get uni-modal selection set the InfCriterion or
    %   the Oracle at random and use the probabilisticFuser. Note that
    %   setting both to random does not equal to a completely random
    %   selector (a completely random selector has not been implemented).
    %
    %   [1] SALIC: Social Active Learning for Image Classification
    
    
    properties
        oldIDs; % ids selected in previous iterations
        errIDs; % ids of images for which feature extraction did not work
        zeroIDs; % ids of images that do not have any meaningful tags
    end
    
    methods
        % constructor
        function obj = MaxSelector(probs)
            if nargin > 0
                obj.probs = probs;
            end
            obj.nsamples = 50;
            obj.oldIDs = [];
            obj.errIDs = [];
            obj.zeroIDs = [];
        end
        
        function [selected_ids] = SelectSamples(obj)
            pr = obj.probs;
            pr(obj.oldIDs) = 0; % remove used images
            pr(obj.errIDs) = 0; % remove images that have no features
            pr(obj.zeroIDs) = 0; % remove images that have no tags
            [~,IX] = sort(pr,'descend');
            avail_samples = numel(pr)-sum(pr==0); % find available samples
            batchsize = min(obj.nsamples,avail_samples);
            if batchsize > 0 % if there are available samples
                selected_ids = IX(1:batchsize); % select new samples
            else % otherwise return empty arrays and display warning message
                selected_ids = [];
                disp('Warning: There are no available samples')
            end
            obj.selected_ids = selected_ids; % store info in the object
            
            % Finally add the selected_ids to the oldIDs
            obj.oldIDs = sort(unique([obj.oldIDs;obj.selected_ids]));
        end
    end
    
end