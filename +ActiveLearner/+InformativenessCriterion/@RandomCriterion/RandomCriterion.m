classdef RandomCriterion < ActiveLearner.InformativenessCriterion.BaseCriterion
    %RANDOMSELECTOR Summary of this class goes here
    % Return 0.5 as a probability of selecting an image regardless of
    % feature values to accomodate random selection based on fusion tactics
    % Use this for unimodal selection based on P(S|T).
    
    properties
    end
    
    methods
        function obj = RandomCriterion() % constructor
        end
        function Informativeness_values = GetInformativeness(obj,feats)
            obj.Informativeness_values = 0.5*ones(size(feats,1),1); 
            Informativeness_values = obj.Informativeness_values;
        end
        
    end
    
end


