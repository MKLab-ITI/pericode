classdef ActiveLearning < ActiveLearner.InformativenessCriterion.BaseCriterion
    %ACTIVELEARNING Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Classifier;
    end
    
    methods
        function obj = ActiveLearning(Classifier) % constructor
            obj.Classifier = Classifier; % link to the classifier. 
            % Note that the model must have been already trained
        end
        function Informativeness_values = GetInformativeness(obj,feats)
            [~,V] = obj.Classifier.predict(feats); % Get distances from the hyperplane
            obj.Informativeness_values = 1-abs(V); % Compute informativess based on [1]
            obj.Informativeness_values(abs(V)>1) = 0; % remove samples out of the margin
            Informativeness_values = obj.Informativeness_values;
        end
    end
    
end

