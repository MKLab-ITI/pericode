classdef BoWOracle < ActiveLearner.Oracle.BaseOracle
    %BOWORACLE Implements the bag-of-words based oracle
    %   The class requires as input the text based "Classifier" built prior
    %   and the "class" for which to return the confidence values. Assumes
    %   that binary classificaction is applied and "class" can take as
    %   values integers:
    %   "1" for positive class
    %   "-1" for negative class
    %
    %   The function GetOracleConfidence returns the variable
    %   confidence_values, which are:
    %   probabilistic if Classifier.b = 1 and
    %   distances from the hyperplane otherwise
    
    properties
        Classifier; % type = ClassifierBase
        class; % the class for which to return confidence values. 
    end
    
    methods
        function obj = BoWOracle(Classifier) % constructor
            obj.Classifier = Classifier; % link to the classifier. 
            % Note that the model must have been already trained
        end
        function confidence_values = GetOracleConfidence(obj,feats)
            [~,obj.confidence_values] = obj.Classifier.predict(feats);
            if obj.class == -1 % if scores for negative class
                if obj.Classifier.b == 1 % if probabilistic return the complement
                    obj.confidence_values = 1-obj.confidence_values;
                else % else return the additive inverse
                    obj.confidence_values = -obj.confidence_values;
                end
            end
            confidence_values = obj.confidence_values;
        end
    end
    
end

