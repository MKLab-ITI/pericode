classdef RandomOracle  < ActiveLearner.Oracle.BaseOracle
    %RANDOMORACLE Random Oracle return 0.5 for every instance in feats
    %   If P(S|T)=0.5 (i.e. confidence_values here), the P(S|V,T)=P(S|V) if
    %   the probabilistic fusion is used. Use this to get the unimodal
    %   versions of this paper.
    
    properties
    end
    
    methods
        function obj = RandomOracle() % constructor
        end
        function confidence_values = GetOracleConfidence(obj,feats)
            Nsize = size(feats,1);
            obj.confidence_values = 0.5*ones(Nsize,1);
            confidence_values = obj.confidence_values;
        end
    end
    
end

