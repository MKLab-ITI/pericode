classdef (Abstract) BaseCriterion < handle
    %BASECRITERION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Informativeness_values;
    end
    
    methods(Abstract = true)
        Informativeness_values = GetInformativeness(obj,feats);
    end
    
end

