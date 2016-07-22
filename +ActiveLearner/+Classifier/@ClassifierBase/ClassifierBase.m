
classdef (Abstract) ClassifierBase < handle
    
    properties
        models;
    end
    
    methods (Abstract = true)
        [labels, scores] = predict(obj, instances, labels);
        obj = train(obj, instances);
    end
    
end