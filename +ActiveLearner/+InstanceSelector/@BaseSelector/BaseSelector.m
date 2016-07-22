classdef (Abstract) BaseSelector < handle
    %BASESELECTOR Abstract class for instance selectors
    %   In order to use a different selector, the function SelectSamples
    %   must be implemented returning the ids of the selected samples
    %   corresponding to the Dataset ids, and the corresponding labels
    
    properties
        probs; % the scores based on hich instances will be selected
        nsamples; % number os samples to be selected
        selected_ids; % nsamplesx1 the ids of the nsamples selected samples; 
        % they refer to the Dataset feats
    end
    
    methods (Abstract = true)
        selected_ids = SelectSamples(obj); % The fucntion that selects new samples
    end
    
end

