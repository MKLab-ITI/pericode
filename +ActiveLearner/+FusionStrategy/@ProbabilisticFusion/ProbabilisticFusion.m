classdef ProbabilisticFusion < ActiveLearner.FusionStrategy.BaseFusion
    %PROBABILISTICFUSION Implements the probabilistic fusion presented in
    %[1]
    %   psv = P(S|V), pst = P(S|T), psvt = P(S|V,T), ps = P(S)
    %   Implements equation xx from [1].
    
    properties
        ps;
    end
    
    methods
        function obj = ProbabilisticFusion(psv,pst) %constructor
            if nargin == 2
                obj.psv = psv;
                obj.pst = pst;
            elseif nargin ~= 0
                disp('Error: Wrong arguments for the ProbabilisticFusion class');
                exit;
            end
            obj.ps = 0.5;
        end
        
        function psvt = Fuse(obj)
            pt = obj.pst;
            pv = obj.psv;
            pv(pv>=0.99) = 0.99; % deal with probs on the limit of 1
            pt(pt>=1) = 0.99999999999999; % deal with probs on the limit of 1
            psvt = (1-obj.ps).*pv.*pt./(pv.*pt-obj.ps.*pv-obj.ps.*pt+obj.ps);
            obj.psvt = psvt;
        end
    end
    
end

