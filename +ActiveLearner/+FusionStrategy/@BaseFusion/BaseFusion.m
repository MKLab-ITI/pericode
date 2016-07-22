classdef (Abstract) BaseFusion < handle
    %BASEFUSION Fuses the the two quantities psv and pst to one, psvt
    %   The function Fuse should be implemented so that give pst, psv as 2
    %   column vectors of equal size the function returns a fused quantity
    %   as a column vector psvt
    
    properties
        pst;
        psv;
        psvt;
    end
    
    methods (Abstract = true)
        psvt = Fuse(obj);
    end
    
end

