classdef LIBSVMClassifier < ActiveLearner.Classifier.ClassifierBase
    %LIBSVMCLASSIFIER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        kernel; % The svm kernel
        cost; % The cost parameter
        gamma; % The gamma parameter (not required for linear kernel)
        %models; % The trained models
        b; % the probabilities flag (for Platt's sigmoid)
        w; % the normal vector
        bias; % the bias of the model
    end
    
    methods
        function obj = LIBSVMClassifier(kernel,cost,gamma,b)
            % Default prms
            obj.kernel = 'linear';
            obj.cost = 1.0;
            obj.gamma = 0.01;
            obj.b = 0;
            if nargin > 0
                obj.kernel = kernel;
            end
            if nargin > 1
                obj.cost = cost;
            end
            if nargin > 2
                obj.gamma = gamma;
            end
            if nargin > 3
                obj.b = b;
            end
        end
        
        function obj = train(obj, feats, labels)
            if numel(unique(labels))~=2
                disp('Error in classification, data from 2 classes are required');
                exit;
            end
            % Builds the classification model
            %disp('Computing kernel...');
            K = double(obj.computeKernel(feats')); %computekernel wants a matrix MxD, where D is the feature dimensionality
            %disp('Training...');
            obj.models = svmtrain(labels, [(1:size(K,1))', K+eye(size(K,1))*realmin],...
                sprintf(' -t 4 -c %f -b %d -q', obj.cost,obj.b));
            
            featdim = size(feats, 1);
            obj.bias = obj.models.rho;
            W = feats(:,full(obj.models.SVs)).*repmat(obj.models.sv_coef',[featdim 1]);
            obj.w = sum(W,2);
            if obj.models.Label(1) == -1
                obj.bias = -obj.bias;
                obj.w = -obj.w;
            end
        end
        
        function [labels, scores] = predict(obj,feats)
            %input = instance matrix rows = instances, cols = attributes
            %output = predicted class
            %probabilities = probability for predicted class
            %ranking = propabilities for all classes (e.g. to use with mAP)
            
            %TODO:should print an error if 'build' has not been called
            % [~,numinstance] = size(feats);
            %predict using the stored models
            scores = obj.w'*feats-repmat(obj.bias',1,size(feats,2));
            scores = scores';
            labels = -ones(size(scores));
            if obj.b==1
                scores = obj.Platt(scores);
                labels(scores>0.5) = 1;
            else
                labels(scores>0) = 1;
            end
        end
        
        function K = computeKernel(obj,instances)
            switch obj.kernel
                case 'linear'
                    K = instances*instances';
                case 'rbf'
                    dist = pdist2(instances,instances).^2;
                    K = exp(-obj.gamma.*dist);
                case 'chi'
                    m = size(instances,1);
                    n = size(instances,1);
                    mOnes = ones(1,m); D = zeros(m,n);
                    for i=1:n
                        yi = instances(i,:);  yiRep = yi( mOnes, : );
                        s = yiRep + instances;    d = yiRep - instances;
                        D(:,i) = sum( d.^2 ./ (s+eps), 2 );
                    end
                    D = D/2;
                    K = exp(-obj.gamma.*D);
                case 'xcorr'
                    K = zeros(size(instances,1));
                    if size(instances,2) < 500 % if memory allows it go for the vectorized version
                        a = xcorr(instances',maxlag,scaleopt);
                        c = reshape(a, 2*maxlag+1, size(instances,1), size(instances,1));
                        if size(c,1) == 1 % no max is required
                            K = squeeze(c);
                        else
                            K = squeeze(max(c));
                        end
                    else
                        for i=1:size(instances,1)
                            for j=i:size(instances,1)
                                K(i,j)=max(xcorr(instances(i,:),instances(j,:),maxlag,scaleopt));
                                K(j,i)=K(i,j);
                            end
                        end
                    end
                case {'spearman','correlation','cosine'}
                    dist = pdist2(instances,instances,obj.kernel);
                    K = 1-dist;
                case {'euclidean','seuclidean','mahalanobis'}
                    dist = pdist2(instances,instances,obj.kernel).^2;
                    K = exp(-obj.gamma.*dist);
                otherwise % if not one of the above, it can either be any value of distance in pdist2 or a function handle
                    dist = pdist2(instances,instances,obj.kernel);
                    K = exp(-obj.gamma.*dist);
            end
            
        end
        
        function scores = Platt(obj,scores)
            A = obj.models.ProbA(1);
            B = obj.models.ProbB(1);
            y = scores.*A+B;
            z=exp(-y)./(1+exp(-y));
            z(y<0) = 1./(1+exp(y(y<0)));
            scores = z;
        end
        
    end
    
end