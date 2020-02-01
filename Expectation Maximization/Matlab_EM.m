% A-Vzer
function [] = Matlab_EM()

    %% how many clusters
    N_CLUSTERS = 5;
    
    %% how many EM iterations
    MAX_ITERS = 5;
    
    %% generate a random dataset
    samples    = generateDataset( N_CLUSTERS*500, N_CLUSTERS );
    N_SAMPLES  = size(samples,2);
         
    %% choose initial values
    %% mu    is the 2xN_CLUSTERS         vector containing the 2D means
    %% sigma is the 2x2xN_CLUSTERS       matrix containing the 2D covariance matrices
    %% prior is the 1xN_CLUSTERS         vector containing the cluster priors p(cluster)
    %% w     is the N_CLUSTERSxN_SAMPLES matrix containing the cluster membership probabilities for each sample p(cluster|sample)
    idx   = randperm(N_SAMPLES);
    mu    = samples(:,idx(1:N_CLUSTERS));
    sigma = repmat([10 0; 0 10],[1 1 N_CLUSTERS]);
    prior = ones( 1, N_CLUSTERS ) / N_CLUSTERS;
    w     = [ones(1,N_SAMPLES); zeros( N_CLUSTERS-1, N_SAMPLES)];
        
    %% show initial clusters
    clr = ['r','g','b','c','m','y'];
    clr = repmat( clr, 1, 1000 );
    figure(1)
    clf
    hold on
    plot(samples(1,:),samples(2,:),'ko')
    for nc = 1:N_CLUSTERS
        plotcov2( mu(:,nc), sigma(:,:,nc), 'Color', clr(nc), 'LineWidth', 3, 'conf', 0.75 );
    end   
    axis equal
    drawnow    
    
    %% start the EM loop
    for iter = 1:MAX_ITERS

        %% do Expectation step

            %% for the cluster membership probabilities
            c = 0;
            for z = 1:N_SAMPLES
                for i = 1:N_CLUSTERS  
                    c = c +(normalPDF(mu(:,i), sigma(:,:,i), samples(:,z)) *  prior(i)); 
                end
                for p = 1:N_CLUSTERS
                    w(p,z) = ((normalPDF(mu(:,p), sigma(:,:,p), samples(:,z)) *  prior(p)) / c );               
                end
            end

            
        %% do Maximization step
        
        
            %% for the cluster priors
            for i = 1:N_CLUSTERS
                w_i = 0;
                for k = 1: N_SAMPLES
                    w_i = w_i + w(i,k); 
                end
                prior(i) = (w_i / N_SAMPLES);
            end
            
        
            %% for the cluster means
            for i = 1:N_CLUSTERS
                w_i = 0;
                w_in = 0;
                for k = 1: N_SAMPLES
                    w_i = w_i + w(i,k); 
                    w_in = w_in + w(i,k)*samples(:,k);
                end       
            mu(:,i) = w_in/w_i;
            end
            %% for the cluster covariances
            for i = 1:N_CLUSTERS
                w_i = 0;
                w_ic = 0;
                for k = 1: N_SAMPLES
                    w_i = w_i + w(i,k); 
                    w_ic = w_ic + w(i,k)*( (samples(:,k)-mu(:,i)) * transpose(samples(:,k)-mu(:,i)));
                end
            sigma(:,:,i) = w_ic/w_i; 
            end
                
        %% assign samples to clusters
        [mx class] = max( w, [], 1 );
                
        %% show clusters
        figure(1)
        clf
        hold on        
        for nc = 1:N_CLUSTERS
            
            %% plot mean and cov
            plotcov2( mu(:,nc), sigma(:,:,nc), 'Color', clr(nc), 'LineWidth', 3, 'conf', 0.75 );
            
            %% plot samples
            idx = find( class == nc );
            plot(samples(1,idx),samples(2,idx),[clr(nc),'o'])
            
        end                   
        axis equal
        drawnow 
      disp(iter)
    end

end



%% compute probability of samples using a normal pdf
function [p] = normalPDF( mu, sigma, samples )
    p = mvnpdf( samples', mu', sigma ); 
end



%% generates random samples
function [samples] = generateDataset( nsamples, nclusters )

    %% generate random means
    mean = 40*(rand(2,nclusters)-0.5);
    
    %% generate normalized covariance
    covariance = 2*(rand(1,nclusters)-0.5);
    
    %% generate the standard deviations
    std = 10*rand(2,nclusters);
    
    %% generate the bias
    prior = rand(1,nclusters);
    prior = prior/sum(prior,2);
    prior = round(prior * nsamples); 
    
    %% for each cluster
    samples = [];
    for nc = 1:nclusters
     
        %% get the mean
        mu = mean(:,nc);
        
        %% construct the covariance matrix
        sigma = [std(1,nc) 0; 0 std(2,nc)] * [1 covariance(nc); covariance(nc) 1] * [std(1,nc) 0; 0 std(2,nc)];
        
        %% generate samples according to mu and sigma
        R = chol(sigma);
        z = repmat(mu,1,prior(nc)) + R*randn(2,prior(nc));
        
        %% add
        samples = [samples z];
       
    end
    
    
    %% construct the covariance matrices
    
    
   
end



% PLOTCOV2 - Plots a covariance ellipse with major and minor axes
%            for a bivariate Gaussian distribution.
%
% Usage:
%   h = plotcov2(mu, Sigma[, OPTIONS]);
% 
% Inputs:
%   mu    - a 2 x 1 vector giving the mean of the distribution.
%   Sigma - a 2 x 2 symmetric positive semi-definite matrix giving
%           the covariance of the distribution (or the zero matrix).
%
% Options:
%   'conf'    - a scalar between 0 and 1 giving the confidence
%               interval (i.e., the fraction of probability mass to
%               be enclosed by the ellipse); default is 0.9.
%   'num-pts' - the number of points to be used to plot the
%               ellipse; default is 100.
%
% This function also accepts options for PLOT.
%
% Outputs:
%   h     - a vector of figure handles to the ellipse boundary and
%           its major and minor axes
%
% See also: PLOTCOV3

% Copyright (C) 2002 Mark A. Paskin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = plotcov2(mu, Sigma, varargin)

    if size(Sigma) ~= [2 2], error('Sigma must be a 2 by 2 matrix'); end
    if length(mu) ~= 2, error('mu must be a 2 by 1 vector'); end

    [p, ...
     n, ...
     plot_opts] = process_options(varargin, 'conf', 0.9, ...
                        'num-pts', 100);
    h = [];
    holding = ishold;
    if (Sigma == zeros(2, 2))
      z = mu;
    else
      % Compute the Mahalanobis radius of the ellipsoid that encloses
      % the desired probability mass.
      k = chi2inv(p, 2);
      % The major and minor axes of the covariance ellipse are given by
      % the eigenvectors of the covariance matrix.  Their lengths (for
      % the ellipse with unit Mahalanobis radius) are given by the
      % square roots of the corresponding eigenvalues.
      if (issparse(Sigma))
        [V, D] = eigs(Sigma);
      else
        [V, D] = eig(Sigma);
      end
      % Compute the points on the surface of the ellipse.
      t = linspace(0, 2*pi, n);
      u = [cos(t); sin(t)];
      w = (k * V * sqrt(D)) * u;
      z = repmat(mu, [1 n]) + w;
      % Plot the major and minor axes.
      L = k * sqrt(diag(D));
%       h = plot([mu(1); mu(1) + L(1) * V(1, 1)], ...
%            [mu(2); mu(2) + L(1) * V(2, 1)], plot_opts{:});
%       hold on;
%       h = [h; plot([mu(1); mu(1) + L(2) * V(1, 2)], ...
%                [mu(2); mu(2) + L(2) * V(2, 2)], plot_opts{:})];
    end
    
    h = [h; plot(z(1, :), z(2, :), plot_opts{:})];
    if (~holding) hold off; end
    
end



function [varargout] = process_options(args, varargin)

    % Check the number of input arguments
    n = length(varargin);
    if (mod(n, 2))
      error('Each option must be a string/value pair.');
    end

    % Check the number of supplied output arguments
    if (nargout < (n / 2))
      error('Insufficient number of output arguments given');
    elseif (nargout == (n / 2))
      warn = 1;
      nout = n / 2;
    else
      warn = 0;
      nout = n / 2 + 1;
    end

    % Set outputs to be defaults
    varargout = cell(1, nout);
    for i=2:2:n
      varargout{i/2} = varargin{i};
    end

    % Now process all arguments
    nunused = 0;
    for i=1:2:length(args)
      found = 0;
      for j=1:2:n
        if strcmpi(args{i}, varargin{j})
          varargout{(j + 1)/2} = args{i + 1};
          found = 1;
          break;
        end
      end
      if (~found)
        if (warn)
          warning(sprintf('Option ''%s'' not used.', args{i}));
          args{i}
        else
          nunused = nunused + 1;
          unused{2 * nunused - 1} = args{i};
          unused{2 * nunused} = args{i + 1};
        end
      end
    end

    % Assign the unused arguments
    if (~warn)
      if (nunused)
        varargout{nout} = unused;
      else
        varargout{nout} = cell(0);
      end
    end
end

