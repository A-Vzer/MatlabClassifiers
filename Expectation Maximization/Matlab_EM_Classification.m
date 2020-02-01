%A-Vzer
function [] = Matlab_EM_Classification()
    %% !! MORE THAN 10 ITERATIONS CAUSES AN ERROR (SIGMA must be a square, symmetric, positive definite matrix.)
    %% !! HAS PROBABLY TO DO WITH NUMERICAL EVALUATION OF MATLAB.
    %% !! HIGHER ITERATIONS (LIKE 30) SHOULD PERFROM WAY BETTER 
    
    %% how many EM iterations
    MAX_ITERS = 10;
    
    %% how many clusters per class
    N_CLUSTERS = 30;
    
    %% generate a random dataset
    [samples_c1 samples_c2] = generateDataset();
    N_SAMPLES  = size(samples_c1,2);
         
    %% choose initial values
    %% mu    is the 2xN_CLUSTERS         vector containing the 2D means
    %% sigma is the 2x2xN_CLUSTERS       matrix containing the 2D covariance matrices
    %% prior is the 1xN_CLUSTERS         vector containing the cluster priors p(cluster)
    %% w     is the N_CLUSTERSxN_SAMPLES matrix containing the cluster membership probabilities for each sample p(sample|cluster)p(cluster)
    idx      = randperm(N_SAMPLES);
    
    %% for class 1
    mu_c1    = samples_c1(:,idx(1:N_CLUSTERS));
    sigma_c1 = repmat([2 0; 0 2],[1 1 N_CLUSTERS]);
    prior_c1 = ones( 1, N_CLUSTERS ) / N_CLUSTERS;
    w_c1     = [ones(1,N_SAMPLES); zeros( N_CLUSTERS-1, N_SAMPLES)];
     
    %% for class 2
    mu_c2    = samples_c2(:,idx(1:N_CLUSTERS));
    sigma_c2 = repmat([2 0; 0 2],[1 1 N_CLUSTERS]);
    prior_c2 = ones( 1, N_CLUSTERS ) / N_CLUSTERS;
    w_c2     = [ones(1,N_SAMPLES); zeros( N_CLUSTERS-1, N_SAMPLES)];
    
    
    %% show initial clusters for both classes
    figure(1)
    clf
    hold on
    plot(samples_c1(1,:),samples_c1(2,:),'ro')
    plot(samples_c2(1,:),samples_c2(2,:),'bo')
    for nc = 1:N_CLUSTERS
        plotcov2( mu_c1(:,nc), sigma_c1(:,:,nc), 'Color', 'r', 'LineWidth', 3, 'conf', 0.75 );
        plotcov2( mu_c2(:,nc), sigma_c2(:,:,nc), 'Color', 'b', 'LineWidth', 3, 'conf', 0.75 );
    end   
    grid on
    axis equal
    drawnow 
    pause(1)    
    
    %% start the EM loop
    for iter = 1:MAX_ITERS

        %% do Expectation step

            %% for the cluster membership probabilities
                %% for class 1
                c = 0;
                for z = 1:N_SAMPLES
                    for i = 1:10  
                        c = c +(normalPDF(mu_c1(:,i), sigma_c1(:,:,i), samples_c1(:,z)) *  prior_c1(i)); 
                    end
                    for p = 1:N_CLUSTERS
                        w_c1(p,z) = ((normalPDF(mu_c1(:,p), sigma_c1(:,:,p), samples_c1(:,z)) *  prior_c1(p)) / c );               
                    end
                end
                
                %% for class 2
                c = 0;
                for z = 1:N_SAMPLES
                    for i = 1:10  
                        c = c +(normalPDF(mu_c2(:,i), sigma_c2(:,:,i), samples_c2(:,z)) *  prior_c2(i)); 
                    end
                    for p = 1:N_CLUSTERS
                        w_c2(p,z) = ((normalPDF(mu_c2(:,p), sigma_c2(:,:,p), samples_c2(:,z)) *  prior_c2(p)) / c );               
                    end
                end
                
        
            
            
        %% do Maximization step
        
        
            %% for the cluster priors
                %% for class 1
                for i = 1:N_CLUSTERS
                    w_i = 0;
                    for k = 1: N_SAMPLES
                        w_i = w_i + w_c1(i,k); 
                    end
                    prior_c1(i) = (w_i / N_SAMPLES);
                end
                %% for class 2
                for i = 1:N_CLUSTERS
                    w_i = 0;
                    for k = 1: N_SAMPLES
                        w_i = w_i + w_c2(i,k); 
                    end
                    prior_c2(i) = (w_i / N_SAMPLES);
                end
        
            %% for the cluster means
                %% for class 1
                for i = 1:N_CLUSTERS
                    w_i = 0;
                    w_in = 0;
                    for k = 1: N_SAMPLES
                        w_i = w_i + w_c1(i,k); 
                        w_in = w_in + w_c1(i,k)*samples_c1(:,k);
                    end       
                mu_c1(:,i) = w_in/w_i;
                end
                %% for class 2
                for i = 1:N_CLUSTERS
                    w_i = 0;
                    w_in = 0;
                    for k = 1: N_SAMPLES
                        w_i = w_i + w_c2(i,k); 
                        w_in = w_in + w_c2(i,k)*samples_c2(:,k);
                    end       
                mu_c2(:,i) = w_in/w_i;
                end
        
            %% for the cluster covariances
                %% for class 1
                for i = 1:N_CLUSTERS
                    w_i = 0;
                    w_ic = 0;
                    for k = 1: N_SAMPLES
                        w_i = w_i + w_c1(i,k); 
                        w_ic = w_ic + w_c1(i,k)*( (samples_c1(:,k)-mu_c1(:,i)) * transpose(samples_c1(:,k)-mu_c1(:,i)));
                    end
                sigma_c1(:,:,i) = w_ic/w_i;  
                end
                %% for class 2
                for i = 1:N_CLUSTERS
                    w_i = 0;
                    w_ic = 0;
                    for k = 1: N_SAMPLES
                        w_i = w_i + w_c2(i,k); 
                        w_ic = w_ic + w_c2(i,k)*( (samples_c2(:,k)-mu_c2(:,i)) * transpose(samples_c2(:,k)-mu_c2(:,i)));
                    end
                sigma_c2(:,:,i) = w_ic/w_i;  
                end
                
        %% show clusters for both classes
        figure(1)
        clf
        hold on
        plot(samples_c1(1,:),samples_c1(2,:),'ro')
        plot(samples_c2(1,:),samples_c2(2,:),'bo')
        for nc = 1:N_CLUSTERS
            plotcov2( mu_c1(:,nc), sigma_c1(:,:,nc), 'Color', 'r', 'LineWidth', 3, 'conf', 0.75 );
            plotcov2( mu_c2(:,nc), sigma_c2(:,:,nc), 'Color', 'b', 'LineWidth', 3, 'conf', 0.75 );
        end   
        grid on
        axis equal
        drawnow 
        pause(1) 
      
    end

    %% generate test samples
    [X Y]     = meshgrid(-15:0.25:10,-8:0.25:12);
    X         = X(:)';
    Y         = Y(:)';
    samples   = [X;Y];
    N_SAMPLES = size(samples,2);
    
    
    %% compute probability of each sample wrt each cluster of each class
    w_c1 = zeros( N_CLUSTERS, N_SAMPLES);
    w_c2 = zeros( N_CLUSTERS, N_SAMPLES);
                %% for class 1
                c = 0;
                for z = 1:N_SAMPLES
                    for i = 1:N_CLUSTERS  
                        c = c +(normalPDF(mu_c1(:,i), sigma_c1(:,:,i), samples(:,z)) *  prior_c1(i)); 
                    end
                    for p = 1:N_CLUSTERS
                        w_c1(p,z) = ((normalPDF(mu_c1(:,p), sigma_c1(:,:,p), samples(:,z)) *  prior_c1(p)) / c );               
                    end
                end
                
                %% for class 2
                c = 0;
                for z = 1:N_SAMPLES
                    for i = 1:N_CLUSTERS  
                        c = c +(normalPDF(mu_c2(:,i), sigma_c2(:,:,i), samples(:,z)) *  prior_c2(i)); 
                    end
                    for p = 1:N_CLUSTERS
                        w_c2(p,z) = ((normalPDF(mu_c2(:,p), sigma_c2(:,:,p), samples(:,z)) *  prior_c2(p)) / c );               
                    end
                end
                
    %% do joint normalization (over all clusters of all classes)

        
    %% classify samples
    mx_c1      = sum( w_c1, 1 );
    mx_c2      = sum( w_c2, 1 ); 
    [mx class] = max( [mx_c1; mx_c2], [], 1 );
    
    
    %% show test samples
    figure(2)
    clf
    hold on
    idx = find( class == 1 );
    plot(samples(1,idx),samples(2,idx),'ro');
    idx = find( class == 2 );
    plot(samples(1,idx),samples(2,idx),'bo');
    for nc = 1:N_CLUSTERS
        plotcov2( mu_c1(:,nc), sigma_c1(:,:,nc), 'Color', 'r', 'LineWidth', 3, 'conf', 0.75 );
        plotcov2( mu_c2(:,nc), sigma_c2(:,:,nc), 'Color', 'b', 'LineWidth', 3, 'conf', 0.75 );
    end
    grid on
    axis equal
    drawnow 
    pause(1) 
        
end



%% compute probability of samples using a normal pdf
function [p] = normalPDF( mu, sigma, samples )
    p = mvnpdf( samples', mu', sigma ); 
end



%% generate two class swirl dataset
function [samples_c1 samples_c2] = generateDataset()

    angles = -pi:0.01:pi;
    radius = 10/size(angles,2):10/size(angles,2):10;
    
    c1         = [cos(angles).*radius; sin(angles).*radius];
    samples_c1 =  c1 + randn(2,size(c1,2))/2;
    samples_c1 = [ samples_c1 c1 + randn(2,size(c1,2))/2];
    samples_c1 = [ samples_c1 c1 + randn(2,size(c1,2))/2];
    
    c2         = [cos(angles).*(radius+2); sin(angles).*(radius+2)];
    samples_c2 =  c2 + randn(2,size(c2,2))/2;
    samples_c2 = [ samples_c2 c2 + randn(2,size(c2,2))/2];
    samples_c2 = [ samples_c2 c2 + randn(2,size(c2,2))/2];
    
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

