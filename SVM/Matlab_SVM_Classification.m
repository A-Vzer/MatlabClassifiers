%A-Vzer
function [] = Matlab_SVM_Classification()
        
    %% generate a random dataset
    [samples_c1 samples_c2] = generateDataset();
    N_SAMPLES = size(samples_c1,2);
         
    %% show samples
    figure(1)
    clf
    hold on
    plot(samples_c1(1,:),samples_c1(2,:),'ro')
    plot(samples_c2(1,:),samples_c2(2,:),'bo')   
    grid on
    axis equal
    drawnow


    %% change the sample parameterization
    new_samples_c1 = zeros(2,N_SAMPLES);
    new_samples_c2 = zeros(2,N_SAMPLES);
    %% radius 
    for i = 1:N_SAMPLES
        new_samples_c1(2,i) = (sqrt(samples_c1(1,i).^2 + samples_c1(2,i).^2));
        new_samples_c2(2,i) = (sqrt(samples_c2(1,i).^2 + samples_c2(2,i).^2));

    %% angle
        new_samples_c1(1,i) = (atan2(samples_c1(2,i),samples_c1(1,i)));
        new_samples_c2(1,i) = (atan2(samples_c2(2,i),samples_c2(1,i)));
    end
    disp('done')    
    %% show samples in new parametrization
    figure(2)
    clf
    hold on
    plot(new_samples_c1(1,:),new_samples_c1(2,:),'ro')
    plot(new_samples_c2(1,:),new_samples_c2(2,:),'bo')   
    grid on
    axis equal
    drawnow 

    
    
    %% fit the linear SVM (hint: use fitcsvm)
    X = horzcat(new_samples_c1,new_samples_c2);
    y = zeros(1,N_SAMPLES*2);
    for i = 1:N_SAMPLES
        y(1,i) = 1;
    end
    for k = N_SAMPLES+1:N_SAMPLES*2
        y(1,k) = -1;
    end
    
    SVMModel = fitcsvm(X',y);
        
    
    

    
    %% get beta and bias from the linear SVM model (hint: SVMModel.Beta and SVMModel.Bias)
    beta = SVMModel.Beta;
    bias = SVMModel.Bias;
    disp(size(beta))

    
    
    
    %% generate test samples
    [X Y]     = meshgrid(-15:0.25:10,-8:0.25:12);
    X         = X(:)';
    Y         = Y(:)';
    samples   = [X;Y];
    N_SAMPLES = size(samples,2);
    
    %% test samples to alternative parametrization
    new_samples = zeros(2,N_SAMPLES);
    

    for i = 1:N_SAMPLES
    %% radius 
        new_samples(2,i) = (sqrt(samples(1,i).^2 + samples(2,i).^2));

    %% angle
        new_samples(1,i) = (atan2(samples(2,i),samples(1,i)));
    end

    
    
    
    %% clasify test samples
    class = zeros(1,N_SAMPLES);    
    %% use class = sample'.beta + bias
    for i = 1:N_SAMPLES
        class(1,i) = dot(new_samples(:,i),beta) + bias;
    end
    
    
    %% show classified samples
    figure(3)
    clf
    hold on
    idx = find( class >= 0 );
    plot(samples(1,idx),samples(2,idx),'ro');
    idx = find( class < 0 );
    plot(samples(1,idx),samples(2,idx),'bo'); 
    grid on
    axis equal
    drawnow 
    
end



%% generate two class swirl dataset
function [samples_c1 samples_c2] = generateDataset()

    angles = -pi+0.75:0.01:pi-0.1;
    radius = 10/size(angles,2):10/size(angles,2):10;
    std    = 1/2.5;
    
    c1         = [cos(angles).*(radius+2); sin(angles).*(radius+2)];
    samples_c1 = [c1 + randn(2,size(c1,2))*std c1 + randn(2,size(c1,2))*std];
    
    c2         = [cos(angles).*(radius+4); sin(angles).*(radius+4)];
    samples_c2 = [c2 + randn(2,size(c2,2))*std c2 + randn(2,size(c2,2))*std];   
end



