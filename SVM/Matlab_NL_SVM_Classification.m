%A-Vzer
function [] = Matlab_NL_SVM_Classification()
        
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
    
    
    
    %% fit the non-linear SVM (hint: use fitcsvm with a rbf kernelfunction)
    X = zeros(N_SAMPLES*2,2);
    y = zeros(1,N_SAMPLES*2);
    for i = 1:N_SAMPLES
        X(i,1) = samples_c1(1,i);
        X(i,2) = samples_c1(2,i);
        X(i+N_SAMPLES,1) = samples_c2(1,i);
        X(i+N_SAMPLES,2) = samples_c2(2,i);
        y(1,i) = 1;
        y(1,i+N_SAMPLES) = -1;
    end
    Mdl = fitcsvm(X,y,'KernelFunction','rbf');
            
    
    %% generate test samples
    [X Y]     = meshgrid(-15:0.25:10,-8:0.25:12);
    X         = X(:)';
    Y         = Y(:)';
    samples   = [X;Y];
    N_SAMPLES = size(samples,2);
    
    %% classify the samples 
    class = zeros(1,N_SAMPLES);
    %% Now use the predict function for convenience
    %%for i = 1:N_SAMPLES
    %%    samples_k(i,1) = samples(1,i);
    %%    samples_k(i,2) = samples(2,i);
    %%    samples_k(i,3) = exp(-(samples(1,i)*samples(2,i))^2); 
    %%end
    class = predict(Mdl,samples');
    

       
        
    %% show classified samples
    figure(2)
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



