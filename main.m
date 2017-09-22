% Author: Arjun Nataraj

close all
clear
clc

dataset = input('Select your dataset: \n(1) simple.m \n(2) pose.m\n');

if dataset == 1
     load '../data/data.mat'
     %Define variables
     k = 200;% Number of classes
     n = 3;% Number of images per class
elseif dataset == 2
     load '../pose/pose.mat'
     face = pose;
    %Define variables
     k = 68;% Number of classes
     n = 13;% Number of images per class
else
    display('Input not valid');
end

testNum = input('Enter the number of the face for which you want to find the match: ');
classType = input('Choose (1) for SVM or (2) for Euclidean Distance: ');
[nRow nCol M] = size(face);
T = reshape(face,[nRow*nCol M]);
mTot = mean(T,2);
A = T-repmat(mTot,1,M);
[V,D] = eig(A'*A);
eval = diag(D);

peval = [];
pevec = [];

for i = M:-1:k+1
    peval = [peval eval(i)];
    pevec = [pevec V(:,i)];
end

% Obtaining the eigenvectors
U = A * pevec;

% Obtaining PCA weights
 Wpca = U'*A;

 % Obtaining Sb and Sw
 cMean = zeros(M-k,M-k);
 Sb = zeros(M-k,M-k);
 Sw = zeros(M-k,M-k);

 pcaMean = mean(Wpca,2);

 for i = 1:k
     cMean = mean(Wpca(:,n*i-(n-1):n*i),2);
     Sb = Sb + (cMean-pcaMean)*(cMean-pcaMean)';
 end

 Sb = n*Sb;

 for i = 1:k
     cMean = mean(Wpca(:,n*i-(n-1):n*i),2);
     for j = n*i-(n-1):n*i
          Sw = Sw + (Wpca(:,j)-cMean)*(Wpca(:,j)-cMean)';
     end
 end
