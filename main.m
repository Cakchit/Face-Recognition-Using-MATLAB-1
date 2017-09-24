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

 % Obtaining Fisher eigenvectors and eigenvalues
 [Vf, Df] = eig(Sb,Sw);

 % Calculating weights
  Df = fliplr(diag(Df));
  Vf = fliplr(Vf);

 % Calculating fisher weights
 Wf = Vf'*Wpca;

 % Support Vector Machine (LIBSCM Dependency)

 if classType == 1

     % testNum = 13*20;

     % Reshape the selected face
     Tr = reshape(face(:,:,testNum),[nRow*nCol 1]);
     Ar = Tr-mTot;

     % Obtain the weights of the normalized selected face
     Wrec = Vf'*U'*Ar;

     % SVM parameters. The kernel is a polynomial
     c=1e9;
     params=[' -t ' int2str(1) ' -c ' int2str(c)];

     % prevArray starts with an array containing each class. Winner array is the
     %classes that are selected in the binary tree
     prevArr = [1:k];
     winnerArr = [];

     for max = 1:1000
         winnerArr = [];
         for winRep = 1:2:length(prevArr)

             % Selects the two classes to train the SVM
             if winRep >= length(prevArr)
                 i = prevArr(winRep) ;
                 j = prevArr(winRep-1) ;
             else
                 i = prevArr(winRep)  ;
                 j = prevArr(winRep+1)  ;
             end

             % Selects the features of the 2 classes
             feature = [Wf(1:2,n*i-(n-1):n*i),Wf(1:2,n*j-(n-1):n*j)]';
