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
