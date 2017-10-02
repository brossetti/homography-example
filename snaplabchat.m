function snaplabchat(imgpath, flter)
%SNAPLABCHAT A MATLAB version of SnapChat's filters
%   This set of MATLAB functions detects facial features in an image and
%   uses RANSAC to match these points with keypoints on the selected
%   filter.
% 
%   imgpath - path to input image file
%   filter -  "moustache"
%             "tophat"

% handle input arguments
if nargin < 2
    flter = 'moustache';
end

% read in image
img = imread(imgpath);

% load filters and their keypoints
load('filter.m');

switch flter
    case 'moustache'
        f = 1;
    case 'tophat'
        f = 2;
    otherwise
        error('Error: Filter type does not exist');
end

% find facial keypoints
kps = facialkps(img);

% apply filter to each face in the image
for i = 1:length(kps)
    % match facial keypoints with the filter's keypoints
    params = matchkps(kps(i), flter(f).kps);
    
    % add filter to current face
    img = overlayfilter(img, params);
end

% display results
imshow(img)

end

function kps = facialkps(img)
%FACIALKPS Finds faces and their four facial keypoints.
%   This function finds four facial keypoints (left eye, right eye, nose,
%   and mouth) for each face in the input image. The function output is an
%   array of 4x2 keypoint matrices for each face.

% create detectors
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
mouthDetector = vision.CascadeObjectDetector('Mouth');
noseDetector = vision.CascadeObjectDetector('Nose');
leftEyeDetector = vision.CascadeObjectDetector('LeftEye');
rightEyeDetector = vision.CascadeObjectDetector('RightEye');

% find faces
bbox_faces = step(faceDetector, img);

% display detected faces (useful for debugging
img_faces = insertObjectAnnotation(img, 'rectangle', bbox_faces, 'Face');   
figure; imshow(img_faces); title('Detected Faces');

% find facial keypoints for each face
bbox2centroid = @(x) [x(1) + x(3)/2, x(2) + x(4)/2];
nfaces = size(bbox_faces, 1);
bbox = zeros(nfaces,1);
for i = 1:nfaces
    % find four facialy keypoits
    bbox_mouth = step(mouthDetector,img);
    bbox_nose = step(noseDetector,img);
    bbox_leftEye = step(leftEyeDetector,img);
    bbox_rightEye = step(rightEyeDetector,img);
    
    % convert from bbox to centroids
    
    
end

end

function matchkps(kps1, kps2)
%MATCHKPS

end

function tform = ransac(data,num,iter,threshDist,inlierRatio)
% RANSAC 
% data: a 2xn dataset with #n data points
% num: the minimum number of points. For line fitting problem, num=2
% iter: the number of iterations
% threshDist: the threshold of the distances between points and the fitting line
% inlierRatio: the threshold of the number of inliers 

    %% Plot the data points
    figure;plot(data(1,:),data(2,:),'o');hold on;
    number = size(data,2); % Total number of points
    bestInNum = 0; % Best fitting line with largest number of inliers
    bestParameter1=0;bestParameter2=0; % parameters for best fitting line
    for i=1:iter
    %% Randomly select 2 points
     idx = randperm(number,num); sample = data(:,idx);   
    %% Compute the distances between all points with the fitting line 
     kLine = sample(:,2)-sample(:,1);% two points relative distance
     kLineNorm = kLine/norm(kLine);
     normVector = [-kLineNorm(2),kLineNorm(1)];%Ax+By+C=0 A=-kLineNorm(2),B=kLineNorm(1)
     distance = normVector*(data - repmat(sample(:,1),1,number));
    %% Compute the inliers with distances smaller than the threshold
     inlierIdx = find(abs(distance)<=threshDist);
     inlierNum = length(inlierIdx);
    %% Update the number of inliers and fitting model if better model is found     
     if inlierNum>=round(inlierRatio*number) && inlierNum>bestInNum
         bestInNum = inlierNum;
         parameter1 = (sample(2,2)-sample(2,1))/(sample(1,2)-sample(1,1));
         parameter2 = sample(2,1)-parameter1*sample(1,1);
         bestParameter1=parameter1; bestParameter2=parameter2;
     end
    end
end

function img = overlayfilter(img, params)
    
end