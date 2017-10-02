function snaplabchat(imgpath, flter)
%SNAPLABCHAT A MATLAB version of SnapChat's filters
%   This set of MATLAB functions detects facial features in an image and
%   uses RANSAC to match these points with keypoints on the selected
%   filter.
% 
%   imgpath - path to input image file
%   filter -  "moustache"
%             "tophat"

close all;

% handle input arguments
if nargin < 1
    imgpath = 'example.jpg';
    flter = 'moustache';
elseif nargin < 2
    flter = 'moustache';
end

% read in image
img = imread(imgpath);

% load filters and their keypoints
load('filters.mat');

switch flter
    case 'moustache'
        f = 1;
    otherwise
        error('Error: Filter type does not exist');
end

% find facial keypoints
kps = facialkps(img);

% apply filter to each face in the image
for i = 1:length(kps)
    % match facial keypoints with the filter's keypoints
    tform = matchkps(kps{i}, filters.kps);
    
    % add filter to current face
    img = overlayfilter(img, params);
end

% display results
imshow(img)

end

function kps = facialkps(img)
%FACIALKPS Finds faces and their four facial keypoints.
%   This function finds four facial keypoints (mouth, nose, left eye, and
%   right eye) for each face in the input image. The function output is a
%   cell array of 4x2 keypoint matrices for each face.

% create detectors
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
mouthDetector = vision.CascadeObjectDetector('Mouth', 'MergeThreshold',16, 'UseROI', true);
noseDetector = vision.CascadeObjectDetector('Nose', 'MergeThreshold', 16, 'UseROI', true);
eyesDetector = vision.CascadeObjectDetector('EyePairSmall', 'MergeThreshold', 16, 'UseROI', true);

% find faces
faces_bbox = step(faceDetector, img);

% find facial keypoints for each face
bbox2centroid = @(x) [x(1) + x(3)/2, x(2) + x(4)/2];
nfaces = size(faces_bbox, 1);
kps = cell(nfaces,1);
img_all = img;
for i = 1:nfaces
    % grow face bbox
    scale = [.1, .75];
    face_bbox = faces_bbox(i,:);
    border = (face_bbox(3:4) .* scale) ./ 2;
    face_bbox = face_bbox + [-border, border];
    face_bbox(face_bbox < 1) = 1;
    
    % find facialy keypoints
    mouth_bbox = step(mouthDetector,img, face_bbox);
    nose_bbox = step(noseDetector,img, face_bbox);
    eyes_bbox = step(eyesDetector,img, face_bbox);
    
    % skip to next face if we didn't find all the features
    if isempty(mouth_bbox) || isempty(nose_bbox) || isempty(eyes_bbox)
        continue;
    end
    
    % remove eyes detected as mouths
    mouth_bbox = sortrows(mouth_bbox, -2);
    mouth_bbox = mouth_bbox(1,:);
    
    % display detected features (useful for debugging)
    img_all = insertObjectAnnotation(img_all, 'rectangle', face_bbox, 'face');   
    img_all = insertObjectAnnotation(img_all, 'rectangle', mouth_bbox, 'mouth'); 
    img_all = insertObjectAnnotation(img_all, 'rectangle', nose_bbox, 'nose');  
    img_all = insertObjectAnnotation(img_all, 'rectangle', eyes_bbox, 'eyes');  
    
    % convert from bbox to centroids
    mouth_kps = bbox2centroid(mouth_bbox);
    nose_kps = bbox2centroid(nose_bbox);
    leftEye_kps = bbox2centroid(eyes_bbox) - [eyes_bbox(3)/4 , 0];
    rightEye_kps = bbox2centroid(eyes_bbox) + [eyes_bbox(3)/4, 0];
    
    % construct keypoint matrix
    kps{i} = [mouth_kps; nose_kps; leftEye_kps; rightEye_kps];
end

% remove any empty cells
kps = kps(~cellfun('isempty', kps));

% plot if in debug mode
figure; imshow(img_all); title('Detected Faces');

end

function tform = matchkps(kps1, kps2)
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