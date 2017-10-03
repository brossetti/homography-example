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
figure; subplot(1,2,1); imshow(img);
img_overlay = img;
for i = 1:length(kps)
    % match facial keypoints with the filter's keypoints
    tform = homography(kps{i}, filters.kps);
    
    % plot current keypoints
    hold on;
    plot(kps{i}(1,1),kps{i}(1,2), 'ro');     % mouth
    plot(kps{i}(2,1),kps{i}(2,2), 'yo');     % nose
    plot(kps{i}(3:4,1),kps{i}(3:4,2), 'go'); % eyes
    hold off;
    
    % add filter to current face
    img_overlay = overlayfilter(img_overlay, filters.img{f}, tform);
end

% display results
subplot(1,2,2);
imshow(img_overlay);

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
    
    % keep lowest nose
    nose_bbox = sortrows(nose_bbox, -2);
    nose_bbox = nose_bbox(1,:);
    
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

end

function tform = homography(kps1, kps2)
%HOMOGRAPHY Finds the homography between two corresponding point sets
%    This function find the homographic matrix, H, 

% Solve equations using SVD
x = kps1(:,1)'; 
y = kps1(:,2)'; 
X = kps2(:,1)'; 
Y = kps2(:,2)';
rows0 = zeros(3, 4);
rowsXY = -[X; Y; ones(1,4)];
hx = [rowsXY; rows0; x.*X; x.*Y; x];
hy = [rows0; rowsXY; y.*X; y.*Y; y];
h = [hx hy];

[U, ~, ~] = svd(h);

H = (reshape(U(:,9), 3, 3)).';

tform = projective2d(H');

end

function img_overlay = overlayfilter(img, flter, tform)
    flter_warp = imwarp(flter, tform, 'OutputView', imref2d(size(img(:,:,1))));
    img_overlay = imfuse(img, flter_warp, 'diff');
end