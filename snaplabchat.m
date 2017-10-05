function snaplabchat(imgpath, lens)
%SNAPLABCHAT A MATLAB version of SnapChat's lenses
%   This set of MATLAB functions detects facial features in an image and
%   uses the Direct Linear Transform algorithm to find a homography 
%   between two sets of corresponding keypoints. 
% 
%   imgpath - path to input image file
%   lens -  "moustache"
%   
%   This demo uses functions provided by the wonderful Peter Kovesi. More
%   code from Peter can be found on his website:
%   (http://www.peterkovesi.com/matlabfns/)
%   

% cleanup
clear; close all;

% handle input arguments
if nargin < 1
    imgpath = 'example.jpg';
    lens = 'moustache';
elseif nargin < 2
    lens = 'moustache';
end

% read in image
img = imread(imgpath);

% load lenses and their keypoints
load('lenses.mat');

switch lens
    case 'moustache'
        f = 1;
    otherwise
        error('Error: Filter type does not exist');
end

% find facial keypoints
kps = facialkps(img);

% apply lens to each face in the image
figure; 
subplot(1,2,1); imshow(img);
subplot(1,2,2); imshow(img);
for i = 1:length(kps)
    % match facial keypoints with the lens' keypoints
    H = homography2d([lenses.kps; ones(1, 7)], [kps{i}; ones(1,7)]);
    
    % construct MATLAB transform
    tform = projective2d(H');
    
    % plot current keypoints
    subplot(1,2,1);
    hold on;
    plot(kps{i}(1,1:4),kps{i}(2,1:4), 'ro'); % mouth
    plot(kps{i}(1,5),kps{i}(2,5), 'yo');     % nose
    plot(kps{i}(1,6:7),kps{i}(2,6:7), 'go'); % eyes
    hold off;
    
    % add lens to current face
    subplot(1,2,2);
    hold on;
    overlaylens(lenses.img{f}, tform, size(img(:,:,1)));
    hold off;
end

end

function kps = facialkps(img)
%FACIALKPS Finds faces and their seven facial keypoints.
%   This function finds seven facial keypoints (mouth, nose, and eyes) for 
%   each face in the input image. The function output is a cell array of
%   2x7 keypoint matrices (one for each face).

% create detectors
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
mouthDetector = vision.CascadeObjectDetector('Mouth', 'MergeThreshold',16, 'UseROI', true);
noseDetector = vision.CascadeObjectDetector('Nose', 'MergeThreshold', 16, 'UseROI', true);
eyesDetector = vision.CascadeObjectDetector('EyePairSmall', 'MergeThreshold', 16, 'UseROI', true);

% find faces
faces_bbox = step(faceDetector, img);

% find facial keypoints for each face
bbox2centroid = @(x) [x(1) + x(3)/2; x(2) + x(4)/2];
nfaces = size(faces_bbox, 1);
kps = cell(nfaces,1);
for i = 1:nfaces
    % grow face bbox
    scale = [.1, .75];
    face_bbox = faces_bbox(i,:);
    border = (face_bbox(3:4) .* scale) ./ 2;
    face_bbox = face_bbox + [-border, border];
    face_bbox(face_bbox < 1) = 1;
    
    % find face parts
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
    
    % convert from bbox to corners, centroids, or focii
    moutht_kps = bbox2centroid(mouth_bbox) - [0; mouth_bbox(4)/2];
    mouthb_kps = bbox2centroid(mouth_bbox) + [0; mouth_bbox(4)/2];
    mouthl_kps = bbox2centroid(mouth_bbox) - [mouth_bbox(4)/2; 0];
    mouthr_kps = bbox2centroid(mouth_bbox) + [mouth_bbox(4)/2; 0];
    nose_kps = bbox2centroid(nose_bbox);
    leftEye_kps = bbox2centroid(eyes_bbox) - [eyes_bbox(3)/4 ; 0];
    rightEye_kps = bbox2centroid(eyes_bbox) + [eyes_bbox(3)/4; 0];
    
    % construct keypoint matrix
    kps{i} = [moutht_kps, mouthb_kps, mouthl_kps, mouthr_kps, nose_kps, leftEye_kps, rightEye_kps];
end

% remove any empty cells
kps = kps(~cellfun('isempty', kps));

end

function overlaylens(lens, tform, s)
%OVERLAYLENS Warps and overlays a lens onto a plotted image
%    This function uses a given transform to warp a lens and overlay
%    it on an existing image.

    % warp the lens
    lens_warp = imwarp(lens, tform, 'OutputView', imref2d(s));
    
    % overlay
    h = imshow(lens_warp(:,:,1:3));
    set(h, 'AlphaData', squeeze(lens_warp(:,:,4)));
end