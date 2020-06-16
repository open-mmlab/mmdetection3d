% Copyright (c) Facebook, Inc. and its affiliates.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.

%% Dump SUNRGBD data to our format
% for each sample, we have RGB image, 2d boxes.
% point cloud (in camera coordinate), calibration and 3d boxes.
%
% Extract using V1 labels.
%
% Author: Charles R. Qi
%
clear; close all; clc;
addpath(genpath('.'))
addpath('../OFFICIAL_SUNRGBD/SUNRGBDtoolbox')
%% V1 2D&3D BB and Seg masks
load('../OFFICIAL_SUNRGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')
% load('./Metadata/SUNRGBD2Dseg.mat')

%% Create folders
det_label_folder = '../sunrgbd_trainval/label_v1/';
mkdir(det_label_folder);
%% Read
for imageId = 1:10335
    imageId
try
data = SUNRGBDMeta(imageId);
data.depthpath(1:16) = '';
data.depthpath = strcat('../OFFICIAL_SUNRGBD/SUNRGBD', data.depthpath);
data.rgbpath(1:16) = '';
data.rgbpath = strcat('../OFFICIAL_SUNRGBD/SUNRGBD', data.rgbpath);

% MAT files are 3x smaller than TXT files. In Python we can use
% scipy.io.loadmat('xxx.mat')['points3d_rgb'] to load the data.
mat_filename = strcat(num2str(imageId,'%06d'), '.mat');
txt_filename = strcat(num2str(imageId,'%06d'), '.txt');

% Write 2D and 3D box label
data2d = data;
fid = fopen(strcat(det_label_folder, txt_filename), 'w');
for j = 1:length(data.groundtruth3DBB)
    centroid = data.groundtruth3DBB(j).centroid;
    classname = data.groundtruth3DBB(j).classname;
    orientation = data.groundtruth3DBB(j).orientation;
    coeffs = abs(data.groundtruth3DBB(j).coeffs);
    box2d = data2d.groundtruth2DBB(j).gtBb2D;
    fprintf(fid, '%s %d %d %d %d %f %f %f %f %f %f %f %f\n', classname, box2d(1), box2d(2), box2d(3), box2d(4), centroid(1), centroid(2), centroid(3), coeffs(1), coeffs(2), coeffs(3), orientation(1), orientation(2));
end
fclose(fid);

catch
end

end
