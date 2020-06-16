% Copyright (c) Facebook, Inc. and its affiliates.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.

%% Dump SUNRGBD data to our format
% for each sample, we have RGB image, 2d boxes.
% point cloud (in camera coordinate), calibration and 3d boxes.
%
% Compared to extract_rgbd_data.m in frustum_pointents, use v2 2D and 3D
% bboxes.
%
% Author: Charles R. Qi
%
clear; close all; clc;
addpath(genpath('.'))
addpath('../OFFICIAL_SUNRGBD/SUNRGBDtoolbox/readData')
%% V1 2D&3D BB and Seg masks
% load('./Metadata/SUNRGBDMeta.mat')
% load('./Metadata/SUNRGBD2Dseg.mat')

%% V2 3DBB annotations (overwrites SUNRGBDMeta)
load('../OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat');
load('../OFFICIAL_SUNRGBD/SUNRGBDMeta2DBB_v2.mat');
%% Create folders
depth_folder = '../sunrgbd_trainval/depth/';
image_folder = '../sunrgbd_trainval/image/';
calib_folder = '../sunrgbd_trainval/calib/';
det_label_folder = '../sunrgbd_trainval/label/';
seg_label_folder = '../sunrgbd_trainval/seg_label/';
mkdir(depth_folder);
mkdir(image_folder);
mkdir(calib_folder);
mkdir(det_label_folder);
mkdir(seg_label_folder);
%% Read
parfor imageId = 1:10335
    imageId
try
data = SUNRGBDMeta(imageId);
data.depthpath(1:16) = '';
data.depthpath = strcat('../OFFICIAL_SUNRGBD', data.depthpath);
data.rgbpath(1:16) = '';
data.rgbpath = strcat('../OFFICIAL_SUNRGBD', data.rgbpath);

% Write point cloud in depth map
[rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
rgb(isnan(points3d(:,1)),:) = [];
points3d(isnan(points3d(:,1)),:) = [];
points3d_rgb = [points3d, rgb];

% MAT files are 3x smaller than TXT files. In Python we can use
% scipy.io.loadmat('xxx.mat')['points3d_rgb'] to load the data.
mat_filename = strcat(num2str(imageId,'%06d'), '.mat');
txt_filename = strcat(num2str(imageId,'%06d'), '.txt');
parsave(strcat(depth_folder, mat_filename), points3d_rgb);

% Write images
copyfile(data.rgbpath, sprintf('%s/%06d.jpg', image_folder, imageId));

% Write calibration
dlmwrite(strcat(calib_folder, txt_filename), data.Rtilt(:)', 'delimiter', ' ');
dlmwrite(strcat(calib_folder, txt_filename), data.K(:)', 'delimiter', ' ', '-append');

% Write 2D and 3D box label
data2d = SUNRGBDMeta2DBB(imageId);
fid = fopen(strcat(det_label_folder, txt_filename), 'w');
for j = 1:length(data.groundtruth3DBB)
    centroid = data.groundtruth3DBB(j).centroid;
    classname = data.groundtruth3DBB(j).classname;
    orientation = data.groundtruth3DBB(j).orientation;
    coeffs = abs(data.groundtruth3DBB(j).coeffs);
    box2d = data2d.groundtruth2DBB(j).gtBb2D;
    assert(strcmp(data2d.groundtruth2DBB(j).classname, classname));
    fprintf(fid, '%s %d %d %d %d %f %f %f %f %f %f %f %f\n', classname, box2d(1), box2d(2), box2d(3), box2d(4), centroid(1), centroid(2), centroid(3), coeffs(1), coeffs(2), coeffs(3), orientation(1), orientation(2));
end
fclose(fid);

catch
end

end

function parsave(filename, instance)
save(filename, 'instance');
end
