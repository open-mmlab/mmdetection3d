% Modified from
% https://github.com/facebookresearch/votenet/blob/master/sunrgbd/matlab/extract_split.m
% Copyright (c) Facebook, Inc. and its affiliates.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.

%% Dump train/val split.
% Author: Charles R. Qi

addpath('../OFFICIAL_SUNRGBD/SUNRGBDtoolbox')

%% Construct Hash Map
hash_train = java.util.Hashtable;
hash_val = java.util.Hashtable;

split = load('../OFFICIAL_SUNRGBD/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat');

N_train = length(split.alltrain);
N_val = length(split.alltest);

for i = 1:N_train
    folder_path = split.alltrain{i};
    folder_path(1:16) = '';
    hash_train.put(folder_path,0);
end
for i = 1:N_val
    folder_path = split.alltest{i};
    folder_path(1:16) = '';
    hash_val.put(folder_path,0);
end

%% Map data to train or val set.
load('../OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat');
if exist('../sunrgbd_trainval','dir')==0
    mkdir('../sunrgbd_trainval');
end
fid_train = fopen('../sunrgbd_trainval/train_data_idx.txt', 'w');
fid_val = fopen('../sunrgbd_trainval/val_data_idx.txt', 'w');

for imageId = 1:10335
    data = SUNRGBDMeta(imageId);
    depthpath = data.depthpath;
    depthpath(1:16) = '';
    [filepath,name,ext] = fileparts(depthpath);
    [filepath,name,ext] = fileparts(filepath);
    if hash_train.containsKey(filepath)
        fprintf(fid_train, '%d\n', imageId);
    elseif hash_val.containsKey(filepath)
        fprintf(fid_val, '%d\n', imageId);
    else
        a = 1;
    end
end
fclose(fid_train);
fclose(fid_val);
