function R = get_radon_matrix(img_sz, det_len, theta)
%% GET_RADON_MATRIX Get a radon transformation matrix.
%   This function computes the system matrix for a radon transformation.
%   That is, given vectorized CT image IMG, this function compute a sparse
%   matrix R such that a verctorized sinogram PROJ can be given by PROJ =
%   R * IMG. 
%   
%   Note this function assumes that the detector spacing equals to the
%   CT image spacing and the CT image is a square image. The system matrix
%   R behaves slightly different from MATLAB's official radon
%   transformation as the official implementation upsamples the CT image
%   before projecting.
%
%   INPUTS:
%       IMG_SZ  size of the CT image
%       DET_LEN detector length
%       THETA   an array of projection angles
%   OUTPUTS:
%       R       the system matrix of radon transformation
%
%
%   EXAMPLE
%   -------
%       img = phantom(256);
%       proj = radon(img, theta);
% 
%       theta = 0:179;
%       img_sz = size(img, 1);
%       det_len = size(proj, 1);
%       R = get_radon_matrix(img_sz, det_len, theta); 
% 
%       proj_ = R * double(img(:));
%       proj_ = reshape(proj_, size(proj));
% 
%       figure; imshow(proj, []);
%       figure; imshow(proj_, []);
%
%   Author: Haofu Liao (liaohaofu@gmail.com)
%   Date: 01/06/2019

num_ang = size(theta, 2);

img_ctr = floor((img_sz - 1) / 2);
x_left = -img_ctr;
y_top = img_ctr;
xs = x_left + (0:img_sz-1);
ys = y_top - (0:img_sz-1);
img_offsets = (0:img_sz-1) * img_sz;

det_ctr = floor(det_len / 2);
det_offsets = det_ctr + det_len * (0:num_ang) + 2;
s = sin(theta * pi / 180);
c = cos(theta * pi / 180);

i = zeros(num_ang * img_sz * img_sz, 1); % i indices of the sparse matrix
j = zeros(num_ang * img_sz * img_sz, 1); % j indices of the sparse matrix
p = zeros(num_ang * img_sz * img_sz, 1); % value a i, j
idx = 1;
for theta_idx=1:num_ang
    det_offset = det_offsets(theta_idx);
    st = s(theta_idx);
    ct = c(theta_idx);
    for x_idx=1:img_sz
        x = xs(x_idx);
        img_offset = img_offsets(x_idx);
        for y_idx=1:img_sz
            y = ys(y_idx);

            % projection location of point (x, y) at angle theta
            det = x * ct + y * st;
            det_flr = floor(det);
            
            % i = floor(det) + det_ctr + det_len * (theta_idx - 1) + 1
            % j = y_idx + img_sz * (x_idx - 1) + 1
            i(idx) = det_flr + det_offset;
            j(idx) = y_idx + img_offset;
            p(idx) = det - det_flr;
            idx = idx + 1;
        end
    end
end
% get the right side interpolation values
i = [i; i - 1];
j = [j; j];
p = [p; 1 - p];
R = sparse(i, j, p, det_len * num_ang, img_sz * img_sz);
