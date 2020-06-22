function data = simulate_sparse_view(data_file, num_views, down_rate)

% load full-dose CT
meta = dicominfo(data_file);
pixel_spacing = mean(meta.PixelSpacing) / 10;
gt_CT = dicomread(data_file);
gt_CT = single(gt_CT) * meta.RescaleSlope + meta.RescaleIntercept;
gt_CT = helper.convert2coefficient(gt_CT);

dtheta = 360 / num_views;
dv_theta = linspace(0, 360-dtheta, num_views);
dv_sinogram = radon(gt_CT, dv_theta) * pixel_spacing;

sv_theta = dv_theta(1:down_rate:end);
sv_sinogram = dv_sinogram(:, 1:down_rate:end);

% LI
Y = 1:size(dv_sinogram, 1);
X = 1:down_rate:size(dv_sinogram, 2);
Xq = 1:size(dv_sinogram, 2);
F = griddedInterpolant({Y, X}, sv_sinogram);
LI_sinogram = F({Y, Xq});

dv_CT = iradon(dv_sinogram, dv_theta, 512) / pixel_spacing;
sv_CT = iradon(sv_sinogram, sv_theta, 512) / pixel_spacing;
LI_CT = iradon(LI_sinogram, dv_theta, 512) / pixel_spacing;

data.spacing = single(pixel_spacing);
data.dv_sinogram = single(dv_sinogram);
data.sv_sinogram = single(sv_sinogram);
data.LI_sinogram = single(LI_sinogram);
data.dv_CT = single(dv_CT);
data.sv_CT = single(sv_CT);
data.LI_CT = single(LI_CT);

end