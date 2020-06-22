function results = compute_metrics(lq_image, hq_image, varargin)
    psnr_res = psnr(lq_image, hq_image, max(hq_image(:)));
    ssim_res = ssim(lq_image, hq_image, 'DynamicRange', max(max(hq_image(:)), eps));

    if nargin > 2
        results = varargin{1};
        results.psnr(end+1) = psnr_res;
        results.ssim(end+1) = ssim_res;
    else
        results.psnr = psnr_res;
        results.ssim = ssim_res;
    end
end
