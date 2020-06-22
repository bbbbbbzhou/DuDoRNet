function recon = forward_wavresnet(image, net, opts)
    % default options
    defaults.patch_size = [55, 55];
    defaults.wgt = 1000;
    defaults.lv = [1, 2, 3];
    defaults.dflt = 'vk';
    defaults.overlap = 10;
    opts = helper.set_default_opts(opts, defaults);

    [ny, nx] = size(image);
    py = opts.patch_size(1);
    px = opts.patch_size(2);
    ys = 1:py-opts.overlap:ny-1;
    xs = 1:px-opts.overlap:nx-1;

    wgtMap = zeros(ny, nx, 'single');
    imageCoeffs = helper.image2wavelet(image, opts);
    reconCoeffs = zeros(size(imageCoeffs),'single');
    imagePatches = zeros(py, px, size(imageCoeffs, 3), numel(xs), 'single');

    if opts.gpus > 0
        imageCoeffs = gpuArray(imageCoeffs);
        reconCoeffs = gpuArray(reconCoeffs);
        imagePatches = gpuArray(imagePatches);
    end

    for ii=1:numel(ys)
        yy = min(ys(ii), ny-py+1);

        for jj = 1:numel(xs)
            xx = min(xs(jj),nx-px+1);
            imagePatches(:, :, :, jj) = imageCoeffs(yy:yy+py-1, xx:xx+px-1, :);
        end

        res = vl_simplenn_modified(net, imagePatches, [], [], ...
            'mode','test', 'conserveMemory', 1, 'cudnn', opts.gpus > 0);
        reconPatches = res(end-1).x;

        for jj = 1:numel(xs)
            xx = min(xs(jj),nx-px+1);

            reconCoeffs(yy:yy+py-1, xx:xx+px-1, :) = ...
                reconCoeffs(yy:yy+py-1, xx:xx+px-1, :) + reconPatches(:, :, :, jj);
            wgtMap(yy:yy+py-1, xx:xx+px-1, :) = ...
                wgtMap(yy:yy+py-1, xx:xx+px-1, :) + 1;
        end 
    end
    
    wgtMap = repmat(wgtMap, [1, 1, size(reconCoeffs, 3)]);
    if opts.gpus > 0
        reconCoeffs = gather(reconCoeffs);
    end
    reconCoeffs = reconCoeffs ./ wgtMap + imageCoeffs;
    
    recon = helper.wavelet2image(reconCoeffs, opts);
end