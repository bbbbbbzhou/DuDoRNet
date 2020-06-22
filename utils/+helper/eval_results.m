function results=eval_results(results_dir)
    filenames = dir(fullfile(results_dir, '*.mat'));
    num_files = numel(filenames);
    results.psnr = zeros(0, 2);
    results.ssim = zeros(0, 2);
    for ii=1:num_files
        result_file = fullfile(results_dir, filenames(ii).name);
        load(result_file, ...
            'hq_image', 'lq_image', 'recon_image', 'metrics');
        metrics_before = helper.compute_metrics(lq_image, hq_image);
        metrics_after = helper.compute_metrics(recon_image, hq_image);

        results.psnr(end+1, :) = [metrics_before.psnr, metrics_after.psnr];
        results.ssim(end+1, :) = [metrics_before.ssim, metrics_after.ssim];

        fprintf('PSNR: %.3f/%.3f SSIM: %.3f/%.3f\n', ...
            mean(results.psnr(:, 1)),  mean(results.psnr(:, 2)), ...
            mean(results.ssim(:, 1)),  mean(results.ssim(:, 2)));
    end
end