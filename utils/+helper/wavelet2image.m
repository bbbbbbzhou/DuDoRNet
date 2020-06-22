function img = wavelet2image(img, opts)
    img = img ./ opts.wgt;
    img = single(cnn_wavelet_recon( ...
        double(img), opts.lv, opts.dflt));
    img = helper.convert2hu(img);
end