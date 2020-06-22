function img = image2wavelet(img, opts)
    img = helper.convert2coefficient(img);
    img = single(cnn_wavelet_decon( ...
    double(img), opts.lv, opts.dflt));
    img = img .* opts.wgt;
end
