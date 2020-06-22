function [hq_image, lq_image, varargout] = load_data( ...
    data_file, dataset_name, with_sinogram)
    if nargin < 3
        with_sinogram = false;
    end

    switch dataset_name
        case 'aapm_official'
            load(data_file, 'full_dose', 'quarter_dose', ...
            'full_rescale_params', 'quarter_rescale_params');
            hq_image = single(full_dose) * full_rescale_params(2) + ...
                full_rescale_params(1);
            lq_image = single(quarter_dose) * quarter_rescale_params(2) + ...
                quarter_rescale_params(1);
        case {'ellipsoid', 'aapm_sparse'}
            if with_sinogram
                load(data_file, 'dense_sinogram', 'dense_view', ...
                    'sparse_sinogram', 'sparse_view');
                hq_image = single(dense_view);
                lq_image = single(sparse_view);
                hq_sinogram = single(dense_sinogram);
                lq_sinogram = single(sparse_sinogram);
                varargout{1} = hq_sinogram;
                varargout{2} = lq_sinogram;
            else
                load(data_file, 'dense_view', 'sparse_view');
                hq_image = single(dense_view);
                lq_image = single(sparse_view);
            end
        case 'aapm_learn'
            load( ...
                data_file, 'dense_view', 'sparse_view', 'sparse_sinogram');
            hq_image = single(dense_view);
            lq_image = single(sparse_view);
            varargout{1} = single(sparse_sinogram);
        case 'aapm_wavelet'
            load(data_file, 'full_dose', 'low_dose', ...
                'full_dose_coeffs', 'low_dose_coeffs');
            hq_image = single(full_dose);
            lq_image = single(low_dose);
            varargout{1} = single(full_dose_coeffs);
            varargout{2} = single(low_dose_coeffs);
        otherwise
            error('Unsupported dataset name: %s', dataset_name);    
    end
end