function data_files = read_dataset_dir(dataset_dir, dataset_name)
    switch dataset_name
        case {'aapm_official', 'aapm_sparse', 'aapm_learn', ...
                'aapm_wavelet'}
            data_files = read_aapm_dir(dataset_dir);
        case 'ellipsoid'
            data_files = read_ellipsoid_dir(dataset_dir);
        otherwise
            error('Unsupported dataset name: %s', dataset_name);
    end
    data_files = string(data_files);
end

function data_files = read_ellipsoid_dir(data_dir)
    data_names = dir(fullfile(data_dir, '*.mat'));
    data_files = strings(size(data_names));
    for i=1:size(data_files, 1)
        data_files(i) = fullfile(data_dir, data_names(i).name);
    end
    data_files = data_files';
end

function data_files = read_aapm_dir(dataset_dir)
    study_names = dir(fullfile(dataset_dir, 'L*'));
    study_dirs = strings(size(study_names));
    for i=1:size(study_names, 1)
        study_dirs(i) = fullfile(dataset_dir, study_names(i).name);
    end

    data_files = {};
    for i=1:size(study_dirs, 1)
        instance_names = dir(fullfile(study_dirs(i), '*.mat'));

        for j=1:size(instance_names, 1)
            data_files{end+1} = fullfile( ...
                study_dirs(i), instance_names(j).name);
        end
    end
end
