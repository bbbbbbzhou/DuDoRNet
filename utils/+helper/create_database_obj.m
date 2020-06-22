function imdb = create_database_obj(dataset_dir, dataset_name)
    splits = {'train', 'val', 'test'};
    data_files = [];
    set = [];
    for i=1:size(splits, 2)
        split_dir = fullfile(dataset_dir, splits{i});
        if ~isfolder(split_dir), continue; end
        
        split_files = helper.read_dataset_dir(split_dir, dataset_name);
        data_files = [data_files; split_files'];
        
        set = [set; i*ones(size(split_files, 2), 1)];
    end
    
    imdb.images.data_files = data_files;
    imdb.images.set = set;
    imdb.meta.dataset_name = dataset_name;
    imdb.meta.dataset_dir = dataset_dir;
end
