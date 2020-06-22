function checkpoint = find_last_checkpoint(exp_dir)
    list = dir(fullfile(exp_dir, 'net-epoch-*.mat')) ;
    tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
    epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
    epoch = max([epoch 0]) ;
    checkpoint = fullfile(exp_dir, sprintf('net-epoch-%d.mat', epoch));
end