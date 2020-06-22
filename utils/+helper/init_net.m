function net = init_net(net)
    for t=1:numel(net.layers)
        layer_type = net.layers{t}.type;
        switch layer_type
            case 'bnorm'
                net.layers{t}.weights{1} = ...
                    ones(size(net.layers{t}.weights{1}), 'single');
                net.layers{t}.weights{2} = ...
                    zeros(size(net.layers{t}.weights{2}), 'single');
                net.layers{t}.weights{3} = ...
                    zeros(size(net.layers{t}.weights{3}), 'single');
            case 'conv'
                kernel_size = size(net.layers{t}.weights{1});
                w = sqrt(2 / prod(kernel_size(1:3)));

                net.layers{t}.weights{1} = ...
                    randn(kernel_size, 'single') * w;
                if numel(net.layers{t}.weights) == 2
                    net.layers{t}.weights{2} = ...
                        zeros(kernel_size(4), 1, 'single');
                end
        end
    end
end