function opts=set_default_opts(opts, defaults, varargin)
    if nargin < 3
        overwrite = 0;

    opt_names = fieldnames(defaults);
    for t=1:numel(opt_names)
        opt_name = opt_names{t};
        if ~isfield(opts, opt_name) || overwrite
            opts.(opt_name) = defaults.(opt_name);
        end
    end
end