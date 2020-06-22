__all__ = ['update_from_yaml', 'save_args', 'merge_args', 'get_nondefaults']

import yaml
import argparse


def update_from_yaml(config_file, parser, args, key=None):
    with open(config_file) as f:
        config = yaml.load(f)
        if key:
            if type(key) is str:
                key = [key]
            while key:
                k = key.pop()
                if k in config:
                    config = config[k]
                else:
                    raise ValueError(
                        "{} not found in {}".format(key, config_file))
        args = merge_args(parser, args, config)
    return args


def save_args(args, output_file, print_args=True):
    args_str = yaml.dump(args.__dict__, default_flow_style=False)
    with open(output_file, "w") as f:
        f.write(args_str)
    if print_args:
        print("------------------- Options -------------------")
        print(args_str[:-1])
        print("-----------------------------------------------\n")


def merge_args(parser, args1, args2):
    if args2 is None:
        return args1

    if type(args1) is argparse.Namespace:
        args1 = args1.__dict__

    for k, v in args2.items():
        if k in args1: args1[k] = v
    args = parser.parse_args(convert_dict2args(args1))
    return args


def convert_dict2args(opts):
    args = []
    for key, val in opts.items():
        if val is not None:
            if val is not False:
                args.append("--{}".format(key))
                if type(val) is not bool:
                    if type(val) is list:
                        args.append(" ".join(map(str, val)))
                    else:
                        args.append(str(val))
    return args


def get_nondefaults(parser, args, output_defaults=False):
    args = args.__dict__
    nondefaults = {}
    defaults = {}
    for k, v in args.items():
        v_ = parser.get_default(k)
        if v != v_:
            nondefaults[k] = v
        else:
            defaults[k] = v
    if output_defaults:
        return nondefaults, defaults
    else:
        return nondefaults