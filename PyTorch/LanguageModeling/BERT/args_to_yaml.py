from collections import OrderedDict
import yaml


def yaml_ordered_save(fname, ordered_dict):
    def ordered_dict_representer(self, value):
        return self.represent_mapping('tag:yaml.org,2002:map', value.items())

    yaml.add_representer(OrderedDict, ordered_dict_representer)

    with open(fname, 'w') as f:
        yaml.dump(ordered_dict, f, default_flow_style=False)


def yaml_ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """Function to load YAML file using an OrderedDict

    See: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    return yaml.load(stream, OrderedLoader)


def args_to_yaml(args, yaml_path):
    yaml_ordered_save(yaml_path, vars(args))


def args_from_yaml(yaml_path):
    from argparse import Namespace
    return Namespace(**yaml_ordered_load(open(yaml_path)))
