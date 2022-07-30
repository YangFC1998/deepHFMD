from dataloader.simple_single_dimension import build as simple_single_dimension_builder


def build_dataset(args):
    if args.dataset_type == 'simple_single_dimension':
        return simple_single_dimension_builder(args)

    raise ValueError(f'dataset {args.dataset_file} not supported')