import argparse


def base_options(**defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Command line options for experiments.')

    # Arguments common to all cases.
    parser.add_argument('--batch_size', type=int, default=1, help='Mini-batch size.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of processes used for data pre-processing in the DataLoader.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Which GPU to use. Default is None, which indicates CPU.')

    # File paths.
    file_paths = parser.add_argument_group(title='file_paths', description='Group for common file paths.')
    file_paths.add_argument('--record_path', type=str, default='../records',
                            help='Common root path for logs and checkpoint files.')
    file_paths.add_argument('--data_path', type=str, default='../data', help='Data root path.')

    parser.set_defaults(**defaults)  # set_defaults must be at the end to override other settings.
    return parser


def train_options(**defaults) -> argparse.ArgumentParser:
    parser = base_options(**defaults)

    # Training specific arguments.
    training = parser.add_argument_group(title='training', description='Training specific parameters.')
    training.add_argument('--lr', default=0.001, type=float, help='Initial learning rate.')
    training.add_argument('--momentum', default=0.9, type=float, help='Momentum to use in SGD optimizer.')
    training.add_argument('--nesterov', action='store_true',
                          help='Whether to use nesterov momentum in SGD. Defaults to False.')
    training.add_argument('--weight_decay', default=0.0001, type=float, help='Weight decay for SGD optimizer.')
    training.add_argument('--num_epochs', default=400, type=int, help='Number of epochs to train the model.')

    parser.set_defaults(**defaults)
    return parser


def classification_options(**defaults) -> argparse.ArgumentParser:
    parser = train_options(**defaults)
    parser.set_defaults(**defaults)
    return parser


def knowledge_distillation_options(**defaults) -> argparse.ArgumentParser:
    parser = train_options(**defaults)

    # Knowledge Distillation specific arguments.
    kd = parser.add_argument_group(title='knowledge_distillation',
                                   description='Arguments specific to knowledge distillation.')
    kd.add_argument('--teacher_checkpoint', type=str,
                    help='Checkpoint file location of teacher to be used for knowledge distillation.')

    kd.add_argument('--distill_ratio', type=float, help='Value deciding split between training loss between '
                                                        'ground truth targets and teacher predictions.')
    kd.add_argument('--temperature', default=1., type=float,
                    help='Temperature value deciding entropy of teacher predictions. Helps ease learning.')

    parser.set_defaults(**defaults)
    return parser
