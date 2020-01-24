import argparse


def get_train_options(**defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Command line options for training and testing.')

    # Arguments common to all cases.
    parser.add_argument('--batch_size', type=int, default=1, help='Mini-batch size.')
    parser.add_argument('--gpu', type=int, default=None, help='Which GPU to use. Default is None, which indicates CPU.')

    # File paths.
    file_paths = parser.add_argument_group(title='file_paths', description='Group for common file paths.')
    file_paths.add_argument('--record_path', type=str, default='../records',
                            help='Common root path for logs and checkpoint files.')
    file_paths.add_argument('--data_path', type=str, default='../data', help='Data root path.')

    # Training specific arguments.
    training = parser.add_argument_group(title='training', description='Training specific parameters.')
    training.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    training.add_argument('--num_epochs', type=int, help='Number of epochs to train the model.')

    parser.set_defaults(**defaults)
    return parser

