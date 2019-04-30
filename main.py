from datasets.mnist import read_datasets
import model
import nn_module
import train
import test
import argparse
import utils
import sys


# TODO: make this more flexible, such as parsing a json config
# just a simple example demo
def build_and_train_model(batch_size, epochs, lr, start_epoch_decay, decay_rate, train_data, valid_data):
    modules_list = [
        (nn_module.Linear, {'n_in': 28*28, 'n_out': 200}),
        (nn_module.Tanh, {}),
        (nn_module.Linear, {'n_in': 200, 'n_out': 10}),
        (nn_module.Softmax, {})
    ]

    net_model = model.NNModel(modules_list)
    loss = nn_module.CrossEntropyLoss()
    train.train(net_model, loss, batch_size, epochs, lr, start_epoch_decay, decay_rate, train_data, valid_data)
    return net_model


def main():
    """Main entry point function. This is where it begins"""

    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument("--model_file", type=str, default='./model.pkl', help='Pickle file for model (dir has to exist)')
    parser.add_argument('--data_dir', type=str, default='./mnist_data', help='data dir for downloaded dataset')
    parser.add_argument('--train', type='bool', nargs='?', const=True, default=False, help='Run training')
    parser.add_argument('--test', type='bool', nargs='?', const=True, default=False, help='Run test')
    parser.add_argument('--log_file', type=str, default=None, help='log file dir (dir has to exist)')

    # hyperparameters
    parser.add_argument('--lr', type=float, default=0.7, help='initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='minibatch size')
    parser.add_argument('--decay_rate', type=float, default=0.3, help='the decay of the learning rate')
    parser.add_argument('--start_epoch_decay', type=float, default=5, help='start lr decay epoch')

    args = parser.parse_args()

    if args.log_file:
        sys.stdout = open(args.log_file, 'w')

    # read data
    dataset_dir = args.data_dir
    datasets = read_datasets(dataset_dir)

    net_model = None
    if args.train:
        net_model = build_and_train_model(args.batch_size,
                                          args.num_epochs,
                                          args.lr,
                                          args.start_epoch_decay,
                                          args.decay_rate,
                                          datasets.train,
                                          datasets.valid)
        # save model for testing later
        if net_model and args.model_file:
            utils.save_model(args.model_file, net_model)

    if args.test:
        # in case we called test later so we need to check if there exists an already trained model
        if not net_model:
            if args.model_file:
                net_model = utils.load_model(args.model_file)
            else:
                raise Exception('No trained model for testing')

        # model is loaded
        test.test_model(net_model, 1000, datasets.test)

    sys.stdout.close()


if __name__ == '__main__':
    main()
