import argparse
from data_management import load_data
import model_management

parser = argparse.ArgumentParser(description='Training a neural network on a given dataset')
parser.add_argument('data_directory')
parser.add_argument('--save_dir')
parser.add_argument('--arch')
parser.add_argument('--learning_rate')
parser.add_argument('--hidden_units')
parser.add_argument('--epochs')
parser.add_argument('--gpu', action='store_true')


args = parser.parse_args()


save_dir = '' if args.save_dir is None else args.save_dir
network_architecture = 'densenet121' if args.arch is None else args.arch
learning_rate = 0.003 if args.learning_rate is None else float(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else float(args.hidden_units)
epochs = 8 if args.epochs is None else int(args.epochs)
gpu = False if args.gpu is None else True


train_data, trainloader, validloader, testloader = load_data(args.data_directory)


model = model_management.build_network(network_architecture, hidden_units)
model.class_to_idx = train_data.class_to_idx

model, criterion = model_management.train_network(model, epochs, learning_rate, trainloader, validloader, gpu)
model_management.evaluate_model(model, testloader, criterion, gpu)
model_management.save_model(model, network_architecture, hidden_units, epochs, learning_rate, save_dir)