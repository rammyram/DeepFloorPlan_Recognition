from torch.utils.data import dataset
import wandb
import configuration
import train

params = dict(epochs = configuration.training_config.number_epochs, batch_size = configuration.training_config.batch_size, lr = configuration.training_config.learning_rate, dataset_size = configuration.train_data_config.training_data_size)

if __name__ == "__main__":
    wandb.login()
    model = train.wandb_initializer(params)