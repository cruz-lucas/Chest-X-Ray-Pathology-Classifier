import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.dataset import CheXpertDataset
from GDRAM.model import GDRAM

import torch

device = torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(123)

# Data paths default values
RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed/'

@click.command()
@click.option('--input_filepath', '-i', default=RAW_DATA_PATH, type=str, help='Path to input data.')
@click.option('--output_filepath', '-o', default=PROCESSED_DATA_PATH, type=click.Path(exists=True), help='Path to output data.')
@click.option('--uncertainty_policy', '-u', type=str,
    help='Policy to handle uncertainty.According the CheXpert original paper, policies are "U-Ignore", "U-Zeros", "U-Ones", "U-SelfTrained", and "U-MultiClass".')
@click.option('--batch_size', '-b', default=128, type=int, help='Batch size in training.')
@click.option('--img_size', '-s', default=128, type=int, help='Image size.')
@click.option('--resume', '-r', default=False, type=bool, help='Flag to resume previous started training.')
@click.option('--learning_rate', '-lr', default=1e-3, type=float, help='Learning rate for training.')
@click.option('--epochs', '-e', default=200, type=int, help='Epochs for training.')
@click.option('--checkpoint', '-cp', default=None, type=str, help='Checkpoint to resume training.')
def train(input_filepath: str,
          output_filepath: str,
          uncertainty_policy: str,
          batch_size: int,
          img_size: int,
          resume: bool,
          learning_rate: float,
          epochs: int) -> None:
    #TODO: docstring and integrate dataloader to training loop; include lr optim param and model params on hyperparam tunning
    logger = logging.getLogger(__name__)

    logger.info(f'\nStart training with:\n- Batch size:\t\t{batch_size}\n- Uncertainty Policy:\t"{uncertainty_policy}".')
    #dataloader = CheXpertDataset(data_path=input_filepath, uncertainty_policy=uncertainty_policy, logger=logger)
    
    model = GDRAM(device=device, Fast=False).to(device)

    if resume:
        model.load_state_dict(torch.load(checkpoint))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, verbose=True, patience=5)

    predtion_loss_fn = nn.CrossEntropyLoss()
    best_valid_accuracy, test_accuracy = 0, 0
    for epoch in range(1, epochs + 1):
        #TODO: review training loop and the need to test inside loop
        accuracy = test(model, device, epoch, valid_loader, len(valid_loader.dataset))
        scheduler.step(accuracy)
        print('====> Validation set accuracy: {:.2%}'.format(accuracy))
        if accuracy > best_valid_accuracy:
            best_valid_accuracy = accuracy
            #test_accuracy = test(epoch, test_loader, len(test_loader.dataset))

            torch.save(model.state_dict(), f"checkpoints/CheXpert__u_{uncertainty_policy}_b_{batch_size}_s_{img_size}_lr_{learning_rate}_e_{epoch}_of_{epochs}.pth")

            print('====> Test set accuracy: {:.2%}'.format(test_accuracy))

        # Training loop
        model.train()
        train_loss = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            action_logits, loc, location_log_probs, baselines, _ = model(data)
            labels = labels.unsqueeze(dim=1).to(device)
            loss = loss_function(labels, action_logits, location_log_probs, baselines)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), train_size,
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / train_size))
    
    return best_valid_accuracy

def test(model, device, epoch, data_source, size):
    #TODO: type hint and docstring function
    model.eval()
    total_correct = 0.0
    with torch.no_grad():
        for i, (data, labels) in enumerate(data_source):
            data = data.to(device)
            action_logits, _,  _, _, _= model(data)
            predictions = torch.argmax(action_logits, dim=1)
            labels = labels.to(device)
            total_correct += torch.sum((labels == predictions)).item()
    accuracy = total_correct / size

    image = data[0:1]
    _, locations, _, _, weights = model(image)
    draw_locations(image.cpu().numpy()[0], locations.detach().cpu().numpy()[0], weights=weights, epoch=epoch)
    return accuracy

def loss_function(labels, action_logits, location_log_probs, baselines):
    #TODO: type hint and docstring
    pred_loss = predtion_loss_fn(action_logits, labels.squeeze())
    predictions = torch.argmax(action_logits, dim=1, keepdim=True)
    num_repeats = baselines.size(-1)
    rewards = (labels == predictions.detach()).float().repeat(1, num_repeats)

    baseline_loss = F.mse_loss(rewards, baselines)
    b_rewards = rewards - baselines.detach()
    reinforce_loss = torch.mean(
        torch.sum(-location_log_probs * b_rewards, dim=1))

    return pred_loss + baseline_loss + reinforce_loss

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    train()
