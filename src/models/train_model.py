import click
import logging
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import time
import numpy as np

from src.data.dataset import get_dataloader
from src.models.utils import load_checkpoint, save_checkpoint
from resnest import get_model

import torch
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

# Data paths default values
RAW_DATA_PATH = 'data/raw/'
CHECKPOINT_PATH = 'models/ckpt/'

@click.command()
@click.option('--input_filepath', '-i', default=RAW_DATA_PATH, type=str, help='Path to input data.')
@click.option('--uncertainty_policy', '-u', type=str,
    help='Policy to handle uncertainty.According the CheXpert original paper, policies are "U-Ignore", "U-Zeros", "U-Ones", "U-SelfTrained", and "U-MultiClass".')
@click.option('--resume', '-r', default=False, type=bool, help='Flag to resume training given checkpoint.')
@click.option('--path_to_checkpoint', '-c', default=CHECKPOINT_PATH, type=str, help='Path to checkpoint folder.')   
def train(input_filepath: str,
          uncertainty_policy: str,
          resume: bool,
          path_to_checkpoint: str) -> None:
    #TODO: docstring; include model/train params on hyperparam tunning, decay lr, dropout
    logger = logging.getLogger(__name__)    

    # Hyperparameters
    BATCH_SIZE = 64
    RESIZE_SHAPE = (224,224)
    LEARNING_RATE = 0.01
    EPOCHS = 200

    logger.info(f' \
        \n\n\
        ------------------------- \n\
        \n\
        Start training with: \n\
        - Batch size:\t\t{BATCH_SIZE} \n\
        - Uncertainty Policy:\t"{uncertainty_policy}" \n\
        \n\
        ------------------------- \n')

    # Fetch model
    model = get_model()

    use_cuda = False

    np.random.seed(1)
    torch.manual_seed(1)
    pin_memory = False
    num_workers = 1
    if torch.cuda.is_available():
        logger.info("Cuda is available")
        use_cuda = True
        pin_memory = True
        num_workers = 0
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)
        torch.cuda.manual_seed(1)
        model.cuda()

    # Data loader
    train_data_loader = get_dataloader(data_path=input_filepath,
                                       uncertainty_policy=uncertainty_policy,
                                       logger=logger,
                                       train=True,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       #pathologies=config.pathologies,
                                       pin_memory=pin_memory,
                                       resize_shape=RESIZE_SHAPE,
                                       downsampled=True)
    valid_data_loader = get_dataloader(data_path=input_filepath,
                                       uncertainty_policy=uncertainty_policy,
                                       logger=logger,
                                       train=False,
                                       batch_size=BATCH_SIZE,
                                       #pathologies=config.pathologies,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory,
                                       resize_shape=RESIZE_SHAPE,
                                       downsampled=True)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # create losses (criterion in pytorch)
    criterion_BCE = torch.nn.BCELoss()

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0

    if resume:
        ckpt = load_checkpoint(path_to_checkpoint) # custom method for loading last checkpoint
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        loss = ckpt['loss']
        optim.load_state_dict(ckpt['optimizer_state_dict'])
        logger.info("last checkpoint restored")

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter('models/runs')

    n_iter = start_n_iter
    for epoch in range(start_epoch, EPOCHS):
        model.train()

        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(train_data_loader),
                    total=len(train_data_loader))
        start_time = time.time()

        correct = 0.
        total = 0.
        # for loop going through dataset
        for i, data in pbar:
            # data preparation
            img, label = data
            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            
            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time-time.time()
            
            # forward and backward pass
            out = model(img)
            loss = criterion_BCE(out, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            predicted = np.round(out.detach().data.cpu()) #keep in mind that np.round() is a round to even function
            total += label.size(0) * label.size(1)
            #calculate how many images were correctly classified
            correct += (predicted == label.cpu()).sum().item()
                    
            # compute computation time and *compute_efficiency*
            process_time = start_time-time.time()-prepare_time
            compute_efficiency = process_time/(process_time+prepare_time)
            pbar.set_description(
                f'Compute efficiency: {compute_efficiency:.2f}, ' 
                f'loss: {loss.item():.2f},  epoch: {epoch}/{EPOCHS}')
            start_time = time.time()
            writer.add_scalar('loss/train', loss.item(), n_iter)
            n_iter+=1
        
        train_accuracy = 100 * correct / total
        # udpate tensorboardX
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            
        # maybe do a test pass every N=1 epochs
        if epoch % 1 == 0:
            # bring models to evaluation mode
            model.eval()

            correct = 0
            total = 0

            pbar = tqdm(enumerate(valid_data_loader),
                    total=len(valid_data_loader)) 
            with torch.no_grad():
                for i, data in pbar:
                    # data preparation
                    img, label = data
                    if use_cuda:
                        img = img.cuda()
                        label = label.cuda()
                    
                    out = model(img)
                    predicted = np.round(out.data.cpu()) #keep in mind that np.round() is a round to even function
                    total += label.size(0) * label.size(1)
                    #calculate how many images were correctly classified
                    correct += (predicted == label.cpu()).sum().item()

            valid_accuracy = 100 * correct / total

            # udpate tensorboardX
            writer.add_scalar('Accuracy/valid', valid_accuracy, epoch)
            
            if epoch % 5 == 0:
                # save checkpoint if needed
                cpkt = {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'n_iter': n_iter,
                    'optim': optim.state_dict()
                }
                save_checkpoint(cpkt, 'models/ckpt/model_checkpoint.ckpt')

    return None


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train()
