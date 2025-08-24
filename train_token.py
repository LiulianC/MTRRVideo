"""
Updated training script for TokenMTRRNet with:
- Disabled net_c (use input directly)
- Auxiliary supervision for intermediate scales
- Visualization taps for debug outputs
- Support for token-only architecture
"""
import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm
import time
from torchvision.utils import save_image 
from multiprocessing import freeze_support
import shutil
import datetime
import csv
from torch.utils.data import ConcatDataset

# Import new token architecture
import sys
sys.path.insert(0, '.')
import mamba_ssm_mock
sys.modules['mamba_ssm'] = mamba_ssm_mock

from MTRRNet_token import TokenMTRREngine
from early_stop import EarlyStopping
from customloss import CustomLoss
from dataset.quality_index import *
from dataset.new_dataset1 import *
from torch import amp
scaler = amp.GradScaler()
from set_seed import set_seed 
from fix_mamba_init import apply_improved_init
import torch
torch.autograd.set_detect_anomaly(True)

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--model_dir', type=str, default='./model', help='the model dir')
parser.add_argument('--save_dir', type=str, default='./results', help='the results saving dir')
parser.add_argument("--host", type=bool, default="127.0.0.1")
parser.add_argument("--port", default=57117)
opts = parser.parse_args()
opts.batch_size = 4
opts.shuffle = False
opts.display_id = -1  
opts.num_workers = 0

opts.always_print = 1
opts.debug_monitor_layer_stats = 0  # Disable by default for token architecture
opts.debug_monitor_layer_grad = 0  # Disable by default for token architecture
opts.draw_attention_map = False
opts.sampler_size1 = 0
opts.sampler_size2 = 0
opts.sampler_size3 = 800
opts.test_size = [200,0,0]
opts.epoch = 40
opts.model_path = None  # Start fresh with token architecture
# opts.model_path='./model_fit/model_91.pth'  # Comment out to start fresh
current_lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TokenMTRREngine(opts, device)

# Apply improved initialization
print("Applying improved initialization to TokenMTRRNet...")
model.netG_T = apply_improved_init(model.netG_T)

if opts.debug_monitor_layer_stats or opts.debug_monitor_layer_grad:
    opts.epoch = 200
    opts.batch_size = 8
    opts.sampler_size1 = 0
    opts.sampler_size2 = 0
    opts.sampler_size3 = 100*opts.batch_size
    opts.test_size = [200,0,0]

# Sample data setup (minimal for testing)
# For full training, use the original data loaders
test_data = torch.utils.data.TensorDataset(
    torch.randn(200, 3, 256, 256),  # input
    torch.randn(200, 3, 256, 256),  # target_t
    torch.randn(200, 3, 256, 256)   # target_r
)

class MockDataset:
    def __init__(self, tensor_dataset):
        self.dataset = tensor_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input_tensor, target_t, target_r = self.dataset[idx]
        return {
            'input': input_tensor,
            'target_t': target_t,
            'target_r': target_r,
            'fn': f'mock_sample_{idx}'
        }

train_data = MockDataset(test_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=opts.batch_size, shuffle=opts.shuffle, 
                                         num_workers=opts.num_workers, drop_last=False, pin_memory=True)

test_data = MockDataset(test_data)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, 
                                        num_workers=opts.num_workers, drop_last=False, pin_memory=True)

total_train_step = 0
total_test_step = 0

loss_function = CustomLoss().to(device)

min_loss=1000
max_psnr=0
max_ssim=0

tensorboard_writer = SummaryWriter("./logs")

def save_debug_visualizations(debug_outputs, epoch, step, output_dir):
    """Save debug visualization outputs"""
    if not debug_outputs:
        return
        
    for key, tensor in debug_outputs.items():
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 4:  # (B, C, H, W)
            try:
                save_image(tensor, 
                          os.path.join(output_dir, f'epoch{epoch}_step{step}_{key}.png'), 
                          nrow=4, normalize=True)
            except Exception as e:
                print(f"Warning: Could not save debug output {key}: {e}")

def compute_auxiliary_loss(aux_outputs, target_t, weight=0.03):
    """Compute auxiliary supervision loss"""
    aux_loss = 0.0
    aux_count = 0
    
    for key, aux_pred in aux_outputs.items():
        if 'aux_T' in key:  # Auxiliary predictions for target_t
            # Downsample and blur target for supervision
            target_down = F.avg_pool2d(target_t, kernel_size=4, stride=4)
            target_up = F.interpolate(target_down, size=target_t.shape[2:], mode='bilinear', align_corners=False)
            
            # Compute MSE + SSIM loss
            mse_loss = F.mse_loss(aux_pred, target_up)
            
            # Simple SSIM approximation (for efficiency)
            ssim_loss = 1.0 - torch.mean((aux_pred * target_up) / (torch.norm(aux_pred, dim=1, keepdim=True) * torch.norm(target_up, dim=1, keepdim=True) + 1e-8))
            
            aux_loss += weight * (mse_loss + 0.1 * ssim_loss)
            aux_count += 1
    
    return aux_loss / max(aux_count, 1)

if __name__ == '__main__':
    set_seed(42)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join('./logs', current_time)
    train_loss_path = os.path.join("./indexcsv",f"{current_time}_train_loss_logs_token.csv")
    index_file_path = os.path.join("./indexcsv",f"{current_time}_index_token.csv")
    os.makedirs('./indexcsv', exist_ok=True)
    os.makedirs('./model_fit', exist_ok=True)
    os.makedirs('./debug', exist_ok=True)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    output_dir = './img_results'
    output_dir_test = os.path.join(output_dir, f'output_test_token_{current_time}')
    output_dir_train = os.path.join(output_dir, f'output_train_token_{current_time}')
    output_dir_debug = os.path.join(output_dir, f'debug_token_{current_time}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_test, exist_ok=True)
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(output_dir_debug, exist_ok=True)

    # Optimizer setup for token architecture
    norm_names = ['norm', 'bn', 'running_mean', 'running_var']
    decay, no_decay = [], []
    for n, p in model.netG_T.named_parameters():
        if (p.dim()==1 and 'weight' in n) or any(x in n.lower() for x in ['norm','bn','base_scale','delta_scale']):
            no_decay.append(p)
        else:
            decay.append(p)
    
    optimizer = torch.optim.Adam([
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': 1e-4},
    ], lr=current_lr, betas=(0.5,0.999), eps=1e-8)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-8
    )

    early_stopping = EarlyStopping(patience=20, delta=1e-4, verbose=True)

    # Load checkpoint if available
    if opts.model_path is not None and os.path.exists(opts.model_path):
        epoch_last_num = model.load_checkpoint(optimizer)
    else:
        epoch_last_num = None

    train_begin = False
    epoch_start_num = 0
    if epoch_last_num is not None:
        if epoch_last_num < opts.epoch:
            train_begin = True
            epoch_start_num = epoch_last_num + 1
        else:
            print("Model already trained, no need to continue")
            exit(0)

    print("Starting training with TokenMTRRNet (net_c disabled)")
    print(f"Base scale: {model.netG_T.base_scale.item():.3f}")
    print(f"Delta scale: {model.netG_T.delta_scale.item():.3f}")

    for i in range(epoch_start_num, opts.epoch):
        t1 = time.time()
        print(f"-----------Epoch {i + 1} training started-----------")
        print(f"Train data length: {len(train_loader)*opts.batch_size} batch size: {opts.batch_size}")
        
        total_train_loss = 0
        train_pbar = tqdm(
            train_loader,
            desc="Training",
            total=len(train_loader),
            ncols=170,
            dynamic_ncols=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        
        for t, data1 in enumerate(train_pbar):
            model.set_input(data1)
            train_file_name = str(data1['fn'])

            # Forward pass (net_c disabled)
            model.inference()

            visuals = model.get_current_visuals()
            train_input = visuals['I'].to(device)
            train_label1 = visuals['T'].to(device)
            train_label2 = visuals['R'].to(device)
            train_fake_Ts = visuals['fake_T'].to(device)
            train_fake_Rs = visuals['fake_R'].to(device)
            train_rcmaps = visuals['c_map'].to(device)

            # Get auxiliary outputs
            aux_outputs = model.get_aux_outputs()

            # Main loss (note: using train_input directly since net_c is disabled)
            loss_table, mse_loss, vgg_loss, ssim_loss, color_loss, all_loss = loss_function(
                train_fake_Ts, train_label1, train_input, train_rcmaps, train_fake_Rs, train_label2
            )
            
            # Add auxiliary supervision loss
            if aux_outputs:
                aux_loss = compute_auxiliary_loss(aux_outputs, train_label1, weight=0.03)
                all_loss = all_loss + aux_loss
                loss_table['aux_loss'] = aux_loss

            total_train_loss += all_loss.item()

            if torch.isnan(all_loss):
                print("⚠️  Loss is NaN! input:", train_file_name)

            # Log losses
            file_exists = os.path.isfile(train_loss_path)
            with open(train_loss_path, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["step"] + list(loss_table.keys()))
                if not file_exists:
                    writer.writeheader()
                row = {"step": total_train_step}
                for k, v in loss_table.items():
                    row[k] = v.item() if hasattr(v, "item") else float(v)
                writer.writerow(row)

            optimizer.zero_grad()
            all_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.netG_T.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_step += 1

            # Save training images and debug outputs
            if i % 5 == 0 and total_train_step % 10 == 0:
                save_image(train_input, os.path.join(output_dir_train, f'epoch{i}_{total_train_step}_input.png'), nrow=4)
                save_image(train_label1, os.path.join(output_dir_train, f'epoch{i}_{total_train_step}_target.png'), nrow=4)
                save_image(train_fake_Ts, os.path.join(output_dir_train, f'epoch{i}_{total_train_step}_fake_T.png'), nrow=4)
                save_image(train_fake_Rs, os.path.join(output_dir_train, f'epoch{i}_{total_train_step}_fake_R.png'), nrow=4)
                
                # Save debug visualizations
                debug_outputs = model.get_debug_outputs()
                save_debug_visualizations(debug_outputs, i, total_train_step, output_dir_debug)

            if i % 1 == 0:
                current_lr = optimizer.param_groups[0]['lr']

            train_pbar.set_postfix({
                'loss': all_loss.item(),
                'mse': mse_loss.item(), 
                'vgg': vgg_loss.item(), 
                'ssim': ssim_loss.item(),
                'color': color_loss.item(),
                'aux': aux_outputs and loss_table.get('aux_loss', 0),
                'lr': current_lr,
                'base_scale': model.netG_T.base_scale.item(),
                'delta_scale': model.netG_T.delta_scale.item()
            })
            train_pbar.update(1)
        
        train_pbar.close()

        # Validation loop
        total_test_loss = 0
        total_test_step = 0
        total_test_psnr = 0
        total_test_ssim = 0

        with torch.no_grad():
            print(f"Test data length: {len(test_data)} batch size: {opts.batch_size}")
            test_pbar = tqdm(
                test_loader,
                desc="Validating",
                total=len(test_loader),
                ncols=150,
                dynamic_ncols=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )
            
            for n1, test_data1 in enumerate(test_pbar):
                model.set_input(test_data1)
                model.inference()
                
                visuals_test = model.get_current_visuals()
                test_imgs = visuals_test['I'].to(device)
                test_label1 = visuals_test['T'].to(device)
                test_label2 = visuals_test['R'].to(device)
                test_fake_Ts = visuals_test['fake_T'].to(device)
                test_fake_Rs = visuals_test['fake_R'].to(device)
                test_rcmaps = visuals_test['c_map'].to(device)

                # Test loss (using test_imgs directly since net_c is disabled)
                _, _, _, _, _, loss = loss_function(test_fake_Ts, test_label1, test_imgs, test_rcmaps, test_fake_Rs, test_label2)
                total_test_loss += loss.item()

                # Quality metrics
                index = quality_assess(test_fake_Ts.to('cpu'), test_label1.to('cpu'))
                file_name, psnr, ssim, lmse, ncc = test_data1['fn'], index['PSNR'], index['SSIM'], index['LMSE'], index['NCC']
                
                res = {'file': str(file_name), 'PSNR': psnr, 'SSIM': ssim, 'LMSE': lmse, 'NCC': ncc}
                total_test_psnr += res['PSNR']
                total_test_ssim += res['SSIM']

                # Log test results
                file_exists1 = os.path.isfile(index_file_path)
                with open(index_file_path, "a", newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=["epoch"] + list(res.keys()))
                    if not file_exists1:
                        writer.writeheader()
                    row = {"epoch": i}
                    for k, v in res.items():
                        if type(v) == str:
                            row[k] = v
                        else:
                            row[k] = v.item() if hasattr(v, "item") else float(v)
                    writer.writerow(row)

                # Save test images
                if i % 5 == 0 and total_test_step % 10 == 0:
                    save_image(test_imgs, os.path.join(output_dir_test, f'epoch{i}_{total_test_step}_input.png'), nrow=4)
                    save_image(test_label1, os.path.join(output_dir_test, f'epoch{i}_{total_test_step}_target.png'), nrow=4)
                    save_image(test_fake_Ts, os.path.join(output_dir_test, f'epoch{i}_{total_test_step}_fake_T.png'), nrow=4)

                total_test_step += 1
                test_pbar.set_postfix({
                    'loss': loss.item(),
                    'psnr': res['PSNR'], 
                    'ssim': res['SSIM'], 
                    'lmse': res['LMSE'],
                    'ncc': res['NCC']
                })
                test_pbar.update(1)

            scheduler.step(total_test_loss)
            test_pbar.close()

        # Calculate averages
        avg_test_loss = total_test_loss / total_test_step
        avg_test_psnr = total_test_psnr / total_test_step
        avg_test_ssim = total_test_ssim / total_test_step
        
        # Track best metrics
        if avg_test_psnr > max_psnr:
            print(f"PSNR improved from {max_psnr:.5f} to {avg_test_psnr:.5f}")
            max_psnr = avg_test_psnr
        else:
            print(f"PSNR did not improve: best {max_psnr:.5f} now {avg_test_psnr:.5f}")

        if avg_test_ssim > max_ssim:
            print(f"SSIM improved from {max_ssim:.5f} to {avg_test_ssim:.5f}")
            max_ssim = avg_test_ssim
        else:
            print(f"SSIM did not improve: best {max_ssim:.5f} now {avg_test_ssim:.5f}")

        # Save checkpoint
        state = {
            'epoch': i,
            'netG_T': model.netG_T.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': current_lr,
            'base_scale': model.netG_T.base_scale.item(),
            'delta_scale': model.netG_T.delta_scale.item(),
        }

        # Early stopping check
        early_stopping(avg_test_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {i}!")
            break

        # Save best model
        if avg_test_loss < min_loss:
            min_loss = avg_test_loss
            print(f"New best model at epoch {i} with loss {min_loss:.4f}")
            torch.save(state, "./model_fit/model_token_best.pth")
        else:
            print(f"Epoch {i} did not improve. Best loss: {min_loss:.4f} now: {avg_test_loss:.4f}")

        # Regular checkpoint
        t2 = time.time()
        if i % 1 == 0:
            print(f'Epoch {i + 1} completed, {(t2 - t1)/60:.2f} mins elapsed')
            torch.save(state, "./model_fit/model_token_latest.pth")
            print("Model saved")

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    torch.save(state, "./model_fit/model_token_final.pth")
    print("Final model saved")

tensorboard_writer.close()