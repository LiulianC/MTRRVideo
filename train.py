import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
import torch.nn.functional as F
import torchvision.models as models
# from location_aware_sirr_model1 import LocationAwareSIRR # 单尺度拉普拉斯
from tqdm import tqdm
import time
from torchvision.utils import save_image 
from multiprocessing import freeze_support
import shutil
import datetime
import csv
from torch.utils.data import ConcatDataset

from MTRRNet import MTRREngine
from early_stop import EarlyStopping
from customloss import CustomLoss
from dataset.quality_index import *
from dataset.new_dataset1 import *
from torch import amp
scaler = amp.GradScaler()
from set_seed import set_seed 
from fix_mamba_init import apply_improved_init  # 导入改进的初始化


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
opts.debug_monitor_layer_stats = 1 # debug模式开启时 epoch和size都要为1 要load模型 可同时打开
opts.debug_monitor_layer_grad = 1 # # debug模式开启时 epoch和size都要为1 要load模型bash
opts.draw_attention_map = False # 注册cbam钩子 画注意力热力图 训练数据集要改 epoch和size都要为1 要load模型 batchsize要改1
opts.sampler_size1 = 0
opts.sampler_size2 = 0
opts.sampler_size3 = 800
opts.test_size = [200,0,0]
opts.epoch = 40
opts.model_path='./model_fit/model_latest.pth'  
opts.model_path=None  #如果要load就注释我
current_lr = 1e-4 # 不可大于1e-5 否则会引起深层网络的梯度爆炸

# nohup /home/gzm/cp310pt26/bin/python /home/gzm/gzm-MTRRVideo/train.py > /home/gzm/gzm-MTRRVideo/train.log 2>&1 &

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MTRREngine(opts, device)
# model.count_parameters()
# 应用改进的初始化
print("Applying improved initialization...")
model.netG_T = apply_improved_init(model.netG_T)

if opts.debug_monitor_layer_stats or opts.debug_monitor_layer_grad:
    opts.epoch = 100
    opts.batch_size = 8
    opts.sampler_size1 = 0
    opts.sampler_size2 = 0
    opts.sampler_size3 = 100*opts.batch_size
    opts.test_size = [200,0,0]
    if opts.debug_monitor_layer_stats:
        os.remove('./debug/state.log') if os.path.exists('./debug/state.log') else None
        model.monitor_layer_stats()# 注册
    if opts.debug_monitor_layer_grad:
        os.remove('./debug/grad.log') if os.path.exists('./debug/grad.log') else None



fit_datadir = '/home/gzm/gzm-RDNet1/dataset/laparoscope_gen'
fit_data = DSRTestDataset(datadir=fit_datadir, fns='/home/gzm/gzm-RDNet1/dataset//laparoscope_gen_index/train1.txt',size=opts.sampler_size1, enable_transforms=False,if_align=True,real=False, HW=[256,256])

tissue_gen = './data/tissue_gen'
tissue_gen_data = DSRTestDataset(datadir=tissue_gen, fns='./data/tissue_gen_index/train1.txt',size=opts.sampler_size2, enable_transforms=False,if_align=True,real=False, HW=[256,256])

tissue_dir = './data/tissue_real'
tissue_data = DSRTestDataset(datadir=tissue_dir,fns='./data/tissue_real_index/train1.txt',size=opts.sampler_size3, enable_transforms=True,if_align=True,real=False, HW=[256,256])

# 使用ConcatDataset方法合成数据集 能自动跳过空数据集
train_data = ConcatDataset([fit_data, tissue_gen_data, tissue_data])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=opts.batch_size, shuffle=opts.shuffle, num_workers = opts.num_workers, drop_last=False, pin_memory=True)



test_data_dir1 = './data/tissue_real'
test_data1 = DSRTestDataset(datadir=test_data_dir1, fns='./data/tissue_real_index/eval1.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=opts.test_size[0])

test_data_dir2 = './data/hyperK_000'
test_data2 = TestDataset(datadir=test_data_dir2, fns='./data/hyperK_000_list.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=opts.test_size[1])

test_data_dir3 = '/home/gzm/gzm-RDNet1/dataset/laparoscope_gen'
test_data3 = DSRTestDataset(datadir=test_data_dir3, fns='/home/gzm/gzm-RDNet1/dataset//laparoscope_gen_index/eval1.txt', enable_transforms=False, if_align=True, real=True, HW=[256,256], size=opts.test_size[2])

test_data = ConcatDataset([test_data1, test_data2, test_data3])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, drop_last=False, pin_memory=True)



total_train_step = 0

total_test_step = 0

run_times = []
tabel=[]

loss_function = CustomLoss().to(device)

min_loss=1000 # 初始loss 尽可能大
max_psnr=0
max_ssim=0


tensorboard_writer = SummaryWriter("./logs")

if __name__ == '__main__':
    set_seed(42)  # 设置随机种子 使得checkpoint和训练结果可复现
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join('./logs', current_time)
    train_loss_path = os.path.join("./indexcsv",f"{current_time}_train_loss_logs_compare.csv")
    index_file_path = os.path.join("./indexcsv",f"{current_time}_index_compare.csv")
    os.makedirs('./indexcsv', exist_ok=True)
    os.makedirs('./model_fit', exist_ok=True)

    channel_weights=[]
    spatial_weights=[]
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    output_dir = './img_results'
    output_dir6 = os.path.join(output_dir,f'./output_test_{current_time}')
    output_dir7 = os.path.join(output_dir,f'./output_train_{current_time}')
    os.makedirs(output_dir, exist_ok=True)
    os.mkdir(output_dir6)
    os.mkdir(output_dir7)

    # 定义优化器
    parameter_groups = [
        {'params': [p for n, p in model.named_parameters() if 'proj.2.weight' in n], 'weight_decay': 0.01},  # PReLU参数
        {'params': [p for n, p in model.named_parameters() if n.endswith('scale_raw')], 'weight_decay': 0.1},  # scale参数
        {'params': [p for n, p in model.named_parameters() if 'alpha' in n], 'weight_decay': 0.05},  # alpha参数
        {'params': [p for n, p in model.named_parameters() if not any(x in n for x in ['proj.2.weight', 'scale_raw', 'alpha'])], 'weight_decay': 0.0001}  # 其他参数
    ]
    optimizer = torch.optim.Adam(parameter_groups, lr=current_lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-5)
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # 监控的 quantity 是 loss，我们希望它减小
        factor=0.5,           # 学习率乘以 0.5
        patience=7,           # 等待 4 个 epoch 没有 improvement 后才触发 LR 衰减
        threshold=1e-4,       # 可选：认为 loss 没有显著下降的阈值
        threshold_mode='rel', # 使用相对阈值
        cooldown=0,           # 每次衰减后冷却期（可不设）
        min_lr=1e-8,          # 学习率下限
        eps=1e-8              # 学习率更新的最小变化
    )

    # 定义早停
    early_stopping = EarlyStopping(patience=20, delta=1e-4, verbose=True)

    # 网络load 以及继承上次的epoch和学习参数
    if opts.model_path is not None and os.path.exists(opts.model_path):
        epoch_last_num = model.load_checkpoint(optimizer)
    else:
        epoch_last_num = None

    train_begin=False
    epoch_start_num = 0
    if epoch_last_num is not None:
        if epoch_last_num < opts.epoch :
            train_begin=True
            epoch_start_num=epoch_last_num+1 # 上一轮是n 下一轮要+1
        else:
            print("模型last_epoch>epoch,模型已经训练完毕,不需要继续训练")
            exit(0)

    if opts.num_workers > 0:  # 多线程
        freeze_support()

    for i in range(epoch_start_num, opts.epoch):
        t1 = time.time()
        print("-----------第{}轮训练开始-----------".format(i + 1))
        print(" train data length: {} batch size: {}".format((len(train_loader))*opts.batch_size, opts.batch_size))
        
        total_train_loss=0
        train_pbar = tqdm(
            train_loader,
            desc="Training",
            total=len(train_loader),
            ncols=170,  # 建议宽度根据指标数量调整
            dynamic_ncols=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )        
        for t, data1 in enumerate(train_pbar):
            # print('\n')
            # print('input mean:',data1['input'].mean())
            # print('input std:',data1['input'].std())
            # print('input shape:',data1['input'].shape)
            model.set_input(data1)
            train_file_name = str(data1['fn']) 



            if opts.draw_attention_map:
                model.register_cbam_hooks()
                channel_weights,spatial_weights=model.get_attention_matix() # 打印cbam钩子
                for i,spatial_weight in enumerate(spatial_weights):
                    # scale = 256 / spatial_weight.size(0)
                    spatial_weights[i] = F.interpolate(spatial_weight, size=(256, 256), mode='bilinear', align_corners=False)
                    save_image(spatial_weights[i], os.path.join('./毕业paper/Spatial_attention/map', f'{train_file_name}-spatial_weight_{i:02d}.png'), normalize=True)                
            
            else: 
                # with amp.autocast(device_type='cuda'):
                model.inference()



            visuals = model.get_current_visuals()
            train_input =   visuals['I'].to(device)
            train_ipt =   visuals['Ic'].to(device)
            train_label1 =  visuals['T'].to(device)
            train_label2 =  visuals['R'].to(device)
            # 列表的最后一个元素 shape B C H W
            train_fake_Ts = visuals['fake_T'].to(device)
            train_fake_Rs = visuals['fake_R'].to(device)
            train_rcmaps =  visuals['c_map'].to(device)
 
            with amp.autocast(device_type='cuda'):
                loss_table, mse_loss, vgg_loss, ssim_loss, color_loss, all_loss = loss_function(train_fake_Ts, train_label1, train_ipt, train_rcmaps, train_fake_Rs, train_label2)
            total_train_loss +=all_loss.item()

            if torch.isnan(all_loss):
                print("⚠️  Loss is NaN! input:", train_file_name)

            # Log loss_table to a CSV file
            file_exists = os.path.isfile(train_loss_path)
            # Write header if file does not exist
            with open(train_loss_path, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["step"] + list(loss_table.keys()))
                if not file_exists:
                    writer.writeheader()
                row = {"step": total_train_step}
                # Convert all tensor values to float
                for k, v in loss_table.items():
                    row[k] = v.item() if hasattr(v, "item") else float(v)
                writer.writerow(row)



            optimizer.zero_grad()
            scaler.scale(all_loss).backward()
            # 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            scaler.step(optimizer)
            scaler.update()            
            # 打印每层梯度
            if opts.debug_monitor_layer_grad :
                model.monitor_layer_grad()

            total_train_step += 1

            if i % 10 == 0 & total_train_step % 1 == 0:

                save_image(train_input, os.path.join(output_dir7, f'epoch{i}+{total_train_step}-train_imgs.png'), nrow=4)
                save_image(train_ipt, os.path.join(output_dir7, f'epoch{i}+{total_train_step}-train_ipt.png'), nrow=4)
                save_image(train_label1, os.path.join(output_dir7, f'epoch{i}+{total_train_step}-train_label1.png'), nrow=4)
                save_image(train_label2, os.path.join(output_dir7, f'epoch{i}+{total_train_step}-train_reflection.png'), nrow=4)

                train_fake_TList = visuals['fake_T']
                train_fake_TList_cat = train_fake_TList
                save_image(train_fake_TList_cat, os.path.join(output_dir7, f'epoch{i}+{total_train_step}-train_fakeT.png'), nrow=4)

                train_fake_RList = visuals['fake_R']
                train_fake_RList_cat = train_fake_RList
                save_image(train_fake_RList_cat, os.path.join(output_dir7, f'epoch{i}+{total_train_step}-train_FakeR.png'), nrow=4)

                train_rcmaps_List = visuals['c_map']
                train_rcmaps_List_cat = train_rcmaps_List
                save_image(train_rcmaps_List_cat, os.path.join(output_dir7, f'epoch{i}+{total_train_step}-train_Rcmaps_List.png'), nrow=4)



                        
            if i % 1 == 0:
                current_lr = optimizer.param_groups[0]['lr']

            # if total_train_step % 50 == 0:
            #     model.apply_weight_constraints()


            train_pbar.set_postfix({'loss':all_loss.item(),'mseloss':mse_loss.item(), 'vggloss':vgg_loss.item(), 'ssimloss':ssim_loss.item(),'colorloss':color_loss.item(),'current_lr': current_lr})
            train_pbar.update(1)
        train_pbar.close()


        total_test_loss = 0
        total_test_step = 0
        total_test_psnr = 0
        total_test_ssim = 0

        with torch.no_grad():
            print("test data length: {} batch size: {}".format(len(test_data),opts.batch_size))
            test_pbar = tqdm(
                test_loader,
                desc="Validating",
                total=len(test_loader),
                ncols=150,  # 建议宽度根据指标数量调整
                dynamic_ncols=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )                  
            for n1 , test_data1 in enumerate(test_pbar):
                model.set_input(test_data1)
                model.inference()
                visuals_test = model.get_current_visuals()
                test_imgs = visuals_test['I'].to(device)
                test_ipt = visuals_test['Ic'].to(device)
                test_label1 = visuals_test['T'].to(device)
                test_label2 = visuals_test['R'].to(device)

                test_fake_Ts = visuals_test['fake_T'].to(device)
                 
                test_fake_Rs = visuals_test['fake_R'].to(device)
                
                test_rcmaps = visuals_test['c_map'].to(device)
                
                _,_,_,_,_,loss = loss_function(test_fake_Ts, test_label1, test_ipt, test_rcmaps, test_fake_Rs, test_label2)
                total_test_loss += loss.item()


                # 计算psnr与ssim与NCC与LMSN
                index = quality_assess(test_fake_Ts.to('cpu'), test_label1.to('cpu'))
                file_name, psnr, ssim, lmse, ncc = test_data1['fn'], index['PSNR'], index['SSIM'], index['LMSE'], index['NCC']
                # 数据集返回时 只要batchsize不为0 就返回的是列表
                res = {'file':str(file_name),'PSNR':psnr,'SSIM':ssim,'LMSE':lmse,'NCC':ncc}
                total_test_psnr = total_test_psnr + res['PSNR']
                total_test_ssim = total_test_ssim + res['SSIM']
                
                # 检查文件是否存在，不存在则写入表头
                file_exists1 = os.path.isfile(index_file_path)
                with open(index_file_path, "a", newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=["epoch"] + list(res.keys()))
                    if not file_exists1:
                        writer.writeheader()
                    row = {"epoch": i}
                    # Convert all tensor values to float
                    for k, v in res.items():
                        if type(v) == str:
                            row[k] = v
                        else:
                            row[k] = v.item() if hasattr(v, "item")  else float(v)
                    writer.writerow(row)


                if i % 1 == 0 & total_test_step % 1 == 0:
                    save_image(test_imgs, os.path.join(output_dir6, f'epoch{i}+{total_test_step}-test_imgs.png'), nrow=4)
                    save_image(test_ipt, os.path.join(output_dir6, f'epoch{i}+{total_test_step}-test_ipt.png'), nrow=4)
                    save_image(test_label1, os.path.join(output_dir6, f'epoch{i}+{total_test_step}-test_label1.png'), nrow=4)
                    save_image(test_label2, os.path.join(output_dir6, f'epoch{i}+{total_test_step}-test_reflection.png'), nrow=4)

                    test_fake_TList = visuals_test['fake_T']
                    test_fake_TList_cat = test_fake_TList
                    save_image(test_fake_TList_cat, os.path.join(output_dir6, f'epoch{i}+{total_test_step}-test_fakeT.png'), nrow=4)
                    # torch.save(test_fake_TList_cat,os.path.join(output_dir6,f'epoch{i}+{total_test_step}+fakeT-tensor.pt'))

                    test_fake_RList = visuals_test['fake_R']
                    test_fake_RList_cat = test_fake_RList
                    save_image(test_fake_RList_cat, os.path.join(output_dir6, f'epoch{i}+{total_test_step}-test_FakeR.png'), nrow=4)

                    test_rcmaps_List = visuals_test['c_map']
                    test_rcmaps_List_cat = test_rcmaps_List
                    save_image(test_rcmaps_List_cat, os.path.join(output_dir6, f'epoch{i}+{total_test_step}-test_Rcmaps_List.png'), nrow=4)

                total_test_step += 1
                test_pbar.set_postfix({'loss':all_loss.item(),'psnr':res['PSNR'], 'ssim':res['SSIM'], 'lmse':res['LMSE'],'ncc': res['NCC']})
                test_pbar.update(1)


            # 更新学习率
            scheduler.step(total_test_loss)
            test_pbar.close()

            epoch_num = {"epoch":i}
            # model.state_dict.update(epoch)

        avg_test_loss = total_test_loss / total_test_step
        avg_test_psnr = total_test_psnr / total_test_step
        avg_test_ssim = total_test_ssim / total_test_step
        if avg_test_psnr > max_psnr:
            print(f"psnr from {max_psnr:.5f} improve {avg_test_psnr:.5f} ")
            max_psnr = avg_test_psnr
        else:
            print(f"psnr did not improve : best {max_psnr:.5f} now {avg_test_psnr:.5f} ")
            
        if avg_test_ssim > max_ssim:
            print(f"ssim from {max_ssim:.5f} improve {avg_test_ssim:.5f} ")
            max_ssim = avg_test_ssim
        else:
            print(f"ssim did not improve : best {max_ssim:.5f} now {avg_test_ssim:.5f} ")
        
        


        state = {
            'epoch': i, 
            'net_c': model.net_c.state_dict(),
            'netG_T': model.netG_T.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': current_lr,
        }

        # 早停检查
        early_stopping(avg_test_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {i}!")
            break

        if avg_test_loss<min_loss:
            min_loss = avg_test_loss 
            print(f"New best model at epoch {opts.epoch} with loss {min_loss:.4f}")
            torch.save(state, "./model_fit/model_{}.pth".format(i + 1))        
        else:
            print(f"Epoch {opts.epoch} did not improve. Best loss:{min_loss:.4f}  now: {avg_test_loss:.4f}")   

        t2 = time.time()
        run_times.append(t2 - t1)
        if (i) % 1 == 0:
            print('processing the {} epoch, {} mins passed by'.format(i + 1, run_times[-1]/60))
            torch.save(state, "./model_fit/model_latest.pth".format(i + 1))
            print("模型已保存")         

        # 清理缓存
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    torch.save(state, "./model_fit/model_latest.pth".format(opts.epoch))
    print("模型已保存")

tensorboard_writer.close()













