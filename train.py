import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data

from tqdm import tqdm
import os
import argparse
import random
from torchtoolbox.nn.init import KaimingInitializer
import numpy as np

from models import get_model
from loaders import get_loader
from utils import dict2string,mkdir,get_lr,get_acc
from warmup import GradualWarmupScheduler


# torch.backends.cudnn.enable =True
# torch.backends.cudnn.benchmark = True

type_list = ['p',"0","1","2","3","4","5","6","7","8","9",'s','e']



def train(args):

    ### Log file:
    mkdir(args.logdir)
    mkdir(os.path.join(args.logdir,args.experiment_name))
    log_file_path=os.path.join(args.logdir,args.experiment_name,'log.txt')
    log_file=open(log_file_path,'a')
    log_file.write('\n---------------  '+args.experiment_name+'  ---------------\n')
    log_file.close()
    ### Setup tensorboard for visualization
    if args.tboard:
        writer = SummaryWriter(os.path.join(args.logdir,args.experiment_name,'runs'),args.experiment_name)

    ### Setup Dataloader
    data_loader_lmdb = get_loader('lmdb')
    t_loader = data_loader_lmdb(db_path='./dataset/WMeter5K_lmdb', db_name='train')
    v_loader = data_loader_lmdb(db_path='./dataset/WMeter5K_lmdb', db_name='test')
    trainloader = data.DataLoader(t_loader, batch_size=120, num_workers=6, shuffle=True,pin_memory=True)
    valloader = data.DataLoader(v_loader, batch_size=36, num_workers=6)

    ### Setup Model
    model = get_model('transformer_im_distribution').cuda()
    model_im = get_model('cnn_extractor_pretrain_distribution').cuda()
    model_im.load_state_dict(torch.load('./checkpoints/im.pkl')['model_state'])

    initializer = KaimingInitializer()
    model.apply(initializer)

    # ### Optimizer
    optimizer= torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=5e-4)

    ## for paper writing total epoch=500
    scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[400],gamma=0.5, verbose=True)
    sched = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=25, after_scheduler=scheduler_steplr)


    optimizer.zero_grad()
    optimizer.step()


    # ### load checkpoint
    epoch_start=0
    if args.resume is not None:                                         
        print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume,map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        epoch_start=checkpoint['epoch']
        print("Loaded checkpoint '{}' (epoch {})".format(args.resume, epoch_start))
    if args.resume_im is not None:                                         
        print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume_im,map_location='cpu')
        model_im.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        epoch_start=checkpoint['epoch']
        print("Loaded checkpoint '{}' (epoch {})".format(args.resume, epoch_start))
    # ###-----------------------------------------Training-----------------------------------------
    # ##initialize
    loss_dict = {}
    total_step = 0
    CE = nn.CrossEntropyLoss()
    best = 0.8
    # ## start training
    for epoch in range(epoch_start,args.n_epoch):
        print(epoch, optimizer.param_groups[0]['lr'])
        sched.step(epoch)
        model.train()
        model_im.eval()
        for i, (src, tgt, src_im,src_x,src_y) in enumerate(trainloader):
            src = src.permute(1,0).cuda()
            tgt = tgt.permute(1,0).cuda()
            src_x = src_x.permute(1,0).cuda()
            src_y = src_y.permute(1,0).cuda()
            with torch.no_grad():
                im_embedding,distribution = model_im(src_im.float().cuda())
            output = model(src,distribution,tgt[:-1,:],im_embedding,src_x,src_y)
            loss_ce = CE(output.reshape(-1,13), tgt[1:, :].reshape(-1))
            loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_step += 1
            loss_dict['ce']=loss_ce.item()


            ## log
            if (i+1) % 5 == 0:
                ## print
                print('epoch[{}/{}], batch[{}/{}] -- '.format(epoch+1,args.n_epoch,i+1,len(trainloader))+dict2string(loss_dict))
                ## tbord
                if args.tboard:
                    for key,value in loss_dict.items():
                        writer.add_scalar('Train '+key+'/Iterations', value, total_step)
                ## logfile
                with open(log_file_path,'a') as f:
                    f.write('epoch[{}/{}],batch [{}/{}]--'.format(epoch+1,args.n_epoch,i+1,len(trainloader))+dict2string(loss_dict)+'\n')
        
        ## Evaluate
        if (epoch+1)%5==0:
        # if True:
            model.eval()
            model_im.eval()
            metric_dict={'acc':0,'acc_last':0,'real_acc':0,'real_acc_last':0,'ce':0}
            if (epoch+1) % 1 == 0: 
                for i, (src, tgt, src_im,src_x,src_y) in enumerate(tqdm(valloader)):
                    src_org, tgt_org, src_im_org,src_x_org,src_y_org = src.clone(), tgt.clone(), src_im.clone(),src_x.clone(),src_y.clone()
                    with torch.no_grad():
                        im_embedding,distribution = model_im(src_im.float().cuda())
                        src = src.permute(1,0).cuda()
                        l,b = src.shape[:2]
                        src_x = src_x.permute(1,0).cuda()
                        src_y = src_y.permute(1,0).cuda()
                        tgt = tgt.permute(1,0).cuda()

                        start = [11]
                        start = torch.LongTensor(start).unsqueeze(0)
                        start = torch.tile(start,(b,1))
                        start = start.permute(1,0).cuda()

                        pred_temp = start
                        for j in range(13):
                            output_temp = model(src,distribution,pred_temp,im_embedding,src_x,src_y)
                            pred_temp = output_temp.argmax(2)
                            pred_temp = torch.cat((start,pred_temp),dim=0)
                            # out_num = output_temp.argmax(2)[-1].item()
                            # pred.append(out_num)
                        acc,acc_last = get_acc(output_temp.permute(1,0,2),tgt[1:, :].permute(1,0),0)
                        metric_dict['real_acc'] += acc
                        metric_dict['real_acc_last'] += acc_last

                        src_im = src_im_org.float().cuda()
                        src = src_org.permute(1,0).cuda()
                        tgt = tgt_org.permute(1,0).cuda()
                        src_x = src_x_org.permute(1,0).cuda()
                        src_y = src_y_org.permute(1,0).cuda()
                        im_embedding,distribution = model_im(src_im)
                        output = model(src,distribution,tgt[:-1,:],im_embedding,src_x,src_y)
                        loss_ce = CE(output.reshape(-1,13), tgt[1:, :].reshape(-1))
                        metric_dict['ce']+=loss_ce
                        acc,acc_last = get_acc(output.permute(1,0,2),tgt[1:, :].permute(1,0),0)
                        metric_dict['acc'] += acc
                        metric_dict['acc_last'] += acc_last

        #         ## log
                for key,value in metric_dict.items():
                    metric_dict[key] = value / len(valloader)
                lrate=get_lr(optimizer)
        #         ## print
                print('Testing epoch {}, lr {} -- '.format(epoch+1,lrate)+dict2string(metric_dict))
                ## tbord
                if args.tboard:
                    for key,value in metric_dict.items():
                        writer.add_scalar('Eval '+key+'/Epoch', value, epoch+1)
                ## logfile
                with open(log_file_path,'a') as f:
                    f.write('Testing epoch {}, lr {} -- '.format(epoch+1,lrate)+dict2string(metric_dict)+'\n')

                
                if metric_dict['real_acc_last'] > best:
                    best=metric_dict['real_acc_last']
                    state = {'epoch': epoch+1,
                            'model_state': model.state_dict(),
                            'optimizer_state' : optimizer.state_dict(),}
                    state_im = {'epoch': epoch+1,
                            'model_state': model_im.state_dict(),
                            'optimizer_state' : optimizer.state_dict(),}
                    if not os.path.exists(os.path.join(args.logdir,args.experiment_name)):
                        os.system('mkdir '+ os.path.join(args.logdir,args.experiment_name))
                    print('saving the best model at epoch {} with -- '.format(epoch+1)+dict2string(metric_dict))
                    torch.save(state, os.path.join(args.logdir,args.experiment_name,'{}_best_'.format(epoch+1)+dict2string(metric_dict)+'_tr.pkl').replace(', ','_').replace(' ','-'))
                    torch.save(state_im, os.path.join(args.logdir,args.experiment_name,'{}_best_'.format(epoch+1)+dict2string(metric_dict)+'_im.pkl').replace(', ','_').replace(' ','-'))
                # sched.step(metric_dict['real_acc_last'])

        if (epoch+1) % 100 == 0:
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            state_im = {'epoch': epoch+1,
                     'model_state': model_im.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            if not os.path.exists(os.path.join(args.logdir,args.experiment_name)):
                 os.system('mkdir ' + os.path.join(args.logdir,args.experiment_name))
            torch.save(state, os.path.join(args.logdir,args.experiment_name,"{}_tr.pkl".format(epoch+1)))
            torch.save(state_im, os.path.join(args.logdir,args.experiment_name,"{}_im.pkl".format(epoch+1)))
        
    #     ## test
             


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=500, 
                        help='# of the epochs')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--resume_im', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.add_argument('--experiment_name', nargs='?', type=str,default='temp',
                        help='the name of this experiment')
    parser.set_defaults(tboard=True)
    args = parser.parse_args()

    seed = 11
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


    train(args)