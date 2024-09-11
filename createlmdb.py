from torch.utils.data import DataLoader,Dataset
import os 
import pyarrow
import lmdb
import pickle
from tqdm import tqdm
from loaders import get_loader

type_list = ['p',"0","1","2","3","4","5","6","7","8","9",'s','e']

def generate_lmdb_dataset(data_set: Dataset, save_dir: str, name: str, num_workers=0, max_size_rate=1.0, write_frequency=5000):
    data_loader = DataLoader(data_set, batch_size=1, num_workers=6, collate_fn=lambda x: x)
    num_samples = len(data_set)*1
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    lmdb_path = os.path.join(save_dir, '{}.lmdb'.format(name))

    db = lmdb.open(lmdb_path,
                   subdir=False,
                   map_size=int(1099511627776 * max_size_rate),
                   readonly=False,
                   meminit=True,
                   map_async=True)
    
    idx = 0
    for _ in range(1):
        for _, data in enumerate(tqdm(data_loader)):
            ## data -> can be a list: [ndarray,str,tuple,int,...,float], or any single element in [ndarray,str,tuple,int,...,float]
            with db.begin(write=True) as txn:
                # txn.put(str(idx).encode(),pyarrow.serialize(data).to_buffer())
                txn.put(str(idx).encode(),pickle.dumps(data))
                idx+=1
    with db.begin(write=True) as txn:
        # txn.put('__len__'.encode(), pyarrow.serialize(num_samples).to_buffer())
        txn.put('__len__'.encode(), pickle.dumps(num_samples))


class ImageLMDB(Dataset):
    """
    LMDB format for image folder.
    """
    def __init__(self, db_path, db_name, transform=None, target_transform=None, backend='cv2'):
        self.env = lmdb.open(os.path.join(db_path, '{}.lmdb'.format(db_name)),
                             subdir=False,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin() as txn:
            self.length = pyarrow.deserialize(txn.get(b'__len__'))

        self.map_list = [str(i).encode() for i in range(self.length)]
        self.transform = transform
        self.target_transform = target_transform
        self.backend = backend

    def __len__(self):
        return self.length*1

    def __getitem__(self, item):
        with self.env.begin() as txn:
            byteflow = txn.get(self.map_list[item])
        unpacked = pyarrow.deserialize(byteflow)
        ## unpacked is exectly the data list that you convert to lmdb
        data = unpacked[0]
        return data



if __name__ == '__main__':

    ## image to lmdb
    ### Setup Dataloader
    save_dir = './dataset/WMeter5K_lmdb' 
    data_loader = get_loader('lmdb_create')
    t_loader = data_loader(['./dataset/WMeter5K/label_train.txt'])
    v_loader = data_loader('./dataset/WMeter5K/label_val.txt')
    generate_lmdb_dataset(data_set=v_loader,save_dir=save_dir,name='test',num_workers=6)
    generate_lmdb_dataset(data_set=t_loader,save_dir=save_dir,name='train',num_workers=6)


    ## lmdb to image
    # dt = ImageLMDB(db_path='dataset/WMeter5K_lmdb/',db_name='test')
    # trainloader = DataLoader(dt, batch_size=2, num_workers=4, shuffle=True,drop_last = True,pin_memory=True)
    # for i, data in enumerate(trainloader):
    #     image = data[0]
