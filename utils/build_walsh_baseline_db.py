import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
import random
import glob
import random
from lungmask import LMInferer

inferer = LMInferer(modelname="R231CovidWeb")

def create_montages(input_list, img_dir, train):
    labels = []
    paths = []
    patient_id = []
    pt_idx = 0
    for patient in input_list:
        data = np.load(patient, allow_pickle=True)
        # case_uid = data['case']
        ct = data['ct']
        # ct = np.flip(ct, axis=1)
        label = data['label'] - 1  # c['label'] is from 1 to 4 but needs to be 0 to 3
        try:
            lung = data['lung']
        except:
            lung = inferer.apply(ct)
        print(patient)
        print(lung.shape)
        print(lung)
        print('#'*100)
        ct[ct < -1000] = -1000
        ct[ct > 200] = 200
        ct = (((ct + 1000) / 1200) * 255).astype(np.uint8)
        ct_masked = ct * lung
        # crop to only lung area
        cb = [(np.min(a), np.max(a)) for a, m in zip(np.where(lung), list(lung.shape))]
        ct_masked = ct_masked[cb[0][0]:cb[0][1], cb[1][0]:cb[1][1], cb[2][0]:cb[2][1]]
        # eliminate apical 10% of slices
        axial_value = ct_masked.shape[0]
        # Get 10% value for channels, divide by two, convert to int
        filter_value = int(axial_value / 10 / 2)
        # Filter 10% of image out
        lower_bound = filter_value
        upper_bound = axial_value - filter_value
        ct_masked = ct_masked[lower_bound:upper_bound, :, :]
        num_blocks = 4
        block_size = ct_masked.shape[0] // num_blocks
        montage_size = (350, 350)
        for n in range(500):
            montage_slices = []
            used_slices = set()
            for j in range(num_blocks):
                # Randomly select a slice from each block
                while True:
                    slice_index = random.randint(j * block_size, (j + 1) * block_size - 1)
                    if slice_index not in used_slices:
                        used_slices.add(slice_index)
                        break
                slice_data = np.flip(ct_masked[slice_index, :, :], axis=0)
                montage_slices.append(slice_data)
            montage_1 = np.concatenate([montage_slices[0], montage_slices[1]], axis=1)
            montage_2 = np.concatenate([montage_slices[2], montage_slices[3]], axis=1)
            montage = np.concatenate([montage_1,montage_2], axis=0)
            montage = np.concatenate([montage[:,:,None], montage[:,:,None], montage[:,:,None]], axis=-1)
            path = '{}/{}_{}.jpg'.format(img_dir, pt_idx, n)
            im = Image.fromarray(montage)
            im = im.resize(montage_size)
            im.save(path)
            paths.append(path)
            labels.append(label)
            patient_id.append(patient)
        pt_idx +=1
    df = pd.DataFrame(list(zip(patient_id, paths, labels)), columns=['id', 'filename', 'label'])
    csv_path = os.path.join(img_dir, 'split.csv')
    df.to_csv(csv_path, index=False)


def build_db(data_path='data/intact_original/INTACT_Nii_npz_only', seed=1):
    patients = glob.glob(data_path + '/*.npz')
    random.seed(seed)
    random.shuffle(patients)
    # split_index = int(0.7 * len(patients))
    # train_list = patients[:split_index]
    # val_list = patients[split_index:]
    img_dir = f'data/walsh_db/seed{seed}'

    create_montages(patients, img_dir, train=True)
    # create_montages(val_list, img_dir, train=False)

if __name__ == "__main__":
    seeds = [1]
    path = ''
    for seed in seeds:
        build_db(data_path=path, seed=seed)