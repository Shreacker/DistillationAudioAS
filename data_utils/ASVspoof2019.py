import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

TTS_ATTACKS = ("A01", "A02", "A03", "A04")
VC_ATTACKS = ("A05", "A06")

def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    d_attack = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, attack, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
            d_attack[key] = attack
        return d_meta, file_list, d_attack

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


def select_files(d_meta: dict, d_attack: dict, seed: int, tts_percents=(30, 50, 70)):
    pools = {a: [] for a in TTS_ATTACKS + VC_ATTACKS}
    bonafide = []

    for key, label in d_meta.items():
        if label == 1:
            bonafide.append(key)

        else:
            attack = d_attack[key]
            pools[attack].append(key)

    for i, attack in enumerate(sorted(pools)):
        keys = sorted(pools[attack])
        rs = np.random.RandomState([seed, i])
        pools[attack] = [keys[j] for j in rs.permutation(len(keys))]

    cap_tts = len(TTS_ATTACKS) * min(len(pools[a]) for a in TTS_ATTACKS)
    cap_vc = len(VC_ATTACKS) * min(len(pools[a]) for a in VC_ATTACKS)

    n_spoof = min(
        min(
            cap_tts * 100 // p, cap_vc * 100 // (100 - p)
        )
        for p in tts_percents
    )

    subsets = {}
    info = {
        "n_spoof": n_spoof,
        "n_bonafide": len(bonafide),
        "groups": {}
    }
    for p in tts_percents:
        n_tts = round(n_spoof * p / 100)
        n_vc = n_spoof - n_tts

        if n_tts > cap_tts or n_vc > cap_vc:
            raise ValueError("TTS/VC vượt quá giới hạn dữ liệu")

        selected = list(bonafide)
        for attacks, n in ((TTS_ATTACKS, n_tts), (VC_ATTACKS, n_vc)):
            base, rem = divmod(n, len(attacks))
            limits = [base + (1 if i < rem else 0) for i in range(len(attacks))]
            for attack, limit in zip(attacks, limits):
                selected += pools[attack][:limit]

        name = f"tts{p}_vc{100 - p}"
        subsets[name] = sorted(selected)
        info["groups"][name] = {"tts": n_tts, "vc": n_vc}

    return subsets, info

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key