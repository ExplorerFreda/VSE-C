from torch.utils.data.dataset import Dataset

import numpy as np
import jacinle.random as random
from jacinle.utils.enum import JacEnum
from jacinle.io import load_json, load
from jactorch.data.dataloader import JacDataLoader
from jactorch.data.collate import VarLengthCollate

__all__ = ['CompletionDataset', 'make_dataloader']


class CompletionDatasetMode(JacEnum):
    SAMPLE = 'sample'
    ALL = 'all'
    NOUN = 'noun'
    PREP = 'prep'


class CompletionDataset(Dataset):
    def __init__(self, vocab, image_embeddings, captions, mode='sample'):
        self.vocab = vocab
        self.image_embeddings = image_embeddings
        self.captions = captions
        self.mode = CompletionDatasetMode.from_string(mode)
        self.load_files()

    def load_files(self):
        print('Loading captions and precomputed image embeddings')
        self.image_embeddings = load(self.image_embeddings)
        self.captions = load_json(self.captions)
        assert len(self.captions) == len(self.image_embeddings)
        # self.captions = [self.captions[i] for i in non_empty_inds]
        # self.image_embeddings = self.image_embeddings[non_empty_inds]
        if self.mode is CompletionDatasetMode.SAMPLE:
            self.non_empty_inds = [i for i, c in enumerate(self.captions) if len(c['replace']) > 0]
        else:
            if self.mode is CompletionDatasetMode.ALL:
                replace = lambda c: c['replace']
            elif self.mode is CompletionDatasetMode.NOUN:
                replace = lambda c: c['replace_noun']
            elif self.mode is CompletionDatasetMode.PREP:
                replace = lambda c: c['replace_prep']
            self.all_inds = [(i, r) for i, c in enumerate(self.captions) for r in replace(c)]

    def __getitem__(self, item):
        if self.mode is CompletionDatasetMode.SAMPLE:
            item, split = self.non_empty_inds[item], None
        else:
            item, split = self.all_inds[item]

        image = self.image_embeddings[item]
        caption = self.captions[item]
        sent = caption['sentence'].split()
        if split is None:
            split = random.choice_list(caption['replace'])
        f = ' '.join(sent[:split])
        b = ' '.join(reversed(sent[split + 1:]))

        return dict(
                sent_f=np.array(self.vocab(f)), 
                sent_b=np.array(self.vocab(b)), 
                image=image, 
                label=self.vocab(sent[split], is_sent=False)[0]
        )

    def __len__(self):
        if self.mode is CompletionDatasetMode.SAMPLE:
            return len(self.non_empty_inds)
        else:
            return len(self.all_inds)


def make_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False):
    return JacDataLoader(dataset, 
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
            pin_memory=pin_memory, drop_last=drop_last,
            collate_fn=VarLengthCollate(['sent_f', 'sent_b'], 'pad')
    )

