import torch
import aedat
import glob
import os


class DVSGesture(torch.utils.data.Dataset):
    def __init__(self, root):
        super(DVSGesture).__init__()

        self.root = root
        files = glob.glob(os.path.join(self.root, "*.aedat"))
        self.data = aedat.DVSGestureData()

        for events, labels in [(x, f"{x[:-6]}_labels.csv") for x in files]:
            self.data.load(events, labels)

    def __getitem__(self, index):
        return (
            aedat.convert_polarity_events(self.data.datapoints[index].events),
            self.data.datapoints[index].label,
        )

    def __len__(self):
        return len(self.data.datapoints)
