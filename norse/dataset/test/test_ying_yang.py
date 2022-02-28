from norse.dataset.yingyang import YinYangDataset

def test_generation():
    dataset_train = YinYangDataset(size=20, seed=42)
    dataset_validation = YinYangDataset(size=30, seed=41)
    dataset_test = YinYangDataset(size=40, seed=40)