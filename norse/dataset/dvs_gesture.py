import torch
import aedat
import os

import logging

from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class DVSGesture(torch.utils.data.Dataset):
    """
    The IBM gesture dataset containing 11 hand gestures from 29 subjects under
    3 illumination conditions recorded from a DVS128 camera

    Unless otherwise specified, the dataset will be downloaded to the default
    (or given) path

    **Depends** on the `AEDAT <https://github.com/norse/aedat>`_ library

    Source: http://www.research.ibm.com/dvsgesture/

    Parameters:
        root (str): The root of the dataset directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    mappings_file = "gesture_mapping.csv"
    tensor_size = (10000000, 128, 128)
    tgz_md5 = "8a5c71fb11e24e5ca5b11866ca6c00a1"
    filename = "ibm_dvs.tar.gz"
    data_dir = "DvsGesture"
    url = "https://kth.box.com/shared/static/7fk64kjuit6exwlzbzfetwoa93qsuoy4.gz"

    class_to_idx = {
        "hand_clapping": 1,
        "right_hand_wave": 2,
        "left_hand_wave": 3,
        "right_arm_clockwise": 4,
        "right_arm_counter_clockwise": 5,
        "left_arm_clockwise": 6,
        "left_arm_counter_clockwise": 7,
        "arm_roll": 8,
        "air_drums": 9,
        "air_guitar": 10,
        "other_gestures": 11,
    }
    classes = class_to_idx.values

    test_list = [
        ["ff883f9b52c6f84d14be18674d7e8a85", "user02_fluorescent.aedat"],
        ["c82fa62429ee51858985ff07afccd8b9", "user02_fluorescent_led.aedat"],
        ["84c62a8f3336b01f4a2f78c24ad0ea0e", "user02_lab.aedat"],
        ["696db3d8da3defbff96d2cb3d5da78a8", "user02_led.aedat"],
        ["08bb21aa108cb4f5c4bd5899666ec4b2", "user02_natural.aedat"],
        ["07bf070c98057eb91805f03f7e39dd14", "user11_fluorescent.aedat"],
        ["bc73c49520f4eeecbfddb7121e1bbb48", "user11_fluorescent_led.aedat"],
        ["2005eade67ae4180b54b900eca08fc14", "user11_natural.aedat"],
        ["42e215ea69d8468afd01b4e18f261b12", "user17_fluorescent.aedat"],
        ["03c8f7860bda6517bec0ae27f9c95715", "user17_fluorescent_led.aedat"],
        ["d1ced90bed55b58ba37c419a61224b26", "user17_lab.aedat"],
        ["52ff6c6a4dd55a03049687613548b2ae", "user17_led.aedat"],
        ["9146316b5b74891f5c2c6bc37181f838", "user17_natural.aedat"],
        ["ba6a8dc4e95da3b53c5a52ac08766a12", "user18_fluorescent.aedat"],
        ["729b4f6ad024d3ce101e36a50581129d", "user18_fluorescent_led.aedat"],
        ["39ea291876e716b6f7f416b5a7848256", "user18_lab.aedat"],
        ["97ed078c4104dc0928036ede88466de4", "user18_led.aedat"],
        ["c1bf9480f97c2098d79995edce5b359d", "user23_fluorescent.aedat"],
        ["efa48186c9435edfe0aadaf2c2766ce7", "user23_fluorescent_led.aedat"],
        ["e6b2fcda56b47da869b2b4b48c587447", "user23_lab.aedat"],
        ["520f05beef4b623d34bb6039ac752ee9", "user23_led.aedat"],
        ["5ddcb8a3b94cb8e5f2ec18104e808948", "user26_fluorescent.aedat"],
        ["08604845038463d0897f91e35a69ea45", "user26_fluorescent_led.aedat"],
        ["a52e368c850380615b310074e8166941", "user26_lab.aedat"],
        ["147836193bc15a0786d2f6f08d9958e2", "user26_led.aedat"],
        ["bcf0488fdd7ee6daf47c9db80aad1daf", "user26_natural.aedat"],
    ]
    train_list = [
        ["e113517a64ab66cf2259dd8428058222", "user01_fluorescent.aedat"],
        ["7280eac9ec165cf0ff8e14f56679a9f8", "user01_fluorescent_led.aedat"],
        ["e7e1d66e2b9d11164851fbd7f1f8ac1f", "user01_lab.aedat"],
        ["dc50d295cb1af9277486f22a556c3153", "user01_led.aedat"],
        ["401ff23f0c91f095ed8a0701ab24794a", "user01_natural.aedat"],
        ["26cdbc971fec225d132054b37f3c8a55", "user03_fluorescent.aedat"],
        ["3fc93616aa6d3ee6f7ed36bc5a199875", "user03_fluorescent_led.aedat"],
        ["47e772f3ee80edd58e30493755a6ebbc", "user03_led.aedat"],
        ["49dba7cbf22a6e209392869d02f8187b", "user03_natural.aedat"],
        ["96727cc1c64183bc4db5a28db6427289", "user04_fluorescent.aedat"],
        ["ba33603a7e560ec08e5b62f567542658", "user04_fluorescent_led.aedat"],
        ["8c3ece51743db0a6191ca5e5c9903da7", "user04_led.aedat"],
        ["0cffd708c224df0cbf061644174df9de", "user04_natural.aedat"],
        ["b33a87123d204e591a5bbbe51a9883a2", "user05_fluorescent.aedat"],
        ["bec648630aece183c1e33e7f69397f25", "user05_fluorescent_led.aedat"],
        ["6d56c1fc1cfa56e908d0a35efbd19823", "user05_lab.aedat"],
        ["9bd2586003386938704d4e0d84f6e393", "user05_led.aedat"],
        ["c4a60c769fcc1cecf12b210338b2ec23", "user05_natural.aedat"],
        ["824d7e00f20fbb4145475e4c1db24b05", "user06_fluorescent.aedat"],
        ["9623d0caed540bf3735470e27ad1a550", "user06_fluorescent_led.aedat"],
        ["8f6d70570d0c0192b57a98be64deaf96", "user06_lab.aedat"],
        ["e92c7e2649261fc326f849783c21acc9", "user06_led.aedat"],
        ["f7c2102462d126300915e6a1aa266957", "user06_natural.aedat"],
        ["9ca2aec9e9b80671dba00223c36a0203", "user07_fluorescent.aedat"],
        ["b0589d8d271a67cfb859c39d9d237c7c", "user07_fluorescent_led.aedat"],
        ["340dd0cbce9a311057ffcd34dfc9dc8a", "user07_lab.aedat"],
        ["a2276802aaf84bce0bb564470adf3fa2", "user07_led.aedat"],
        ["8da2c7748c07669fb11326c7422eeecf", "user08_fluorescent.aedat"],
        ["ec2ba4a7071910530b4c40a77120bb2b", "user08_fluorescent_led.aedat"],
        ["0e199262643a851d29287a335924c99c", "user08_lab.aedat"],
        ["d0bd827564492d62960acc2318058877", "user08_led.aedat"],
        ["64e0c9e329061c09df74a65a39310de5", "user09_fluorescent.aedat"],
        ["fa0b7ed7d7b99ce5e9f25d2f739d59c9", "user09_fluorescent_led.aedat"],
        ["2cb402d45cfe94f0ad97386d3f688a17", "user09_lab.aedat"],
        ["69addacc47fed56aef710b8b8d47fe55", "user09_led.aedat"],
        ["670e32f14a2d0165f3260d756ae1d23d", "user09_natural.aedat"],
        ["889a722076ceb0c062e9b0fc97666ee1", "user10_fluorescent.aedat"],
        ["61428512d590714fa33ada9e2137201d", "user10_fluorescent_led.aedat"],
        ["dbd79c303461407862161134a0bdee4a", "user10_lab.aedat"],
        ["ee5f1b5ee13a3d61ce418bc03f8297e9", "user10_led.aedat"],
        ["a8ad16870e944f5dfc1e288b8bd931db", "user12_fluorescent_led.aedat"],
        ["cf9e1979c3daf8c048b00e3e1fda8964", "user12_led.aedat"],
        ["bd305f1f4112cac4a060c7dd34340213", "user13_fluorescent.aedat"],
        ["9b9399d723c5f6a5168bfa8fd0c30941", "user13_fluorescent_led.aedat"],
        ["6e5b4ae30b095ac44c969480e16ddbdf", "user13_lab.aedat"],
        ["5bae5bbd8d38375f1e082aad87e59032", "user13_led.aedat"],
        ["9b902e35ef6352d151ef0cd313e2f0f4", "user13_natural.aedat"],
        ["f63461d3be511b8a0225e380eed8c00c", "user14_fluorescent.aedat"],
        ["c0a0bb30ad897fcb33b7e5dbda53aa67", "user14_fluorescent_led.aedat"],
        ["3426123998de30b62d14577e713692b7", "user14_led.aedat"],
        ["101d0c8c4c79e0e79fe9279566847c95", "user14_natural.aedat"],
        ["5a4df5a4703f94f54e59415436e2f2f1", "user15_fluorescent.aedat"],
        ["b17b1401643fac32679b2b30f0bf5216", "user15_fluorescent_led.aedat"],
        ["55e8efeafa4aa5002f80b4d1ac716e77", "user15_lab.aedat"],
        ["a7af361ccdd1a3d0ca30ed3487b327ca", "user15_led.aedat"],
        ["0a289cd4774d0cb02d7ad823a6d4ee3f", "user15_natural.aedat"],
        ["e91b9a4bdeaadbca8b7d57de0305ce15", "user16_fluorescent.aedat"],
        ["ab3c405c6ce16746ddc4960c6dd420ce", "user16_lab.aedat"],
        ["78cc668d66af4460101090bdd7dda7a7", "user16_led.aedat"],
        ["fa97da22138daca004ed608dfc53c8fd", "user16_natural.aedat"],
        ["078d675669cac51ed27e66eb09f26588", "user19_fluorescent.aedat"],
        ["87cd29d3133173083a2d4f6c2293e66b", "user19_fluorescent_led.aedat"],
        ["120d0079824d8ae4a557068d06ca707c", "user19_lab.aedat"],
        ["b97e441db4ac06a809f7c091d3d8ba37", "user19_led.aedat"],
        ["81d7f4081034e2a005e3e5f243145abc", "user19_natural.aedat"],
        ["39603316e10bf095461196234ba6c3da", "user20_fluorescent.aedat"],
        ["bd7a4fc388ae050a7d2047041b8638ce", "user20_fluorescent_led.aedat"],
        ["4b78a4c6cb444e0bee2d29d9fed7b19d", "user20_led.aedat"],
        ["d9fd906b2d0cc8f125280d3b6ff160c4", "user21_fluorescent.aedat"],
        ["a142ce4e5b9b725d5244a19f5e6cb77e", "user21_fluorescent_led.aedat"],
        ["0442ba221fd5af22769ca34d575eee23", "user21_lab.aedat"],
        ["b86a9f4cb959a959ed036fbfda076120", "user21_natural.aedat"],
        ["1e161012c767220db83b08ebf0364efa", "user22_fluorescent.aedat"],
        ["0145ebf561d861fc95a5a7df315c770d", "user22_fluorescent_led.aedat"],
        ["66ab68a4743213de0862f7e430f96b4c", "user22_lab.aedat"],
        ["6ceafd193d838d4371fb6615dbf2d36f", "user22_led.aedat"],
        ["49f7df28f49c2a3f15beae96987592e1", "user22_natural.aedat"],
        ["f89a2526dcf2826b9bf902d8d285a5ec", "user24_fluorescent.aedat"],
        ["e4f9171c7868c5f7f1fe861644a97e81", "user24_fluorescent_led.aedat"],
        ["e7c34ba86e17de56ca5336f6f02f8b02", "user24_led.aedat"],
        ["91ad11ed0a229d51e11cdd9a7eadce19", "user25_fluorescent.aedat"],
        ["290c4801a674439a7f8822ad74dbe8dd", "user25_led.aedat"],
        ["566dfaa35b1d67877eb779444a459b63", "user27_fluorescent.aedat"],
        ["b788383a93bf92825af81ed554c98577", "user27_fluorescent_led.aedat"],
        ["92d60850c7fbb32d21d3032817c5f930", "user27_led.aedat"],
        ["ae7b4eef347c2f72008af1899908b36f", "user27_natural.aedat"],
        ["b9c9f09a6d4a907b08ead198f48992dd", "user28_fluorescent.aedat"],
        ["be834d6b872d60244c1f2615793328ce", "user28_fluorescent_led.aedat"],
        ["983f84996a8d8051d0a10b591adc90f7", "user28_lab.aedat"],
        ["d61b157c1d0985f805b5549d36296ad6", "user28_led.aedat"],
        ["35cd06ca7b820592355bd26407aab23b", "user28_natural.aedat"],
        ["fbf320693883263cb06ce5d848c22004", "user29_fluorescent.aedat"],
        ["5bf8cf03464b2676bffc37e00dfeb656", "user29_fluorescent_led.aedat"],
        ["b953a9e5af1393ded645e8e4b51b8bba", "user29_lab.aedat"],
        ["043d9e7935bb1318251b5c320b87d9eb", "user29_led.aedat"],
        ["f8c2c6a1374e44f18bf5d1f9f292e40e", "user29_natural.aedat"],
    ]

    def __init__(self, root, train=True, download=False):
        super(DVSGesture).__init__()

        self.root = root
        self.train = train

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        logging.info("DVS Gesture dataset downloaded and verified")

        self.data = aedat.DVSGestureData()

        self.current_list = self.train_list if self.train else self.test_list
        self.current_files = [
            (
                os.path.join(self.root, self.data_dir, x[1]),
                (os.path.join(self.root, self.data_dir, f"{x[1][:-6]}_labels.csv")),
            )
            for x in self.current_list
        ]

        for event_file, label_file in self.current_files:
            self.data.load(event_file, label_file)

    def _check_integrity(self):
        for fentry in self.train_list + self.test_list:
            md5, filename = fentry[0], fentry[1]
            fpath = os.path.join(self.root, self.data_dir, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

    def __getitem__(self, index):
        return (
            aedat.convert_polarity_events(
                self.data.datapoints[index].events,
                self.tensor_size,  # Force to specific width for torch.stack
            ),
            self.data.datapoints[index].label,
        )

    def __len__(self):
        return len(self.data.datapoints)
