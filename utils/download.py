import gdown
import zipfile
import argparse

data_samples = {
    "sample@200": "https://drive.google.com/uc?id=1FfV7YyDJvNUCDP5r3-8iQfZ2-xJp_pgb",
    "sample@500": "https://drive.google.com/uc?id=1dHwUqpmSogEdjAB9rwDUL-OKFRUcVXte",
    "sample@1000": "https://drive.google.com/uc?id=1DPZrHrj3Bdte5Dc6NCZ33CAqMG-Oipa2",
    "sample@2000": "https://drive.google.com/uc?id=1PB7uGd-dUnZKnKZpZl-HvE1DVcWgX50F",
    "sample@3000": "https://drive.google.com/uc?id=1_yre5K9YYvJgSrT4xvrI8eD_htucIywA",
    "sample@4000_images": "https://drive.google.com/uc?id=1dqVB8EozEpwWzyuU80AauoQmsiw3Gtm2",
    "sample@20000": "https://drive.google.com/uc?id=1MTDpLzpmhSiZq2jSdmHx2UDPn9FC8gzO",
    "val-voets-tf": "https://drive.google.com/uc?id=1VzVgMGTkBBPG2qbzLunD9HvLzH6tcyrv",
    "train_voets": "https://drive.google.com/uc?id=1AmcFh1MOOZ6aqKm2eO7XEdgmIEqHKTZ5",
    "voets_test_images": "https://drive.google.com/uc?id=15S_V3B_Z3BOjCT3AbO2c887FyS5B0Lyd",
}


def download(data, url):
    # Download dataset
    url = url
    output = "data/{}.zip".format(data)
    gdown.download(url, output, quiet=False)

    # Uncompress dataset
    local_zip = "data/{}.zip".format(data)
    zip_ref = zipfile.ZipFile(local_zip, "r")
    zip_ref.extractall(path="data")
    zip_ref.close()


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset")
    return parser


def main():
    parser = init_argparse()
    args = parser.parse_args()

    UNLABELED = args.dataset

    URL_UNLABELED = data_samples[UNLABELED]
    download(UNLABELED, URL_UNLABELED)

if __name__ == "__main__":
    main()
