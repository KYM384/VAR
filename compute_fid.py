from cleanfid import fid
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="generated_freqar_shift0p1_5")
    args = parser.parse_args()

    path = args.path

    fid_score = fid.compute_fid(
            fdir1=path,
            fdir2=None,
            mode="legacy_tensorflow",
            num_workers=12,
            dataset_name="CIFAR10",
            dataset_res=32,
        )

    print(path, fid_score)
