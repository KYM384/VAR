from cleanfid import fid


path = "generated_var_p95_cfg2"

fid = fid.compute_fid(
    fdir1=path,
    fdir2=None,
    mode="legacy_tensorflow",
    num_workers=12,
    dataset_name="CIFAR10",
    dataset_res=32,
)

print(path, fid)