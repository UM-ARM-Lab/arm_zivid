import numpy as np

def get_local_hostname():
    import socket
    hostname = socket.getfqdn()
    if len(hostname) == 0:
        hostname = "Unknown"
    if not ".local" in hostname:
        hostname += ".local"
    return hostname

# Saving as H5 utility copied from zxhuang97/force_tool
def store_h5_dict(file_path, data_dict, compression="lzf", **ds_kwargs):
    import h5py
    def is_field_homogeneous(arr):
        if not isinstance(arr[0], np.ndarray):
            return True
        base_s = arr[0].shape
        same_shape = np.array([base_s == arr[i].shape for i in range(len(arr))])
        return same_shape.all()

    with h5py.File(file_path, 'w') as hf:
        for k, v in data_dict.items():
            if is_field_homogeneous(v):
                hf.create_dataset(k, data=v, **ds_kwargs)
            else:
                # Deal with PC
                grp = hf.create_group(k)
                for i, element in enumerate(data_dict[k]):  # ← now store as list, not np.stack
                    grp.create_dataset(f"{i:05d}", data=element)

def store_zarr_dict(file_path, data_dict, compression="lz4", **ds_kwargs):
    import zarr
    
    def is_field_homogeneous(arr):
        if not isinstance(arr[0], np.ndarray):
            return True
        base_s = arr[0].shape
        same_shape = np.array([base_s == arr[i].shape for i in range(len(arr))])
        return same_shape.all()

    # Ensure zarr v2 compatibility
    store = zarr.DirectoryStore(file_path)
    root = zarr.group(store=store, overwrite=True)
    
    for k, v in data_dict.items():
        if is_field_homogeneous(v):
            # Use zarr v2 compatible syntax
            root.create_dataset(k, data=np.array(v), compressor=zarr.Blosc(cname=compression), **ds_kwargs)
        else:
            # Deal with PC - create group for variable-sized arrays
            grp = root.create_group(k)
            for i, element in enumerate(data_dict[k]):
                grp.create_dataset(f"{i:05d}", data=element, compressor=zarr.Blosc(cname=compression))

def get_file_extension(output_format):
    """Get file extension based on output format"""
    if output_format.lower() == "zarr":
        return ".zarr"
    elif output_format.lower() == "h5":
        return ".h5"
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def store_data_dict(file_path, data_dict, output_format="h5", **kwargs):
    """Universal store function that chooses format based on output_format"""
    if output_format.lower() == "zarr":
        # Remove compression from kwargs for zarr and handle separately
        zarr_kwargs = {k: v for k, v in kwargs.items() if k != 'compression'}
        compression = kwargs.get('compression', 'lz4')
        store_zarr_dict(file_path, data_dict, compression=compression, **zarr_kwargs)
    elif output_format.lower() == "h5":
        store_h5_dict(file_path, data_dict, **kwargs)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")