import h5py

with h5py.File('/home/ubuntu/lab8_lan_pham_ws/model.h5', 'r') as f:
    def print_group(name, obj):
        print(name, '->', 'Group' if isinstance(obj, h5py.Group) else 'Dataset', 
              f"{obj.shape if hasattr(obj, 'shape') else ''}")
    f.visititems(print_group)