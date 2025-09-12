import xarray as xr
from pathlib import Path

data_root = Path("ocean-data/data/incois")

print("=== SURVEYING ALL NETCDF FILES ===\n")

total_files = 0
total_profiles = 0

for date_folder in sorted(data_root.iterdir()):
    if date_folder.is_dir():
        print(f"Folder: {date_folder.name}")
        
        nc_files = list(date_folder.glob("*.nc"))
        if nc_files:
            for nc_file in nc_files:
                try:
                    ds = xr.open_dataset(nc_file)
                    n_profiles = ds.sizes['n_prof']
                    platform = str(ds.platform_number.values[0]).strip("b' ")
                    
                    print(f"  {nc_file.name}: {n_profiles} profiles, Platform {platform}")
                    total_files += 1
                    total_profiles += n_profiles
                    ds.close()
                except Exception as e:
                    print(f"  ERROR reading {nc_file.name}: {e}")
        else:
            print(f"  No .nc files found")
        print()

print(f"\n=== SUMMARY ===")
print(f"Total files: {total_files}")
print(f"Total profiles: {total_profiles}")