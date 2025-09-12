import xarray as xr
import numpy as np

# Open one file
file_path = "ocean-data/data/incois/1900121/nodc_1900121_prof.nc"
ds = xr.open_dataset(file_path)

print("=== ALL VARIABLES AND THEIR DETAILS ===\n")

for var_name in ds.variables:
    var = ds[var_name]
    print(f"Variable: {var_name}")
    print(f"  Shape: {var.shape}")
    print(f"  Dimensions: {var.dims}")
    print(f"  Data type: {var.dtype}")
    
    # Show sample data for different variable types
    try:
        if var.size == 1:
            print(f"  Value: {var.values}")
        elif len(var.dims) == 1 and var.dims[0] == 'n_prof':
            print(f"  Sample values: {var.values[:3]}")  # First 3 profiles
        elif len(var.dims) == 2 and 'n_prof' in var.dims and 'n_levels' in var.dims:
            print(f"  Sample (first profile, first 3 levels): {var.values[0, :3]}")
        else:
            print(f"  Sample: {str(var.values)[:100]}...")
    except:
        print(f"  Could not show sample data")
    
    # Show attributes if any
    if var.attrs:
        print(f"  Attributes: {dict(var.attrs)}")
    
    print()

ds.close()