import xarray as xr

# Open one of your NetCDF files (adjust the path to match your structure)
file_path = "ocean-data/data/incois/1900121/nodc_1900121_prof.nc"

print("Opening NetCDF file...")
ds = xr.open_dataset(file_path)

print("\n=== DATASET OVERVIEW ===")
print(ds)

print("\n=== VARIABLES ===")
print(list(ds.variables.keys()))

print("\n=== DIMENSIONS ===")
print(ds.dims)