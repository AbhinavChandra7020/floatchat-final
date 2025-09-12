import xarray as xr

file_path = "ocean-data/data/incois/1900121/nodc_1900121_prof.nc"
ds = xr.open_dataset(file_path)

print("=== KEY INFORMATION ===")
print(f"Number of profiles: {ds.dims['n_prof']}")
print(f"Number of depth levels: {ds.dims['n_levels']}")
print(f"Platform number: {ds.platform_number.values}")

print("\n=== FIRST PROFILE DATA ===")
print(f"Latitude: {ds.latitude.values[0]}")
print(f"Longitude: {ds.longitude.values[0]}")
print(f"Date: {ds.juld.values[0]}")

print("\n=== SAMPLE MEASUREMENTS (first 5 levels of first profile) ===")
print("Pressure | Temperature | Salinity")
for i in range(5):
    pres = ds.pres.values[0, i] if not ds.pres.isnull().values[0, i] else "N/A"
    temp = ds.temp.values[0, i] if not ds.temp.isnull().values[0, i] else "N/A"
    psal = ds.psal.values[0, i] if not ds.psal.isnull().values[0, i] else "N/A"
    print(f"{pres:8.1f} | {temp:11.3f} | {psal:8.3f}")