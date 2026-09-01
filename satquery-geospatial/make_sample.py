import os
import numpy as np
import rasterio

from rasterio.transform import from_origin


width = 500
height = 500

output_folder = "data/sample"

os.makedirs(
    output_folder,
    exist_ok=True
)

transform = from_origin(
    72.8,
    19.2,
    0.0001,
    0.0001
)

np.random.seed(42)

blue = np.random.randint(
    500,
    1500,
    size=(height, width),
    dtype=np.uint16
)

green = np.random.randint(
    700,
    2000,
    size=(height, width),
    dtype=np.uint16
)

red = np.random.randint(
    600,
    1800,
    size=(height, width),
    dtype=np.uint16
)

nir = np.random.randint(
    1000,
    2500,
    size=(height, width),
    dtype=np.uint16
)

red[100:350, 100:350] = 700
nir[100:350, 100:350] = 4000
green[100:350, 100:350] = 1800

green[350:450, 50:300] = 2500
nir[350:450, 50:300] = 400
red[350:450, 50:300] = 500
blue[350:450, 50:300] = 1800


def save_band(filename, data):
    path = os.path.join(
        output_folder,
        filename
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform
    ) as dst:
        dst.write(
            data,
            1
        )

    print(
        f"Created {path}"
    )


save_band(
    "B02.tif",
    blue
)

save_band(
    "B03.tif",
    green
)

save_band(
    "B04.tif",
    red
)

save_band(
    "B08.tif",
    nir
)

print(
    "Sample GeoTIFF files created."
)