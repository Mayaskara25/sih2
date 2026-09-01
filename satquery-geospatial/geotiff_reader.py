import os
import sys
import rasterio


def read_metadata(path):
    with rasterio.open(path) as src:
        return {
            "file": os.path.basename(path),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs) if src.crs else None,
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top
            },
            "resolution": src.res,
            "dtype": src.dtypes,
            "nodata": src.nodata,
            "driver": src.driver
        }


def print_metadata(path):
    try:
        metadata = read_metadata(
            path
        )

        print(
            f'File: {metadata["file"]}'
        )

        print(
            f'Width: {metadata["width"]}'
        )

        print(
            f'Height: {metadata["height"]}'
        )

        print(
            f'Bands: {metadata["bands"]}'
        )

        print(
            f'CRS: {metadata["crs"]}'
        )

        print(
            f'Bounds: {metadata["bounds"]}'
        )

        print(
            f'Resolution: {metadata["resolution"]}'
        )

        print(
            f'Data type: {metadata["dtype"]}'
        )

        print(
            f'NoData: {metadata["nodata"]}'
        )

        print(
            f'Driver: {metadata["driver"]}'
        )

    except Exception as error:
        print(
            f"Error: {error}"
        )


if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = "data/sample/B04.tif"


print_metadata(
    file_path
)