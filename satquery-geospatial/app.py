import io
import json

import numpy as np
import pydeck as pdk
import streamlit as st

from PIL import Image
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling, transform_bounds


st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

:root {
    --bg: #070b0c;
    --surface: #0d1314;
    --surface-2: #11191a;
    --border: #293233;
    --lime: #baff00;
    --lime-soft: #d4ff65;
    --text: #f4f7f3;
    --muted: #899493;
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 20% 0%, rgba(186,255,0,0.045), transparent 24%),
        radial-gradient(circle at 80% 20%, rgba(186,255,0,0.025), transparent 22%),
        linear-gradient(180deg, #070b0c 0%, #090d0e 100%);
    color: var(--text);
}

.block-container {
    max-width: 1380px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

[data-testid="stSidebar"] {
    background: #090e0f;
    border-right: 1px solid #202829;
}

[data-testid="stSidebar"] * {
    color: #eef4ef;
}

[data-testid="stSidebar"] textarea {
    background: #101617 !important;
    color: white !important;
    border: 1px solid #2a3435 !important;
    border-radius: 13px !important;
}

[data-testid="stSidebar"] textarea:focus {
    border-color: #baff00 !important;
    box-shadow: 0 0 0 3px rgba(186,255,0,0.12) !important;
}

h1, h2, h3 {
    color: #f7faf5 !important;
    letter-spacing: -0.5px;
}

.hero {
    padding: 42px 44px;
    border-radius: 22px;
    border: 1px solid #242c2d;
    background:
        radial-gradient(circle at 85% 15%, rgba(186,255,0,0.09), transparent 28%),
        linear-gradient(135deg, #0d1314 0%, #090e0f 100%);
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -90px;
    top: -100px;
    border: 1px solid rgba(186,255,0,0.12);
    border-radius: 50%;
}

.hero::before {
    content: "";
    position: absolute;
    width: 140px;
    height: 140px;
    right: 40px;
    top: 20px;
    border: 1px solid rgba(186,255,0,0.08);
    border-radius: 50%;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #baff00;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.eyebrow::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #baff00;
    box-shadow: 0 0 10px 2px rgba(186,255,0,0.6);
}

.hero-title {
    color: #f5f7f3;
    font-size: 46px;
    line-height: 1.08;
    font-weight: 800;
    max-width: 720px;
    letter-spacing: -1.5px;
}

.hero-title span {
    color: #baff00;
}

.hero-text {
    max-width: 640px;
    color: #929c9b;
    font-size: 15px;
    line-height: 1.7;
    margin-top: 16px;
}

.feature-row {
    display: grid;
    grid-template-columns: 1fr 1.15fr 1fr;
    gap: 14px;
    margin-bottom: 30px;
}

.feature {
    min-height: 170px;
    padding: 24px;
    background: #0e1415;
    border: 1px solid #2a3233;
    border-radius: 18px;
    transition: border-color 0.15s ease, transform 0.15s ease;
}

.feature:hover {
    border-color: #3a4546;
}

.feature.highlight {
    background: #baff00;
    border-color: #baff00;
    transform: translateY(-7px);
}

.feature.highlight:hover {
    transform: translateY(-9px);
}

.feature-number {
    font-size: 15px;
    font-weight: 800;
    color: #baff00;
    font-variant-numeric: tabular-nums;
}

.feature.highlight .feature-number,
.feature.highlight .feature-title,
.feature.highlight .feature-text {
    color: #111511;
}

.feature-title {
    margin-top: 18px;
    color: white;
    font-size: 17px;
    line-height: 1.35;
    font-weight: 750;
}

.feature-text {
    margin-top: 8px;
    color: #788382;
    font-size: 12.5px;
    line-height: 1.55;
}

.workspace-title {
    font-size: 28px;
    font-weight: 800;
    margin: 18px 0 4px 0;
    letter-spacing: -0.5px;
}

.workspace-title span {
    color: #baff00;
}

.workspace-sub {
    color: #788382;
    margin-bottom: 22px;
    font-size: 14px;
}

[data-testid="stFileUploader"] {
    background: #0d1314;
    border: 1px dashed #384344;
    border-radius: 16px;
    padding: 12px;
    transition: border-color 0.15s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #baff00;
}

[data-testid="stMetric"] {
    background: #0e1415;
    padding: 18px;
    border: 1px solid #273031;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    color: #75807f !important;
}

[data-testid="stMetricValue"] {
    color: #f5f7f3 !important;
    font-weight: 750;
}

button[data-baseweb="tab"] {
    background: #0d1314;
    border: 1px solid #273031;
    border-radius: 12px;
    margin-right: 8px;
    padding: 9px 16px;
    color: #98a2a1;
    font-weight: 700;
    transition: all 0.15s ease;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: #baff00;
    color: #101510;
    border-color: #baff00;
}

.stButton > button {
    background: #baff00;
    color: #111511 !important;
    border: none;
    border-radius: 11px;
    font-weight: 800;
    padding: 10px 20px;
    min-height: 43px;
    box-shadow: none;
    transition: background 0.15s ease, transform 0.1s ease;
}

.stButton > button:hover {
    background: #d2ff52;
    color: #0b100b !important;
    border: none;
    transform: translateY(-1px);
}

.stDownloadButton > button {
    background: #111718;
    color: #baff00 !important;
    border: 1px solid #374142;
    border-radius: 11px;
    font-weight: 750;
    transition: all 0.15s ease;
}

.stDownloadButton > button:hover {
    border-color: #baff00;
    background: #151d1e;
}

div[data-baseweb="select"] > div {
    background: #111718;
    color: white;
    border: 1px solid #303a3b;
    border-radius: 11px;
}

div[data-baseweb="base-input"] {
    background: #111718 !important;
    border: 1px solid #303a3b !important;
    border-radius: 11px !important;
}

input {
    color: #ffffff !important;
}

[data-testid="stExpander"] {
    background: #0e1415;
    border: 1px solid #293233;
    border-radius: 13px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #293233;
    border-radius: 12px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

.soft-card {
    background: #0e1415;
    border: 1px solid #283132;
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0 22px 0;
}

.lime-text {
    color: #baff00;
}

.small-label {
    color: #717c7b;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.auto-pick {
    background: #0e1415;
    border: 1px solid #273031;
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.6;
}

.query-box {
    margin-top: 15px;
    padding: 13px 15px;
    background: rgba(186,255,0,0.07);
    border: 1px solid rgba(186,255,0,0.25);
    border-radius: 11px;
    color: #d9ff73;
    font-size: 13px;
    line-height: 1.5;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
<div class="eyebrow">Satellite intelligence workspace</div>
<div class="hero-title">Understand Earth data.<br><span>Without the complexity.</span></div>
<div class="hero-text">Upload geospatial imagery, inspect its structure, generate multispectral composites, analyse vegetation and water, and transform image detections into real-world coordinates.</div>
</div>
<div class="feature-row">
<div class="feature">
<div class="feature-number">01.</div>
<div class="feature-title">Multispectral<br>visualisation.</div>
<div class="feature-text">Convert raw satellite bands into clear RGB composites with automatic spatial alignment.</div>
</div>
<div class="feature highlight">
<div class="feature-number">02.</div>
<div class="feature-title">GeoTIFF intelligence<br>built in.</div>
<div class="feature-text">Extract CRS, dimensions, resolution, bounds and spectral information directly from geospatial imagery.</div>
</div>
<div class="feature">
<div class="feature-number">03.</div>
<div class="feature-title">Geographic<br>precision.</div>
<div class="feature-text">Translate pixel detections into latitude, longitude and downloadable GeoJSON regions.</div>
</div>
</div>
""", unsafe_allow_html=True)


def get_bytes(uploaded_file):
    return uploaded_file.getvalue()


def read_metadata(file_bytes):
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            return {
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": str(src.crs) if src.crs else None,
                "resolution_x": abs(src.res[0]),
                "resolution_y": abs(src.res[1]),
                "dtype": ", ".join(src.dtypes),
                "driver": src.driver,
                "nodata": src.nodata,
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top
            }


def read_band(file_bytes, band_number=1):
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            if band_number < 1 or band_number > src.count:
                raise ValueError("Selected band does not exist.")

            return {
                "array": src.read(band_number).astype(np.float32),
                "transform": src.transform,
                "crs": src.crs,
                "nodata": src.nodata,
                "profile": src.profile.copy()
            }


def align_to_reference(source, reference):
    if (
        source["array"].shape == reference["array"].shape
        and source["crs"] == reference["crs"]
        and source["transform"] == reference["transform"]
    ):
        return source["array"]

    if source["crs"] is None or reference["crs"] is None:
        raise ValueError("Automatic spatial alignment requires georeferencing.")

    destination = np.full(
        reference["array"].shape,
        np.nan,
        dtype=np.float32
    )

    reproject(
        source=source["array"],
        destination=destination,
        src_transform=source["transform"],
        src_crs=source["crs"],
        dst_transform=reference["transform"],
        dst_crs=reference["crs"],
        src_nodata=source["nodata"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear
    )

    return destination


def normalize_band(array):
    valid = array[np.isfinite(array)]

    if len(valid) == 0:
        return np.zeros(array.shape, dtype=np.uint8)

    low = np.percentile(valid, 2)
    high = np.percentile(valid, 98)

    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)

    normalized = (array - low) / (high - low)
    normalized = np.clip(normalized, 0, 1)
    normalized = np.nan_to_num(normalized)

    return (normalized * 255).astype(np.uint8)


def create_rgb(red_file, red_band, green_file, green_band, blue_file, blue_band):
    red = read_band(red_file, red_band)
    green = read_band(green_file, green_band)
    blue = read_band(blue_file, blue_band)

    green_array = align_to_reference(green, red)
    blue_array = align_to_reference(blue, red)

    return np.dstack([
        normalize_band(red["array"]),
        normalize_band(green_array),
        normalize_band(blue_array)
    ])


def rgb_to_png(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def calculate_index(first_file, first_band, second_file, second_band):
    first = read_band(first_file, first_band)
    second = read_band(second_file, second_band)

    second_array = align_to_reference(second, first)

    first_array = first["array"]
    denominator = first_array + second_array

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.abs(denominator) > 1e-8,
            (first_array - second_array) / denominator,
            np.nan
        )

    return np.clip(result, -1, 1), first["profile"]


def ndvi_preview(array):
    array = np.nan_to_num(array)

    output = np.zeros((*array.shape, 3), dtype=np.uint8)

    output[array < 0] = [45, 75, 120]
    output[(array >= 0) & (array < 0.2)] = [175, 160, 95]
    output[(array >= 0.2) & (array < 0.5)] = [135, 210, 60]
    output[array >= 0.5] = [175, 255, 0]

    return output


def ndwi_preview(array):
    array = np.nan_to_num(array)

    output = np.zeros((*array.shape, 3), dtype=np.uint8)

    output[array <= 0] = [100, 110, 75]
    output[(array > 0) & (array < 0.3)] = [90, 180, 220]
    output[array >= 0.3] = [35, 125, 235]

    return output


def get_index_stats(array):
    valid = array[np.isfinite(array)]

    if len(valid) == 0:
        return None

    return {
        "minimum": float(np.min(valid)),
        "mean": float(np.mean(valid)),
        "maximum": float(np.max(valid))
    }


def index_to_geotiff(index_array, profile):
    output_profile = profile.copy()

    output_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="deflate"
    )

    with MemoryFile() as memfile:
        with memfile.open(**output_profile) as dst:
            dst.write(index_array.astype(np.float32), 1)

        return memfile.read()


def pixel_to_latlon(file_bytes, x, y):
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            if src.crs is None:
                raise ValueError("This TIFF contains no geographic CRS.")

            if x < 0 or x >= src.width or y < 0 or y >= src.height:
                raise ValueError("Pixel is outside the image.")

            map_x, map_y = src.xy(int(y), int(x))

            transformer = Transformer.from_crs(
                src.crs,
                "EPSG:4326",
                always_xy=True
            )

            lon, lat = transformer.transform(map_x, map_y)

            return lat, lon


def bbox_to_geojson(file_bytes, x1, y1, x2, y2):
    corners = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
        (x1, y1)
    ]

    coordinates = []

    for x, y in corners:
        lat, lon = pixel_to_latlon(file_bytes, x, y)
        coordinates.append([lon, lat])

    return {
        "type": "Feature",
        "properties": {
            "label": "Detected Region"
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates]
        }
    }


def get_wgs84_bounds(file_bytes):
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            if src.crs is None:
                return None

            return transform_bounds(
                src.crs,
                "EPSG:4326",
                *src.bounds,
                densify_pts=21
            )


def calculate_overlap(bounds1, bounds2):
    if bounds1 is None or bounds2 is None:
        return None

    left = max(bounds1[0], bounds2[0])
    bottom = max(bounds1[1], bounds2[1])
    right = min(bounds1[2], bounds2[2])
    top = min(bounds1[3], bounds2[3])

    if right <= left or top <= bottom:
        return 0

    intersection = (right - left) * (top - bottom)

    area1 = (
        (bounds1[2] - bounds1[0])
        *
        (bounds1[3] - bounds1[1])
    )

    area2 = (
        (bounds2[2] - bounds2[0])
        *
        (bounds2[3] - bounds2[1])
    )

    smaller = min(area1, area2)

    if smaller == 0:
        return 0

    return intersection / smaller * 100


def validate_pair(file1, file2):
    metadata1 = read_metadata(file1)
    metadata2 = read_metadata(file2)

    overlap = calculate_overlap(
        get_wgs84_bounds(file1),
        get_wgs84_bounds(file2)
    )

    return {
        "same_crs": metadata1["crs"] == metadata2["crs"],
        "same_dimensions": (
            metadata1["width"] == metadata2["width"]
            and metadata1["height"] == metadata2["height"]
        ),
        "same_resolution": np.allclose(
            [
                metadata1["resolution_x"],
                metadata1["resolution_y"]
            ],
            [
                metadata2["resolution_x"],
                metadata2["resolution_y"]
            ]
        ),
        "overlap": overlap
    }


def find_default_file(names, token):
    for index, name in enumerate(names):
        if token.upper() in name.upper():
            return index

    return 0


def choose_band(files, label, token, key):
    names = list(files.keys())
    default_index = find_default_file(names, token)

    file_state_key = f"{key}_file"
    band_state_key = f"{key}_band"

    if file_state_key not in st.session_state:
        st.session_state[file_state_key] = names[default_index]

    if band_state_key not in st.session_state:
        st.session_state[band_state_key] = 1

    filename = st.session_state[file_state_key]
    if filename not in names:
        filename = names[default_index]
        st.session_state[file_state_key] = filename

    band_number = int(st.session_state[band_state_key])

    st.markdown(
        f'<div class="auto-pick"><span class="small-label">{label}</span><br>'
        f'<b>{filename}</b> · band {band_number}</div>',
        unsafe_allow_html=True
    )

    with st.expander(f"Use a different file or band for {label}"):
        filename = st.selectbox(
            f"{label} file",
            names,
            index=names.index(filename),
            key=f"{key}_file_picker",
            help="Which uploaded file holds this colour channel."
        )

        metadata = read_metadata(files[filename])

        band_number = st.number_input(
            f"{label} band",
            min_value=1,
            max_value=metadata["bands"],
            value=min(band_number, metadata["bands"]),
            step=1,
            key=f"{key}_band_picker",
            help="Most files only have 1 band — leave this as 1 unless you know it's different."
        )

        st.session_state[file_state_key] = filename
        st.session_state[band_state_key] = int(band_number)

    return files[filename], int(band_number)


def create_light_map(latitude, longitude, layers, zoom=13):
    return pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=zoom
        ),
        layers=layers
    )


def show_metadata(metadata):
    st.markdown(
        f"""
<div class="workspace-title">GeoTIFF <span>overview.</span></div>
<div class="workspace-sub">Core spatial properties extracted directly from the uploaded file.</div>
""",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Width", f'{metadata["width"]} px')
    col2.metric("Height", f'{metadata["height"]} px')
    col3.metric("Bands", metadata["bands"])
    col4.metric("Format", metadata["driver"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "CRS",
        metadata["crs"] or "Unavailable"
    )

    col2.metric(
        "X Resolution",
        f'{metadata["resolution_x"]:.6f}'
    )

    col3.metric(
        "Y Resolution",
        f'{metadata["resolution_y"]:.6f}'
    )

    with st.expander("Geographic bounds"):
        st.dataframe(
            [
                {
                    "Direction": "Left",
                    "Coordinate": metadata["left"]
                },
                {
                    "Direction": "Bottom",
                    "Coordinate": metadata["bottom"]
                },
                {
                    "Direction": "Right",
                    "Coordinate": metadata["right"]
                },
                {
                    "Direction": "Top",
                    "Coordinate": metadata["top"]
                }
            ],
            hide_index=True,
            use_container_width=True
        )


def single_image_mode():
    st.markdown(
        """
<div class="workspace-title">Start your <span>analysis.</span></div>
<div class="workspace-sub">Upload one GeoTIFF or multiple spectral-band TIFF files.</div>
""",
        unsafe_allow_html=True
    )

    uploads = st.file_uploader(
        "Upload satellite imagery",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        help="Drop in one .tif file, or several band files (like B02.tif, B03.tif, B04.tif, B08.tif)."
    )

    if not uploads:
        st.info("👆 Upload one or more .tif / .tiff files above to get started — nothing else to set up.")
        return

    files = {
        uploaded.name: get_bytes(uploaded)
        for uploaded in uploads
    }

    names = list(files.keys())
    default_index = find_default_file(names, "B04")

    if "primary_name" not in st.session_state or st.session_state["primary_name"] not in names:
        st.session_state["primary_name"] = names[default_index]

    st.markdown(
        f'<div class="auto-pick" style="margin-bottom:14px;">'
        f'<span class="small-label">Main image</span><br>'
        f'<b>{st.session_state["primary_name"]}</b>'
        f'<span style="color:#788382;"> — used for the overview stats, map and pixel lookups below</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if len(names) > 1:
        with st.expander("Uploaded a different file you'd rather use as the main one? Change it here"):
            primary_name = st.selectbox(
                "Main image",
                names,
                index=names.index(st.session_state["primary_name"]),
                label_visibility="collapsed"
            )
            st.session_state["primary_name"] = primary_name

    primary_name = st.session_state["primary_name"]
    primary_file = files[primary_name]
    metadata = read_metadata(primary_file)

    show_metadata(metadata)

    st.markdown(
        """
<div class="workspace-title">Explore <span>your data.</span></div>
<div class="workspace-sub">Generate composites, spectral indices and geographic outputs.</div>
""",
        unsafe_allow_html=True
    )

    rgb_tab, ndvi_tab, ndwi_tab, coordinates_tab, geojson_tab = st.tabs(
        [
            "RGB Composite",
            "NDVI",
            "NDWI",
            "Coordinates",
            "GeoJSON"
        ]
    )

    with rgb_tab:
        st.caption(
            "Makes a normal-looking colour photo from your files. "
            "We've already matched the Red / Green / Blue files below — just hit Generate."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            red = choose_band(
                files,
                "Red",
                "B04",
                "rgb_red"
            )

        with col2:
            green = choose_band(
                files,
                "Green",
                "B03",
                "rgb_green"
            )

        with col3:
            blue = choose_band(
                files,
                "Blue",
                "B02",
                "rgb_blue"
            )

        if st.button("Generate RGB Composite"):
            try:
                rgb = create_rgb(
                    red[0],
                    red[1],
                    green[0],
                    green[1],
                    blue[0],
                    blue[1]
                )

                st.image(
                    rgb,
                    caption="Natural colour composite",
                    use_container_width=True
                )

                st.download_button(
                    "Download RGB Image",
                    rgb_to_png(rgb),
                    "satquery_rgb.png",
                    "image/png"
                )

            except Exception as error:
                st.error(str(error))

    with ndvi_tab:
        st.caption(
            "Highlights healthy plants and vegetation. "
            "Bright green/lime = lots of vegetation, blue = water or bare ground."
        )

        col1, col2 = st.columns(2)

        with col1:
            nir = choose_band(
                files,
                "NIR",
                "B08",
                "ndvi_nir"
            )

        with col2:
            red = choose_band(
                files,
                "Red",
                "B04",
                "ndvi_red"
            )

        if st.button("Calculate NDVI"):
            try:
                ndvi, profile = calculate_index(
                    nir[0],
                    nir[1],
                    red[0],
                    red[1]
                )

                st.image(
                    ndvi_preview(ndvi),
                    caption="NDVI vegetation map",
                    use_container_width=True
                )

                stats = get_index_stats(ndvi)

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Minimum",
                    f'{stats["minimum"]:.3f}'
                )

                col2.metric(
                    "Mean",
                    f'{stats["mean"]:.3f}'
                )

                col3.metric(
                    "Maximum",
                    f'{stats["maximum"]:.3f}'
                )

                st.download_button(
                    "Download NDVI GeoTIFF",
                    index_to_geotiff(
                        ndvi,
                        profile
                    ),
                    "ndvi.tif",
                    "image/tiff"
                )

            except Exception as error:
                st.error(str(error))

    with ndwi_tab:
        st.caption(
            "Highlights water — rivers, lakes, flooding. "
            "Blue = water, olive/tan = dry land."
        )

        col1, col2 = st.columns(2)

        with col1:
            green = choose_band(
                files,
                "Green",
                "B03",
                "ndwi_green"
            )

        with col2:
            nir = choose_band(
                files,
                "NIR",
                "B08",
                "ndwi_nir"
            )

        if st.button("Calculate NDWI"):
            try:
                ndwi, profile = calculate_index(
                    green[0],
                    green[1],
                    nir[0],
                    nir[1]
                )

                st.image(
                    ndwi_preview(ndwi),
                    caption="NDWI water map",
                    use_container_width=True
                )

                stats = get_index_stats(ndwi)

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Minimum",
                    f'{stats["minimum"]:.3f}'
                )

                col2.metric(
                    "Mean",
                    f'{stats["mean"]:.3f}'
                )

                col3.metric(
                    "Maximum",
                    f'{stats["maximum"]:.3f}'
                )

                st.download_button(
                    "Download NDWI GeoTIFF",
                    index_to_geotiff(
                        ndwi,
                        profile
                    ),
                    "ndwi.tif",
                    "image/tiff"
                )

            except Exception as error:
                st.error(str(error))

    with coordinates_tab:
        st.caption(
            "Click a spot in your image and find out what GPS location it is. "
            "X = how far from the left edge, Y = how far from the top, in pixels."
        )

        col1, col2 = st.columns(2)

        with col1:
            x = st.number_input(
                "Pixel X",
                min_value=0,
                max_value=max(
                    metadata["width"] - 1,
                    0
                ),
                value=0
            )

        with col2:
            y = st.number_input(
                "Pixel Y",
                min_value=0,
                max_value=max(
                    metadata["height"] - 1,
                    0
                ),
                value=0
            )

        if st.button("Locate Pixel"):
            try:
                latitude, longitude = pixel_to_latlon(
                    primary_file,
                    x,
                    y
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Latitude",
                    f"{latitude:.6f}"
                )

                col2.metric(
                    "Longitude",
                    f"{longitude:.6f}"
                )

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[
                        {
                            "latitude": latitude,
                            "longitude": longitude
                        }
                    ],
                    get_position=[
                        "longitude",
                        "latitude"
                    ],
                    get_radius=80,
                    get_fill_color=[
                        90,
                        160,
                        240
                    ],
                    pickable=True
                )

                st.pydeck_chart(
                    create_light_map(
                        latitude,
                        longitude,
                        [layer],
                        zoom=14
                    ),
                    use_container_width=True
                )

            except Exception as error:
                st.error(str(error))

    with geojson_tab:
        st.caption(
            "Draw a rectangle over an area of interest (like a field or building) "
            "and turn it into a downloadable geographic shape file."
        )

        col1, col2, col3, col4 = st.columns(4)

        x1 = col1.number_input(
            "Top-left X",
            0,
            max(metadata["width"] - 1, 0),
            0,
            help="Pixels from the left edge"
        )

        y1 = col2.number_input(
            "Top-left Y",
            0,
            max(metadata["height"] - 1, 0),
            0,
            help="Pixels from the top edge"
        )

        x2 = col3.number_input(
            "Bottom-right X",
            0,
            max(metadata["width"] - 1, 0),
            max(metadata["width"] - 1, 0),
            help="Pixels from the left edge"
        )

        y2 = col4.number_input(
            "Bottom-right Y",
            0,
            max(metadata["height"] - 1, 0),
            max(metadata["height"] - 1, 0),
            help="Pixels from the top edge"
        )

        if st.button("Generate Geographic Region"):
            try:
                geojson = bbox_to_geojson(
                    primary_file,
                    x1,
                    y1,
                    x2,
                    y2
                )

                polygon = geojson[
                    "geometry"
                ]["coordinates"][0]

                longitude = np.mean([
                    point[0]
                    for point in polygon
                ])

                latitude = np.mean([
                    point[1]
                    for point in polygon
                ])

                layer = pdk.Layer(
                    "PolygonLayer",
                    data=[
                        {
                            "polygon": polygon
                        }
                    ],
                    get_polygon="polygon",
                    filled=True,
                    stroked=True,
                    get_fill_color=[
                        186,
                        255,
                        0,
                        70
                    ],
                    get_line_color=[
                        80,
                        110,
                        30
                    ],
                    get_line_width=4
                )

                st.pydeck_chart(
                    create_light_map(
                        latitude,
                        longitude,
                        [layer]
                    ),
                    use_container_width=True
                )

                st.download_button(
                    "Download GeoJSON",
                    json.dumps(
                        geojson,
                        indent=2
                    ),
                    "region.geojson",
                    "application/geo+json"
                )

            except Exception as error:
                st.error(str(error))


def pair_mode(title, first_label, second_label, key):
    st.markdown(
        f"""
<div class="workspace-title">{title}</div>
<div class="workspace-sub">Verify whether the two geospatial images are spatially compatible.</div>
""",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        first = st.file_uploader(
            first_label,
            type=["tif", "tiff"],
            key=f"{key}_first"
        )

    with col2:
        second = st.file_uploader(
            second_label,
            type=["tif", "tiff"],
            key=f"{key}_second"
        )

    if not first or not second:
        st.info("Upload both GeoTIFFs to continue.")
        return

    result = validate_pair(
        get_bytes(first),
        get_bytes(second)
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "CRS Match",
        "YES" if result["same_crs"] else "NO"
    )

    col2.metric(
        "Dimensions",
        "MATCH" if result["same_dimensions"] else "DIFFER"
    )

    col3.metric(
        "Resolution",
        "MATCH" if result["same_resolution"] else "DIFFER"
    )

    overlap = result["overlap"]

    col4.metric(
        "Overlap",
        (
            f"{overlap:.1f}%"
            if overlap is not None
            else "N/A"
        )
    )

    if overlap is not None and overlap > 80:
        st.success(
            "Strong geographic compatibility detected."
        )

    elif overlap is not None and overlap > 0:
        st.warning(
            "The images only partially overlap."
        )

    elif overlap == 0:
        st.error(
            "These images do not overlap geographically."
        )


st.sidebar.markdown("## SATQUERY")

st.sidebar.caption(
    "REMOTE SENSING WORKSPACE"
)

mode = st.sidebar.radio(
    "Input mode",
    [
        "Single Image",
        "Bi-Temporal Pair",
        "Optical + SAR"
    ]
)

query = st.sidebar.text_area(
    "Natural language query",
    placeholder=(
        "e.g. Show vegetation in this area"
    ),
    height=120
)

if query:
    st.sidebar.markdown(
        f"""
<div class="query-box">Query ready<br><b>{query}</b></div>
""",
        unsafe_allow_html=True
    )


if mode == "Single Image":
    single_image_mode()

elif mode == "Bi-Temporal Pair":
    pair_mode(
        "Compare two observations.",
        "Earlier GeoTIFF",
        "Later GeoTIFF",
        "temporal"
    )

else:
    pair_mode(
        "Combine sensor perspectives.",
        "Optical GeoTIFF",
        "SAR GeoTIFF",
        "optical_sar"
    )