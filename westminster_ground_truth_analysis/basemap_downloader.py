"""Basemap downloader and comparison utilities."""

import math
import time
import requests
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, calculate_default_transform
import io
from pathlib import Path
import cv2
from skimage.metrics import structural_similarity as ssim


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to lat/lon of top-left corner."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def get_tile_url(xtile: int, ytile: int, zoom: int, source: str = "openstreetmap") -> str:
    """Get URL for a tile."""
    if source == "openstreetmap":
        return f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
    elif source == "esri_world_imagery":
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    else:
        raise ValueError(f"Unknown tile source: {source}")


def download_tile(xtile: int, ytile: int, zoom: int, source: str = "openstreetmap", 
                  verbose: bool = False, retries: int = 2) -> Optional[Image.Image]:
    """Download a single tile with retry logic."""
    url = get_tile_url(xtile, ytile, zoom, source)
    
    headers = {
        'User-Agent': 'westminster-analysis/0.1.0'
    }
    
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            return img
        except requests.exceptions.HTTPError as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'unknown'
                print(f"Warning: HTTP error downloading tile {zoom}/{xtile}/{ytile}: {e} (Status: {status_code})")
            return None
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                print(f"Warning: Request error downloading tile {zoom}/{xtile}/{ytile}: {e}")
            return None
        except Exception as e:
            if attempt < retries:
                time.sleep(0.1 * (attempt + 1))
                continue
            if verbose:
                print(f"Warning: Failed to download tile {zoom}/{xtile}/{ytile}: {e}")
            return None
    
    return None


def calculate_zoom_level(bbox: Tuple[float, float, float, float], 
                         max_tiles: int = 16, 
                         target_resolution: Optional[float] = None) -> int:
    """
    Calculate appropriate zoom level based on bounding box size or target resolution.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        max_tiles: Maximum number of tiles to download
        target_resolution: Target resolution in meters per pixel (optional)
        
    Returns:
        Zoom level
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    if target_resolution:
        # Calculate zoom based on target resolution
        # Approximate: at equator, 1 tile at zoom z covers ~156543 meters / 2^z
        center_lat = (min_lat + max_lat) / 2
        meters_per_pixel_at_equator = 156543.03392
        meters_per_pixel = meters_per_pixel_at_equator * math.cos(math.radians(center_lat))
        
        for zoom in range(1, 20):
            tile_size_meters = meters_per_pixel * 256 / (2 ** zoom)
            if tile_size_meters <= target_resolution:
                return zoom
        return 18
    
    # Calculate based on bounding box size
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    
    for zoom in range(1, 20):
        xtile_min, ytile_min = deg2num(min_lat, min_lon, zoom)
        xtile_max, ytile_max = deg2num(max_lat, max_lon, zoom)
        
        num_tiles = (xtile_max - xtile_min + 1) * (ytile_max - ytile_min + 1)
        if num_tiles <= max_tiles:
            return zoom
    
    return 18


def download_basemap(
    bbox: Tuple[float, float, float, float],
    output_path: str,
    source: str = "esri_world_imagery",
    zoom: Optional[int] = None,
    target_resolution: Optional[float] = None
) -> str:
    """
    Download basemap tiles and create a GeoTIFF.
    
    Args:
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        output_path: Path to save GeoTIFF
        source: Tile source ('openstreetmap' or 'esri_world_imagery')
        zoom: Zoom level (auto-calculated if None)
        target_resolution: Target resolution in meters per pixel
        
    Returns:
        Path to saved GeoTIFF
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    
    if zoom is None:
        zoom = calculate_zoom_level(bbox, target_resolution=target_resolution)
    
    print(f"Downloading basemap at zoom level {zoom}...")
    
    # Calculate tile range
    # Note: In tile coordinates, Y increases as latitude decreases (south)
    # So min_lat (south) gives larger Y, max_lat (north) gives smaller Y
    xtile_min, ytile_south = deg2num(min_lat, min_lon, zoom)  # South = larger Y
    xtile_max, ytile_north = deg2num(max_lat, max_lon, zoom)  # North = smaller Y
    
    # Ensure correct order for range (ytile_min <= ytile_max)
    ytile_min = min(ytile_north, ytile_south)
    ytile_max = max(ytile_north, ytile_south)
    
    # Also ensure X is in correct order
    xtile_min, xtile_max = min(xtile_min, xtile_max), max(xtile_min, xtile_max)
    
    print(f"Tile range: X [{xtile_min}, {xtile_max}], Y [{ytile_min}, {ytile_max}]")
    
    # Download tiles
    tiles = []
    if ytile_max < ytile_min:
        raise ValueError(f"Invalid Y tile range: ytile_min={ytile_min}, ytile_max={ytile_max}")
    
    if xtile_max < xtile_min:
        raise ValueError(f"Invalid X tile range: xtile_min={xtile_min}, xtile_max={xtile_max}")
    
    print(f"Downloading tiles: {xtile_max - xtile_min + 1} columns x {ytile_max - ytile_min + 1} rows")
    
    for y in range(ytile_min, ytile_max + 1):
        row = []
        for x in range(xtile_min, xtile_max + 1):
            tile = download_tile(x, y, zoom, source, verbose=False)
            if tile is None:
                # Create blank tile
                tile = Image.new('RGB', (256, 256), color=(128, 128, 128))
            row.append(tile)
        tiles.append(row)
    
    # Validate tiles were downloaded
    if not tiles:
        raise ValueError(f"No tile rows created. Tile range: X [{xtile_min}, {xtile_max}], Y [{ytile_min}, {ytile_max}]")
    
    if not tiles[0]:
        raise ValueError(f"No tiles in first row. Tile range: X [{xtile_min}, {xtile_max}], Y [{ytile_min}, {ytile_max}]")
    
    # Stitch tiles together
    
    tile_height = tiles[0][0].height
    tile_width = tiles[0][0].width
    
    stitched = Image.new('RGB', 
                        ((xtile_max - xtile_min + 1) * tile_width,
                         (ytile_max - ytile_min + 1) * tile_height))
    
    for y_idx, row in enumerate(tiles):
        for x_idx, tile in enumerate(row):
            x_pos = (x_idx) * tile_width
            y_pos = (y_idx) * tile_height
            stitched.paste(tile, (x_pos, y_pos))
    
    # Get bounds of stitched image
    top_left_lat, top_left_lon = num2deg(xtile_min, ytile_min, zoom)
    bottom_right_lat, bottom_right_lon = num2deg(xtile_max + 1, ytile_max + 1, zoom)
    
    # Crop to requested bounds
    # Calculate pixel positions
    pixels_per_degree_lon = stitched.width / (bottom_right_lon - top_left_lon)
    pixels_per_degree_lat = stitched.height / (top_left_lat - bottom_right_lat)
    
    left_pixel = int((min_lon - top_left_lon) * pixels_per_degree_lon)
    top_pixel = int((top_left_lat - max_lat) * pixels_per_degree_lat)
    right_pixel = int((max_lon - top_left_lon) * pixels_per_degree_lon)
    bottom_pixel = int((top_left_lat - min_lat) * pixels_per_degree_lat)
    
    left_pixel = max(0, left_pixel)
    top_pixel = max(0, top_pixel)
    right_pixel = min(stitched.width, right_pixel)
    bottom_pixel = min(stitched.height, bottom_pixel)
    
    cropped = stitched.crop((left_pixel, top_pixel, right_pixel, bottom_pixel))
    
    # Save as GeoTIFF
    width, height = cropped.size
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    
    array = np.array(cropped)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=array.dtype,
        crs=CRS.from_epsg(4326),  # WGS84
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(array.transpose(2, 0, 1))
    
    print(f"Basemap saved to {output_path}")
    return output_path


def compare_orthomosaic_to_basemap(
    ortho_path: str,
    basemap_path: str,
    output_dir: str = "outputs"
) -> dict:
    """
    Compare orthomosaic to basemap and calculate accuracy metrics.
    
    Args:
        ortho_path: Path to orthomosaic GeoTIFF
        basemap_path: Path to basemap GeoTIFF
        output_dir: Directory for output visualizations
        
    Returns:
        Dictionary of comparison metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Comparing orthomosaic to basemap...")
    
    # Load rasters
    with rasterio.open(ortho_path) as ortho_src:
        ortho_data = ortho_src.read()
        ortho_crs = ortho_src.crs
        ortho_bounds = ortho_src.bounds
    
    with rasterio.open(basemap_path) as basemap_src:
        basemap_data = basemap_src.read()
        basemap_crs = basemap_src.crs
        basemap_bounds = basemap_src.bounds
    
    # Reproject to common CRS and resolution
    # First, check if CRS are the same
    if ortho_crs != basemap_crs:
        print(f"CRS mismatch: Ortho={ortho_crs}, Basemap={basemap_crs}")
        print("Transforming bounds to common CRS for overlap calculation...")
        
        # Transform basemap bounds to ortho CRS for overlap calculation
        from rasterio.warp import transform_bounds
        basemap_bounds_transformed = transform_bounds(
            basemap_crs,
            ortho_crs,
            basemap_bounds.left,
            basemap_bounds.bottom,
            basemap_bounds.right,
            basemap_bounds.top
        )
        basemap_left, basemap_bottom, basemap_right, basemap_top = basemap_bounds_transformed
    else:
        basemap_left = basemap_bounds.left
        basemap_bottom = basemap_bounds.bottom
        basemap_right = basemap_bounds.right
        basemap_top = basemap_bounds.top
    
    # Calculate overlap in ortho CRS
    target_left = max(ortho_bounds.left, basemap_left)
    target_bottom = max(ortho_bounds.bottom, basemap_bottom)
    target_right = min(ortho_bounds.right, basemap_right)
    target_top = min(ortho_bounds.top, basemap_top)
    
    # Check for valid overlap
    if target_right <= target_left or target_top <= target_bottom:
        raise ValueError(
            f"No valid overlap between orthomosaic and basemap. "
            f"Ortho bounds (in {ortho_crs}): {ortho_bounds}, "
            f"Basemap bounds (transformed to {ortho_crs}): ({basemap_left}, {basemap_bottom}, {basemap_right}, {basemap_top}), "
            f"Target bounds: ({target_left}, {target_bottom}, {target_right}, {target_top})"
        )
    
    target_crs = ortho_crs
    
    # Calculate dimensions with a reasonable resolution
    # Use the resolution from the orthomosaic if available, otherwise use 0.5m
    try:
        ortho_resolution = abs(ortho_src.transform[0])  # Pixel width in CRS units
        target_resolution = min(ortho_resolution, 0.5)  # Use smaller of the two
    except:
        target_resolution = 0.5  # Default to 0.5m
    
    # Calculate width and height based on target resolution
    width = max(1, int((target_right - target_left) / target_resolution))
    height = max(1, int((target_top - target_bottom) / target_resolution))
    
    # Calculate transform
    transform, width, height = calculate_default_transform(
        basemap_crs,
        target_crs,
        width,
        height,
        left=target_left,
        bottom=target_bottom,
        right=target_right,
        top=target_top
    )
    
    # Reproject basemap
    basemap_reprojected = np.zeros((3, height, width), dtype=basemap_data.dtype)
    reproject(
        source=basemap_data,
        destination=basemap_reprojected,
        src_transform=basemap_src.transform,
        src_crs=basemap_crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )
    
    # Reproject orthomosaic
    ortho_reprojected = np.zeros((3, height, width), dtype=ortho_data.dtype)
    reproject(
        source=ortho_data,
        destination=ortho_reprojected,
        src_transform=ortho_src.transform,
        src_crs=ortho_crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )
    
    # Convert to grayscale for comparison
    basemap_gray = cv2.cvtColor(basemap_reprojected.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    ortho_gray = cv2.cvtColor(ortho_reprojected.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    
    # Calculate metrics
    # RMSE
    valid_mask = (basemap_gray > 0) & (ortho_gray > 0)
    if valid_mask.sum() > 0:
        diff = basemap_gray[valid_mask].astype(float) - ortho_gray[valid_mask].astype(float)
        rmse = np.sqrt(np.mean(diff ** 2))
        
        # MAE
        mae = np.mean(np.abs(diff))
        
        # Correlation
        correlation = np.corrcoef(basemap_gray[valid_mask], ortho_gray[valid_mask])[0, 1]
        
        # SSIM
        ssim_value = ssim(basemap_gray, ortho_gray, data_range=255)
    else:
        rmse = mae = correlation = ssim_value = 0.0
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'ssim': ssim_value,
        'valid_pixels': int(valid_mask.sum()),
        'total_pixels': int(valid_mask.size)
    }
    
    print(f"Comparison Metrics:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  SSIM: {ssim_value:.4f}")
    
    return metrics

