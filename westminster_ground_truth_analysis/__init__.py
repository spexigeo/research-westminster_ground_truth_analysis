"""Westminster Ground Truth Analysis Package."""

__version__ = "0.1.0"

from .gcp_parser import GCPParser, GroundControlPoint
from .dji_metadata import DJIMetadataParser, TimestampEntry, NavigationData
from .orthomosaic_pipeline import OrthomosaicPipeline, ImageMetadata, CameraPose, Match
from .basemap_downloader import download_basemap, compare_orthomosaic_to_basemap
from .visualization import (
    visualize_matches,
    visualize_reprojection_errors,
    visualize_camera_poses,
    create_match_quality_report
)

__all__ = [
    'GCPParser',
    'GroundControlPoint',
    'DJIMetadataParser',
    'TimestampEntry',
    'NavigationData',
    'OrthomosaicPipeline',
    'ImageMetadata',
    'CameraPose',
    'Match',
    'download_basemap',
    'compare_orthomosaic_to_basemap',
    'visualize_matches',
    'visualize_reprojection_errors',
    'visualize_camera_poses',
    'create_match_quality_report',
]

