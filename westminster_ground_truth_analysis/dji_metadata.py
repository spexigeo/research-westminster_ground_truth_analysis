"""Parser for DJI metadata files (.nav, .obs, .bin, timestamp.MRK)."""

import struct
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class TimestampEntry:
    """Represents a timestamp entry from .MRK file."""
    image_name: str
    timestamp: float
    frame_number: int


@dataclass
class NavigationData:
    """Represents navigation data from .nav file."""
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    roll: float
    pitch: float
    yaw: float


class DJIMetadataParser:
    """Parser for DJI metadata files."""
    
    def __init__(self, directory: str):
        """
        Initialize DJI metadata parser.
        
        Args:
            directory: Directory containing DJI files
        """
        self.directory = Path(directory)
        self.timestamps: List[TimestampEntry] = []
        self.nav_data: List[NavigationData] = []
        self._parse_timestamps()
        self._parse_nav()
    
    def _parse_timestamps(self):
        """Parse timestamp.MRK file."""
        mrk_files = list(self.directory.glob("*Timestamp.MRK")) + \
                   list(self.directory.glob("*.MRK"))
        
        if not mrk_files:
            print(f"Warning: No timestamp.MRK file found in {self.directory}")
            return
        
        mrk_file = mrk_files[0]
        print(f"Parsing timestamp file: {mrk_file}")
        
        try:
            with open(mrk_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Try to parse as text-based format
            # Format may vary, try common patterns
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try different patterns
                # Pattern 1: "image_name timestamp frame"
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        image_name = parts[0]
                        timestamp = float(parts[1])
                        frame_number = int(parts[2]) if len(parts) > 2 else 0
                        
                        entry = TimestampEntry(
                            image_name=image_name,
                            timestamp=timestamp,
                            frame_number=frame_number
                        )
                        self.timestamps.append(entry)
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"Warning: Could not parse timestamp file: {e}")
    
    def _parse_nav(self):
        """Parse .nav file (navigation data)."""
        nav_files = list(self.directory.glob("*.nav"))
        
        if not nav_files:
            print(f"Warning: No .nav file found in {self.directory}")
            return
        
        nav_file = nav_files[0]
        print(f"Parsing navigation file: {nav_file}")
        
        try:
            # Try to parse as text first
            with open(nav_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Try to parse space or comma separated values
                    parts = re.split(r'[,\s]+', line)
                    if len(parts) >= 7:
                        try:
                            nav = NavigationData(
                                timestamp=float(parts[0]),
                                latitude=float(parts[1]),
                                longitude=float(parts[2]),
                                altitude=float(parts[3]),
                                roll=float(parts[4]),
                                pitch=float(parts[5]),
                                yaw=float(parts[6])
                            )
                            self.nav_data.append(nav)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            print(f"Warning: Could not parse nav file as text: {e}")
            # Could try binary parsing here if needed
    
    def get_timestamp_for_image(self, image_name: str) -> Optional[float]:
        """Get timestamp for a specific image."""
        # Try exact match first
        for entry in self.timestamps:
            if entry.image_name == image_name:
                return entry.timestamp
        
        # Try partial match (filename only)
        image_basename = Path(image_name).name
        for entry in self.timestamps:
            if Path(entry.image_name).name == image_basename:
                return entry.timestamp
        
        return None
    
    def get_nav_data_for_timestamp(self, timestamp: float, tolerance: float = 1.0) -> Optional[NavigationData]:
        """Get navigation data closest to a given timestamp."""
        if not self.nav_data:
            return None
        
        closest = min(self.nav_data, key=lambda x: abs(x.timestamp - timestamp))
        if abs(closest.timestamp - timestamp) <= tolerance:
            return closest
        return None
    
    def get_image_pose(self, image_name: str) -> Optional[Dict]:
        """Get pose (position and orientation) for an image."""
        timestamp = self.get_timestamp_for_image(image_name)
        if timestamp is None:
            return None
        
        nav = self.get_nav_data_for_timestamp(timestamp)
        if nav is None:
            return None
        
        return {
            'latitude': nav.latitude,
            'longitude': nav.longitude,
            'altitude': nav.altitude,
            'roll': nav.roll,
            'pitch': nav.pitch,
            'yaw': nav.yaw,
            'timestamp': timestamp
        }

