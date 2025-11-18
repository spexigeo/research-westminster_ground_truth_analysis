"""Parser for ground control point CSV files."""

import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class GroundControlPoint:
    """Represents a ground control point."""
    id: int
    x: float  # UTM X coordinate
    y: float  # UTM Y coordinate
    z: float  # Elevation
    name: str
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z])


class GCPParser:
    """Parser for GCP CSV files."""
    
    def __init__(self, csv_path: str):
        """
        Initialize GCP parser.
        
        Args:
            csv_path: Path to the GCP CSV file
        """
        self.csv_path = Path(csv_path)
        self.gcps: List[GroundControlPoint] = []
        self._parse()
    
    def _parse(self):
        """Parse the CSV file."""
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5:
                    try:
                        gcp_id = int(row[0])
                        x = float(row[1])
                        y = float(row[2])
                        z = float(row[3])
                        name = row[4].strip()
                        
                        gcp = GroundControlPoint(
                            id=gcp_id,
                            x=x,
                            y=y,
                            z=z,
                            name=name
                        )
                        self.gcps.append(gcp)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping invalid row: {row}, error: {e}")
                        continue
    
    def get_gcps(self) -> List[GroundControlPoint]:
        """Get all parsed GCPs."""
        return self.gcps
    
    def get_gcp_by_name(self, name: str) -> GroundControlPoint:
        """Get a GCP by its name."""
        for gcp in self.gcps:
            if gcp.name == name:
                return gcp
        raise ValueError(f"GCP with name '{name}' not found")
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box of all GCPs.
        
        Returns:
            (min_x, min_y, max_x, max_y)
        """
        if not self.gcps:
            raise ValueError("No GCPs available")
        
        xs = [gcp.x for gcp in self.gcps]
        ys = [gcp.y for gcp in self.gcps]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of all GCPs."""
        min_x, min_y, max_x, max_y = self.get_bounds()
        return ((min_x + max_x) / 2, (min_y + max_y) / 2)

