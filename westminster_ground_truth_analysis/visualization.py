"""Visualization utilities for orthomosaic analysis."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Optional
import cv2

from .orthomosaic_pipeline import OrthomosaicPipeline, Match


def visualize_matches(
    pipeline: OrthomosaicPipeline,
    match_idx: int,
    output_path: Optional[str] = None,
    max_matches: int = 50
):
    """
    Visualize feature matches between two images.
    
    Args:
        pipeline: OrthomosaicPipeline instance
        match_idx: Index of match to visualize
        output_path: Path to save visualization
        max_matches: Maximum number of matches to show
    """
    if match_idx >= len(pipeline.matches):
        raise ValueError(f"Match index {match_idx} out of range")
    
    match = pipeline.matches[match_idx]
    img1_path = pipeline.images[match.image1_idx].path
    img2_path = pipeline.images[match.image2_idx].path
    
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        raise ValueError("Could not load images")
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Sample matches
    n_matches = min(len(match.points1), max_matches)
    indices = np.random.choice(len(match.points1), n_matches, replace=False)
    
    pts1 = match.points1[indices]
    pts2 = match.points2[indices]
    
    # Create visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Resize if needed
    max_h = 800
    if h1 > max_h:
        scale1 = max_h / h1
        img1 = cv2.resize(img1, (int(w1 * scale1), int(h1 * scale1)))
        pts1 = pts1 * scale1
        h1, w1 = img1.shape[:2]
    
    if h2 > max_h:
        scale2 = max_h / h2
        img2 = cv2.resize(img2, (int(w2 * scale2), int(h2 * scale2)))
        pts2 = pts2 * scale2
        h2, w2 = img2.shape[:2]
    
    # Combine images
    combined_h = max(h1, h2)
    combined_w = w1 + w2
    combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:] = img2
    
    # Draw matches
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(combined)
    ax.axis('off')
    
    # Draw lines and points
    colors = plt.cm.rainbow(np.linspace(0, 1, n_matches))
    
    for i, (pt1, pt2, color) in enumerate(zip(pts1, pts2, colors)):
        x1, y1 = pt1
        x2, y2 = pt2 + np.array([w1, 0])
        
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.6)
        ax.plot(x1, y1, 'o', color=color, markersize=4)
        ax.plot(x2, y2, 'o', color=color, markersize=4)
    
    ax.set_title(f'Feature Matches: {pipeline.images[match.image1_idx].path.name} <-> {pipeline.images[match.image2_idx].path.name}\n'
                 f'Showing {n_matches} of {len(match.points1)} matches',
                 fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Match visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_reprojection_errors(
    pipeline: OrthomosaicPipeline,
    output_path: Optional[str] = None
):
    """
    Visualize reprojection error distribution.
    
    Args:
        pipeline: OrthomosaicPipeline instance
        output_path: Path to save visualization
    """
    if not pipeline.reprojection_errors:
        print("No reprojection errors available")
        return
    
    errors = np.array(pipeline.reprojection_errors)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}px')
    axes[0].axvline(np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.2f}px')
    axes[0].set_xlabel('Reprojection Error (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Reprojection Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(errors, vert=True)
    axes[1].set_ylabel('Reprojection Error (pixels)')
    axes[1].set_title('Reprojection Error Box Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Reprojection error visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_camera_poses(
    pipeline: OrthomosaicPipeline,
    output_path: Optional[str] = None
):
    """
    Visualize camera poses in 3D.
    
    Args:
        pipeline: OrthomosaicPipeline instance
        output_path: Path to save visualization
    """
    valid_poses = [(i, p) for i, p in enumerate(pipeline.camera_poses) if p is not None]
    
    if not valid_poses:
        print("No valid camera poses to visualize")
        return
    
    positions = np.array([p.position for _, p in valid_poses])
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
              c=range(len(positions)), cmap='viridis', s=50, alpha=0.7)
    
    # Draw camera orientations
    for i, (img_idx, pose) in enumerate(valid_poses):
        # Draw camera frame
        frame_length = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)) * 0.05
        
        # Camera axes
        x_axis = pose.rotation @ np.array([frame_length, 0, 0])
        y_axis = pose.rotation @ np.array([0, frame_length, 0])
        z_axis = pose.rotation @ np.array([0, 0, -frame_length])  # Negative Z (camera looks down)
        
        pos = pose.position
        ax.plot([pos[0], pos[0] + x_axis[0]], 
               [pos[1], pos[1] + x_axis[1]], 
               [pos[2], pos[2] + x_axis[2]], 'r-', linewidth=2)
        ax.plot([pos[0], pos[0] + y_axis[0]], 
               [pos[1], pos[1] + y_axis[1]], 
               [pos[2], pos[2] + y_axis[2]], 'g-', linewidth=2)
        ax.plot([pos[0], pos[0] + z_axis[0]], 
               [pos[1], pos[1] + z_axis[1]], 
               [pos[2], pos[2] + z_axis[2]], 'b-', linewidth=2)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Camera Poses ({len(valid_poses)} cameras)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Camera pose visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_match_quality_report(
    pipeline: OrthomosaicPipeline,
    output_dir: str = "outputs"
):
    """
    Create a comprehensive report of match quality.
    
    Args:
        pipeline: OrthomosaicPipeline instance
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating match quality report...")
    
    # Calculate match statistics
    match_stats = []
    for i, match in enumerate(pipeline.matches):
        n_matches = len(match.points1)
        match_stats.append({
            'index': i,
            'image1': pipeline.images[match.image1_idx].path.name,
            'image2': pipeline.images[match.image2_idx].path.name,
            'num_matches': n_matches
        })
    
    # Sort by number of matches
    match_stats.sort(key=lambda x: x['num_matches'], reverse=True)
    
    # Visualize top matches
    n_visualize = min(5, len(pipeline.matches))
    for i in range(n_visualize):
        match_idx = match_stats[i]['index']
        output_path = output_dir / f"match_visualization_{i+1}.png"
        visualize_matches(pipeline, match_idx, str(output_path), max_matches=100)
    
    # Visualize worst matches
    if len(match_stats) > n_visualize:
        for i in range(len(match_stats) - n_visualize, len(match_stats)):
            match_idx = match_stats[i]['index']
            output_path = output_dir / f"match_visualization_worst_{i+1}.png"
            visualize_matches(pipeline, match_idx, str(output_path), max_matches=50)
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    num_matches = [m['num_matches'] for m in match_stats]
    ax.bar(range(len(num_matches)), num_matches, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Match Pair Index (sorted by quality)')
    ax.set_ylabel('Number of Matches')
    ax.set_title('Match Quality Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "match_quality_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Match quality report saved to {output_dir}")

