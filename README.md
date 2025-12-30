# Westminster Ground Truth Analysis

This repository contains tools for creating orthomosaics from DJI drone imagery and evaluating their accuracy using ground control points (GCPs) and basemap comparisons.

## Overview

The project processes drone imagery from two flight missions (DJI_202510060955_017_25-3288 and DJI_202510060955_019_25-3288) to create orthomosaics with and without ground control points, then compares them against basemaps for accuracy assessment.

## Features

- **Orthomosaic Creation**: Feature detection, matching, bundle adjustment, and triangulation
- **GCP Integration**: Incorporates ground control points for improved accuracy
- **DJI Metadata Parsing**: Attempts to use .nav, .obs, .bin, and timestamp.MRK files
- **Basemap Comparison**: Downloads basemaps and quantifies absolute accuracy
- **Visualization**: Comprehensive notebooks with match quality analysis and error metrics

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

See the Jupyter notebooks:

### MetaShape Analysis Notebooks
- `test_westminster_analysis_metashape_spexi_data.ipynb` - Local Jupyter notebook for Spexi data analysis
- `test_westminster_analysis_metashape_spexi_data_colab.ipynb` - Google Colab version
- `test_westminster_analysis_metashape_rtk_ground_truth.ipynb` - Local Jupyter notebook for RTK ground truth analysis
- `test_westminster_analysis_metashape_rtk_ground_truth_colab.ipynb` - Google Colab version

### Orthomosaic Matching Notebooks
- `test_orthos_gcp_matching.ipynb` - Local Jupyter notebook for GCP-based orthomosaic matching
- `test_orthos_gcp_matching_colab.ipynb` - Google Colab version

## Project Structure

```
research-westminster_ground_truth_analysis/
├── westminster_ground_truth_analysis/
│   ├── __init__.py
│   ├── orthomosaic_pipeline.py    # Main orthomosaic creation pipeline
│   ├── gcp_parser.py               # GCP CSV parser
│   ├── dji_metadata.py             # DJI metadata file parsers
│   ├── basemap_downloader.py       # Basemap download and comparison
│   └── visualization.py            # Visualization utilities
├── outputs/                        # Generated orthomosaics and results
├── test_westminster_analysis_metashape_spexi_data.ipynb
├── test_westminster_analysis_metashape_spexi_data_colab.ipynb
├── test_westminster_analysis_metashape_rtk_ground_truth.ipynb
├── test_westminster_analysis_metashape_rtk_ground_truth_colab.ipynb
├── test_orthos_gcp_matching.ipynb
└── test_orthos_gcp_matching_colab.ipynb
```

## Data

The input data should be located at:
`/Users/mauriciohessflores/Documents/Code/Data/New Westminster Oct _25/`

## License

MIT License

