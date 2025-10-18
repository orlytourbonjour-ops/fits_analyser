#!/usr/bin/env python3
"""
Complete Astrophotography Analysis Program
Advanced version with recursive analysis, target separation and light calculations
"""

# Check Python version compatibility
import sys
if sys.version_info < (3, 8):
    print("⚠️  WARNING: This program requires Python 3.8 or higher.")
    print("   Current Python version:", sys.version)
    print("   Execution continues, but issues may occur...")
    print("   " + "="*60)

import os
import re
import math
import requests
from pathlib import Path
from collections import defaultdict, Counter
from datetime import timedelta, datetime
import json
import argparse
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: Matplotlib/NumPy/Pandas not installed. Charts disabled.")
    print("   Install with: pip install matplotlib numpy pandas")

try:
    from astropy.io import fits
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import astropy.units as units
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("WARNING: Astropy not installed. Cannot read FITS headers.")
    print("   Install with: pip install astropy")

# Suppress common FITS warnings to reduce noise (only if astropy is available)
if ASTROPY_AVAILABLE:
    try:
        import warnings
        from astropy.io.fits.util import UserWarning as FitsUserWarning
        warnings.filterwarnings('ignore', category=FitsUserWarning, message='.*non-ASCII characters.*')
        warnings.filterwarnings('ignore', category=FitsUserWarning, message='.*null bytes.*')
        warnings.filterwarnings('ignore', category=FitsUserWarning, message='.*Header block contains null bytes.*')
        warnings.filterwarnings('ignore', category=FitsUserWarning, message='.*FITS-compliant.*')
    except ImportError:
        # If warnings module or FitsUserWarning not available, continue silently
        pass

try:
    from tqdm import tqdm
    import os
    # Force tqdm to use colors and disable dynamic columns
    os.environ['TQDM_DISABLE'] = '0'
    os.environ['TQDM_COLOUR'] = '#00FF00'
    # Disable dynamic columns to prevent bar from disappearing
    os.environ['TQDM_DYNAMIC_NCOLS'] = '0'
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("WARNING: tqdm not available, progress bars disabled.")
    print("   Install with: pip install tqdm")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("WARNING: reportlab not available, PDF generation without LaTeX disabled.")
    print("   Install with: pip install reportlab")

# Recognized filters with their central wavelengths (nm)
FILTERS_INFO = {
    'HA': {'name': 'Hydrogen Alpha', 'lambda': 656.3, 'width': 4.5},
    'Ha': {'name': 'Hydrogen Alpha', 'lambda': 656.3, 'width': 4.5},  # Alternative spelling
    'OIII': {'name': 'Oxygen III', 'lambda': 500.7, 'width': 4.5},
    'O3': {'name': 'Oxygen III', 'lambda': 500.7, 'width': 4.5},  # Alternative spelling
    'SII': {'name': 'Sulfur II', 'lambda': 672.4, 'width': 4.5},
    'S2': {'name': 'Sulfur II', 'lambda': 672.4, 'width': 4.5},  # Alternative spelling
    'L': {'name': 'Luminance', 'lambda': 550.0, 'width': 400.0},
    'LUM': {'name': 'Luminance', 'lambda': 550.0, 'width': 400.0},  # Alternative spelling
    'R': {'name': 'Red', 'lambda': 650.0, 'width': 100.0},
    'G': {'name': 'Green', 'lambda': 550.0, 'width': 100.0},
    'B': {'name': 'Blue', 'lambda': 450.0, 'width': 100.0},
    'RGB': {'name': 'Color (OSC RGB)', 'lambda': 550.0, 'width': 400.0},
    'OSC': {'name': 'One-Shot Color', 'lambda': 550.0, 'width': 400.0},
    # Additional narrowband filters
    'HBETA': {'name': 'Hydrogen Beta', 'lambda': 486.1, 'width': 4.5},
    'HB': {'name': 'Hydrogen Beta', 'lambda': 486.1, 'width': 4.5},
    'H-BETA': {'name': 'Hydrogen Beta', 'lambda': 486.1, 'width': 4.5},
    'HGAMMA': {'name': 'Hydrogen Gamma', 'lambda': 434.0, 'width': 4.5},
    'HG': {'name': 'Hydrogen Gamma', 'lambda': 434.0, 'width': 4.5},
    'H-GAMMA': {'name': 'Hydrogen Gamma', 'lambda': 434.0, 'width': 4.5},
    'HDELTA': {'name': 'Hydrogen Delta', 'lambda': 410.2, 'width': 4.5},
    'HD': {'name': 'Hydrogen Delta', 'lambda': 410.2, 'width': 4.5},
    'H-DELTA': {'name': 'Hydrogen Delta', 'lambda': 410.2, 'width': 4.5},
    'HEPSILON': {'name': 'Hydrogen Epsilon', 'lambda': 397.0, 'width': 4.5},
    'HZETA': {'name': 'Hydrogen Zeta', 'lambda': 388.9, 'width': 4.5},
    'HETA': {'name': 'Hydrogen Eta', 'lambda': 383.5, 'width': 4.5},
    'HTHETA': {'name': 'Hydrogen Theta', 'lambda': 379.8, 'width': 4.5},
    'HIOTA': {'name': 'Hydrogen Iota', 'lambda': 377.1, 'width': 4.5},
    'HKAPPA': {'name': 'Hydrogen Kappa', 'lambda': 375.0, 'width': 4.5},
    'HLAMBDA': {'name': 'Hydrogen Lambda', 'lambda': 373.4, 'width': 4.5},
    'HMU': {'name': 'Hydrogen Mu', 'lambda': 371.2, 'width': 4.5},
    'HNU': {'name': 'Hydrogen Nu', 'lambda': 369.7, 'width': 4.5},
    'HXI': {'name': 'Hydrogen Xi', 'lambda': 368.3, 'width': 4.5},
    'HOMICRON': {'name': 'Hydrogen Omicron', 'lambda': 367.1, 'width': 4.5},
    'HPI': {'name': 'Hydrogen Pi', 'lambda': 366.0, 'width': 4.5},
    'HRHO': {'name': 'Hydrogen Rho', 'lambda': 365.0, 'width': 4.5},
    'HSIGMA': {'name': 'Hydrogen Sigma', 'lambda': 364.1, 'width': 4.5},
    'HTAU': {'name': 'Hydrogen Tau', 'lambda': 363.3, 'width': 4.5},
    'HUPSILON': {'name': 'Hydrogen Upsilon', 'lambda': 362.6, 'width': 4.5},
    'HPHI': {'name': 'Hydrogen Phi', 'lambda': 361.9, 'width': 4.5},
    'HCHI': {'name': 'Hydrogen Chi', 'lambda': 361.3, 'width': 4.5},
    'HPSI': {'name': 'Hydrogen Psi', 'lambda': 360.7, 'width': 4.5},
    'HOMEGA': {'name': 'Hydrogen Omega', 'lambda': 360.1, 'width': 4.5},
    'NII': {'name': 'Nitrogen II', 'lambda': 658.4, 'width': 4.5},
    'N2': {'name': 'Nitrogen II', 'lambda': 658.4, 'width': 4.5},
    'OI': {'name': 'Oxygen I', 'lambda': 630.0, 'width': 4.5},
    'OII': {'name': 'Oxygen II', 'lambda': 372.7, 'width': 4.5},
    'SIII': {'name': 'Sulfur III', 'lambda': 906.9, 'width': 10.0},
    'S3': {'name': 'Sulfur III', 'lambda': 906.9, 'width': 10.0},
    'HEII': {'name': 'Helium II', 'lambda': 468.6, 'width': 4.5},
    # Broadband photometric (Johnson-Cousins)
    'U': {'name': 'Ultraviolet (Johnson U)', 'lambda': 365.0, 'width': 60.0},
    'V': {'name': 'Visual (Johnson V)', 'lambda': 551.0, 'width': 88.0},
    'I': {'name': 'Infrared (Cousins I)', 'lambda': 806.0, 'width': 149.0},
    'RC': {'name': 'Red (Cousins R)', 'lambda': 658.0, 'width': 138.0},
    'IC': {'name': 'Infrared (Cousins I)', 'lambda': 806.0, 'width': 149.0},
    # Sloan/SDSS
    'U_SDSS': {'name': 'Sloan u′', 'lambda': 355.1, 'width': 59.0},
    'G_SDSS': {'name': 'Sloan g′', 'lambda': 475.9, 'width': 138.0},
    'R_SDSS': {'name': 'Sloan r′', 'lambda': 622.3, 'width': 138.0},
    'I_SDSS': {'name': 'Sloan i′', 'lambda': 763.2, 'width': 152.0},
    'Z_SDSS': {'name': 'Sloan z′', 'lambda': 905.5, 'width': 94.0},
    'U_SLOAN': {'name': 'Sloan u′', 'lambda': 355.1, 'width': 59.0},
    'G_SLOAN': {'name': 'Sloan g′', 'lambda': 475.9, 'width': 138.0},
    'R_SLOAN': {'name': 'Sloan r′', 'lambda': 622.3, 'width': 138.0},
    'I_SLOAN': {'name': 'Sloan i′', 'lambda': 763.2, 'width': 152.0},
    'Z_SLOAN': {'name': 'Sloan z′', 'lambda': 905.5, 'width': 94.0},
    # Utility/clear filters
    'CLEAR': {'name': 'Clear', 'lambda': 550.0, 'width': 400.0},
    'IRCUT': {'name': 'IR Cut', 'lambda': 550.0, 'width': 400.0},
    'UVIR': {'name': 'UV/IR Block', 'lambda': 550.0, 'width': 400.0},
    'UV-IR': {'name': 'UV/IR Block', 'lambda': 550.0, 'width': 400.0},
    # Rare noble gas lines (Argon)
    'ARIII': {'name': 'Argon III', 'lambda': 713.6, 'width': 10.0},
    'ARIV': {'name': 'Argon IV', 'lambda': 474.0, 'width': 10.0},
    'ARV': {'name': 'Argon V', 'lambda': 700.0, 'width': 10.0},
    'ARGON': {'name': 'Argon (generic)', 'lambda': 706.7, 'width': 10.0},
    # Other rare/line filters
    'NEON': {'name': 'Neon (generic)', 'lambda': 640.2, 'width': 10.0},
    'KR': {'name': 'Krypton (generic)', 'lambda': 758.7, 'width': 10.0},
    'KRYPTON': {'name': 'Krypton (generic)', 'lambda': 758.7, 'width': 10.0},
    'XE': {'name': 'Xenon (generic)', 'lambda': 823.2, 'width': 10.0},
    'XENON': {'name': 'Xenon (generic)', 'lambda': 823.2, 'width': 10.0},
    'HEI': {'name': 'Helium I', 'lambda': 587.6, 'width': 10.0},
    'NA': {'name': 'Sodium D', 'lambda': 589.3, 'width': 6.0},
    'SODIUM': {'name': 'Sodium D', 'lambda': 589.3, 'width': 6.0},
    'K': {'name': 'Potassium', 'lambda': 769.9, 'width': 6.0},
    'CAK': {'name': 'Calcium K', 'lambda': 393.4, 'width': 2.0},
    'CAH': {'name': 'Calcium H', 'lambda': 396.8, 'width': 2.0},
    'OI_5577': {'name': 'Oxygen I (airglow)', 'lambda': 557.7, 'width': 3.0},
    'OI_6300': {'name': 'Oxygen I', 'lambda': 630.0, 'width': 3.0},
    'OI_6364': {'name': 'Oxygen I', 'lambda': 636.4, 'width': 3.0},
    'SIII_9531': {'name': 'Sulfur III', 'lambda': 953.1, 'width': 10.0},
    'CH4': {'name': 'Methane', 'lambda': 889.0, 'width': 10.0},
    # Light pollution suppression and multiband filters (typical central bands ref.)
    'CLS': {'name': 'City Light Suppression (CLS)', 'lambda': 550.0, 'width': 200.0},
    'UHC': {'name': 'Ultra High Contrast (UHC)', 'lambda': 500.0, 'width': 200.0},
    'LPRO': {'name': 'Optolong L-Pro', 'lambda': 550.0, 'width': 300.0},
    'LPRO_OPTOLONG': {'name': 'Optolong L-Pro', 'lambda': 550.0, 'width': 300.0},
    'LEHNANCE': {'name': 'Optolong L-eNhance (dual-band)', 'lambda': 600.0, 'width': 20.0},
    'LEXTREME': {'name': 'Optolong L-eXtreme (dual-band)', 'lambda': 600.0, 'width': 14.0},
    'LULTIMATE': {'name': 'Optolong L-Ultimate (dual-band)', 'lambda': 600.0, 'width': 7.0},
    'IDAS_LPS': {'name': 'IDAS LPS', 'lambda': 550.0, 'width': 200.0},
    'IDAS_LPS_D1': {'name': 'IDAS LPS D1', 'lambda': 550.0, 'width': 200.0},
    'IDAS_LPS_D2': {'name': 'IDAS LPS D2', 'lambda': 550.0, 'width': 200.0},
    'IDAS_LPS_NBZ': {'name': 'IDAS NBZ (dual-band)', 'lambda': 600.0, 'width': 20.0},
    'NBZ': {'name': 'IDAS NBZ (dual-band)', 'lambda': 600.0, 'width': 20.0},
    'TRIBAND': {'name': 'Tri-band', 'lambda': 600.0, 'width': 30.0},
    'QUAD_BAND': {'name': 'Quad-band', 'lambda': 600.0, 'width': 40.0}
}

def create_enhanced_progress_bar(iterable, total, desc, unit="file"):
    """Create an enhanced progress bar with better visibility and formatting"""
    if not TQDM_AVAILABLE:
        return iterable
    
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        bar_format='{l_bar}🟢{bar}🟢| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        colour='green',
        leave=True,
        position=0,
        ncols=100,
        mininterval=0.05,
        maxinterval=0.5,
        dynamic_ncols=False,
        ascii=False,
        smoothing=0.1
    )

def print_progress_info(adu_tasks, non_adu_tasks, total_tasks):
    """Print detailed progress information"""
    print(f"📊 Progress Details:")
    if total_tasks > 0:
        print(f"   ⚡ Fast processing: {total_tasks} files (100.0%)")
    else:
        print(f"   ⚡ Fast processing: {total_tasks} files (0.0%)")
    print(f"   📁 Total: {total_tasks} files")
    print(f"   🎯 Progress bar will show real-time status...")
    print()

def print_progress_completion():
    """Print completion message for progress"""
    print()
    print("✅ File processing completed!")
    print("   📈 Progress bar will remain visible until analysis is finished.")
    print()

def parse_args():
    parser = argparse.ArgumentParser(description="Complete Astrophotography Analysis (CLI)")
    parser.add_argument("--folder", type=str, default=None, help="FITS folder to analyze (default: script folder)")
    parser.add_argument("--mode", type=int, choices=[1], default=1, help="Analysis mode: 1=fast (theoretical calculation only)")
    parser.add_argument("--region-size", type=int, default=100, help="Size (px) of advanced SNR regions")
    parser.add_argument("--output", type=str, default=None, help="Output folder")
    parser.add_argument("--no-graphs", action="store_true", help="Do not generate graphs")
    parser.add_argument("--no-latex", action="store_true", help="Do not generate LaTeX files")
    parser.add_argument("--auto-install", action="store_true", help="Automatically install missing Python packages")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: auto-detect CPU cores)")
    parser.add_argument("--export-csv", action="store_true", help="Export CSV summaries")
    parser.add_argument("--zip-output", action="store_true", help="Compress output folder to .zip")
    return parser.parse_args()

def export_csv(data_by_target, global_data, output_folder):
    import csv
    global_path = os.path.join(output_folder, "global_summary.csv")
    with open(global_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["total_files","targets","instruments","telescopes","total_time"])
        w.writerow([
            global_data['total_files'],
            len(global_data['targets_found']),
            len(global_data['instruments_used']),
            len(global_data['telescopes_used']),
            global_data['total_time']
        ])
    targets_path = os.path.join(output_folder, "targets_summary.csv")
    with open(targets_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target","nb_files","filters","instruments","telescopes","total_duration_s"])
        for target, d in data_by_target.items():
            nb = len(d['files'])
            filters = ",".join(sorted(d['time_by_filter'].keys()))
            w.writerow([
                target, nb, filters,
                len(d['instruments']), len(d['telescopes']),
                sum(d['time_by_filter'].values())
            ])
    print(f"CSV files generated: {global_path}, {targets_path}")

# Recognized image types
RECOGNIZED_TYPES = ['LIGHT', 'DARK', 'BIAS']

# Configuration to optimize analysis speed
FAST_ANALYSIS = True  # Fast mode enabled (no ADU analysis)
ADU_SAMPLE_PER_FILTER = 0  # No ADU analysis in fast mode
ADU_ANALYSIS_ENABLED = True  # New variable to control ADU analysis

# Path to calibration files (no personal default)
BIAS_DARK_PATH = None

# Configuration file to save parameters
CONFIG_FILE = "astro_config.json"
DEFAULT_REGION_SIZE = 100

# Cache for displayed warnings to avoid repetition
_displayed_warnings = set()

# Alternative cache using a simple flag file
_warning_cache_file = "telescope_warning_shown.txt"

def has_warning_been_shown(warning_type="telescope_unknown"):
    """Check if a warning has already been shown using file-based cache"""
    try:
        if os.path.exists(_warning_cache_file):
            with open(_warning_cache_file, 'r') as f:
                content = f.read()
                return warning_type in content
    except:
        pass
    return False

def mark_warning_as_shown(warning_type="telescope_unknown"):
    """Mark a warning as shown using file-based cache"""
    try:
        with open(_warning_cache_file, 'a') as f:
            f.write(f"{warning_type}\n")
    except:
        pass

def open_fits_for_data(file_path):
    """Opens a FITS file choosing memmap according to header.
    If BZERO/BSCALE/BLANK are present, uses memmap=False (required by astropy).
    Returns an HDUList ready for .data and .header reading.
    """
    try:
        hdul = fits.open(file_path, memmap=True, ignore_missing_simple=True)
        header = hdul[0].header
        if ('BZERO' in header) or ('BSCALE' in header) or ('BLANK' in header):
            hdul.close()
            return fits.open(file_path, memmap=False, ignore_missing_simple=True)
        return hdul
    except Exception as e:
        # Handle specific FITS file issues
        if "Header missing END card" in str(e):
            raise Exception(f"Header missing END card")
        elif "non-ASCII characters" in str(e):
            raise Exception(f"Non-ASCII characters in header")
        elif "null bytes" in str(e):
            raise Exception(f"Non-compliant FITS header (null bytes)")
        else:
            # Safe fallback for other errors
            return fits.open(file_path, memmap=False, ignore_missing_simple=True)

# CCD/CMOS sensor database with their characteristics
SENSORS_DATABASE = {
    # ZWO Cameras
    'ASI1600MM': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 3.8, 'quantum_efficiency': 0.6, 'width_px': 4656, 'height_px': 3520},
    'ASI1600MM-Pro': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 3.8, 'quantum_efficiency': 0.6, 'width_px': 4656, 'height_px': 3520},
    'ASI2600MM': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 6248, 'height_px': 4176},
    'ASI2600MM-Pro': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 6248, 'height_px': 4176},
    'ASI6200MM': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 9576, 'height_px': 6388},
    'ASI6200MM-Pro': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 9576, 'height_px': 6388},
    'ASI294MM': {'gain': 120, 'read_noise': 1.2, 'full_well': 42000, 'pixel_size': 4.63, 'quantum_efficiency': 0.75, 'width_px': 4144, 'height_px': 2822},
    'ASI294MM-Pro': {'gain': 120, 'read_noise': 1.2, 'full_well': 42000, 'pixel_size': 4.63, 'quantum_efficiency': 0.75, 'width_px': 4144, 'height_px': 2822},
    'ASI533MM': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 3008, 'height_px': 3008},
    'ASI533MM-Pro': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 3008, 'height_px': 3008},
    'ASI183MM': {'gain': 111, 'read_noise': 1.1, 'full_well': 15000, 'pixel_size': 2.4, 'quantum_efficiency': 0.54, 'width_px': 5496, 'height_px': 3672},
    'ASI183MM-Pro': {'gain': 111, 'read_noise': 1.1, 'full_well': 15000, 'pixel_size': 2.4, 'quantum_efficiency': 0.54, 'width_px': 5496, 'height_px': 3672},
    'ASI178MM': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 2.4, 'quantum_efficiency': 0.6, 'width_px': 1920, 'height_px': 1080},
    'ASI178MM-Pro': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 2.4, 'quantum_efficiency': 0.6, 'width_px': 1920, 'height_px': 1080},
    'ASI174MM': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 5.86, 'quantum_efficiency': 0.6, 'width_px': 1920, 'height_px': 1200},
    'ASI174MM-Pro': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 5.86, 'quantum_efficiency': 0.6, 'width_px': 1920, 'height_px': 1200},
    'ASI290MM': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 2.9, 'quantum_efficiency': 0.6, 'width_px': 1920, 'height_px': 1080},
    'ASI290MM-Pro': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 2.9, 'quantum_efficiency': 0.6, 'width_px': 1920, 'height_px': 1080},
    'ASI224MC': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 3.75, 'quantum_efficiency': 0.6, 'width_px': 1280, 'height_px': 960},
    'ASI224MC-Pro': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 3.75, 'quantum_efficiency': 0.6, 'width_px': 1280, 'height_px': 960},
    'ASI385MC': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 3.75, 'quantum_efficiency': 0.6, 'width_px': 1280, 'height_px': 960},
    'ASI385MC-Pro': {'gain': 139, 'read_noise': 1.2, 'full_well': 20000, 'pixel_size': 3.75, 'quantum_efficiency': 0.6, 'width_px': 1280, 'height_px': 960},
    
    # QHY Cameras
    'QHY600M': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 9576, 'height_px': 6388},
    'QHY600M-P': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 9576, 'height_px': 6388},
    'QHY268M': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 6248, 'height_px': 4176},
    'QHY268M-P': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 6248, 'height_px': 4176},
    'QHY163M': {'gain': 111, 'read_noise': 1.1, 'full_well': 15000, 'pixel_size': 3.8, 'quantum_efficiency': 0.54, 'width_px': 4656, 'height_px': 3520},
    'QHY163M-P': {'gain': 111, 'read_noise': 1.1, 'full_well': 15000, 'pixel_size': 3.8, 'quantum_efficiency': 0.54, 'width_px': 4656, 'height_px': 3520},
    'QHY367C': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 6240, 'height_px': 4160},
    'QHY367C-P': {'gain': 100, 'read_noise': 1.0, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 6240, 'height_px': 4160},
    
    # FLI Cameras
    'FLI-ML16200': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 6.0, 'quantum_efficiency': 0.6, 'width_px': 4096, 'height_px': 4096},
    'FLI-ML8300': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 5.4, 'quantum_efficiency': 0.6, 'width_px': 3326, 'height_px': 2504},
    'FLI-ML11002': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 9.0, 'quantum_efficiency': 0.6, 'width_px': 4008, 'height_px': 2672},
    'FLI-ML16803': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 9.0, 'quantum_efficiency': 0.6, 'width_px': 4096, 'height_px': 4096},
    
    # SBIG Cameras
    'STF-8300M': {'gain': 0.37, 'read_noise': 9.3, 'full_well': 25000, 'pixel_size': 5.4, 'quantum_efficiency': 0.6, 'width_px': 3326, 'height_px': 2504},
    'STF-8300C': {'gain': 0.37, 'read_noise': 9.3, 'full_well': 25000, 'pixel_size': 5.4, 'quantum_efficiency': 0.6, 'width_px': 3326, 'height_px': 2504},
    'STT-8300M': {'gain': 0.37, 'read_noise': 9.3, 'full_well': 25000, 'pixel_size': 5.4, 'quantum_efficiency': 0.6, 'width_px': 3326, 'height_px': 2504},
    'STT-8300C': {'gain': 0.37, 'read_noise': 9.3, 'full_well': 25000, 'pixel_size': 5.4, 'quantum_efficiency': 0.6, 'width_px': 3326, 'height_px': 2504},
    
    # Atik Cameras
    'ATIK460EXM': {'gain': 0.26, 'read_noise': 6.5, 'full_well': 18000, 'pixel_size': 4.54, 'quantum_efficiency': 0.6, 'width_px': 1280, 'height_px': 1024},
    'ATIK460EXC': {'gain': 0.26, 'read_noise': 6.5, 'full_well': 18000, 'pixel_size': 4.54, 'quantum_efficiency': 0.6, 'width_px': 1280, 'height_px': 1024},
    'ATIK383L+': {'gain': 0.26, 'read_noise': 6.5, 'full_well': 18000, 'pixel_size': 5.4, 'quantum_efficiency': 0.6, 'width_px': 3326, 'height_px': 2504},
    'ATIK383L+ Mono': {'gain': 0.26, 'read_noise': 6.5, 'full_well': 18000, 'pixel_size': 5.4, 'quantum_efficiency': 0.6, 'width_px': 3326, 'height_px': 2504},
    
    # Moravian Cameras
    'G3-16200': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 6.0, 'quantum_efficiency': 0.6, 'width_px': 4096, 'height_px': 4096},
    'G3-8300': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 5.4, 'quantum_efficiency': 0.6, 'width_px': 3326, 'height_px': 2504},
    'G3-11002': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 9.0, 'quantum_efficiency': 0.6, 'width_px': 4008, 'height_px': 2672},
    'G3-16803': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 9.0, 'quantum_efficiency': 0.6, 'width_px': 4096, 'height_px': 4096},
    
    # Moravian G1 series (representative models)
    'G1-0300': {'gain': 1.0, 'read_noise': 7.0, 'full_well': 20000, 'pixel_size': 5.6, 'quantum_efficiency': 0.5, 'width_px': 640, 'height_px': 480},
    'G1-1200': {'gain': 1.0, 'read_noise': 7.0, 'full_well': 20000, 'pixel_size': 3.75, 'quantum_efficiency': 0.5, 'width_px': 1280, 'height_px': 960},

    # Moravian G4 series (large CCDs)
    'G4-16000': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 9.0, 'quantum_efficiency': 0.6, 'width_px': 4096, 'height_px': 4096},
    'G4-9000': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 120000, 'pixel_size': 12.0, 'quantum_efficiency': 0.55, 'width_px': 3056, 'height_px': 3056},
    'G4-11000': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 80000, 'pixel_size': 9.0, 'quantum_efficiency': 0.55, 'width_px': 4008, 'height_px': 2672},

    # Moravian G5 series (modern large-format)
    'G5-16200': {'gain': 1.4, 'read_noise': 8.5, 'full_well': 90000, 'pixel_size': 6.0, 'quantum_efficiency': 0.6, 'width_px': 4500, 'height_px': 3600},
    'G5-16803': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 9.0, 'quantum_efficiency': 0.6, 'width_px': 4096, 'height_px': 4096},

    # Additional ZWO Cameras
    'ASI6200MM': {'gain': 0.1, 'read_noise': 0.8, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.91, 'width_px': 9576, 'height_px': 6388},
    'ASI6200MC': {'gain': 0.1, 'read_noise': 0.8, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.91, 'width_px': 9576, 'height_px': 6388},
    'ASI2600MM': {'gain': 0.1, 'read_noise': 0.8, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.91, 'width_px': 6240, 'height_px': 4160},
    'ASI2600MC': {'gain': 0.1, 'read_noise': 0.8, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.91, 'width_px': 6240, 'height_px': 4160},
    'ASI2400MM': {'gain': 0.1, 'read_noise': 0.8, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.91, 'width_px': 6240, 'height_px': 4160},
    'ASI2400MC': {'gain': 0.1, 'read_noise': 0.8, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.91, 'width_px': 6240, 'height_px': 4160},
    'ASI533MM': {'gain': 0.1, 'read_noise': 0.8, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.91, 'width_px': 3008, 'height_px': 3008},
    'ASI533MC': {'gain': 0.1, 'read_noise': 0.8, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.91, 'width_px': 3008, 'height_px': 3008},
    'ASI462MM': {'gain': 0.5, 'read_noise': 1.8, 'full_well': 20000, 'pixel_size': 2.9, 'quantum_efficiency': 0.84, 'width_px': 1920, 'height_px': 1080},
    'ASI462MC': {'gain': 0.5, 'read_noise': 1.8, 'full_well': 20000, 'pixel_size': 2.9, 'quantum_efficiency': 0.84, 'width_px': 1920, 'height_px': 1080},
    'ASI482MM': {'gain': 0.5, 'read_noise': 1.8, 'full_well': 20000, 'pixel_size': 2.9, 'quantum_efficiency': 0.84, 'width_px': 1920, 'height_px': 1080},
    'ASI482MC': {'gain': 0.5, 'read_noise': 1.8, 'full_well': 20000, 'pixel_size': 2.9, 'quantum_efficiency': 0.84, 'width_px': 1920, 'height_px': 1080},
    'ASI485MM': {'gain': 0.5, 'read_noise': 1.8, 'full_well': 20000, 'pixel_size': 2.9, 'quantum_efficiency': 0.84, 'width_px': 1920, 'height_px': 1080},
    'ASI485MC': {'gain': 0.5, 'read_noise': 1.8, 'full_well': 20000, 'pixel_size': 2.9, 'quantum_efficiency': 0.84, 'width_px': 1920, 'height_px': 1080},
    'ASI678MC': {'gain': 0.5, 'read_noise': 0.7, 'full_well': 16000, 'pixel_size': 2.0, 'quantum_efficiency': 0.85, 'width_px': 3840, 'height_px': 2160},
    'ASI662MC': {'gain': 0.5, 'read_noise': 0.8, 'full_well': 18000, 'pixel_size': 2.9, 'quantum_efficiency': 0.85, 'width_px': 1920, 'height_px': 1080},
    'ASI585MC': {'gain': 0.5, 'read_noise': 1.2, 'full_well': 30000, 'pixel_size': 2.9, 'quantum_efficiency': 0.9, 'width_px': 3840, 'height_px': 2160},
    
    # Additional QHY Cameras
    'QHY8L': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 5.4, 'quantum_efficiency': 0.75, 'width_px': 1536, 'height_px': 1024},
    'QHY9M': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 5.4, 'quantum_efficiency': 0.75, 'width_px': 1536, 'height_px': 1024},
    'QHY10': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 4.65, 'quantum_efficiency': 0.75, 'width_px': 1280, 'height_px': 1024},
    'QHY11': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 4.65, 'quantum_efficiency': 0.75, 'width_px': 1280, 'height_px': 1024},
    'QHY12': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 4.65, 'quantum_efficiency': 0.75, 'width_px': 1280, 'height_px': 1024},
    'QHY168M': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.8, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    'QHY168C': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.8, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    'QHY183M': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.4, 'quantum_efficiency': 0.75, 'width_px': 5496, 'height_px': 3672},
    'QHY183C': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.4, 'quantum_efficiency': 0.75, 'width_px': 5496, 'height_px': 3672},
    'QHY294M': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 4.63, 'quantum_efficiency': 0.75, 'width_px': 4144, 'height_px': 2822},
    'QHY294C': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 4.63, 'quantum_efficiency': 0.75, 'width_px': 4144, 'height_px': 2822},
    'QHY533M': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 3008, 'height_px': 3008},
    'QHY533C': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 3008, 'height_px': 3008},
    'QHY268M': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 6248, 'height_px': 4176},
    'QHY268C': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 6248, 'height_px': 4176},
    'QHY410C': {'gain': 1.0, 'read_noise': 1.1, 'full_well': 80000, 'pixel_size': 5.94, 'quantum_efficiency': 0.8, 'width_px': 6000, 'height_px': 4000},
    'QHY367C': {'gain': 1.0, 'read_noise': 1.5, 'full_well': 60000, 'pixel_size': 4.88, 'quantum_efficiency': 0.75, 'width_px': 7376, 'height_px': 4928},
    'QHY247C': {'gain': 1.0, 'read_noise': 1.8, 'full_well': 45000, 'pixel_size': 3.91, 'quantum_efficiency': 0.7, 'width_px': 6000, 'height_px': 4000},
    
    # Additional FLI Cameras
    'FLI-PL16803': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 9.0, 'quantum_efficiency': 0.85, 'width_px': 4096, 'height_px': 4096},
    'FLI-PL11002': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 9.0, 'quantum_efficiency': 0.85, 'width_px': 4008, 'height_px': 2672},
    'FLI-PL09000': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 12.0, 'quantum_efficiency': 0.85, 'width_px': 3056, 'height_px': 3056},
    'FLI-PL4710': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 4.65, 'quantum_efficiency': 0.85, 'width_px': 1024, 'height_px': 1024},
    'FLI-PL230': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 4.65, 'quantum_efficiency': 0.85, 'width_px': 1024, 'height_px': 1024},
    'FLI-ML16803': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 9.0, 'quantum_efficiency': 0.85, 'width_px': 4096, 'height_px': 4096},
    'FLI-ML11002': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 9.0, 'quantum_efficiency': 0.85, 'width_px': 4008, 'height_px': 2672},
    'FLI-ML09000': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 12.0, 'quantum_efficiency': 0.85, 'width_px': 3056, 'height_px': 3056},
    'FLI-ML4710': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 4.65, 'quantum_efficiency': 0.85, 'width_px': 1024, 'height_px': 1024},
    'FLI-ML230': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 4.65, 'quantum_efficiency': 0.85, 'width_px': 1024, 'height_px': 1024},
    
    # Additional Moravian Cameras
    'G2-8300': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 5.4, 'quantum_efficiency': 0.85, 'width_px': 3326, 'height_px': 2504},
    'G2-1600': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 4.4, 'quantum_efficiency': 0.85, 'width_px': 1536, 'height_px': 1024},
    'G2-4000': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 7.4, 'quantum_efficiency': 0.85, 'width_px': 2048, 'height_px': 2048},
    'G2-8300M': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 5.4, 'quantum_efficiency': 0.85, 'width_px': 3326, 'height_px': 2504},
    'G2-1600M': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 4.4, 'quantum_efficiency': 0.85, 'width_px': 1536, 'height_px': 1024},
    'G2-4000M': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 7.4, 'quantum_efficiency': 0.85, 'width_px': 2048, 'height_px': 2048},
    'G3-8300': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 5.4, 'quantum_efficiency': 0.85, 'width_px': 3326, 'height_px': 2504},
    'G3-1600': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 4.4, 'quantum_efficiency': 0.85, 'width_px': 1536, 'height_px': 1024},
    'G3-4000': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 7.4, 'quantum_efficiency': 0.85, 'width_px': 2048, 'height_px': 2048},
    'G3-8300M': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 5.4, 'quantum_efficiency': 0.85, 'width_px': 3326, 'height_px': 2504},
    'G3-1600M': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 4.4, 'quantum_efficiency': 0.85, 'width_px': 1536, 'height_px': 1024},
    'G3-4000M': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 7.4, 'quantum_efficiency': 0.85, 'width_px': 2048, 'height_px': 2048},
    
    # Touptek Cameras
    'Touptek IMX178': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.4, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    'Touptek IMX183': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.4, 'quantum_efficiency': 0.75, 'width_px': 5496, 'height_px': 3672},
    'Touptek IMX294': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 4.63, 'quantum_efficiency': 0.75, 'width_px': 4144, 'height_px': 2822},
    'Touptek IMX533': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 3008, 'height_px': 3008},
    'Touptek IMX571': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 6248, 'height_px': 4176},
    'Touptek IMX455': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 9576, 'height_px': 6388},
    'Touptek IMX461': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.9, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    'Touptek IMX485': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.9, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    'Touptek IMX482': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.9, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    # Player One (examples)
    'PLAYER ONE NEPTUNE-C II': {'gain': 1.0, 'read_noise': 0.8, 'full_well': 20000, 'pixel_size': 2.9, 'quantum_efficiency': 0.85, 'width_px': 1920, 'height_px': 1080},
    'PLAYER ONE URANUS-C': {'gain': 1.0, 'read_noise': 1.0, 'full_well': 20000, 'pixel_size': 2.9, 'quantum_efficiency': 0.85, 'width_px': 1920, 'height_px': 1080},
    'PLAYER ONE APOLLO-MINI C': {'gain': 1.0, 'read_noise': 0.9, 'full_well': 18000, 'pixel_size': 2.4, 'quantum_efficiency': 0.85, 'width_px': 5496, 'height_px': 3672},
    # SVBONY
    'SVBONY SV305': {'gain': 1.0, 'read_noise': 2.0, 'full_well': 15000, 'pixel_size': 2.9, 'quantum_efficiency': 0.7, 'width_px': 1920, 'height_px': 1080},
    'SVBONY SV405CC': {'gain': 1.0, 'read_noise': 1.2, 'full_well': 45000, 'pixel_size': 4.63, 'quantum_efficiency': 0.75, 'width_px': 4144, 'height_px': 2822},
    'SVBONY SV605CC': {'gain': 1.0, 'read_noise': 1.2, 'full_well': 50000, 'pixel_size': 3.76, 'quantum_efficiency': 0.8, 'width_px': 6248, 'height_px': 4176},
    # ATIK/FLI (color variants)
    'ATIK-ONE C': {'gain': 0.26, 'read_noise': 6.5, 'full_well': 18000, 'pixel_size': 4.54, 'quantum_efficiency': 0.5, 'width_px': 3352, 'height_px': 2532},
    'FLI-ML8300C': {'gain': 1.4, 'read_noise': 9.0, 'full_well': 100000, 'pixel_size': 5.4, 'quantum_efficiency': 0.5, 'width_px': 3326, 'height_px': 2504},
    # DSLR / Mirrorless (approximate typical values)
    'CANON 600D': {'gain': 0.9, 'read_noise': 2.7, 'full_well': 25000, 'pixel_size': 4.3, 'quantum_efficiency': 0.5, 'width_px': 5184, 'height_px': 3456},
    'CANON 750D': {'gain': 0.9, 'read_noise': 2.6, 'full_well': 24000, 'pixel_size': 3.7, 'quantum_efficiency': 0.5, 'width_px': 6000, 'height_px': 4000},
    'CANON 5D MARK III': {'gain': 0.9, 'read_noise': 2.5, 'full_well': 60000, 'pixel_size': 6.25, 'quantum_efficiency': 0.5, 'width_px': 5760, 'height_px': 3840},
    'CANON R6': {'gain': 0.9, 'read_noise': 1.5, 'full_well': 70000, 'pixel_size': 6.56, 'quantum_efficiency': 0.6, 'width_px': 5472, 'height_px': 3648},
    'NIKON D5300': {'gain': 0.9, 'read_noise': 1.8, 'full_well': 24000, 'pixel_size': 3.9, 'quantum_efficiency': 0.5, 'width_px': 6000, 'height_px': 4000},
    'NIKON D750': {'gain': 0.9, 'read_noise': 1.7, 'full_well': 60000, 'pixel_size': 5.95, 'quantum_efficiency': 0.5, 'width_px': 6016, 'height_px': 4016},
    'NIKON D850': {'gain': 0.9, 'read_noise': 2.2, 'full_well': 45000, 'pixel_size': 4.35, 'quantum_efficiency': 0.5, 'width_px': 8256, 'height_px': 5504},
    'NIKON Z6': {'gain': 0.9, 'read_noise': 1.4, 'full_well': 70000, 'pixel_size': 5.94, 'quantum_efficiency': 0.6, 'width_px': 6048, 'height_px': 4024},
    'SONY A7S': {'gain': 0.9, 'read_noise': 1.0, 'full_well': 80000, 'pixel_size': 8.4, 'quantum_efficiency': 0.65, 'width_px': 4240, 'height_px': 2832},
    'SONY A7III': {'gain': 0.9, 'read_noise': 1.2, 'full_well': 70000, 'pixel_size': 5.9, 'quantum_efficiency': 0.6, 'width_px': 6000, 'height_px': 4000},
    'SONY A7RIII': {'gain': 0.9, 'read_noise': 1.5, 'full_well': 60000, 'pixel_size': 4.5, 'quantum_efficiency': 0.55, 'width_px': 7952, 'height_px': 5304},
    'SIGMA FP': {'gain': 0.9, 'read_noise': 1.5, 'full_well': 60000, 'pixel_size': 5.97, 'quantum_efficiency': 0.5, 'width_px': 6000, 'height_px': 4000},
    # More Canon
    'CANON 1100D': {'gain': 0.9, 'read_noise': 3.0, 'full_well': 22000, 'pixel_size': 5.2, 'quantum_efficiency': 0.45, 'width_px': 4272, 'height_px': 2848},
    'CANON 700D': {'gain': 0.9, 'read_noise': 2.7, 'full_well': 24000, 'pixel_size': 4.3, 'quantum_efficiency': 0.48, 'width_px': 5184, 'height_px': 3456},
    'CANON 80D': {'gain': 0.9, 'read_noise': 2.4, 'full_well': 25000, 'pixel_size': 3.7, 'quantum_efficiency': 0.5, 'width_px': 6000, 'height_px': 4000},
    'CANON 90D': {'gain': 0.9, 'read_noise': 2.3, 'full_well': 23000, 'pixel_size': 3.2, 'quantum_efficiency': 0.5, 'width_px': 6960, 'height_px': 4640},
    'CANON 6D': {'gain': 0.9, 'read_noise': 2.1, 'full_well': 65000, 'pixel_size': 6.55, 'quantum_efficiency': 0.5, 'width_px': 5472, 'height_px': 3648},
    'CANON 6D MARK II': {'gain': 0.9, 'read_noise': 2.0, 'full_well': 65000, 'pixel_size': 5.76, 'quantum_efficiency': 0.5, 'width_px': 6240, 'height_px': 4160},
    'CANON R5': {'gain': 0.9, 'read_noise': 1.4, 'full_well': 65000, 'pixel_size': 4.4, 'quantum_efficiency': 0.6, 'width_px': 8192, 'height_px': 5464},
    'CANON R7': {'gain': 0.9, 'read_noise': 1.6, 'full_well': 26000, 'pixel_size': 3.2, 'quantum_efficiency': 0.55, 'width_px': 6960, 'height_px': 4640},
    # More Nikon
    'NIKON D3200': {'gain': 0.9, 'read_noise': 2.4, 'full_well': 22000, 'pixel_size': 3.9, 'quantum_efficiency': 0.45, 'width_px': 6016, 'height_px': 4000},
    'NIKON D5600': {'gain': 0.9, 'read_noise': 1.7, 'full_well': 24000, 'pixel_size': 3.9, 'quantum_efficiency': 0.5, 'width_px': 6000, 'height_px': 4000},
    'NIKON D610': {'gain': 0.9, 'read_noise': 1.7, 'full_well': 65000, 'pixel_size': 5.95, 'quantum_efficiency': 0.5, 'width_px': 6016, 'height_px': 4016},
    'NIKON D780': {'gain': 0.9, 'read_noise': 1.5, 'full_well': 70000, 'pixel_size': 5.94, 'quantum_efficiency': 0.55, 'width_px': 6048, 'height_px': 4024},
    'NIKON Z7': {'gain': 0.9, 'read_noise': 1.9, 'full_well': 60000, 'pixel_size': 4.35, 'quantum_efficiency': 0.55, 'width_px': 8256, 'height_px': 5504},
    'NIKON D810': {'gain': 0.9, 'read_noise': 2.2, 'full_well': 45000, 'pixel_size': 4.88, 'quantum_efficiency': 0.5, 'width_px': 7360, 'height_px': 4912},
    'NIKON D810A': {'gain': 0.9, 'read_noise': 2.1, 'full_well': 45000, 'pixel_size': 4.88, 'quantum_efficiency': 0.52, 'width_px': 7360, 'height_px': 4912},
    'NIKON Z8': {'gain': 0.9, 'read_noise': 1.8, 'full_well': 60000, 'pixel_size': 4.35, 'quantum_efficiency': 0.58, 'width_px': 8256, 'height_px': 5504},
    'NIKON Z9': {'gain': 0.9, 'read_noise': 1.8, 'full_well': 60000, 'pixel_size': 4.35, 'quantum_efficiency': 0.58, 'width_px': 8256, 'height_px': 5504},
    # More Sony
    'SONY A6000': {'gain': 0.9, 'read_noise': 1.5, 'full_well': 24000, 'pixel_size': 3.9, 'quantum_efficiency': 0.5, 'width_px': 6000, 'height_px': 4000},
    'SONY A6400': {'gain': 0.9, 'read_noise': 1.4, 'full_well': 26000, 'pixel_size': 3.9, 'quantum_efficiency': 0.55, 'width_px': 6000, 'height_px': 4000},
    'SONY A6500': {'gain': 0.9, 'read_noise': 1.3, 'full_well': 26000, 'pixel_size': 3.9, 'quantum_efficiency': 0.55, 'width_px': 6000, 'height_px': 4000},
    'SONY A7IV': {'gain': 0.9, 'read_noise': 1.3, 'full_well': 65000, 'pixel_size': 5.1, 'quantum_efficiency': 0.6, 'width_px': 7008, 'height_px': 4672},
    
    # SONY IMX Sensors (Direct sensor references)
    'IMX178': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.4, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    'IMX183': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.4, 'quantum_efficiency': 0.75, 'width_px': 5496, 'height_px': 3672},
    'IMX294': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 4.63, 'quantum_efficiency': 0.75, 'width_px': 4144, 'height_px': 2822},
    'IMX533': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 3008, 'height_px': 3008},
    'IMX571': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 6248, 'height_px': 4176},
    'IMX455': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.76, 'quantum_efficiency': 0.75, 'width_px': 9576, 'height_px': 6388},
    'IMX461': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.9, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    'IMX485': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.9, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    'IMX482': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.9, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    'IMX224': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 3.75, 'quantum_efficiency': 0.75, 'width_px': 1280, 'height_px': 960},
    'IMX290': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.9, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    'IMX174': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 5.86, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1200},
    'IMX462': {'gain': 1.0, 'read_noise': 3.5, 'full_well': 15000, 'pixel_size': 2.9, 'quantum_efficiency': 0.75, 'width_px': 1920, 'height_px': 1080},
    
    # Default values for unknown sensors
    'default': {'gain': 100, 'read_noise': 1.0, 'full_well': 20000, 'pixel_size': 3.8, 'quantum_efficiency': 0.6, 'width_px': 1920, 'height_px': 1080}
}

# Messier objects database with common names in English
MESSIER_DATABASE = {
    'M1': 'M 1 (Crab Nebula)',
    'M2': 'M 2 (Globular Cluster)',
    'M3': 'M 3 (Globular Cluster)',
    'M4': 'M 4 (Globular Cluster)',
    'M5': 'M 5 (Globular Cluster)',
    'M6': 'M 6 (Butterfly Cluster)',
    'M7': 'M 7 (Ptolemy Cluster)',
    'M8': 'M 8 (Lagoon Nebula)',
    'M9': 'M 9 (Globular Cluster)',
    'M10': 'M 10 (Globular Cluster)',
    'M11': 'M 11 (Wild Duck Cluster)',
    'M12': 'M 12 (Globular Cluster)',
    'M13': 'M 13 (Hercules Cluster)',
    'M14': 'M 14 (Globular Cluster)',
    'M15': 'M 15 (Globular Cluster)',
    'M16': 'M 16 (Eagle Nebula)',
    'M17': 'M 17 (Omega Nebula)',
    'M18': 'M 18 (Open Cluster)',
    'M19': 'M 19 (Globular Cluster)',
    'M20': 'M 20 (Trifid Nebula)',
    'M21': 'M 21 (Open Cluster)',
    'M22': 'M 22 (Globular Cluster)',
    'M23': 'M 23 (Open Cluster)',
    'M24': 'M 24 (Sagittarius Star Cloud)',
    'M25': 'M 25 (Open Cluster)',
    'M26': 'M 26 (Open Cluster)',
    'M27': 'M 27 (Dumbbell Nebula)',
    'M28': 'M 28 (Globular Cluster)',
    'M29': 'M 29 (Open Cluster)',
    'M30': 'M 30 (Globular Cluster)',
    'M31': 'M 31 (Andromeda Galaxy)',
    'M32': 'M 32 (Andromeda Satellite Galaxy)',
    'M33': 'M 33 (Triangulum Galaxy)',
    'M34': 'M 34 (Open Cluster)',
    'M35': 'M 35 (Open Cluster)',
    'M36': 'M 36 (Open Cluster)',
    'M37': 'M 37 (Open Cluster)',
    'M38': 'M 38 (Open Cluster)',
    'M39': 'M 39 (Open Cluster)',
    'M40': 'M 40 (Double Star)',
    'M41': 'M 41 (Open Cluster)',
    'M42': 'M 42 (Orion Nebula)',
    'M43': 'M 43 (De Mairan Nebula)',
    'M44': 'M 44 (Beehive Cluster)',
    'M45': 'M 45 (Pleiades Cluster)',
    'M46': 'M 46 (Open Cluster)',
    'M47': 'M 47 (Open Cluster)',
    'M48': 'M 48 (Open Cluster)',
    'M49': 'M 49 (Elliptical Galaxy)',
    'M50': 'M 50 (Open Cluster)',
    'M51': 'M 51 (Whirlpool Galaxy)',
    'M52': 'M 52 (Open Cluster)',
    'M53': 'M 53 (Globular Cluster)',
    'M54': 'M 54 (Globular Cluster)',
    'M55': 'M 55 (Globular Cluster)',
    'M56': 'M 56 (Globular Cluster)',
    'M57': 'M 57 (Ring Nebula)',
    'M58': 'M 58 (Spiral Galaxy)',
    'M59': 'M 59 (Elliptical Galaxy)',
    'M60': 'M 60 (Elliptical Galaxy)',
    'M61': 'M 61 (Spiral Galaxy)',
    'M62': 'M 62 (Globular Cluster)',
    'M63': 'M 63 (Sunflower Galaxy)',
    'M64': 'M 64 (Black Eye Galaxy)',
    'M65': 'M 65 (Spiral Galaxy)',
    'M66': 'M 66 (Spiral Galaxy)',
    'M67': 'M 67 (Open Cluster)',
    'M68': 'M 68 (Globular Cluster)',
    'M69': 'M 69 (Globular Cluster)',
    'M70': 'M 70 (Globular Cluster)',
    'M71': 'M 71 (Globular Cluster)',
    'M72': 'M 72 (Globular Cluster)',
    'M73': 'M 73 (Asterism)',
    'M74': 'M 74 (Spiral Galaxy)',
    'M75': 'M 75 (Globular Cluster)',
    'M76': 'M 76 (Little Dumbbell Nebula)',
    'M77': 'M 77 (Cetus Galaxy)',
    'M78': 'M 78 (Nebula)',
    'M79': 'M 79 (Globular Cluster)',
    'M80': 'M 80 (Globular Cluster)',
    'M81': 'M 81 (Bode Galaxy)',
    'M82': 'M 82 (Cigar Galaxy)',
    'M83': 'M 83 (Southern Pinwheel Galaxy)',
    'M84': 'M 84 (Elliptical Galaxy)',
    'M85': 'M 85 (Elliptical Galaxy)',
    'M86': 'M 86 (Elliptical Galaxy)',
    'M87': 'M 87 (Elliptical Galaxy)',
    'M88': 'M 88 (Spiral Galaxy)',
    'M89': 'M 89 (Elliptical Galaxy)',
    'M90': 'M 90 (Spiral Galaxy)',
    'M91': 'M 91 (Spiral Galaxy)',
    'M92': 'M 92 (Globular Cluster)',
    'M93': 'M 93 (Open Cluster)',
    'M94': 'M 94 (Spiral Galaxy)',
    'M95': 'M 95 (Spiral Galaxy)',
    'M96': 'M 96 (Spiral Galaxy)',
    'M97': 'M 97 (Owl Nebula)',
    'M98': 'M 98 (Spiral Galaxy)',
    'M99': 'M 99 (Spiral Galaxy)',
    'M100': 'M 100 (Spiral Galaxy)',
    'M101': 'M 101 (Pinwheel Galaxy)',
    'M102': 'M 102 (Spiral Galaxy)',
    'M103': 'M 103 (Open Cluster)',
    'M104': 'M 104 (Sombrero Galaxy)',
    'M105': 'M 105 (Elliptical Galaxy)',
    'M106': 'M 106 (Spiral Galaxy)',
    'M107': 'M 107 (Globular Cluster)',
    'M108': 'M 108 (Spiral Galaxy)',
    'M109': 'M 109 (Spiral Galaxy)',
    'M110': 'M 110 (Andromeda Satellite Galaxy)'
}

# Extended astronomical objects database with common names
EXTENDED_ASTRONOMICAL_DATABASE = {
    # IC Objects
    'IC 1396': 'IC 1396 (Elephant Trunk Nebula)',
    'IC 1805': 'IC 1805 (Heart Nebula)',
    'IC 405': 'IC 405 (Flaming Star Nebula)',
    'IC 410': 'IC 410 (Tadpole Nebula)',
    'IC 434': 'IC 434 (Horsehead Nebula)',
    'IC 5070': 'IC 5070 (Pelican Nebula)',
    'IC 5146': 'IC 5146 (Cocoon Nebula)',
    
    # NGC Objects
    'NGC 1499': 'NGC 1499 (California Nebula)',
    'NGC 1514': 'NGC 1514 (Crystal Ball Nebula)',
    'NGC 2174': 'NGC 2174 (Monkey Head Nebula)',
    'NGC 281': 'NGC 281 (Pacman Nebula)',
    'NGC 3628': 'NGC 3628 (Hamburger Galaxy)',
    'NGC 4565': 'NGC 4565 (Needle Galaxy)',
    'NGC 5128': 'NGC 5128 (Centaurus A)',
    'NGC 6543': 'NGC 6543 (Cat\'s Eye Nebula)',
    'NGC 6888': 'NGC 6888 (Crescent Nebula)',
    'NGC 6946': 'NGC 6946 (Fireworks Galaxy)',
    'NGC 6960': 'NGC 6960 (Western Veil Nebula)',
    'NGC 6979': 'NGC 6979 (Pickering\'s Triangle)',
    'NGC 7000': 'NGC 7000 (North America Nebula)',
    'NGC 7023': 'NGC 7023 (Iris Nebula)',
    'NGC 7331': 'NGC 7331 (Deer Lick Galaxy)',
    'NGC 7380': 'NGC 7380 (Wizard Nebula)',
    'NGC 7635': 'NGC 7635 (Bubble Nebula / Medusa Nebula)',
    'NGC 891': 'NGC 891 (Silver Sliver Galaxy)',
    
    # Sharpless Objects
    'Sh2-101': 'Sh2-101 (Tulip Nebula)'
}

# Arp catalog database with common names for peculiar galaxies
ARP_DATABASE = {
    # Famous Arp galaxies with common names
    'Arp 86': 'Arp 86 (The Mice)',
    'Arp 87': 'Arp 87 (Galaxy Pair)',
    'Arp 147': 'Arp 147 (Ring Galaxy)',
    'Arp 148': 'Arp 148 (Mayall\'s Object)',
    'Arp 188': 'Arp 188 (Tadpole Galaxy)',
    'Arp 244': 'Arp 244 (Antennae Galaxies)',
    'Arp 273': 'Arp 273 (Rose Galaxy)',
    'Arp 281': 'Arp 281 (Whale Galaxy)',
    'Arp 337': 'Arp 337 (Cigar Galaxy)'
}

# Telescope database with their characteristics
TELESCOPES_DATABASE = {
    'default': {
        'diameter_mm': 200.0,
        'focal_length_mm': 1600.0,
        'f_number': 8.0
    },
    'FSQ-85EDP': {
        'diameter_mm': 85.0,
        'focal_length_mm': 455.0,
        'f_number': 5.35
    },
    'FSQ85EDP': {
        'diameter_mm': 85.0,
        'focal_length_mm': 455.0,
        'f_number': 5.35
    },
    'FSQ-85': {
        'diameter_mm': 85.0,
        'focal_length_mm': 455.0,
        'f_number': 5.35
    },
    'FSQ85': {
        'diameter_mm': 85.0,
        'focal_length_mm': 455.0,
        'f_number': 5.35
    },
    
    # Takahashi Telescopes
    'FSQ-106': {'diameter_mm': 106.0, 'focal_length_mm': 530.0, 'f_number': 5.0},
    'FSQ106': {'diameter_mm': 106.0, 'focal_length_mm': 530.0, 'f_number': 5.0},
    'FSQ-130': {'diameter_mm': 130.0, 'focal_length_mm': 650.0, 'f_number': 5.0},
    'FSQ130': {'diameter_mm': 130.0, 'focal_length_mm': 650.0, 'f_number': 5.0},
    'TOA-130': {'diameter_mm': 130.0, 'focal_length_mm': 1000.0, 'f_number': 7.7},
    'TOA-150': {'diameter_mm': 150.0, 'focal_length_mm': 1100.0, 'f_number': 7.3},
    'TOA-160': {'diameter_mm': 160.0, 'focal_length_mm': 1200.0, 'f_number': 7.5},
    'TSA-102': {'diameter_mm': 102.0, 'focal_length_mm': 816.0, 'f_number': 8.0},
    'TSA-120': {'diameter_mm': 120.0, 'focal_length_mm': 900.0, 'f_number': 7.5},
    'Epsilon-130': {'diameter_mm': 130.0, 'focal_length_mm': 430.0, 'f_number': 3.3},
    'Epsilon-160': {'diameter_mm': 160.0, 'focal_length_mm': 530.0, 'f_number': 3.3},
    'Epsilon-180': {'diameter_mm': 180.0, 'focal_length_mm': 600.0, 'f_number': 3.3},
    
    # Celestron Telescopes
    'C8': {'diameter_mm': 203.2, 'focal_length_mm': 2032.0, 'f_number': 10.0},
    'C9': {'diameter_mm': 235.0, 'focal_length_mm': 2350.0, 'f_number': 10.0},
    'C9.25': {'diameter_mm': 235.0, 'focal_length_mm': 2350.0, 'f_number': 10.0},
    'C11': {'diameter_mm': 279.4, 'focal_length_mm': 2794.0, 'f_number': 10.0},
    'C14': {'diameter_mm': 355.6, 'focal_length_mm': 3911.6, 'f_number': 11.0},
    'EDGEHD8': {'diameter_mm': 203.2, 'focal_length_mm': 2032.0, 'f_number': 10.0},
    'EDGEHD9.25': {'diameter_mm': 235.0, 'focal_length_mm': 2350.0, 'f_number': 10.0},
    'EDGEHD11': {'diameter_mm': 279.4, 'focal_length_mm': 2794.0, 'f_number': 10.0},
    'EDGEHD14': {'diameter_mm': 355.6, 'focal_length_mm': 3911.6, 'f_number': 11.0},
    'RASA8': {'diameter_mm': 203.2, 'focal_length_mm': 400.0, 'f_number': 2.0},
    'RASA11': {'diameter_mm': 279.4, 'focal_length_mm': 620.0, 'f_number': 2.2},
    'RASA14': {'diameter_mm': 355.6, 'focal_length_mm': 650.0, 'f_number': 1.8},
    'STARIZON': {'diameter_mm': 130.0, 'focal_length_mm': 650.0, 'f_number': 5.0},
    'STARIZON-130': {'diameter_mm': 130.0, 'focal_length_mm': 650.0, 'f_number': 5.0},
    'STARIZON-150': {'diameter_mm': 150.0, 'focal_length_mm': 750.0, 'f_number': 5.0},
    'STARIZON-180': {'diameter_mm': 180.0, 'focal_length_mm': 900.0, 'f_number': 5.0},
    
    # PlaneWave Telescopes
    'CDK12': {'diameter_mm': 304.8, 'focal_length_mm': 2438.4, 'f_number': 8.0},
    'CDK14': {'diameter_mm': 355.6, 'focal_length_mm': 2844.8, 'f_number': 8.0},
    'CDK16': {'diameter_mm': 406.4, 'focal_length_mm': 3251.2, 'f_number': 8.0},
    'CDK17': {'diameter_mm': 431.8, 'focal_length_mm': 3454.4, 'f_number': 8.0},
    'CDK20': {'diameter_mm': 508.0, 'focal_length_mm': 4064.0, 'f_number': 8.0},
    'CDK24': {'diameter_mm': 609.6, 'focal_length_mm': 4876.8, 'f_number': 8.0},
    'L-350': {'diameter_mm': 350.0, 'focal_length_mm': 2450.0, 'f_number': 7.0},
    'L-500': {'diameter_mm': 500.0, 'focal_length_mm': 3500.0, 'f_number': 7.0},
    'L-600': {'diameter_mm': 600.0, 'focal_length_mm': 4200.0, 'f_number': 7.0},
    
    # CFF Telescopes (Classical Cassegrain)
    'CFF160': {'diameter_mm': 160.0, 'focal_length_mm': 1280.0, 'f_number': 8.0},
    'CFF185': {'diameter_mm': 185.0, 'focal_length_mm': 1480.0, 'f_number': 8.0},
    'CFF200': {'diameter_mm': 200.0, 'focal_length_mm': 1600.0, 'f_number': 8.0},
    'CFF250': {'diameter_mm': 250.0, 'focal_length_mm': 2000.0, 'f_number': 8.0},
    'CFF300': {'diameter_mm': 300.0, 'focal_length_mm': 2400.0, 'f_number': 8.0},
    'CFF350': {'diameter_mm': 350.0, 'focal_length_mm': 2800.0, 'f_number': 8.0},
    'CFF400': {'diameter_mm': 400.0, 'focal_length_mm': 3200.0, 'f_number': 8.0},
    'CFF500': {'diameter_mm': 500.0, 'focal_length_mm': 4000.0, 'f_number': 8.0},
    
    # TS-Optics Telescopes
    'TS-APO65Q': {'diameter_mm': 65.0, 'focal_length_mm': 420.0, 'f_number': 6.5},
    'TS-APO80Q': {'diameter_mm': 80.0, 'focal_length_mm': 480.0, 'f_number': 6.0},
    'TS-APO102Q': {'diameter_mm': 102.0, 'focal_length_mm': 714.0, 'f_number': 7.0},
    'TS-APO115Q': {'diameter_mm': 115.0, 'focal_length_mm': 805.0, 'f_number': 7.0},
    'TS-APO130Q': {'diameter_mm': 130.0, 'focal_length_mm': 910.0, 'f_number': 7.0},
    'TS-APO140Q': {'diameter_mm': 140.0, 'focal_length_mm': 980.0, 'f_number': 7.0},
    'TS-APO150Q': {'diameter_mm': 150.0, 'focal_length_mm': 1050.0, 'f_number': 7.0},
    'TS-APO160Q': {'diameter_mm': 160.0, 'focal_length_mm': 1120.0, 'f_number': 7.0},
    'TS-APO180Q': {'diameter_mm': 180.0, 'focal_length_mm': 1260.0, 'f_number': 7.0},
    'TS-APO200Q': {'diameter_mm': 200.0, 'focal_length_mm': 1400.0, 'f_number': 7.0},
    'TS-APO250Q': {'diameter_mm': 250.0, 'focal_length_mm': 1750.0, 'f_number': 7.0},
    'TS-APO300Q': {'diameter_mm': 300.0, 'focal_length_mm': 2100.0, 'f_number': 7.0},
    'TS-APO350Q': {'diameter_mm': 350.0, 'focal_length_mm': 2450.0, 'f_number': 7.0},
    'TS-APO400Q': {'diameter_mm': 400.0, 'focal_length_mm': 2800.0, 'f_number': 7.0},
    'TS-APO500Q': {'diameter_mm': 500.0, 'focal_length_mm': 3500.0, 'f_number': 7.0},
    'TS-APO600Q': {'diameter_mm': 600.0, 'focal_length_mm': 4200.0, 'f_number': 7.0},
    'TS-APO700Q': {'diameter_mm': 700.0, 'focal_length_mm': 4900.0, 'f_number': 7.0},
    'TS-APO800Q': {'diameter_mm': 800.0, 'focal_length_mm': 5600.0, 'f_number': 7.0},
    'TS-APO900Q': {'diameter_mm': 900.0, 'focal_length_mm': 6300.0, 'f_number': 7.0},
    'TS-APO1000Q': {'diameter_mm': 1000.0, 'focal_length_mm': 7000.0, 'f_number': 7.0},
    
    # Askar Telescopes
    'ASKAR-50PHQ': {'diameter_mm': 50.0, 'focal_length_mm': 250.0, 'f_number': 5.0},
    'ASKAR-60PHQ': {'diameter_mm': 60.0, 'focal_length_mm': 300.0, 'f_number': 5.0},
    'ASKAR-70PHQ': {'diameter_mm': 70.0, 'focal_length_mm': 350.0, 'f_number': 5.0},
    'ASKAR-80PHQ': {'diameter_mm': 80.0, 'focal_length_mm': 400.0, 'f_number': 5.0},
    'ASKAR-90PHQ': {'diameter_mm': 90.0, 'focal_length_mm': 450.0, 'f_number': 5.0},
    'ASKAR-100PHQ': {'diameter_mm': 100.0, 'focal_length_mm': 500.0, 'f_number': 5.0},
    'ASKAR-120PHQ': {'diameter_mm': 120.0, 'focal_length_mm': 600.0, 'f_number': 5.0},
    'ASKAR-130PHQ': {'diameter_mm': 130.0, 'focal_length_mm': 650.0, 'f_number': 5.0},
    'ASKAR-150PHQ': {'diameter_mm': 150.0, 'focal_length_mm': 750.0, 'f_number': 5.0},
    'ASKAR-180PHQ': {'diameter_mm': 180.0, 'focal_length_mm': 900.0, 'f_number': 5.0},
    'ASKAR-200PHQ': {'diameter_mm': 200.0, 'focal_length_mm': 1000.0, 'f_number': 5.0},
    'ASKAR-250PHQ': {'diameter_mm': 250.0, 'focal_length_mm': 1250.0, 'f_number': 5.0},
    'ASKAR-300PHQ': {'diameter_mm': 300.0, 'focal_length_mm': 1500.0, 'f_number': 5.0},
    'ASKAR-350PHQ': {'diameter_mm': 350.0, 'focal_length_mm': 1750.0, 'f_number': 5.0},
    'ASKAR-400PHQ': {'diameter_mm': 400.0, 'focal_length_mm': 2000.0, 'f_number': 5.0},
    'ASKAR-500PHQ': {'diameter_mm': 500.0, 'focal_length_mm': 2500.0, 'f_number': 5.0},
    'ASKAR-600PHQ': {'diameter_mm': 600.0, 'focal_length_mm': 3000.0, 'f_number': 5.0},
    'ASKAR-700PHQ': {'diameter_mm': 700.0, 'focal_length_mm': 3500.0, 'f_number': 5.0},
    
    # William Optics Telescopes
    'GT81': {'diameter_mm': 81.0, 'focal_length_mm': 478.0, 'f_number': 5.9},
    'GT102': {'diameter_mm': 102.0, 'focal_length_mm': 714.0, 'f_number': 7.0},
    'GT103': {'diameter_mm': 103.0, 'focal_length_mm': 618.0, 'f_number': 6.0},
    'GT105': {'diameter_mm': 105.0, 'focal_length_mm': 735.0, 'f_number': 7.0},
    'GT110': {'diameter_mm': 110.0, 'focal_length_mm': 770.0, 'f_number': 7.0},
    'GT120': {'diameter_mm': 120.0, 'focal_length_mm': 840.0, 'f_number': 7.0},
    'GT130': {'diameter_mm': 130.0, 'focal_length_mm': 910.0, 'f_number': 7.0},
    'GT150': {'diameter_mm': 150.0, 'focal_length_mm': 1050.0, 'f_number': 7.0},
    'GT180': {'diameter_mm': 180.0, 'focal_length_mm': 1260.0, 'f_number': 7.0},
    'GT200': {'diameter_mm': 200.0, 'focal_length_mm': 1400.0, 'f_number': 7.0},
    'GT250': {'diameter_mm': 250.0, 'focal_length_mm': 1750.0, 'f_number': 7.0},
    'GT300': {'diameter_mm': 300.0, 'focal_length_mm': 2100.0, 'f_number': 7.0},
    'GT350': {'diameter_mm': 350.0, 'focal_length_mm': 2450.0, 'f_number': 7.0},
    'GT400': {'diameter_mm': 400.0, 'focal_length_mm': 2800.0, 'f_number': 7.0},
    'GT500': {'diameter_mm': 500.0, 'focal_length_mm': 3500.0, 'f_number': 7.0},
    'GT600': {'diameter_mm': 600.0, 'focal_length_mm': 4200.0, 'f_number': 7.0},
    'GT700': {'diameter_mm': 700.0, 'focal_length_mm': 4900.0, 'f_number': 7.0},
    'GT800': {'diameter_mm': 800.0, 'focal_length_mm': 5600.0, 'f_number': 7.0},
    'GT900': {'diameter_mm': 900.0, 'focal_length_mm': 6300.0, 'f_number': 7.0},
    'GT1000': {'diameter_mm': 1000.0, 'focal_length_mm': 7000.0, 'f_number': 7.0},
    
    # Explore Scientific Telescopes
    'ES80': {'diameter_mm': 80.0, 'focal_length_mm': 480.0, 'f_number': 6.0},
    'ES102': {'diameter_mm': 102.0, 'focal_length_mm': 714.0, 'f_number': 7.0},
    'ES127': {'diameter_mm': 127.0, 'focal_length_mm': 952.0, 'f_number': 7.5},
    'ES152': {'diameter_mm': 152.0, 'focal_length_mm': 1209.0, 'f_number': 8.0},
    'ES203': {'diameter_mm': 203.0, 'focal_length_mm': 1624.0, 'f_number': 8.0},
    'ES254': {'diameter_mm': 254.0, 'focal_length_mm': 2032.0, 'f_number': 8.0},
    'ES305': {'diameter_mm': 305.0, 'focal_length_mm': 2438.0, 'f_number': 8.0},
    'ES356': {'diameter_mm': 356.0, 'focal_length_mm': 2848.0, 'f_number': 8.0},
    'ES406': {'diameter_mm': 406.0, 'focal_length_mm': 3248.0, 'f_number': 8.0},
    'ES457': {'diameter_mm': 457.0, 'focal_length_mm': 3656.0, 'f_number': 8.0},
    'ES508': {'diameter_mm': 508.0, 'focal_length_mm': 4064.0, 'f_number': 8.0},
    'ES610': {'diameter_mm': 610.0, 'focal_length_mm': 4880.0, 'f_number': 8.0},
    'ES711': {'diameter_mm': 711.0, 'focal_length_mm': 5688.0, 'f_number': 8.0},
    'ES813': {'diameter_mm': 813.0, 'focal_length_mm': 6504.0, 'f_number': 8.0},
    'ES914': {'diameter_mm': 914.0, 'focal_length_mm': 7312.0, 'f_number': 8.0},
    'ES1016': {'diameter_mm': 1016.0, 'focal_length_mm': 8128.0, 'f_number': 8.0},
    'ES1118': {'diameter_mm': 1118.0, 'focal_length_mm': 8944.0, 'f_number': 8.0},
    'ES1219': {'diameter_mm': 1219.0, 'focal_length_mm': 9752.0, 'f_number': 8.0},
    'ES1321': {'diameter_mm': 1321.0, 'focal_length_mm': 10568.0, 'f_number': 8.0},
    'ES1422': {'diameter_mm': 1422.0, 'focal_length_mm': 11376.0, 'f_number': 8.0},
    'ES1524': {'diameter_mm': 1524.0, 'focal_length_mm': 12192.0, 'f_number': 8.0},
    'ES1625': {'diameter_mm': 1625.0, 'focal_length_mm': 13000.0, 'f_number': 8.0},
    'ES1727': {'diameter_mm': 1727.0, 'focal_length_mm': 13816.0, 'f_number': 8.0},
    'ES1828': {'diameter_mm': 1828.0, 'focal_length_mm': 14624.0, 'f_number': 8.0},
    'ES1930': {'diameter_mm': 1930.0, 'focal_length_mm': 15440.0, 'f_number': 8.0},
    'ES2032': {'diameter_mm': 2032.0, 'focal_length_mm': 16256.0, 'f_number': 8.0},
    
    # Sky-Watcher Telescopes
    'EQ80': {'diameter_mm': 80.0, 'focal_length_mm': 400.0, 'f_number': 5.0},
    'EQ100': {'diameter_mm': 100.0, 'focal_length_mm': 500.0, 'f_number': 5.0},
    'EQ120': {'diameter_mm': 120.0, 'focal_length_mm': 600.0, 'f_number': 5.0},
    'EQ150': {'diameter_mm': 150.0, 'focal_length_mm': 750.0, 'f_number': 5.0},
    'EQ200': {'diameter_mm': 200.0, 'focal_length_mm': 1000.0, 'f_number': 5.0},
    'EQ250': {'diameter_mm': 250.0, 'focal_length_mm': 1250.0, 'f_number': 5.0},
    'EQ300': {'diameter_mm': 300.0, 'focal_length_mm': 1500.0, 'f_number': 5.0},
    'EQ350': {'diameter_mm': 350.0, 'focal_length_mm': 1750.0, 'f_number': 5.0},
    'EQ400': {'diameter_mm': 400.0, 'focal_length_mm': 2000.0, 'f_number': 5.0},
    'EQ500': {'diameter_mm': 500.0, 'focal_length_mm': 2500.0, 'f_number': 5.0},
    'EQ600': {'diameter_mm': 600.0, 'focal_length_mm': 3000.0, 'f_number': 5.0},
    'EQ700': {'diameter_mm': 700.0, 'focal_length_mm': 3500.0, 'f_number': 5.0},
    'EQ800': {'diameter_mm': 800.0, 'focal_length_mm': 4000.0, 'f_number': 5.0},
    'EQ900': {'diameter_mm': 900.0, 'focal_length_mm': 4500.0, 'f_number': 5.0},
    'EQ1000': {'diameter_mm': 1000.0, 'focal_length_mm': 5000.0, 'f_number': 5.0},
    
    # Sky-Watcher Esprit Series
    'ESPRIT80': {'diameter_mm': 80.0, 'focal_length_mm': 400.0, 'f_number': 5.0},
    'ESPRIT100': {'diameter_mm': 100.0, 'focal_length_mm': 550.0, 'f_number': 5.5},
    'ESPRIT120': {'diameter_mm': 120.0, 'focal_length_mm': 840.0, 'f_number': 7.0},
    'ESPRIT150': {'diameter_mm': 150.0, 'focal_length_mm': 1050.0, 'f_number': 7.0},
    'ESPRIT200': {'diameter_mm': 200.0, 'focal_length_mm': 1400.0, 'f_number': 7.0},
    'ESPRIT250': {'diameter_mm': 250.0, 'focal_length_mm': 1750.0, 'f_number': 7.0},
    'ESPRIT300': {'diameter_mm': 300.0, 'focal_length_mm': 2100.0, 'f_number': 7.0},
    'ESPRIT350': {'diameter_mm': 350.0, 'focal_length_mm': 2450.0, 'f_number': 7.0},
    'ESPRIT400': {'diameter_mm': 400.0, 'focal_length_mm': 2800.0, 'f_number': 7.0},
    'ESPRIT500': {'diameter_mm': 500.0, 'focal_length_mm': 3500.0, 'f_number': 7.0},
    'ESPRIT600': {'diameter_mm': 600.0, 'focal_length_mm': 4200.0, 'f_number': 7.0},
    'ESPRIT700': {'diameter_mm': 700.0, 'focal_length_mm': 4900.0, 'f_number': 7.0},
    'ESPRIT800': {'diameter_mm': 800.0, 'focal_length_mm': 5600.0, 'f_number': 7.0},
    'ESPRIT900': {'diameter_mm': 900.0, 'focal_length_mm': 6300.0, 'f_number': 7.0},
    'ESPRIT1000': {'diameter_mm': 1000.0, 'focal_length_mm': 7000.0, 'f_number': 7.0},
    
    # Sky-Watcher Quattro Series
    'QUATTRO80': {'diameter_mm': 80.0, 'focal_length_mm': 400.0, 'f_number': 5.0},
    'QUATTRO100': {'diameter_mm': 100.0, 'focal_length_mm': 500.0, 'f_number': 5.0},
    'QUATTRO120': {'diameter_mm': 120.0, 'focal_length_mm': 600.0, 'f_number': 5.0},
    'QUATTRO150': {'diameter_mm': 150.0, 'focal_length_mm': 750.0, 'f_number': 5.0},
    'QUATTRO200': {'diameter_mm': 200.0, 'focal_length_mm': 1000.0, 'f_number': 5.0},
    'QUATTRO250': {'diameter_mm': 250.0, 'focal_length_mm': 1250.0, 'f_number': 5.0},
    'QUATTRO300': {'diameter_mm': 300.0, 'focal_length_mm': 1500.0, 'f_number': 5.0},
    'QUATTRO350': {'diameter_mm': 350.0, 'focal_length_mm': 1750.0, 'f_number': 5.0},
    'QUATTRO400': {'diameter_mm': 400.0, 'focal_length_mm': 2000.0, 'f_number': 5.0},
    'QUATTRO500': {'diameter_mm': 500.0, 'focal_length_mm': 2500.0, 'f_number': 5.0},
    'QUATTRO600': {'diameter_mm': 600.0, 'focal_length_mm': 3000.0, 'f_number': 5.0},
    'QUATTRO700': {'diameter_mm': 700.0, 'focal_length_mm': 3500.0, 'f_number': 5.0},
    'QUATTRO800': {'diameter_mm': 800.0, 'focal_length_mm': 4000.0, 'f_number': 5.0},
    'QUATTRO900': {'diameter_mm': 900.0, 'focal_length_mm': 4500.0, 'f_number': 5.0},
    'QUATTRO1000': {'diameter_mm': 1000.0, 'focal_length_mm': 5000.0, 'f_number': 5.0},
    
    # Sky-Watcher Newton Series
    'NEWTON80': {'diameter_mm': 80.0, 'focal_length_mm': 400.0, 'f_number': 5.0},
    'NEWTON100': {'diameter_mm': 100.0, 'focal_length_mm': 500.0, 'f_number': 5.0},
    'NEWTON120': {'diameter_mm': 120.0, 'focal_length_mm': 600.0, 'f_number': 5.0},
    'NEWTON150': {'diameter_mm': 150.0, 'focal_length_mm': 750.0, 'f_number': 5.0},
    'NEWTON200': {'diameter_mm': 200.0, 'focal_length_mm': 1000.0, 'f_number': 5.0},
    'NEWTON250': {'diameter_mm': 250.0, 'focal_length_mm': 1250.0, 'f_number': 5.0},
    'NEWTON300': {'diameter_mm': 300.0, 'focal_length_mm': 1500.0, 'f_number': 5.0},
    'NEWTON350': {'diameter_mm': 350.0, 'focal_length_mm': 1750.0, 'f_number': 5.0},
    'NEWTON400': {'diameter_mm': 400.0, 'focal_length_mm': 2000.0, 'f_number': 5.0},
    'NEWTON500': {'diameter_mm': 500.0, 'focal_length_mm': 2500.0, 'f_number': 5.0},
    'NEWTON600': {'diameter_mm': 600.0, 'focal_length_mm': 3000.0, 'f_number': 5.0},
    'NEWTON700': {'diameter_mm': 700.0, 'focal_length_mm': 3500.0, 'f_number': 5.0},
    'NEWTON800': {'diameter_mm': 800.0, 'focal_length_mm': 4000.0, 'f_number': 5.0},
    'NEWTON900': {'diameter_mm': 900.0, 'focal_length_mm': 4500.0, 'f_number': 5.0},
    'NEWTON1000': {'diameter_mm': 1000.0, 'focal_length_mm': 5000.0, 'f_number': 5.0},
    
    # ZWO Telescopes
    'ZWO80': {'diameter_mm': 80.0, 'focal_length_mm': 400.0, 'f_number': 5.0},
    'ZWO100': {'diameter_mm': 100.0, 'focal_length_mm': 500.0, 'f_number': 5.0},
    'ZWO120': {'diameter_mm': 120.0, 'focal_length_mm': 600.0, 'f_number': 5.0},
    'ZWO150': {'diameter_mm': 150.0, 'focal_length_mm': 750.0, 'f_number': 5.0},
    'ZWO200': {'diameter_mm': 200.0, 'focal_length_mm': 1000.0, 'f_number': 5.0},
    'ZWO250': {'diameter_mm': 250.0, 'focal_length_mm': 1250.0, 'f_number': 5.0},
    'ZWO300': {'diameter_mm': 300.0, 'focal_length_mm': 1500.0, 'f_number': 5.0},
    'ZWO350': {'diameter_mm': 350.0, 'focal_length_mm': 1750.0, 'f_number': 5.0},
    'ZWO400': {'diameter_mm': 400.0, 'focal_length_mm': 2000.0, 'f_number': 5.0},
    'ZWO500': {'diameter_mm': 500.0, 'focal_length_mm': 2500.0, 'f_number': 5.0},
    'ZWO600': {'diameter_mm': 600.0, 'focal_length_mm': 3000.0, 'f_number': 5.0},
    'ZWO700': {'diameter_mm': 700.0, 'focal_length_mm': 3500.0, 'f_number': 5.0},
    'ZWO800': {'diameter_mm': 800.0, 'focal_length_mm': 4000.0, 'f_number': 5.0},
    'ZWO900': {'diameter_mm': 900.0, 'focal_length_mm': 4500.0, 'f_number': 5.0},
    'ZWO1000': {'diameter_mm': 1000.0, 'focal_length_mm': 5000.0, 'f_number': 5.0},
    
    # ZWO APO Series
    'ZWO-APO80': {'diameter_mm': 80.0, 'focal_length_mm': 480.0, 'f_number': 6.0},
    'ZWO-APO100': {'diameter_mm': 100.0, 'focal_length_mm': 600.0, 'f_number': 6.0},
    'ZWO-APO120': {'diameter_mm': 120.0, 'focal_length_mm': 720.0, 'f_number': 6.0},
    'ZWO-APO150': {'diameter_mm': 150.0, 'focal_length_mm': 900.0, 'f_number': 6.0},
    'ZWO-APO200': {'diameter_mm': 200.0, 'focal_length_mm': 1200.0, 'f_number': 6.0},
    'ZWO-APO250': {'diameter_mm': 250.0, 'focal_length_mm': 1500.0, 'f_number': 6.0},
    'ZWO-APO300': {'diameter_mm': 300.0, 'focal_length_mm': 1800.0, 'f_number': 6.0},
    'ZWO-APO350': {'diameter_mm': 350.0, 'focal_length_mm': 2100.0, 'f_number': 6.0},
    'ZWO-APO400': {'diameter_mm': 400.0, 'focal_length_mm': 2400.0, 'f_number': 6.0},
    'ZWO-APO500': {'diameter_mm': 500.0, 'focal_length_mm': 3000.0, 'f_number': 6.0},
    'ZWO-APO600': {'diameter_mm': 600.0, 'focal_length_mm': 3600.0, 'f_number': 6.0},
    'ZWO-APO700': {'diameter_mm': 700.0, 'focal_length_mm': 4200.0, 'f_number': 6.0},
    'ZWO-APO800': {'diameter_mm': 800.0, 'focal_length_mm': 4800.0, 'f_number': 6.0},
    'ZWO-APO900': {'diameter_mm': 900.0, 'focal_length_mm': 5400.0, 'f_number': 6.0},
    'ZWO-APO1000': {'diameter_mm': 1000.0, 'focal_length_mm': 6000.0, 'f_number': 6.0},
    
    # ASA Telescopes (Astro Systeme Austria)
    'ASA-80': {'diameter_mm': 80.0, 'focal_length_mm': 480.0, 'f_number': 6.0},
    'ASA-100': {'diameter_mm': 100.0, 'focal_length_mm': 600.0, 'f_number': 6.0},
    'ASA-120': {'diameter_mm': 120.0, 'focal_length_mm': 720.0, 'f_number': 6.0},
    'ASA-150': {'diameter_mm': 150.0, 'focal_length_mm': 900.0, 'f_number': 6.0},
    'ASA-200': {'diameter_mm': 200.0, 'focal_length_mm': 1200.0, 'f_number': 6.0},
    'ASA-250': {'diameter_mm': 250.0, 'focal_length_mm': 1500.0, 'f_number': 6.0},
    'ASA-300': {'diameter_mm': 300.0, 'focal_length_mm': 1800.0, 'f_number': 6.0},
    'ASA-350': {'diameter_mm': 350.0, 'focal_length_mm': 2100.0, 'f_number': 6.0},
    'ASA-400': {'diameter_mm': 400.0, 'focal_length_mm': 2400.0, 'f_number': 6.0},
    'ASA-500': {'diameter_mm': 500.0, 'focal_length_mm': 3000.0, 'f_number': 6.0},
    'ASA-600': {'diameter_mm': 600.0, 'focal_length_mm': 3600.0, 'f_number': 6.0},
    'ASA-700': {'diameter_mm': 700.0, 'focal_length_mm': 4200.0, 'f_number': 6.0},
    'ASA-800': {'diameter_mm': 800.0, 'focal_length_mm': 4800.0, 'f_number': 6.0},
    'ASA-900': {'diameter_mm': 900.0, 'focal_length_mm': 5400.0, 'f_number': 6.0},
    'ASA-1000': {'diameter_mm': 1000.0, 'focal_length_mm': 6000.0, 'f_number': 6.0},
    
    # ASA Newton Series
    'ASA-NEWTON80': {'diameter_mm': 80.0, 'focal_length_mm': 400.0, 'f_number': 5.0},
    'ASA-NEWTON100': {'diameter_mm': 100.0, 'focal_length_mm': 500.0, 'f_number': 5.0},
    'ASA-NEWTON120': {'diameter_mm': 120.0, 'focal_length_mm': 600.0, 'f_number': 5.0},
    'ASA-NEWTON150': {'diameter_mm': 150.0, 'focal_length_mm': 750.0, 'f_number': 5.0},
    'ASA-NEWTON200': {'diameter_mm': 200.0, 'focal_length_mm': 1000.0, 'f_number': 5.0},
    'ASA-NEWTON250': {'diameter_mm': 250.0, 'focal_length_mm': 1250.0, 'f_number': 5.0},
    'ASA-NEWTON300': {'diameter_mm': 300.0, 'focal_length_mm': 1500.0, 'f_number': 5.0},
    'ASA-NEWTON350': {'diameter_mm': 350.0, 'focal_length_mm': 1750.0, 'f_number': 5.0},
    'ASA-NEWTON400': {'diameter_mm': 400.0, 'focal_length_mm': 2000.0, 'f_number': 5.0},
    'ASA-NEWTON500': {'diameter_mm': 500.0, 'focal_length_mm': 2500.0, 'f_number': 5.0},
    'ASA-NEWTON600': {'diameter_mm': 600.0, 'focal_length_mm': 3000.0, 'f_number': 5.0},
    'ASA-NEWTON700': {'diameter_mm': 700.0, 'focal_length_mm': 3500.0, 'f_number': 5.0},
    'ASA-NEWTON800': {'diameter_mm': 800.0, 'focal_length_mm': 4000.0, 'f_number': 5.0},
    'ASA-NEWTON900': {'diameter_mm': 900.0, 'focal_length_mm': 4500.0, 'f_number': 5.0},
    'ASA-NEWTON1000': {'diameter_mm': 1000.0, 'focal_length_mm': 5000.0, 'f_number': 5.0},
    
    # ASA APO Series
    'ASA-APO80': {'diameter_mm': 80.0, 'focal_length_mm': 480.0, 'f_number': 6.0},
    'ASA-APO100': {'diameter_mm': 100.0, 'focal_length_mm': 600.0, 'f_number': 6.0},
    'ASA-APO120': {'diameter_mm': 120.0, 'focal_length_mm': 720.0, 'f_number': 6.0},
    'ASA-APO150': {'diameter_mm': 150.0, 'focal_length_mm': 900.0, 'f_number': 6.0},
    'ASA-APO200': {'diameter_mm': 200.0, 'focal_length_mm': 1200.0, 'f_number': 6.0},
    'ASA-APO250': {'diameter_mm': 250.0, 'focal_length_mm': 1500.0, 'f_number': 6.0},
    'ASA-APO300': {'diameter_mm': 300.0, 'focal_length_mm': 1800.0, 'f_number': 6.0},
    'ASA-APO350': {'diameter_mm': 350.0, 'focal_length_mm': 2100.0, 'f_number': 6.0},
    'ASA-APO400': {'diameter_mm': 400.0, 'focal_length_mm': 2400.0, 'f_number': 6.0},
    'ASA-APO500': {'diameter_mm': 500.0, 'focal_length_mm': 3000.0, 'f_number': 6.0},
    'ASA-APO600': {'diameter_mm': 600.0, 'focal_length_mm': 3600.0, 'f_number': 6.0},
    'ASA-APO700': {'diameter_mm': 700.0, 'focal_length_mm': 4200.0, 'f_number': 6.0},
    'ASA-APO800': {'diameter_mm': 800.0, 'focal_length_mm': 4800.0, 'f_number': 6.0},
    'ASA-APO900': {'diameter_mm': 900.0, 'focal_length_mm': 5400.0, 'f_number': 6.0},
    'ASA-APO1000': {'diameter_mm': 1000.0, 'focal_length_mm': 6000.0, 'f_number': 6.0},
    
    # ASA Ritchey-Chrétien Series
    'ASA-RC80': {'diameter_mm': 80.0, 'focal_length_mm': 640.0, 'f_number': 8.0},
    'ASA-RC100': {'diameter_mm': 100.0, 'focal_length_mm': 800.0, 'f_number': 8.0},
    'ASA-RC120': {'diameter_mm': 120.0, 'focal_length_mm': 960.0, 'f_number': 8.0},
    'ASA-RC150': {'diameter_mm': 150.0, 'focal_length_mm': 1200.0, 'f_number': 8.0},
    'ASA-RC200': {'diameter_mm': 200.0, 'focal_length_mm': 1600.0, 'f_number': 8.0},
    'ASA-RC250': {'diameter_mm': 250.0, 'focal_length_mm': 2000.0, 'f_number': 8.0},
    'ASA-RC300': {'diameter_mm': 300.0, 'focal_length_mm': 2400.0, 'f_number': 8.0},
    'ASA-RC350': {'diameter_mm': 350.0, 'focal_length_mm': 2800.0, 'f_number': 8.0},
    'ASA-RC400': {'diameter_mm': 400.0, 'focal_length_mm': 3200.0, 'f_number': 8.0},
    'ASA-RC500': {'diameter_mm': 500.0, 'focal_length_mm': 4000.0, 'f_number': 8.0},
    'ASA-RC600': {'diameter_mm': 600.0, 'focal_length_mm': 4800.0, 'f_number': 8.0},
    'ASA-RC700': {'diameter_mm': 700.0, 'focal_length_mm': 5600.0, 'f_number': 8.0},
    'ASA-RC800': {'diameter_mm': 800.0, 'focal_length_mm': 6400.0, 'f_number': 8.0},
    'ASA-RC900': {'diameter_mm': 900.0, 'focal_length_mm': 7200.0, 'f_number': 8.0},
    'ASA-RC1000': {'diameter_mm': 1000.0, 'focal_length_mm': 8000.0, 'f_number': 8.0},
    
    # ASA Cassegrain Series
    'ASA-CASSEGRAIN80': {'diameter_mm': 80.0, 'focal_length_mm': 640.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN100': {'diameter_mm': 100.0, 'focal_length_mm': 800.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN120': {'diameter_mm': 120.0, 'focal_length_mm': 960.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN150': {'diameter_mm': 150.0, 'focal_length_mm': 1200.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN200': {'diameter_mm': 200.0, 'focal_length_mm': 1600.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN250': {'diameter_mm': 250.0, 'focal_length_mm': 2000.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN300': {'diameter_mm': 300.0, 'focal_length_mm': 2400.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN350': {'diameter_mm': 350.0, 'focal_length_mm': 2800.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN400': {'diameter_mm': 400.0, 'focal_length_mm': 3200.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN500': {'diameter_mm': 500.0, 'focal_length_mm': 4000.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN600': {'diameter_mm': 600.0, 'focal_length_mm': 4800.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN700': {'diameter_mm': 700.0, 'focal_length_mm': 5600.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN800': {'diameter_mm': 800.0, 'focal_length_mm': 6400.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN900': {'diameter_mm': 900.0, 'focal_length_mm': 7200.0, 'f_number': 8.0},
    'ASA-CASSEGRAIN1000': {'diameter_mm': 1000.0, 'focal_length_mm': 8000.0, 'f_number': 8.0},
    
    # Omegon Telescopes
    'OMEGON-80': {'diameter_mm': 80.0, 'focal_length_mm': 400.0, 'f_number': 5.0},
    'OMEGON-100': {'diameter_mm': 100.0, 'focal_length_mm': 500.0, 'f_number': 5.0},
    'OMEGON-120': {'diameter_mm': 120.0, 'focal_length_mm': 600.0, 'f_number': 5.0},
    'OMEGON-150': {'diameter_mm': 150.0, 'focal_length_mm': 750.0, 'f_number': 5.0},
    'OMEGON-200': {'diameter_mm': 200.0, 'focal_length_mm': 1000.0, 'f_number': 5.0},
    'OMEGON-250': {'diameter_mm': 250.0, 'focal_length_mm': 1250.0, 'f_number': 5.0},
    'OMEGON-300': {'diameter_mm': 300.0, 'focal_length_mm': 1500.0, 'f_number': 5.0},
    'OMEGON-350': {'diameter_mm': 350.0, 'focal_length_mm': 1750.0, 'f_number': 5.0},
    'OMEGON-400': {'diameter_mm': 400.0, 'focal_length_mm': 2000.0, 'f_number': 5.0},
    'OMEGON-500': {'diameter_mm': 500.0, 'focal_length_mm': 2500.0, 'f_number': 5.0},
    'OMEGON-600': {'diameter_mm': 600.0, 'focal_length_mm': 3000.0, 'f_number': 5.0},
    'OMEGON-700': {'diameter_mm': 700.0, 'focal_length_mm': 3500.0, 'f_number': 5.0},
    'OMEGON-800': {'diameter_mm': 800.0, 'focal_length_mm': 4000.0, 'f_number': 5.0},
    'OMEGON-900': {'diameter_mm': 900.0, 'focal_length_mm': 4500.0, 'f_number': 5.0},
    'OMEGON-1000': {'diameter_mm': 1000.0, 'focal_length_mm': 5000.0, 'f_number': 5.0},
    
    # Omegon APO Series
    'OMEGON-APO80': {'diameter_mm': 80.0, 'focal_length_mm': 480.0, 'f_number': 6.0},
    'OMEGON-APO100': {'diameter_mm': 100.0, 'focal_length_mm': 600.0, 'f_number': 6.0},
    'OMEGON-APO120': {'diameter_mm': 120.0, 'focal_length_mm': 720.0, 'f_number': 6.0},
    'OMEGON-APO150': {'diameter_mm': 150.0, 'focal_length_mm': 900.0, 'f_number': 6.0},
    'OMEGON-APO200': {'diameter_mm': 200.0, 'focal_length_mm': 1200.0, 'f_number': 6.0},
    'OMEGON-APO250': {'diameter_mm': 250.0, 'focal_length_mm': 1500.0, 'f_number': 6.0},
    'OMEGON-APO300': {'diameter_mm': 300.0, 'focal_length_mm': 1800.0, 'f_number': 6.0},
    'OMEGON-APO350': {'diameter_mm': 350.0, 'focal_length_mm': 2100.0, 'f_number': 6.0},
    'OMEGON-APO400': {'diameter_mm': 400.0, 'focal_length_mm': 2400.0, 'f_number': 6.0},
    'OMEGON-APO500': {'diameter_mm': 500.0, 'focal_length_mm': 3000.0, 'f_number': 6.0},
    'OMEGON-APO600': {'diameter_mm': 600.0, 'focal_length_mm': 3600.0, 'f_number': 6.0},
    'OMEGON-APO700': {'diameter_mm': 700.0, 'focal_length_mm': 4200.0, 'f_number': 6.0},
    'OMEGON-APO800': {'diameter_mm': 800.0, 'focal_length_mm': 4800.0, 'f_number': 6.0},
    'OMEGON-APO900': {'diameter_mm': 900.0, 'focal_length_mm': 5400.0, 'f_number': 6.0},
    'OMEGON-APO1000': {'diameter_mm': 1000.0, 'focal_length_mm': 6000.0, 'f_number': 6.0},
    
    # Omegon Newton Series
    'OMEGON-NEWTON80': {'diameter_mm': 80.0, 'focal_length_mm': 400.0, 'f_number': 5.0},
    'OMEGON-NEWTON100': {'diameter_mm': 100.0, 'focal_length_mm': 500.0, 'f_number': 5.0},
    'OMEGON-NEWTON120': {'diameter_mm': 120.0, 'focal_length_mm': 600.0, 'f_number': 5.0},
    'OMEGON-NEWTON150': {'diameter_mm': 150.0, 'focal_length_mm': 750.0, 'f_number': 5.0},
    'OMEGON-NEWTON200': {'diameter_mm': 200.0, 'focal_length_mm': 1000.0, 'f_number': 5.0},
    'OMEGON-NEWTON250': {'diameter_mm': 250.0, 'focal_length_mm': 1250.0, 'f_number': 5.0},
    'OMEGON-NEWTON300': {'diameter_mm': 300.0, 'focal_length_mm': 1500.0, 'f_number': 5.0},
    'OMEGON-NEWTON350': {'diameter_mm': 350.0, 'focal_length_mm': 1750.0, 'f_number': 5.0},
    'OMEGON-NEWTON400': {'diameter_mm': 400.0, 'focal_length_mm': 2000.0, 'f_number': 5.0},
    'OMEGON-NEWTON500': {'diameter_mm': 500.0, 'focal_length_mm': 2500.0, 'f_number': 5.0},
    'OMEGON-NEWTON600': {'diameter_mm': 600.0, 'focal_length_mm': 3000.0, 'f_number': 5.0},
    'OMEGON-NEWTON700': {'diameter_mm': 700.0, 'focal_length_mm': 3500.0, 'f_number': 5.0},
    'OMEGON-NEWTON800': {'diameter_mm': 800.0, 'focal_length_mm': 4000.0, 'f_number': 5.0},
    'OMEGON-NEWTON900': {'diameter_mm': 900.0, 'focal_length_mm': 4500.0, 'f_number': 5.0},
    'OMEGON-NEWTON1000': {'diameter_mm': 1000.0, 'focal_length_mm': 5000.0, 'f_number': 5.0},
    
    # Omegon Ritchey-Chrétien Series
    'OMEGON-RC80': {'diameter_mm': 80.0, 'focal_length_mm': 640.0, 'f_number': 8.0},
    'OMEGON-RC100': {'diameter_mm': 100.0, 'focal_length_mm': 800.0, 'f_number': 8.0},
    'OMEGON-RC120': {'diameter_mm': 120.0, 'focal_length_mm': 960.0, 'f_number': 8.0},
    'OMEGON-RC150': {'diameter_mm': 150.0, 'focal_length_mm': 1200.0, 'f_number': 8.0},
    'OMEGON-RC200': {'diameter_mm': 200.0, 'focal_length_mm': 1600.0, 'f_number': 8.0},
    'OMEGON-RC250': {'diameter_mm': 250.0, 'focal_length_mm': 2000.0, 'f_number': 8.0},
    'OMEGON-RC300': {'diameter_mm': 300.0, 'focal_length_mm': 2400.0, 'f_number': 8.0},
    'OMEGON-RC350': {'diameter_mm': 350.0, 'focal_length_mm': 2800.0, 'f_number': 8.0},
    'OMEGON-RC400': {'diameter_mm': 400.0, 'focal_length_mm': 3200.0, 'f_number': 8.0},
    'OMEGON-RC500': {'diameter_mm': 500.0, 'focal_length_mm': 4000.0, 'f_number': 8.0},
    'OMEGON-RC600': {'diameter_mm': 600.0, 'focal_length_mm': 4800.0, 'f_number': 8.0},
    'OMEGON-RC700': {'diameter_mm': 700.0, 'focal_length_mm': 5600.0, 'f_number': 8.0},
    'OMEGON-RC800': {'diameter_mm': 800.0, 'focal_length_mm': 6400.0, 'f_number': 8.0},
    'OMEGON-RC900': {'diameter_mm': 900.0, 'focal_length_mm': 7200.0, 'f_number': 8.0},
    'OMEGON-RC1000': {'diameter_mm': 1000.0, 'focal_length_mm': 8000.0, 'f_number': 8.0},
    
    # Omegon Cassegrain Series
    'OMEGON-CASSEGRAIN80': {'diameter_mm': 80.0, 'focal_length_mm': 640.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN100': {'diameter_mm': 100.0, 'focal_length_mm': 800.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN120': {'diameter_mm': 120.0, 'focal_length_mm': 960.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN150': {'diameter_mm': 150.0, 'focal_length_mm': 1200.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN200': {'diameter_mm': 200.0, 'focal_length_mm': 1600.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN250': {'diameter_mm': 250.0, 'focal_length_mm': 2000.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN300': {'diameter_mm': 300.0, 'focal_length_mm': 2400.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN350': {'diameter_mm': 350.0, 'focal_length_mm': 2800.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN400': {'diameter_mm': 400.0, 'focal_length_mm': 3200.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN500': {'diameter_mm': 500.0, 'focal_length_mm': 4000.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN600': {'diameter_mm': 600.0, 'focal_length_mm': 4800.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN700': {'diameter_mm': 700.0, 'focal_length_mm': 5600.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN800': {'diameter_mm': 800.0, 'focal_length_mm': 6400.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN900': {'diameter_mm': 900.0, 'focal_length_mm': 7200.0, 'f_number': 8.0},
    'OMEGON-CASSEGRAIN1000': {'diameter_mm': 1000.0, 'focal_length_mm': 8000.0, 'f_number': 8.0},
    
    # Telescope Live Telescopes (from https://app.telescope.live/en/telescopes)
    # Chile Telescopes
    'CHI-1': {'diameter_mm': 610.0, 'focal_length_mm': 3965.0, 'f_number': 6.5},
    'CHI-1-CMOS': {'diameter_mm': 610.0, 'focal_length_mm': 3965.0, 'f_number': 6.5},
    'CHI-1-CCD': {'diameter_mm': 610.0, 'focal_length_mm': 3965.0, 'f_number': 6.5},
    'CHI-2-CCD': {'diameter_mm': 500.0, 'focal_length_mm': 1900.0, 'f_number': 3.8},
    'CHI-3-CCD': {'diameter_mm': 1000.0, 'focal_length_mm': 6800.0, 'f_number': 6.8},
    'CHI-4-CCD': {'diameter_mm': 500.0, 'focal_length_mm': 1900.0, 'f_number': 3.8},
    'CHI-5-CCD': {'diameter_mm': 100.0, 'focal_length_mm': 200.0, 'f_number': 2.0},
    'CHI-6-CCD': {'diameter_mm': 200.0, 'focal_length_mm': 600.0, 'f_number': 3.0},
    
    # Spain Telescopes
    'SPA-1': {'diameter_mm': 106.0, 'focal_length_mm': 381.6, 'f_number': 3.6},
    'SPA-1-CCD': {'diameter_mm': 106.0, 'focal_length_mm': 381.6, 'f_number': 3.6},
    'SPA-1-CMOS': {'diameter_mm': 106.0, 'focal_length_mm': 381.6, 'f_number': 3.6},
    'SPA-2': {'diameter_mm': 710.0, 'focal_length_mm': 5680.0, 'f_number': 8.0},
    'SPA-2-CCD': {'diameter_mm': 710.0, 'focal_length_mm': 5680.0, 'f_number': 8.0},
    'SPA-2-CMOS': {'diameter_mm': 710.0, 'focal_length_mm': 5680.0, 'f_number': 8.0},
    'SPA-3': {'diameter_mm': 106.0, 'focal_length_mm': 382.0, 'f_number': 3.6},
    'SPA-3-CCD': {'diameter_mm': 106.0, 'focal_length_mm': 382.0, 'f_number': 3.6},
    'SPA-3-CMOS': {'diameter_mm': 106.0, 'focal_length_mm': 382.0, 'f_number': 3.6},
    
    # Australia Telescopes
    'AUS-2': {'diameter_mm': 106.0, 'focal_length_mm': 381.6, 'f_number': 3.6},
    'AUS-2-CCD': {'diameter_mm': 106.0, 'focal_length_mm': 381.6, 'f_number': 3.6},
    'AUS-2-CMOS': {'diameter_mm': 106.0, 'focal_length_mm': 381.6, 'f_number': 3.6}
}

def format_time(seconds):
    """Converts seconds to hours:minutes:seconds format"""
    time = timedelta(seconds=int(seconds))
    return str(time)

def format_time_hours_minutes(seconds):
    """Converts seconds to hours:minutes format without days"""
    total_hours = int(seconds // 3600)
    total_minutes = int((seconds % 3600) // 60)
    return f"{total_hours}:{total_minutes:02d}"

def format_time_with_details(seconds):
    """Converts seconds to hours:minutes format with full duration in parentheses"""
    time = timedelta(seconds=int(seconds))
    
    # Calculate hours and minutes
    total_hours = int(seconds // 3600)
    total_minutes = int((seconds % 3600) // 60)
    
    # Format as HhMM
    time_str = f"{total_hours}h{total_minutes:02d}"
    
    # Add full duration in parentheses
    full_duration = str(time)
    return f"{time_str} ({full_duration})"

def ensure_catalog_uppercase(text):
    """Ensures catalog names are uppercase in object names"""
    if not text:
        return ""
    
    import re
    
    # Common astronomical catalogs to convert to uppercase
    catalogs = ['ngc', 'messier', 'm ', 'ic ', 'ugc', 'hd', 'hip', 'tyc', 'gsc', 'usno', '2mass', 'pgc', 'arp', 'c ']
    
    result = text
    
    # Convert catalog names to uppercase
    for catalog in catalogs:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(catalog) + r'\b'
        result = re.sub(pattern, catalog.upper(), result, flags=re.IGNORECASE)
    
    return result

def get_astronomical_sort_key(target_name):
    """Create a sort key that handles astronomical object names properly for alphabetical sorting"""
    import re
    import unicodedata
    
    # Extract main name by removing parenthetical information
    # Example: "Andromeda Galaxy (M 31)" -> "Andromeda Galaxy"
    main_name = re.sub(r'\s*\([^)]*\)', '', target_name).strip()
    
    # Normalize unicode characters (remove accents, etc.)
    normalized = unicodedata.normalize('NFD', main_name.lower())
    
    # Remove accents but keep special characters that are important for astronomical names
    clean_name = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    
    # For astronomical objects, we want to sort by the main name first
    # Handle common prefixes like "NGC", "M", "IC", etc.
    parts = clean_name.split()
    
    # If it starts with a catalog prefix, sort by catalog number
    if len(parts) > 0:
        first_part = parts[0]
        
        # Handle special case where catalog name contains a dash (e.g., "sh2-101")
        if '-' in first_part:
            # Split on dash and check if first part is a catalog prefix
            dash_parts = first_part.split('-')
            if len(dash_parts) >= 2 and dash_parts[0] in ['ngc', 'm', 'ic', 'ugc', 'arp', 'messier', 'abell', 'sh2', 'rcw', 'pgc', 'cl', 'hd', 'barnard', 'c', 'gc']:
                catalog_prefix = dash_parts[0]
                # Extract number from the part after the dash
                number_match = re.search(r'(\d+)', dash_parts[1])
                if number_match:
                    number = int(number_match.group(1))
                    return (catalog_prefix, number, clean_name)
        
        if first_part in ['ngc', 'm', 'ic', 'ugc', 'arp', 'messier', 'abell', 'sh2', 'rcw', 'pgc', 'cl', 'hd', 'barnard', 'c', 'gc']:
            # Extract number for proper numerical sorting
            number_match = re.search(r'(\d+)', first_part + ' '.join(parts[1:]))
            if number_match:
                number = int(number_match.group(1))
                # Return a tuple for consistent sorting
                return (first_part, number, clean_name)
    
    # For common names, sort alphabetically by the main name
    # Return a tuple for consistent sorting (catalog type, 0, name)
    return ('other', 0, clean_name)

def escape_latex(text):
    """Escapes special characters for LaTeX"""
    if not text:
        return ""
    
    # First ensure catalog names are uppercase
    text = ensure_catalog_uppercase(text)
    
    # Replace special characters
    replacements = {
        '\\': '\\textbackslash{}',
        '{': '\\{',
        '}': '\\}',
        '$': '\\$',
        '&': '\\&',
        '#': '\\#',
        '^': '\\textasciicircum{}',
        '_': '\\_',
        '~': '\\textasciitilde{}',
        '%': '\\%',
        '<': '\\textless{}',
        '>': '\\textgreater{}',
        '|': '\\textbar{}'
    }
    
    result = text
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    
    return result

def convert_filter_name_to_greek_matplotlib(filter_name):
    """Convert filter names to Greek characters for matplotlib display"""
    if not filter_name:
        return filter_name
    
    # Mapping of filter codes to Greek matplotlib representations
    greek_mapping = {
        'HA': 'Hα',
        'Ha': 'Hα',
        'H-ALPHA': 'Hα',
        'HALPHA': 'Hα',
        'SII': 'S II',
        'OIII': 'O III'
    }
    
    return greek_mapping.get(filter_name, filter_name)

def convert_filter_name_to_greek_latex(filter_name):
    """Convert filter names to Greek characters for LaTeX display"""
    if not filter_name:
        return filter_name
    
    # Mapping of filter codes to Greek LaTeX representations
    greek_mapping = {
        'HA': 'H$\\alpha$',
        'Ha': 'H$\\alpha$',
        'H-ALPHA': 'H$\\alpha$',
        'HALPHA': 'H$\\alpha$',
        'HBETA': 'H$\\beta$',
        'HB': 'H$\\beta$',
        'H-BETA': 'H$\\beta$',
        'HGAMMA': 'H$\\gamma$',
        'HG': 'H$\\gamma$',
        'H-GAMMA': 'H$\\gamma$',
        'HDELTA': 'H$\\delta$',
        'HD': 'H$\\delta$',
        'H-DELTA': 'H$\\delta$',
        'HEPSILON': 'H$\\epsilon$',
        'HE': 'H$\\epsilon$',
        'H-EPSILON': 'H$\\epsilon$',
        'HZETA': 'H$\\zeta$',
        'HZ': 'H$\\zeta$',
        'H-ZETA': 'H$\\zeta$',
        'HETA': 'H$\\eta$',
        'HET': 'H$\\eta$',
        'H-ETA': 'H$\\eta$',
        'HTHETA': 'H$\\theta$',
        'HTH': 'H$\\theta$',
        'H-THETA': 'H$\\theta$',
        'HIOTA': 'H$\\iota$',
        'HI': 'H$\\iota$',
        'H-IOTA': 'H$\\iota$',
        'HKAPPA': 'H$\\kappa$',
        'HK': 'H$\\kappa$',
        'H-KAPPA': 'H$\\kappa$',
        'HLAMBDA': 'H$\\lambda$',
        'HL': 'H$\\lambda$',
        'H-LAMBDA': 'H$\\lambda$',
        'HMU': 'H$\\mu$',
        'HM': 'H$\\mu$',
        'H-MU': 'H$\\mu$',
        'HNU': 'H$\\nu$',
        'HN': 'H$\\nu$',
        'H-NU': 'H$\\nu$',
        'HXI': 'H$\\xi$',
        'HX': 'H$\\xi$',
        'H-XI': 'H$\\xi$',
        'HOMICRON': 'H$\\omicron$',
        'HO': 'H$\\omicron$',
        'H-OMICRON': 'H$\\omicron$',
        'HPI': 'H$\\pi$',
        'HP': 'H$\\pi$',
        'H-PI': 'H$\\pi$',
        'HRHO': 'H$\\rho$',
        'HR': 'H$\\rho$',
        'H-RHO': 'H$\\rho$',
        'HSIGMA': 'H$\\sigma$',
        'HS': 'H$\\sigma$',
        'H-SIGMA': 'H$\\sigma$',
        'HTAU': 'H$\\tau$',
        'HT': 'H$\\tau$',
        'H-TAU': 'H$\\tau$',
        'HUPSILON': 'H$\\upsilon$',
        'HU': 'H$\\upsilon$',
        'H-UPSILON': 'H$\\upsilon$',
        'HPHI': 'H$\\phi$',
        'HPH': 'H$\\phi$',
        'H-PHI': 'H$\\phi$',
        'HCHI': 'H$\\chi$',
        'HCH': 'H$\\chi$',
        'H-CHI': 'H$\\chi$',
        'HPSI': 'H$\\psi$',
        'HPS': 'H$\\psi$',
        'H-PSI': 'H$\\psi$',
        'HOMEGA': 'H$\\omega$',
        'HOM': 'H$\\omega$',
        'H-OMEGA': 'H$\\omega$',
    }
    
    # Check if the filter name matches any of our Greek mappings
    if filter_name in greek_mapping:
        return greek_mapping[filter_name]
    
    # If no Greek mapping found, return the original name (escaped for LaTeX)
    return escape_latex(filter_name)

def convert_astronomical_object_to_full_name(target_name):
    """Convert astronomical objects to full names with common names"""
    if not target_name:
        return target_name
    
    import re
    
    # Check if the target name already contains a common name (already processed by normalize_target_name)
    # This prevents duplication for objects like "Crab Nebula (M 1)"
    if '(' in target_name and ')' in target_name:
        # Check if it's already a formatted name like "Crab Nebula (M 1)" or "Andromeda Galaxy (M 31)"
        if re.search(r'[A-Za-z\s]+\([A-Z]+\s*\d+\)', target_name):
            return target_name  # Already formatted, don't process further
    
    # Additional check: if the name contains common astronomical terms, it's likely already processed
    common_terms = ['Nebula', 'Galaxy', 'Cluster', 'Star', 'Cloud', 'Object', 'Heart', 'Elephant', 'Flaming', 'Tadpole', 'Horsehead', 'Pelican', 'Cocoon', 'California', 'Crystal', 'Monkey', 'Pacman', 'Hamburger', 'Needle', 'Cat\'s Eye', 'Crescent', 'Fireworks', 'Western Veil', 'Pickering\'s Triangle', 'North America', 'Iris', 'Deer Lick', 'Wizard', 'Bubble', 'Silver Sliver', 'Tulip']
    if any(term in target_name for term in common_terms) and '(' in target_name:
        return target_name  # Already processed, don't modify further
    
    # First check Messier objects
    messier_match = re.search(r'\bM\s*(\d{1,3})\b', target_name, re.IGNORECASE)
    if messier_match:
        messier_num = messier_match.group(1)
        messier_key = f'M{messier_num}'
        
        if messier_key in MESSIER_DATABASE:
            full_name = MESSIER_DATABASE[messier_key]
            original_text = target_name.replace(messier_match.group(0), '').strip()
            if original_text:
                return f"{full_name} {original_text}"
            else:
                return full_name
    
    # Check extended astronomical database
    # Try exact match first
    if target_name in EXTENDED_ASTRONOMICAL_DATABASE:
        return EXTENDED_ASTRONOMICAL_DATABASE[target_name]
    
    # Try to match with various formats
    # NGC objects
    ngc_match = re.search(r'\bNGC\s*(\d+)\b', target_name, re.IGNORECASE)
    if ngc_match:
        ngc_num = ngc_match.group(1)
        ngc_key = f'NGC {ngc_num}'
        if ngc_key in EXTENDED_ASTRONOMICAL_DATABASE:
            full_name = EXTENDED_ASTRONOMICAL_DATABASE[ngc_key]
            original_text = target_name.replace(ngc_match.group(0), '').strip()
            if original_text:
                return f"{full_name} {original_text}"
            else:
                return full_name
    
    # IC objects
    ic_match = re.search(r'\bIC\s*(\d+)\b', target_name, re.IGNORECASE)
    if ic_match:
        ic_num = ic_match.group(1)
        ic_key = f'IC {ic_num}'
        if ic_key in EXTENDED_ASTRONOMICAL_DATABASE:
            full_name = EXTENDED_ASTRONOMICAL_DATABASE[ic_key]
            original_text = target_name.replace(ic_match.group(0), '').strip()
            if original_text:
                return f"{full_name} {original_text}"
            else:
                return full_name
    
    # Sh2 objects
    sh2_match = re.search(r'\bSh2[-\s]*(\d+)\b', target_name, re.IGNORECASE)
    if sh2_match:
        sh2_num = sh2_match.group(1)
        sh2_key = f'Sh2-{sh2_num}'
        if sh2_key in EXTENDED_ASTRONOMICAL_DATABASE:
            full_name = EXTENDED_ASTRONOMICAL_DATABASE[sh2_key]
            original_text = target_name.replace(sh2_match.group(0), '').strip()
            if original_text:
                return f"{full_name} {original_text}"
            else:
                return full_name
    
    # LBN objects
    lbn_match = re.search(r'\bLBN\s*(\d+)\b', target_name, re.IGNORECASE)
    if lbn_match:
        lbn_num = lbn_match.group(1)
        lbn_key = f'LBN {lbn_num}'
        if lbn_key in EXTENDED_ASTRONOMICAL_DATABASE:
            full_name = EXTENDED_ASTRONOMICAL_DATABASE[lbn_key]
            original_text = target_name.replace(lbn_match.group(0), '').strip()
            if original_text:
                return f"{full_name} {original_text}"
            else:
                return full_name
    
    # PGC objects
    pgc_match = re.search(r'\bPGC\s*(\d+)\b', target_name, re.IGNORECASE)
    if pgc_match:
        pgc_num = pgc_match.group(1)
        pgc_key = f'PGC {pgc_num}'
        if pgc_key in EXTENDED_ASTRONOMICAL_DATABASE:
            full_name = EXTENDED_ASTRONOMICAL_DATABASE[pgc_key]
            original_text = target_name.replace(pgc_match.group(0), '').strip()
            if original_text:
                return f"{full_name} {original_text}"
            else:
                return full_name
    
    # Arp objects
    arp_match = re.search(r'\bArp\s*(\d+)\b', target_name, re.IGNORECASE)
    if arp_match:
        arp_num = arp_match.group(1)
        arp_key = f'Arp {arp_num}'
        if arp_key in ARP_DATABASE:
            full_name = ARP_DATABASE[arp_key]
            original_text = target_name.replace(arp_match.group(0), '').strip()
            if original_text:
                return f"{full_name} {original_text}"
            else:
                return full_name
    
    return target_name

def convert_messier_to_full_name(target_name):
    """Convert Messier numbers to full names with common names (legacy function)"""
    return convert_astronomical_object_to_full_name(target_name)

def normalize_sharpless_name(target_name):
    """Normalize Sharpless catalog names to proper format"""
    if not target_name:
        return target_name
    
    import re
    
    # Pattern to match various Sharpless formats:
    # - SH2, Sh2, sh2 (with or without space before number)
    # - SH 2, Sh 2, sh 2 (with space between SH and number)
    # This pattern captures the catalog prefix and number, and any following text
    sharpless_pattern = r'\b(SH|Sh|sh)\s*2\s*(\d+)([A-Za-z]*)'
    match = re.search(sharpless_pattern, target_name, re.IGNORECASE)
    
    if match:
        # Replace with proper format: Sh2-XXX (always use "Sh2" format)
        number = match.group(2)
        following_text = match.group(3)
        
        # Get any additional text after the match
        remaining_text = target_name[match.end():].strip()
        
        # Build the result with correct capitalization
        result = f"Sh2-{number}"
        
        # Add following text (like "Squib") with proper spacing
        if following_text:
            result += f" {following_text}"
        
        # Add any remaining text, but clean up isolated parentheses
        if remaining_text:
            # Remove isolated closing parentheses like " )" at the end
            remaining_text = re.sub(r'\s*\)\s*$', '', remaining_text)
            # Remove isolated opening parentheses like "( " at the beginning
            remaining_text = re.sub(r'^\s*\(\s*', '', remaining_text)
            # Remove any remaining isolated parentheses
            remaining_text = re.sub(r'\s*\(\s*\)\s*', '', remaining_text)
            
            if remaining_text.strip():
                result += f" {remaining_text.strip()}"
        
        return result
    
    return target_name

def format_target_name_for_latex(target_name):
    """Format target names for LaTeX display with proper capitalization"""
    if not target_name:
        return target_name
    
    # First, normalize Sharpless catalog names
    target_name = normalize_sharpless_name(target_name)
    
    # Then, normalize the target name (this handles multi-catalog objects and common names)
    target_name = normalize_target_name(target_name)
    
    # Finally, convert astronomical objects to full names (only for objects not handled by normalize_target_name)
    target_name = convert_astronomical_object_to_full_name(target_name)
    
    # Special handling for LMC to ensure it's always displayed as LMC in uppercase
    if 'Large Magellanic Cloud' in target_name:
        # Replace any lowercase lmc with uppercase LMC in the display
        formatted_name = target_name.replace('(lmc)', '(LMC)')
        formatted_name = formatted_name.replace('(Lmc)', '(LMC)')
        formatted_name = formatted_name.replace('(lMc)', '(LMC)')
        formatted_name = formatted_name.replace('(LmC)', '(LMC)')
        return escape_latex(formatted_name)
    
    # Special handling for Wolf-Rayet stars to ensure they're always displayed as WR in uppercase
    # This handles cases where "Wr" might appear instead of "WR"
    import re
    # Replace "Wr " with "WR " (Wolf-Rayet stars)
    formatted_name = re.sub(r'\bWr\s+(\d+)', r'WR \1', target_name)
    # Also handle cases without space: "Wr134" -> "WR 134"
    formatted_name = re.sub(r'\bWr(\d+)', r'WR \1', formatted_name)
    
    # For other targets, just escape for LaTeX
    return escape_latex(formatted_name)

def detect_mosaic_panel(target_name):
    """Detects if a target name is a mosaic panel and extracts the base object name"""
    if not target_name:
        return None, None
    
    import re
    
    # Patterns for mosaic panels in various languages and formats
    mosaic_patterns = [
        # French patterns
        r'^(.+?)[-_]panneau[-_](\d+)$',
        r'^(.+?)\s+panneau\s+(\d+)$',
        r'^(.+?)\s+panneau\s+panneau\s+(\d+)$',
        # English patterns  
        r'^(.+?)[-_]panel[-_](\d+)$',
        r'^(.+?)\s+panel\s+(\d+)$',
        r'^(.+?)\s+panel\s+panel\s+(\d+)$',
        # Generic patterns
        r'^(.+?)[-_]part[-_](\d+)$',
        r'^(.+?)\s+part\s+(\d+)$',
        r'^(.+?)[-_]section[-_](\d+)$',
        r'^(.+?)\s+section\s+(\d+)$',
        r'^(.+?)[-_]tile[-_](\d+)$',
        r'^(.+?)\s+tile\s+(\d+)$'
    ]
    
    target_clean = target_name.strip()
    
    for pattern in mosaic_patterns:
        match = re.match(pattern, target_clean, re.IGNORECASE)
        if match:
            base_object = match.group(1).strip()
            panel_number = match.group(2)
            return base_object, panel_number
    
    return None, None

def get_mosaic_name(base_object):
    """Gets the appropriate mosaic name for a base object"""
    if not base_object:
        return None
    
    # Normalize the base object name
    base_normalized = normalize_target_name(base_object)
    
    # If it's a known object, use the proper name with "Mosaic" suffix
    if base_normalized and base_normalized != base_object:
        # Extract the catalog part if present
        if '(' in base_normalized and ')' in base_normalized:
            # Keep the catalog designation
            return f"{base_normalized.replace(')', ' Mosaic)')}"
        else:
            return f"{base_normalized} Mosaic"
    else:
        # For unknown objects, just add "Mosaic" to the base name
        return f"{base_object} Mosaic"

def group_normalized_targets(data_by_target):
    """Groups targets that have the same normalized name (e.g., LMC and lmc)"""
    normalized_groups = {}
    
    for target_name, target_data in data_by_target.items():
        # Get the normalized name for this target
        normalized_name = normalize_target_name(target_name)
        
        if normalized_name not in normalized_groups:
            # First occurrence of this normalized name
            normalized_groups[normalized_name] = target_data.copy()
            # Keep track of original names for reference
            normalized_groups[normalized_name]['original_names'] = [target_name]
        else:
            # Merge with existing group
            existing_data = normalized_groups[normalized_name]
            
            # Merge files
            existing_data['files'].extend(target_data['files'])
            
            # Merge time_by_filter
            for filter_name, time_list in target_data['time_by_filter'].items():
                if filter_name in existing_data['time_by_filter']:
                    existing_data['time_by_filter'][filter_name].extend(time_list)
                else:
                    existing_data['time_by_filter'][filter_name] = time_list.copy()
            
            # Merge telescopes and instruments
            if isinstance(existing_data['telescopes'], set):
                existing_data['telescopes'].update(target_data['telescopes'])
            else:
                existing_data['telescopes'] = list(set(existing_data['telescopes'] + target_data['telescopes']))
            
            if isinstance(existing_data['instruments'], set):
                existing_data['instruments'].update(target_data['instruments'])
            else:
                existing_data['instruments'] = list(set(existing_data['instruments'] + target_data['instruments']))
            
            # Merge dates
            if isinstance(existing_data['dates'], set):
                existing_data['dates'].update(target_data['dates'])
            else:
                existing_data['dates'] = list(set(existing_data['dates'] + target_data['dates']))
            
            # Merge files_by_date
            if 'files_by_date' in target_data:
                if 'files_by_date' not in existing_data:
                    existing_data['files_by_date'] = {}
                
                for date, date_data in target_data['files_by_date'].items():
                    if date in existing_data['files_by_date']:
                        # Merge existing date data
                        existing_date_data = existing_data['files_by_date'][date]
                        existing_date_data['files'].extend(date_data['files'])
                        existing_date_data['total_time'] += date_data['total_time']
                        
                        # Merge time_by_filter for this date
                        for filter_name, time_list in date_data['time_by_filter'].items():
                            if filter_name in existing_date_data['time_by_filter']:
                                existing_date_data['time_by_filter'][filter_name].extend(time_list)
                            else:
                                existing_date_data['time_by_filter'][filter_name] = time_list.copy()
                        
                        # Merge exposure_details
                        for filter_name, exp_details in date_data['exposure_details'].items():
                            if filter_name in existing_date_data['exposure_details']:
                                for exp_time, count in exp_details.items():
                                    if exp_time in existing_date_data['exposure_details'][filter_name]:
                                        existing_date_data['exposure_details'][filter_name][exp_time] += count
                                    else:
                                        existing_date_data['exposure_details'][filter_name][exp_time] = count
                            else:
                                existing_date_data['exposure_details'][filter_name] = exp_details.copy()
                    else:
                        # New date, just copy the data
                        existing_data['files_by_date'][date] = date_data.copy()
            
            # Merge other data structures
            if 'received_light' in target_data:
                if 'received_light' not in existing_data:
                    existing_data['received_light'] = {}
                for filter_name, light_list in target_data['received_light'].items():
                    if filter_name in existing_data['received_light']:
                        existing_data['received_light'][filter_name].extend(light_list)
                    else:
                        existing_data['received_light'][filter_name] = light_list.copy()
            
            if 'adu_samples' in target_data:
                if 'adu_samples' not in existing_data:
                    existing_data['adu_samples'] = {}
                for filter_name, samples in target_data['adu_samples'].items():
                    if filter_name in existing_data['adu_samples']:
                        existing_data['adu_samples'][filter_name].extend(samples)
                    else:
                        existing_data['adu_samples'][filter_name] = samples.copy()
            
            if 'adu_counter_by_filter' in target_data:
                if 'adu_counter_by_filter' not in existing_data:
                    existing_data['adu_counter_by_filter'] = {}
                for filter_name, count in target_data['adu_counter_by_filter'].items():
                    if filter_name in existing_data['adu_counter_by_filter']:
                        existing_data['adu_counter_by_filter'][filter_name] += count
                    else:
                        existing_data['adu_counter_by_filter'][filter_name] = count
            
            # Add to original names
            existing_data['original_names'].append(target_name)
    
    # Convert sets to lists for JSON serialization
    for target_data in normalized_groups.values():
        if isinstance(target_data['telescopes'], set):
            target_data['telescopes'] = list(target_data['telescopes'])
        if isinstance(target_data['instruments'], set):
            target_data['instruments'] = list(target_data['instruments'])
        if isinstance(target_data['dates'], set):
            target_data['dates'] = list(target_data['dates'])
    
    return normalized_groups

def group_mosaic_panels(data_by_target):
    """Groups mosaic panels under a unified mosaic name"""
    mosaic_groups = {}
    non_mosaic_targets = {}
    
    for target_name, target_data in data_by_target.items():
        base_object, panel_number = detect_mosaic_panel(target_name)
        
        if base_object and panel_number:
            # This is a mosaic panel
            mosaic_name = get_mosaic_name(base_object)
            
            if mosaic_name not in mosaic_groups:
                mosaic_groups[mosaic_name] = {
                    'files': [],
                    'telescopes': set(),
                    'instruments': set(),
                    'panels': {},
                    'total_time': 0
                }
            
            # Add panel information
            # Calculate total time from time_by_filter if available, otherwise from files
            panel_total_time = 0
            if 'time_by_filter' in target_data:
                for time_list in target_data['time_by_filter'].values():
                    panel_total_time += sum(time_list)
            else:
                # Fallback: try to calculate from files if they have exposure time info
                for file_info in target_data['files']:
                    if isinstance(file_info, dict) and 'info' in file_info:
                        info = file_info['info']
                        if 'exposure_time' in info:
                            panel_total_time += info['exposure_time']
            
            mosaic_groups[mosaic_name]['panels'][panel_number] = {
                'original_name': target_name,
                'files': target_data['files'],
                'total_time': panel_total_time
            }
            
            # Merge data
            mosaic_groups[mosaic_name]['files'].extend(target_data['files'])
            mosaic_groups[mosaic_name]['telescopes'].update(target_data['telescopes'])
            mosaic_groups[mosaic_name]['instruments'].update(target_data['instruments'])
            mosaic_groups[mosaic_name]['total_time'] += panel_total_time
            
        else:
            # This is not a mosaic panel
            non_mosaic_targets[target_name] = target_data
    
    # Convert sets to lists for consistency
    for mosaic_data in mosaic_groups.values():
        mosaic_data['telescopes'] = list(mosaic_data['telescopes'])
        mosaic_data['instruments'] = list(mosaic_data['instruments'])
    
    # Combine mosaic groups and non-mosaic targets
    result = {}
    result.update(non_mosaic_targets)
    
    for mosaic_name, mosaic_data in mosaic_groups.items():
        # Create a complete data structure for the mosaic
        mosaic_result = {
            'files': mosaic_data['files'],
            'telescopes': mosaic_data['telescopes'],
            'instruments': mosaic_data['instruments'],
            'panels': mosaic_data['panels'],
            'time_by_filter': defaultdict(list),
            'received_light': defaultdict(list),
            'adu_samples': defaultdict(list),
            'dates': [],
            'apertures': [],
            'diameters': [],
            'focal_lengths': [],
            'coordinates': []
        }
        
        # Merge time_by_filter from all panels
        for panel_data in mosaic_data['panels'].values():
            for file_info in panel_data['files']:
                if isinstance(file_info, dict) and 'info' in file_info:
                    info = file_info['info']
                    if 'filter' in info and 'exposure_time' in info:
                        mosaic_result['time_by_filter'][info['filter']].append(info['exposure_time'])
                    if 'date_obs' in info:
                        mosaic_result['dates'].append(info['date_obs'])
                    if 'f_number' in info:
                        mosaic_result['apertures'].append(info['f_number'])
                    if 'diameter_mm' in info:
                        mosaic_result['diameters'].append(info['diameter_mm'])
                    if 'focal_length_mm' in info:
                        mosaic_result['focal_lengths'].append(info['focal_length_mm'])
                    if 'ra' in info and 'dec' in info and info['ra'] and info['dec']:
                        mosaic_result['coordinates'].append((info['ra'], info['dec']))
        
        result[mosaic_name] = mosaic_result
    
    return result

def extract_date_from_file(file_info):
    """Extracts date from file information"""
    import re
    from datetime import datetime
    
    # Try to extract date from filename first
    if 'filename' in file_info:
        filename = file_info['filename']
        # Look for date patterns like 2025-08-05, 2025/08/05, 20250805
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # 2025-08-05
            r'(\d{4}/\d{2}/\d{2})',  # 2025/08/05
            r'(\d{4})(\d{2})(\d{2})',  # 20250805
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                if len(match.groups()) == 1:
                    # Format like 2025-08-05 or 2025/08/05
                    return match.group(1).replace('/', '-')
                else:
                    # Format like 20250805
                    year, month, day = match.groups()
                    return f"{year}-{month}-{day}"
    
    # Try to extract date from FITS header info
    if 'info' in file_info and 'DATE-OBS' in file_info['info']:
        date_obs = file_info['info']['DATE-OBS']
        # Extract just the date part (YYYY-MM-DD)
        if isinstance(date_obs, str):
            date_part = date_obs.split('T')[0]  # Remove time part
            return date_part
    
    # Fallback: use current date
    return datetime.now().strftime('%Y-%m-%d')

def group_files_by_date(data):
    """Groups files by date for night-by-night analysis"""
    files_by_date = {}
    
    # Check if we have files_by_date structure (new approach)
    if 'files_by_date' in data:
        # Use the new grouped data structure
        for date_str, date_data in data['files_by_date'].items():
            # Count files by summing the counts from time_by_filter
            total_files = sum(len(time_list) for time_list in date_data['time_by_filter'].values())
            
            files_by_date[date_str] = {
                'files': [],  # Don't store actual files to avoid duplication
                'total_files': total_files,  # Use calculated count
                'total_time': date_data['total_time'],
                'filters': {},
                'exposure_details': date_data.get('exposure_details', {})  # Include exposure details
            }
            
            # Populate filters from time_by_filter for this date
            for filter_name, time_list in date_data['time_by_filter'].items():
                files_by_date[date_str]['filters'][filter_name] = {
                    'time': sum(time_list),
                    'count': len(time_list)
                }
                
    else:
        # Fallback to old approach (single session)
        date_str = "All Observations"
        total_files = sum(len(time_list) for time_list in data['time_by_filter'].values())
        
        files_by_date[date_str] = {
            'files': [],  # Don't store actual files to avoid duplication
            'total_files': total_files,  # Use calculated count
            'total_time': sum(sum(times) for times in data['time_by_filter'].values()),
            'filters': {}
        }
        
        # Populate filters from time_by_filter data
        for filter_name, time_list in data['time_by_filter'].items():
            files_by_date[date_str]['filters'][filter_name] = {
                'time': sum(time_list),
                'count': len(time_list)
            }
    
    return files_by_date

def extract_observation_date(file_path, additional_info):
    """Extracts observation date from FITS file"""
    import re
    from datetime import datetime, timedelta
    
    # Try to extract date and time from filename first
    filename = file_path.name
    date_time_patterns = [
        r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})',  # 2025-08-05_20-30-15
        r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',  # 20250805_203015
    ]
    
    for pattern in date_time_patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            if len(groups) == 6:
                year, month, day, hour, minute, second = groups
                obs_datetime = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                night_date = get_astronomical_night_date(obs_datetime)
                
                
                return night_date
    
    # Try to extract just date from filename
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # 2025-08-05
        r'(\d{4}/\d{2}/\d{2})',  # 2025/08/05
        r'(\d{4})(\d{2})(\d{2})',  # 20250805
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            if len(match.groups()) == 1:
                date_str = match.group(1).replace('/', '-')
            else:
                year, month, day = match.groups()
                date_str = f"{year}-{month}-{day}"
            
            # Assume evening observation (20:00) if no time specified
            obs_datetime = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=20, minute=0, second=0)
            return get_astronomical_night_date(obs_datetime)
    
    # Try to extract date and time from FITS header
    if 'DATE-OBS' in additional_info:
        date_obs = additional_info['DATE-OBS']
        if isinstance(date_obs, str):
            try:
                # Parse ISO format: 2025-08-05T20:30:15.123
                if 'T' in date_obs:
                    date_part, time_part = date_obs.split('T')
                    obs_datetime = datetime.fromisoformat(date_obs.replace('Z', ''))
                    return get_astronomical_night_date(obs_datetime)
                else:
                    # Just date
                    obs_datetime = datetime.strptime(date_obs, '%Y-%m-%d').replace(hour=20, minute=0, second=0)
                    return get_astronomical_night_date(obs_datetime)
            except:
                pass
    
    # Fallback to current date
    return get_astronomical_night_date(datetime.now())

def get_astronomical_night_date(obs_datetime):
    """Determines the astronomical night date based on observation time.
    Astronomical nights are separated at noon (12:00) instead of midnight.
    A night from Aug 1 evening to Aug 2 morning is considered 'Aug 1 night'."""
    
    # Debug: Print observation datetime and resulting night
    night_date = None
    
    # If observation is before noon (12:00), it belongs to the previous night
    if obs_datetime.hour < 12:
        # This is morning observation, belongs to previous night
        previous_day = obs_datetime - timedelta(days=1)
        night_date = f"{previous_day.strftime('%Y-%m-%d')} night"
    else:
        # This is evening observation, belongs to current night
        night_date = f"{obs_datetime.strftime('%Y-%m-%d')} night"
    
    
    return night_date

def format_night_display(date_str):
    """Converts date string to readable night format.
    Example: '2025-04-27 night' -> 'Night 27th to 28th April 2025'"""
    from datetime import datetime
    
    # Remove 'night' suffix and parse date
    date_part = date_str.replace(' night', '')
    night_date = datetime.strptime(date_part, '%Y-%m-%d')
    
    # Calculate next day for the "to" part
    next_day = night_date + timedelta(days=1)
    
    # Format with ordinal day (using LaTeX superscript)
    def get_ordinal(day):
        if 10 <= day % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        return f"{day}$^{{{suffix}}}$"
    
    # Get month name
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    return f"Night {get_ordinal(night_date.day)} to {get_ordinal(next_day.day)} {month_names[night_date.month-1]} {night_date.year}"

def extract_date_from_filename(filename):
    """Extracts date from filename"""
    import re
    
    # Look for date patterns in filename
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # 2025-08-05
        r'(\d{4}/\d{2}/\d{2})',  # 2025/08/05
        r'(\d{4})(\d{2})(\d{2})',  # 20250805
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            if len(match.groups()) == 1:
                return match.group(1).replace('/', '-')
            else:
                year, month, day = match.groups()
                return f"{year}-{month}-{day}"
    
    # Fallback to current date
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d')

def escape_filename_for_latex(filename):
    """Escapes specifically filenames for LaTeX"""
    if not filename:
        return ""
    
    # Replace underscores and other problematic characters
    result = filename.replace('_', '\\_')
    result = result.replace('&', '\\&')
    result = result.replace('#', '\\#')
    result = result.replace('$', '\\$')
    result = result.replace('%', '\\%')
    result = result.replace('^', '\\textasciicircum{}')
    result = result.replace('~', '\\textasciitilde{}')
    result = result.replace('{', '\\{')
    result = result.replace('}', '\\}')
    
    return result

def load_configuration():
    """Load configuration from JSON file"""
    global BIAS_DARK_PATH, SENSORS_DATABASE, TELESCOPES_DATABASE
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Remove personal path fields if present
                if 'chemin_bias_dark' in config:
                    pass  # Ignore deprecated personal path
                
                # Load sensor database if available
                if 'sensors_database' in config:
                    # Merge with existing database (keep default values)
                    for sensor_name, characteristics in config['sensors_database'].items():
                        if sensor_name not in SENSORS_DATABASE or sensor_name != 'default':
                            SENSORS_DATABASE[sensor_name] = characteristics
                    print(f"Sensor database loaded ({len(config['sensors_database'])} sensors)")
                
                # Load telescope database if available
                if 'telescopes_database' in config:
                    # Merge with existing database (keep default values)
                    for telescope_name, characteristics in config['telescopes_database'].items():
                        if telescope_name not in TELESCOPES_DATABASE or telescope_name != 'default':
                            TELESCOPES_DATABASE[telescope_name] = characteristics
                    print(f"Telescope database loaded ({len(config['telescopes_database'])} telescopes)")
                
                return True
    except Exception as e:
        print(f"Error loading configuration: {e}")
    
    return False

def save_configuration():
    """Save configuration to JSON file"""
    try:
        config = {
            'date_sauvegarde': datetime.now().isoformat(),
            'sensors_database': SENSORS_DATABASE,
            'telescopes_database': TELESCOPES_DATABASE
        }
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

def ask_calibration_path():
    """Interactively asks for the path to calibration files"""
    global BIAS_DARK_PATH
    
    print("\n" + "="*80)
    print("CALIBRATION FILES CONFIGURATION")
    print("="*80)
    print("The program needs BIAS and DARK files for advanced SNR calculations.")
    print("These files must be in .fits format and contain calibration images.")
    
    # Check if current path exists
    if BIAS_DARK_PATH and os.path.exists(BIAS_DARK_PATH):
        print(f"Current path: {BIAS_DARK_PATH}")
        response = input("Do you want to use this path? (y/n): ").strip().lower()
        if response in ['y', 'yes', 'o', 'oui']:
            print(f"Using existing path: {BIAS_DARK_PATH}")
            return
    
    while True:
        path = input("\nEnter the path to the folder containing your BIAS/DARK files\n   (ex: C:\\Path\\To\\Calibration or /home/username/astro/calibration): ").strip()
        
        if not path:
            print("Path cannot be empty.")
            continue
        
        # Check if path exists
        if not os.path.exists(path):
            print(f"Path '{path}' does not exist.")
            response = input("Do you want to create this folder? (y/n): ").strip().lower()
            if response in ['y', 'yes', 'o', 'oui']:
                try:
                    os.makedirs(path, exist_ok=True)
                    print(f"Folder created: {path}")
                except Exception as e:
                    print(f"Cannot create folder: {e}")
                    continue
            else:
                continue
        
        # Check if there are .fits files in the folder
        fits_files = list(Path(path).rglob("*.fits")) + list(Path(path).rglob("*.fit"))
        if not fits_files:
            print(f"No .fits files found in '{path}'")
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response not in ['y', 'yes', 'o', 'oui']:
                continue
        else:
            print(f"{len(fits_files)} .fits files found in folder")
        
        BIAS_DARK_PATH = path
        print(f"Calibration path configured: {BIAS_DARK_PATH}")
        
        # Save configuration
        if save_configuration():
            print("Configuration saved for next use")
        
        break

def ask_sensor_characteristics(device_name):
    """Interactively asks for sensor characteristics if not found in database"""
    print(f"\n" + "="*80)
    print(f"SENSOR CONFIGURATION: {device_name}")
    print("="*80)
    print("Sensor not found in database.")
    print("Please provide its technical characteristics.")
    
    while True:
        try:
            print(f"\nSensor characteristics for '{device_name}':")
            
            gain = input("   Gain (e-/ADU, ex: 0.5): ").strip()
            if not gain:
                print("Gain is required.")
                continue
            gain = float(gain)
            
            read_noise = input("   Read noise (e-, ex: 3.5): ").strip()
            if not read_noise:
                print("Read noise is required.")
                continue
            read_noise = float(read_noise)
            
            full_well = input("   Full well (e-, ex: 50000): ").strip()
            if not full_well:
                print("Full well is required.")
                continue
            full_well = float(full_well)
            
            pixel_size = input("   Pixel size (μm, ex: 3.76): ").strip()
            if not pixel_size:
                print("Pixel size is required.")
                continue
            pixel_size = float(pixel_size)
            
            quantum_efficiency = input("   Quantum efficiency (0-1, ex: 0.85): ").strip()
            if not quantum_efficiency:
                print("Quantum efficiency is required.")
                continue
            quantum_efficiency = float(quantum_efficiency)
            
            if not (0 <= quantum_efficiency <= 1):
                print("Quantum efficiency must be between 0 and 1.")
                continue
            
            dark_current = input("   Dark current (e-/pixel/sec, ex: 0.01): ").strip()
            if not dark_current:
                dark_current = 0.01  # default value
            else:
                dark_current = float(dark_current)
            
            # Create sensor characteristics
            characteristics = {
                'gain': gain,
                'read_noise': read_noise,
                'full_well': full_well,
                'pixel_size': pixel_size,
                'quantum_efficiency': quantum_efficiency,
                'dark_current': dark_current
            }
            
            # Display summary
            print(f"\nCharacteristics summary:")
            print(f"   Gain: {gain} e-/ADU")
            print(f"   Read noise: {read_noise} e-")
            print(f"   Full well: {full_well} e-")
            print(f"   Pixel size: {pixel_size} μm")
            print(f"   Quantum efficiency: {quantum_efficiency:.2f}")
            print(f"   Dark current: {dark_current} e-/pixel/sec")
            
            confirmation = input("\nAre these values correct? (y/n): ").strip().lower()
            if confirmation in ['y', 'yes', 'o', 'oui']:
                # Add to database
                SENSORS_DATABASE[device_name] = characteristics
                print(f"✅ Sensor '{device_name}' added to database")
                
                # Save updated configuration
                save_configuration()
                
                return characteristics
            else:
                print("🔄 Entry cancelled, please start over.")
                
        except ValueError as e:
            print(f"❌ Input error: {e}")
            print("Please enter valid numeric values.")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def ask_telescope_characteristics(telescope_name):
    """Interactively asks for telescope characteristics if not found in database"""
    print(f"\n" + "="*80)
    print(f"TELESCOPE CONFIGURATION: {telescope_name}")
    print("="*80)
    print("Telescope not found in database.")
    print("Please provide its technical characteristics.")
    
    while True:
        try:
            print(f"\nTelescope characteristics for '{telescope_name}':")
            
            diameter = input("   Diameter (mm, ex: 200): ").strip()
            if not diameter:
                print("Diameter is required.")
                continue
            diameter = float(diameter)
            
            focal_length = input("   Focal length (mm, ex: 1600): ").strip()
            if not focal_length:
                print("Focal length is required.")
                continue
            focal_length = float(focal_length)
            
            # Calculate f-number
            f_number = focal_length / diameter if diameter > 0 else 8.0
            
            # Display summary
            print(f"\nCharacteristics summary:")
            print(f"   Diameter: {diameter} mm")
            print(f"   Focal length: {focal_length} mm")
            print(f"   Aperture: f/{f_number:.1f}")
            
            confirmation = input("\nAre these values correct? (y/n): ").strip().lower()
            if confirmation in ['y', 'yes', 'o', 'oui']:
                # Create telescope characteristics
                characteristics = {
                    'diameter_mm': diameter,
                    'focal_length_mm': focal_length,
                    'f_number': f_number
                }
                
                # Add to database
                TELESCOPES_DATABASE[telescope_name] = characteristics
                print(f"✅ Telescope '{telescope_name}' added to database")
                
                # Save updated configuration
                save_configuration()
                
                return characteristics
            else:
                print("🔄 Entry cancelled, please start over.")
                
        except ValueError as e:
            print(f"❌ Input error: {e}")
            print("Please enter valid numeric values.")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def get_sensor_characteristics(device_name):
    """Gets sensor characteristics from database or online"""
    global _displayed_warnings
    
    if not device_name:
        return SENSORS_DATABASE['default']
    
    # Clean device name
    device_clean = str(device_name).strip().upper()
    
    # Search in local database
    for sensor_name, characteristics in SENSORS_DATABASE.items():
        if sensor_name.upper() in device_clean:
            return characteristics
    
    # No warning message for unknown sensors
    
    # If not found, return default characteristics instead of asking interactively
    return SENSORS_DATABASE['default']

def get_telescope_characteristics(telescope_name):
    """Gets telescope characteristics from database"""
    global _displayed_warnings
    
    if not telescope_name:
        return TELESCOPES_DATABASE['default']
    
    # Clean telescope name
    telescope_clean = str(telescope_name).strip().upper()
    
    # Search in local database - try exact match first
    if telescope_clean in TELESCOPES_DATABASE:
        return TELESCOPES_DATABASE[telescope_clean]
    
    # Search for partial matches
    for telescope_db_name, characteristics in TELESCOPES_DATABASE.items():
        if telescope_db_name.upper() in telescope_clean or telescope_clean in telescope_db_name.upper():
            return characteristics
    
    # No warning message for unknown telescopes
    
    # If not found, return default
    return TELESCOPES_DATABASE['default']

def detect_sensor_from_fits_header(fits_file_path):
    """Automatically detects sensor from FITS header"""
    try:
        with fits.open(fits_file_path, ignore_missing_simple=True) as hdul:
            header = hdul[0].header
            
            # Search in different header fields
            sensor_keywords = ['INSTRUME', 'CAMERA', 'DETECTOR', 'SENSOR', 'CCD', 'CMOS']
            detected_sensor = None
            
            for keyword in sensor_keywords:
                if keyword in header:
                    value = str(header[keyword]).strip()
                    if value and value != 'Unknown' and value != 'Inconnu':
                        detected_sensor = value
                        print(f"Sensor detected in FITS header: {keyword} = {value}")
                        break
            
            # If not found, try to deduce from filename
            if not detected_sensor:
                filename = fits_file_path.name.upper()
                for sensor_name in SENSORS_DATABASE.keys():
                    if sensor_name.upper() in filename:
                        detected_sensor = sensor_name
                        print(f"Sensor detected from filename: {sensor_name}")
                        break
            
            return detected_sensor
            
    except Exception as e:
        print(f"Error detecting sensor: {e}")
        return None

# Removed: calculate_photons_from_adu function - photon analysis disabled

# Removed: calculate_photons_from_adu_advanced function - photon analysis disabled

# Removed: calculate_light_quantity function - photon analysis disabled

def calculate_advanced_snr(fits_file_path, sensor_characteristics, dark_frame_path=None, bias_frame_path=None, region_size=None):
    if region_size is None:
        region_size = DEFAULT_REGION_SIZE
    """
    Advanced SNR calculation corrected according to astrophotography standards
    Calculates SNR with all noise components and calibration file support
    """
    try:
        with open_fits_for_data(fits_file_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            
            if data is None:
                return None
            
            # Extract exposure parameters from FITS header
            exposure_time = header.get('EXPTIME', 1.0)  # Exposure time in seconds
            gain_header = header.get('GAIN', sensor_characteristics.get('gain', 1.0))
            
            # Load calibration files
            dark_data = None
            bias_data = None
            
            if dark_frame_path and os.path.exists(dark_frame_path):
                try:
                    with open_fits_for_data(dark_frame_path) as dark_hdul:
                        dark_data = dark_hdul[0].data
                        dark_exposure = dark_hdul[0].header.get('EXPTIME', exposure_time)
                        print(f"   Dark frame loaded: {os.path.basename(dark_frame_path)} (exposure: {dark_exposure}s)")
                except Exception as e:
                    print(f"   Dark frame error: {e}")
            
            if bias_frame_path and os.path.exists(bias_frame_path):
                try:
                    with open_fits_for_data(bias_frame_path) as bias_hdul:
                        bias_data = bias_hdul[0].data
                        print(f"   Bias frame loaded: {os.path.basename(bias_frame_path)}")
                except Exception as e:
                    print(f"   Bias frame error: {e}")
            
            # CORRECTED SNR CALCULATION ACCORDING TO ASTROPHOTOGRAPHY STANDARDS
            
            # 1. DATA CALIBRATION (Standard astrophotography method)
            calibrated_data = data.copy()
            
            # Apply bias (offset) first - this is the base level
            if bias_data is not None:
                # Bias is the zero-point, we subtract it to get the signal above bias
                calibrated_data = calibrated_data - bias_data
                # Ensure no negative values after bias subtraction
                calibrated_data = np.maximum(calibrated_data, 0)
            
            # Apply dark (thermal noise) - normalize by exposure time
            if dark_data is not None:
                dark_exposure = dark_hdul[0].header.get('EXPTIME', exposure_time)
                if dark_exposure > 0:
                    if dark_exposure == exposure_time:
                        # Perfect match - use dark as is
                        calibrated_data = calibrated_data - dark_data
                    else:
                        # Normalize dark to match exposure time
                        dark_normalized = dark_data * (exposure_time / dark_exposure)
                        calibrated_data = calibrated_data - dark_normalized
                    
                    # Ensure no negative values after dark subtraction
                    calibrated_data = np.maximum(calibrated_data, 0)
            
            # 2. SENSOR PARAMETERS (in electrons)
            gain_electrons = sensor_characteristics.get('gain', gain_header)  # e-/ADU
            read_noise_electrons = sensor_characteristics.get('read_noise', 5.0)  # e- RMS
            quantum_efficiency = sensor_characteristics.get('quantum_efficiency', 0.6)  # 60% default
            dark_current = sensor_characteristics.get('dark_current', 0.1)  # e-/pixel/s
            
            # 3. ANALYSIS REGION SELECTION
            height, width = calibrated_data.shape
            center_y, center_x = height // 2, width // 2
            
            # Central region (signal of interest)
            y1 = max(0, center_y - region_size // 2)
            y2 = min(height, center_y + region_size // 2)
            x1 = max(0, center_x - region_size // 2)
            x2 = min(width, center_x + region_size // 2)
            
            central_region = calibrated_data[y1:y2, x1:x2]
            
            # Peripheral regions (sky background)
            background_regions = []
            for i in range(4):
                if i == 0:  # Top left corner
                    region = calibrated_data[:region_size, :region_size]
                elif i == 1:  # Top right corner
                    region = calibrated_data[:region_size, -region_size:]
                elif i == 2:  # Bottom left corner
                    region = calibrated_data[-region_size:, :region_size]
                else:  # Bottom right corner
                    region = calibrated_data[-region_size:, -region_size:]
                background_regions.append(region)
            
            # 4. SIGNAL CALCULATIONS (in ADU) - Realistic approach
            central_signal_adu = np.mean(central_region)
            background_mean_adu = np.mean([np.mean(region) for region in background_regions])
            
            # Net signal (signal - background) - this can be negative if background > signal
            net_signal_adu = central_signal_adu - background_mean_adu
            
            # If net signal is negative, it means the background is higher than the signal
            # This is normal for very faint objects or high background
            if net_signal_adu < 0:
                # Use the central signal as is (it contains the object + background)
                net_signal_adu = central_signal_adu
            
            # 5. CONVERSION TO ELECTRONS - Realistic conversion
            net_signal_electrons = net_signal_adu * gain_electrons
            background_electrons = background_mean_adu * gain_electrons
            total_signal_electrons = central_signal_adu * gain_electrons
            
            # Ensure all values are non-negative (physical constraint)
            net_signal_electrons = max(0, net_signal_electrons)
            background_electrons = max(0, background_electrons)
            total_signal_electrons = max(0, total_signal_electrons)
            
            # 6. NOISE COMPONENT CALCULATION (in electrons) - WITH PROTECTION
            
            # Read noise - per pixel
            read_noise_electrons = max(0, read_noise_electrons)
            
            # Photon noise (shot noise) - Poisson - with protection
            photon_noise_signal_electrons = np.sqrt(abs(total_signal_electrons)) if total_signal_electrons > 0 else 0
            photon_noise_background_electrons = np.sqrt(abs(background_electrons)) if background_electrons > 0 else 0
            
            # Dark current noise (thermal) - with protection
            dark_noise_electrons = np.sqrt(abs(dark_current * exposure_time)) if dark_current > 0 and exposure_time > 0 else 0
            
            # Sky background noise (spatial variation) - with protection
            background_std_adu = np.std([np.mean(region) for region in background_regions])
            spatial_background_noise_electrons = max(0, background_std_adu * gain_electrons)
            
            # 7. TOTAL NOISE (quadratic) - WITH PROTECTION
            noise_components = [
                max(0, read_noise_electrons**2),
                max(0, photon_noise_signal_electrons**2),
                max(0, photon_noise_background_electrons**2),
                max(0, dark_noise_electrons**2),
                max(0, spatial_background_noise_electrons**2)
            ]
            total_noise_electrons = np.sqrt(sum(noise_components))
            
            # Protection against NaN or infinite values
            if np.isnan(total_noise_electrons) or np.isinf(total_noise_electrons):
                print(f"   Invalid total noise detected: {total_noise_electrons}")
                print(f"      Components: {noise_components}")
                total_noise_electrons = 0
            
            # 8. FINAL SNR - WITH PROTECTION
            final_snr = net_signal_electrons / total_noise_electrons if total_noise_electrons > 0 else 0
            
            # Protection against NaN or infinite values
            if np.isnan(final_snr) or np.isinf(final_snr):
                final_snr = 0
            
            # 9. SNR BY COMPONENT FOR DIAGNOSTIC - WITH PROTECTION
            snr_read_only = net_signal_electrons / read_noise_electrons if read_noise_electrons > 0 else 0
            if np.isnan(snr_read_only) or np.isinf(snr_read_only):
                snr_read_only = 0
                
            photon_noise_squared = max(0, photon_noise_signal_electrons**2 + photon_noise_background_electrons**2)
            snr_photon_only = net_signal_electrons / np.sqrt(photon_noise_squared) if photon_noise_squared > 0 else 0
            if np.isnan(snr_photon_only) or np.isinf(snr_photon_only):
                snr_photon_only = 0
            
            # 10. ADDITIONAL METRICS - WITH PROTECTION
            contrast = net_signal_electrons / background_electrons if background_electrons > 0 else 0
            if np.isnan(contrast) or np.isinf(contrast):
                contrast = 0
                
            dynamic_range = total_signal_electrons / total_noise_electrons if total_noise_electrons > 0 else 0
            if np.isnan(dynamic_range) or np.isinf(dynamic_range):
                dynamic_range = 0
            
            # 11. NOISE COMPONENT DIAGNOSTIC - WITH PROTECTION
            total_noise_squared = max(0, total_noise_electrons**2)
            
            # Calculate contributions with protection against NaN values
            def safe_contribution(numerator, denominator):
                if denominator > 0 and not np.isnan(numerator) and not np.isinf(numerator):
                    contribution = (numerator / denominator) * 100
                    return max(0, min(100, contribution))  # Limit between 0 and 100%
                return 0
            
            noise_contributions = {
                'read': safe_contribution(read_noise_electrons**2, total_noise_squared),
                'photon_signal': safe_contribution(photon_noise_signal_electrons**2, total_noise_squared),
                'photon_background': safe_contribution(photon_noise_background_electrons**2, total_noise_squared),
                'dark': safe_contribution(dark_noise_electrons**2, total_noise_squared),
                'spatial_background': safe_contribution(spatial_background_noise_electrons**2, total_noise_squared)
            }
            
            return {
                'snr_final': final_snr,
                'snr_read_only': snr_read_only,
                'snr_photon_only': snr_photon_only,
                'signal_net_electrons': net_signal_electrons,
                'signal_net_adu': net_signal_adu,
                'signal_total_electrons': total_signal_electrons,
                'background_electrons': background_electrons,
                'noise_total_electrons': total_noise_electrons,
                'noise_read_electrons': read_noise_electrons,
                'noise_photon_signal_electrons': photon_noise_signal_electrons,
                'noise_photon_background_electrons': photon_noise_background_electrons,
                'noise_dark_electrons': dark_noise_electrons,
                'noise_spatial_background_electrons': spatial_background_noise_electrons,
                'contrast': contrast,
                'dynamic_range': dynamic_range,
                'calibrated': bias_data is not None or dark_data is not None,
                'region_size': region_size,
                'exposure_time': exposure_time,
                'gain_electrons': gain_electrons,
                'noise_contributions': noise_contributions
            }
            
    except Exception as e:
        print(f"   Advanced SNR calculation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_calibration_files(base_path, exposure_time=None, gain=None, sensor_name=None):
    """
    Automatically finds appropriate BIAS and DARK files in all subdirectories
    Selects the best ones based on sensor, gain and exposure time
    """
    bias_files = []
    dark_files = []
    
    if not os.path.exists(base_path):
        return [], []
    
    print(f"   🔍 Searching for DARK files:")
    print(f"      Target exposure: {exposure_time}s")
    print(f"      Target gain: {gain}")
    print(f"      Target sensor: {sensor_name}")
    
    try:
        # Recursive search in all subdirectories
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith('.fits') or file.lower().endswith('.fit'):
                    file_path = os.path.join(root, file)
                    try:
                        with open_fits_for_data(file_path) as hdul:
                            header = hdul[0].header
                            
                            # Extract file information
                            imagetyp = header.get('IMAGETYP', '').upper()
                            file_gain = header.get('GAIN', None)
                            file_camera = header.get('INSTRUME', '').upper()
                            file_exposure = header.get('EXPTIME', 0)
                            
                            # Compatibility score (0 = perfect, higher = less compatible)
                            score = 0
                            
                            # Check gain compatibility
                            if gain and file_gain:
                                gain_diff = abs(float(file_gain) - float(gain))
                                score += gain_diff * 10  # Penalty for gain difference
                            
                            # Check sensor compatibility
                            if sensor_name and file_camera:
                                sensor_name_upper = sensor_name.upper()
                                if sensor_name_upper not in file_camera and file_camera not in sensor_name_upper:
                                    score += 50  # Significant penalty if different sensor
                            
                            # Detect file type and add with score
                            if 'BIAS' in imagetyp or 'BIAS' in file.upper():
                                bias_files.append({
                                    'path': file_path,
                                    'score': score,
                                    'gain': file_gain,
                                    'camera': file_camera,
                                    'exposure': file_exposure
                                })
                            elif 'DARK' in imagetyp or 'DARK' in file.upper():
                                # For darks, prioritize by exposure time difference
                                if exposure_time and file_exposure > 0:
                                    time_diff = abs(file_exposure - exposure_time)
                                    # Use time difference as primary score - smaller is better
                                    score = time_diff
                                
                                print(f"      📁 Found DARK: {file} - {file_exposure}s (diff: {time_diff}s, score: {score:.1f})")
                                
                                dark_files.append({
                                    'path': file_path,
                                    'score': score,
                                    'gain': file_gain,
                                    'camera': file_camera,
                                    'exposure': file_exposure
                                })
                                
                    except Exception as e:
                        continue
        
        # Sort by score (best score first) - ensure perfect matches come first
        bias_files.sort(key=lambda x: x['score'])
        dark_files.sort(key=lambda x: x['score'])
        
        # Debug: verify sorting is correct
        if dark_files and exposure_time:
            print(f"   🔍 After sorting - Top 3 DARK files:")
            for i, dark in enumerate(dark_files[:3]):
                print(f"      {i+1}. {os.path.basename(dark['path'])} - {dark['exposure']}s (score: {dark['score']:.1f})")
        
        # Debug: show all available dark files
        if dark_files and exposure_time:
            print(f"   🔍 Available DARK files for {exposure_time}s exposure:")
            for i, dark in enumerate(dark_files[:5]):  # Show top 5
                print(f"      {i+1}. {os.path.basename(dark['path'])} - {dark['exposure']}s (score: {dark['score']:.1f})")
        
        # Return paths of the 3 best of each type
        bias_paths = [f['path'] for f in bias_files[:3]]
        dark_paths = [f['path'] for f in dark_files[:3]]
        
        # Debug: show selected dark file
        if dark_files and exposure_time:
            best_dark = dark_files[0]
            print(f"   🎯 Selected DARK: {os.path.basename(best_dark['path'])}")
            print(f"      Exposure: {best_dark['exposure']}s (target: {exposure_time}s)")
            print(f"      Score: {best_dark['score']:.1f}")
            
            # Verify this is the correct choice
            if best_dark['exposure'] == exposure_time:
                print(f"      ✅ PERFECT MATCH: Exposure times match exactly!")
            else:
                print(f"      ⚠️  WARNING: Exposure time mismatch! This may cause calibration issues.")
        
        return bias_paths, dark_paths
        
    except Exception as e:
        print(f"   Calibration file search error: {e}")
        return [], []

# Removed: calculate_human_eye_comparison function - human eye comparison disabled

def check_snr_and_suggest_exposure_time(samples, current_exposure_time):
    """
    Checks average SNR of ADU samples and suggests exposure time if necessary
    Uses advanced SNR calculation if available, otherwise classic calculation
    """
    if len(samples) < 2:
        return None, "Insufficient samples for SNR analysis"
    
    # Check if we have advanced SNR data
    advanced_snr_list = []
    classic_snr_list = []
    
    for s in samples:
        # Advanced SNR (priority)
        if 'advanced_snr' in s and s['advanced_snr']:
            advanced_snr_list.append(s['advanced_snr']['snr_final'])
        
        # Classic SNR (fallback)
        if 'adu_stats' in s and 'signal_to_noise' in s['adu_stats']:
            classic_snr_list.append(s['adu_stats']['signal_to_noise'])
    
    # Use advanced SNR if available, otherwise classic
    if advanced_snr_list:
        snr_list = advanced_snr_list
        snr_type = "advanced"
        print(f"     Using advanced SNR ({len(advanced_snr_list)}/{len(samples)} samples)")
    elif classic_snr_list:
        snr_list = classic_snr_list
        snr_type = "classic"
        print(f"     Using classic SNR ({len(classic_snr_list)}/{len(samples)} samples)")
    else:
        return None, "No SNR data available"
    
    average_snr = sum(snr_list) / len(snr_list)
    print(f"     Average {snr_type} SNR of {len(snr_list)} samples: {average_snr:.2f}")
    
    # Check if SNR is sufficient (> 3)
    if average_snr > 3:
        return average_snr, f"{snr_type.capitalize()} SNR sufficient (> 3)"
    
    # If SNR is insufficient, calculate necessary exposure time
    # SNR = signal / total_noise
    # To improve SNR, need to increase signal proportionally
    # New_time = Current_time * (3 / Current_SNR)²
    
    suggested_time = current_exposure_time * (3.0 / average_snr) ** 2
    
    # Round to practical values (30s, 60s, 120s, 300s, 600s, etc.)
    practical_times = [30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600]
    suggested_time_rounded = min(practical_times, key=lambda x: abs(x - suggested_time))
    
    return average_snr, f"{snr_type.capitalize()} SNR insufficient ({average_snr:.2f} < 3). Suggested time: {suggested_time_rounded}s (calculated: {suggested_time:.0f}s)"

def display_noise_component_details(samples):
    """Displays noise component details for advanced SNR samples"""
    if not samples:
        return
    
    advanced_snr_sources = [s for s in samples if 'advanced_snr' in s and s['advanced_snr']]
    if not advanced_snr_sources:
        return
    
    print(f"     NOISE COMPONENT DETAILS:")
    print(f"     {'Component':<20} {'Value':<12} {'% of total':<12}")
    print(f"     {'-'*20} {'-'*12} {'-'*12}")
    
    # Calculate component averages
    read_noise_avg = sum(s['advanced_snr']['noise_read_electrons'] for s in advanced_snr_sources) / len(advanced_snr_sources)
    photon_signal_noise_avg = sum(s['advanced_snr']['noise_photon_signal_electrons'] for s in advanced_snr_sources) / len(advanced_snr_sources)
    photon_background_noise_avg = sum(s['advanced_snr']['noise_photon_background_electrons'] for s in advanced_snr_sources) / len(advanced_snr_sources)
    spatial_background_noise_avg = sum(s['advanced_snr']['noise_spatial_background_electrons'] for s in advanced_snr_sources) / len(advanced_snr_sources)
    dark_noise_avg = sum(s['advanced_snr']['noise_dark_electrons'] for s in advanced_snr_sources) / len(advanced_snr_sources)
    total_noise_avg = sum(s['advanced_snr']['noise_total_electrons'] for s in advanced_snr_sources) / len(advanced_snr_sources)
    
    # Display each component with calculated percentages
    if advanced_snr_sources:
        # Use percentages calculated by advanced SNR
        noise_contributions_avg = {}
        for key in ['read', 'photon_signal', 'photon_background', 'dark', 'spatial_background']:
            values = [s['advanced_snr']['noise_contributions'].get(key, 0) for s in advanced_snr_sources]
            noise_contributions_avg[key] = sum(values) / len(values) if values else 0
        
        components = [
            ('Read noise (e-)', read_noise_avg, noise_contributions_avg.get('read', 0)),
            ('Photon signal (e-)', photon_signal_noise_avg, noise_contributions_avg.get('photon_signal', 0)),
            ('Photon background (e-)', photon_background_noise_avg, noise_contributions_avg.get('photon_background', 0)),
            ('Dark current (e-)', dark_noise_avg, noise_contributions_avg.get('dark', 0)),
            ('Spatial background (e-)', spatial_background_noise_avg, noise_contributions_avg.get('spatial_background', 0))
        ]
        
        for name, value, percentage in components:
            print(f"     {name:<20} {value:<12.2f} {percentage:<12.1f}%")
        
        print(f"     {'-'*20} {'-'*12} {'-'*12}")
        print(f"     {'Total noise (e-)':<20} {total_noise_avg:<12.2f} {'100.0':<12}%")

def calculate_adu_statistics_by_filter(data_by_target):
    """
    Calculates ADU statistics by filter and extrapolates for all files
    """
    print(f"\nADU STATISTICAL ANALYSIS BY FILTER")
    print("=" * 80)
    
    for target, data in data_by_target.items():
        print(f"\nTARGET: {target}")
        print("-" * 60)
        
        for filter_name, samples in data['adu_samples'].items():
            if len(samples) >= 2:  # At least 2 samples for reliable statistics
                print(f"   FILTER: {filter_name}")
                
                # Calculate ADU sample statistics
                photons_list = [s['adu_photons'] for s in samples]
                exposure_list = [s['exposure_time'] for s in samples]
                
                # Calculate photons/time ratio to normalize
                photon_time_ratios = [p/t for p, t in zip(photons_list, exposure_list)]
                
                # Ratio statistics
                average_ratio = sum(photon_time_ratios) / len(photon_time_ratios)
                ratio_std = (sum((r - average_ratio)**2 for r in photon_time_ratios) / len(photon_time_ratios))**0.5
                
                # Photon statistics
                average_photons = sum(photons_list) / len(photons_list)
                photons_std = (sum((p - average_photons)**2 for p in photons_list) / len(photons_list))**0.5
                
                print(f"     Analyzed samples: {len(samples)} files")
                print(f"     Average photons: {average_photons:.2e} ± {photons_std:.2e}")
                print(f"     Photons/time ratio: {average_ratio:.2e} ± {ratio_std:.2e}")
                
                # SNR CHECK AND EXPOSURE TIME SUGGESTION
                average_exposure_time = sum(exposure_list) / len(exposure_list)
                average_snr, snr_message = check_snr_and_suggest_exposure_time(samples, average_exposure_time)
                print(f"     {snr_message}")
                
                # Display noise component details if advanced SNR available
                display_noise_component_details(samples)
                
                # Calculate total time for this filter
                total_filter_time = sum(time for time in data['time_by_filter'][filter_name])
                filter_file_count = len(data['time_by_filter'][filter_name])
                
                # Extrapolate total photon count
                extrapolated_photons = average_ratio * total_filter_time
                
                print(f"     Extrapolation: {extrapolated_photons:.2e} photons for {filter_file_count} files")
                print(f"     Estimated precision: ±{ratio_std * total_filter_time:.2e} photons")
                
                # Update light data with ADU extrapolation
                if filter_name in data['received_light']:
                    # Replace theoretical calculations with ADU extrapolation
                    for light in data['received_light'][filter_name]:
                        # Calculate proportion for this file (simplified without eye comparison)
                        time_proportion = 1.0 / len(data['received_light'][filter_name])  # Equal distribution
                        extrapolated_adu_photons = extrapolated_photons * time_proportion
                        
                        # ADU extrapolation removed (photon analysis disabled)
                        light['source'] = 'basic_info'
                        light['adu_stats'] = {
                            'average_ratio': average_ratio,
                            'ratio_std': ratio_std,
                            'samples_used': len(samples),
                            'extrapolated_photons': extrapolated_photons,
                            'average_snr': average_snr,
                            'snr_message': snr_message
                        }
                
                print(f"     ✅ Statistics calculated for all files")
            else:
                print(f"   FILTER: {filter_name} - Insufficient samples ({len(samples)} < 2)")
                print(f"     ⚠️  Keeping theoretical calculation")
    
    return data_by_target

def extract_fits_header_info_fast(file_path):
    """Fast extraction of basic FITS header info for Phase 1 (optimized for speed)"""
    try:
        with open_fits_for_data(file_path) as hdul:
            header = hdul[0].header
            
            # Extract only essential info for Phase 1
            exposure_time = None
            time_keywords = ['EXPTIME', 'EXPOSURE', 'EXPOSURE_TIME', 'INT_TIME', 'INTEGRATION']
            for keyword in time_keywords:
                if keyword in header:
                    exposure_time = float(header[keyword])
                    break
            
            # Extract image type (simplified)
            image_type = 'LIGHT'  # Default
            if 'IMAGETYP' in header:
                value = str(header['IMAGETYP']).upper().strip()
                if 'FLAT' in value:
                    image_type = 'FLAT'
                elif 'DARK' in value:
                    image_type = 'DARK'
                elif 'BIAS' in value:
                    image_type = 'BIAS'
            
            # Extract filter (simplified)
            filter_found = None
            filter_keywords = ['FILTER', 'FILTRE', 'FILT', 'COLOR']
            for keyword in filter_keywords:
                if keyword in header:
                    filter_found = str(header[keyword]).strip().upper()
                    break
            
            if not filter_found:
                filter_found = 'Unknown'
            
            # Extract target (simplified)
            target = None
            if 'OBJECT' in header:
                target = str(header['OBJECT']).strip()
            elif 'TARGET' in header:
                target = str(header['TARGET']).strip()
            
            # Basic instrument info
            instrument = header.get('INSTRUME', 'Unknown')
            telescope = header.get('TELESCOP', 'Unknown')
            
            return {
                'exposure_time': exposure_time,
                'type': image_type,
                'filter': filter_found,
                'target': target,
                'info': {
                    'instrument': instrument,
                    'telescope': telescope,
                    'date_obs': header.get('DATE-OBS', 'Unknown')
                }
            }
            
    except Exception as e:
        return None

def extract_fits_header_info(file_path, should_analyze_adu=True):
    """Extracts all information from FITS header and calculates photons from ADU"""
    # Remove debug print to avoid interfering with tqdm
    try:
        with open_fits_for_data(file_path) as hdul:
            header = hdul[0].header
            
            # Extract exposure time
            exposure_time = None
            time_keywords = ['EXPTIME', 'EXPOSURE', 'EXPOSURE_TIME', 'INT_TIME', 'INTEGRATION']
            for keyword in time_keywords:
                if keyword in header:
                    exposure_time = float(header[keyword])
                    break
            
            # Extract image type
            image_type = None
            type_keywords = ['IMAGETYP', 'IMTYPE', 'OBJECT', 'OBSTYPE', 'FRAME']
            for keyword in type_keywords:
                if keyword in header:
                    value = str(header[keyword]).upper().strip()
                    if 'LIGHT' in value or 'SCIENCE' in value or 'OBJECT' in value:
                        image_type = 'LIGHT'
                        break
                    elif 'FLAT' in value or 'FLATWIZARD' in value:
                        image_type = 'FLAT'
                        break
                    elif 'DARK' in value:
                        image_type = 'DARK'
                        break
                    elif 'BIAS' in value:
                        image_type = 'BIAS'
                        break
            
            # If no type found, try to deduce from filename
            if image_type is None:
                filename = os.path.basename(file_path).upper()
                if 'FLAT' in filename:
                    image_type = 'FLAT'
                elif 'DARK' in filename:
                    image_type = 'DARK'
                elif 'BIAS' in filename:
                    image_type = 'BIAS'
                else:
                    image_type = 'LIGHT'  # Default to LIGHT for unknown files
            
            # Extract filter
            filter_found = None
            filter_keywords = ['FILTER', 'FILTRE', 'FILTERS', 'FILT', 'COLOR', 'BANDPASS']
            for keyword in filter_keywords:
                if keyword in header:
                    value = str(header[keyword]).upper().strip()
                    
                    # First, try direct mapping for common filter names in headers
                    # Comprehensive mapping for all variants
                    header_filter_mapping = {
                        # RGB filters - all variants
                        'BLUE': 'B', 'blue': 'B', 'Blue': 'B',
                        'GREEN': 'G', 'green': 'G', 'Green': 'G',
                        'RED': 'R', 'red': 'R', 'Red': 'R',
                        
                        # Luminance - all variants
                        'LUMINANCE': 'L', 'luminance': 'L', 'Luminance': 'L',
                        'LUM': 'L', 'lum': 'L', 'Lum': 'L',
                        'LIGHT': 'L', 'light': 'L', 'Light': 'L',
                        'L': 'L', 'l': 'L',
                        
                        # Clear filter
                        'CLEAR': 'CLEAR', 'clear': 'CLEAR', 'Clear': 'CLEAR',
                        
                        # H-Alpha - all variants
                        'H-ALPHA': 'HA', 'H_ALPHA': 'HA', 'H ALPHA': 'HA',
                        'h-alpha': 'HA', 'h_alpha': 'HA', 'h alpha': 'HA',
                        'H-alpha': 'HA', 'H_alpha': 'HA', 'H alpha': 'HA',
                        'HALPHA': 'HA', 'halpha': 'HA', 'Halpha': 'HA',
                        'HA': 'HA', 'ha': 'HA', 'Ha': 'HA',
                        'H-A': 'HA', 'H_A': 'HA', 'H A': 'HA',
                        'h-a': 'HA', 'h_a': 'HA', 'h a': 'HA',
                        'H-a': 'HA', 'H_a': 'HA', 'H a': 'HA',
                        'HYDROGEN ALPHA': 'HA', 'HYDROGEN-ALPHA': 'HA', 'HYDROGEN_ALPHA': 'HA',
                        'hydrogen alpha': 'HA', 'hydrogen-alpha': 'HA', 'hydrogen_alpha': 'HA',
                        'Hydrogen Alpha': 'HA', 'Hydrogen-Alpha': 'HA', 'Hydrogen_Alpha': 'HA',
                        
                        # H-Beta - all variants
                        'H-BETA': 'HBETA', 'H_BETA': 'HBETA', 'H BETA': 'HBETA',
                        'h-beta': 'HBETA', 'h_beta': 'HBETA', 'h beta': 'HBETA',
                        'H-beta': 'HBETA', 'H_beta': 'HBETA', 'H beta': 'HBETA',
                        'HBETA': 'HBETA', 'hbeta': 'HBETA', 'Hbeta': 'HBETA',
                        'HB': 'HBETA', 'hb': 'HBETA', 'Hb': 'HBETA',
                        'H-B': 'HBETA', 'H_B': 'HBETA', 'H B': 'HBETA',
                        'h-b': 'HBETA', 'h_b': 'HBETA', 'h b': 'HBETA',
                        'H-b': 'HBETA', 'H_b': 'HBETA', 'H b': 'HBETA',
                        'HYDROGEN BETA': 'HBETA', 'HYDROGEN-BETA': 'HBETA', 'HYDROGEN_BETA': 'HBETA',
                        'hydrogen beta': 'HBETA', 'hydrogen-beta': 'HBETA', 'hydrogen_beta': 'HBETA',
                        'Hydrogen Beta': 'HBETA', 'Hydrogen-Beta': 'HBETA', 'Hydrogen_Beta': 'HBETA',
                        
                        # OIII - all variants
                        'OIII': 'OIII', 'oiii': 'OIII', 'Oiii': 'OIII',
                        'O3': 'OIII', 'o3': 'OIII', 'O3': 'OIII',
                        'O-3': 'OIII', 'O_3': 'OIII', 'O 3': 'OIII',
                        'o-3': 'OIII', 'o_3': 'OIII', 'o 3': 'OIII',
                        'O-3': 'OIII', 'O_3': 'OIII', 'O 3': 'OIII',
                        'OXYGEN III': 'OIII', 'OXYGEN-III': 'OIII', 'OXYGEN_III': 'OIII',
                        'oxygen iii': 'OIII', 'oxygen-iii': 'OIII', 'oxygen_iii': 'OIII',
                        'Oxygen III': 'OIII', 'Oxygen-III': 'OIII', 'Oxygen_III': 'OIII',
                        'OXYGEN 3': 'OIII', 'OXYGEN-3': 'OIII', 'OXYGEN_3': 'OIII',
                        'oxygen 3': 'OIII', 'oxygen-3': 'OIII', 'oxygen_3': 'OIII',
                        'Oxygen 3': 'OIII', 'Oxygen-3': 'OIII', 'Oxygen_3': 'OIII',
                        
                        # SII - all variants
                        'SII': 'SII', 'sii': 'SII', 'Sii': 'SII',
                        'S2': 'SII', 's2': 'SII', 'S2': 'SII',
                        'S-2': 'SII', 'S_2': 'SII', 'S 2': 'SII',
                        's-2': 'SII', 's_2': 'SII', 's 2': 'SII',
                        'S-2': 'SII', 'S_2': 'SII', 'S 2': 'SII',
                        'SULFUR II': 'SII', 'SULFUR-II': 'SII', 'SULFUR_II': 'SII',
                        'sulfur ii': 'SII', 'sulfur-ii': 'SII', 'sulfur_ii': 'SII',
                        'Sulfur II': 'SII', 'Sulfur-II': 'SII', 'Sulfur_II': 'SII',
                        'SULFUR 2': 'SII', 'SULFUR-2': 'SII', 'SULFUR_2': 'SII',
                        'sulfur 2': 'SII', 'sulfur-2': 'SII', 'sulfur_2': 'SII',
                        'Sulfur 2': 'SII', 'Sulfur-2': 'SII', 'Sulfur_2': 'SII',
                        
                        # NII - all variants
                        'NII': 'NII', 'nii': 'NII', 'Nii': 'NII',
                        'N2': 'NII', 'n2': 'NII', 'N2': 'NII',
                        'N-2': 'NII', 'N_2': 'NII', 'N 2': 'NII',
                        'n-2': 'NII', 'n_2': 'NII', 'n 2': 'NII',
                        'N-2': 'NII', 'N_2': 'NII', 'N 2': 'NII',
                        'NITROGEN II': 'NII', 'NITROGEN-II': 'NII', 'NITROGEN_II': 'NII',
                        'nitrogen ii': 'NII', 'nitrogen-ii': 'NII', 'nitrogen_ii': 'NII',
                        'Nitrogen II': 'NII', 'Nitrogen-II': 'NII', 'Nitrogen_II': 'NII',
                        'NITROGEN 2': 'NII', 'NITROGEN-2': 'NII', 'NITROGEN_2': 'NII',
                        'nitrogen 2': 'NII', 'nitrogen-2': 'NII', 'nitrogen_2': 'NII',
                        'Nitrogen 2': 'NII', 'Nitrogen-2': 'NII', 'Nitrogen_2': 'NII',
                        
                        # HEII - all variants
                        'HEII': 'HEII', 'heii': 'HEII', 'Heii': 'HEII',
                        'HE-II': 'HEII', 'HE_II': 'HEII', 'HE II': 'HEII',
                        'he-ii': 'HEII', 'he_ii': 'HEII', 'he ii': 'HEII',
                        'He-II': 'HEII', 'He_II': 'HEII', 'He II': 'HEII',
                        'HE-2': 'HEII', 'HE_2': 'HEII', 'HE 2': 'HEII',
                        'he-2': 'HEII', 'he_2': 'HEII', 'he 2': 'HEII',
                        'He-2': 'HEII', 'He_2': 'HEII', 'He 2': 'HEII',
                        'HELIUM II': 'HEII', 'HELIUM-II': 'HEII', 'HELIUM_II': 'HEII',
                        'helium ii': 'HEII', 'helium-ii': 'HEII', 'helium_ii': 'HEII',
                        'Helium II': 'HEII', 'Helium-II': 'HEII', 'Helium_II': 'HEII',
                        'HELIUM 2': 'HEII', 'HELIUM-2': 'HEII', 'HELIUM_2': 'HEII',
                        'helium 2': 'HEII', 'helium-2': 'HEII', 'helium_2': 'HEII',
                        'Helium 2': 'HEII', 'Helium-2': 'HEII', 'Helium_2': 'HEII'
                    }
                    
                    # Check direct mapping first
                    if value in header_filter_mapping:
                        filter_found = header_filter_mapping[value]
                        break
                    
                    # More precise filter detection to avoid false positives
                    for filter_name in FILTERS_INFO.keys():
                        # Check for exact match or word boundaries to avoid false positives
                        if (filter_name == value or 
                            f" {filter_name} " in f" {value} " or
                            value.startswith(f"{filter_name} ") or
                            value.endswith(f" {filter_name}") or
                            value == filter_name):
                            filter_found = filter_name
                            break
                    if filter_found:
                        break
            
            # If no filter found in header, check for Bayer pattern FIRST
            if filter_found is None:
                # Check for Bayer pattern in header to detect color cameras FIRST
                bayer_detected = False
                bayer_keywords = ['BAYERPAT', 'BAYERPATN', 'BAYERPATTERN', 'COLORTYP', 'COLORSPACE']
                for bayer_key in bayer_keywords:
                    if bayer_key in header:
                        bayer_value = str(header[bayer_key]).strip().upper()
                        # Check if it's a valid Bayer pattern
                        valid_bayer_patterns = ['RGGB', 'BGGR', 'GRBG', 'GBRG', 'RGB', 'COLOR', 'BAYER']
                        if bayer_value in valid_bayer_patterns:
                            bayer_detected = True
                            break
                
                if bayer_detected:
                    filter_found = 'OSC'  # One Shot Color with Bayer pattern
                else:
                    # If no Bayer pattern, try to extract from filename
                    filt_code, filt_info = extract_filter_from_filename(file_path.name)
                    if filt_code:
                        filter_found = filt_code
                    else:
                        # Fallback: try to detect common filter names in filename
                        filename = os.path.basename(file_path).upper()
                        if 'LUMINANCE' in filename or 'LUM' in filename:
                            filter_found = 'L'
                        elif 'RED' in filename:
                            filter_found = 'R'
                        elif 'GREEN' in filename:
                            filter_found = 'G'
                        elif 'BLUE' in filename:
                            filter_found = 'B'
                        elif 'HALPHA' in filename or 'H-ALPHA' in filename:
                            filter_found = 'HA'
                        elif 'OIII' in filename or 'O3' in filename:
                            filter_found = 'OIII'
                        elif 'SII' in filename or 'S2' in filename:
                            filter_found = 'SII'
                        else:
                            # Check if this is likely a color camera file (OSC)
                            # Look for common OSC indicators in filename or path
                            filename_lower = filename.lower()
                            path_lower = str(file_path).lower()
                            
                            # Common OSC indicators
                            osc_indicators = ['color', 'colour', 'osc', 'one shot', 'oneshot', 'rgb', 'camera', 'cam']
                            if any(indicator in filename_lower or indicator in path_lower for indicator in osc_indicators):
                                filter_found = 'OSC'  # One Shot Color
                            else:
                                filter_found = 'L'  # Default to Luminance for monochrome cameras
            
            # Extract target/object
            target_found = None
            if 'OBJECT' in header:
                target_found = str(header['OBJECT']).strip()
            elif 'TARGET' in header:
                target_found = str(header['TARGET']).strip()
            
            # Normalize target name (case-insensitive, remove extra spaces)
            if target_found and target_found != 'Unknown':
                target_found = normalize_target_name(target_found)
            
            # Extract celestial coordinates
            ra, dec = None, None
            if 'RA' in header and 'DEC' in header:
                try:
                    ra = float(header['RA'])
                    dec = float(header['DEC'])
                except:
                    pass
            elif 'CRVAL1' in header and 'CRVAL2' in header:
                try:
                    ra = float(header['CRVAL1'])
                    dec = float(header['CRVAL2'])
                except:
                    pass
            
            # Extract instrument and telescope
            instrument = header.get('INSTRUME', 'Unknown')
            telescope = header.get('TELESCOP', 'Unknown')
            
            # Normalize telescope name to avoid technical descriptions in reports
            telescope = normalize_telescope_name(telescope)
            
            # Remove debug prints to avoid interfering with tqdm
            
            # Extract instrument diameter
            diameter = None
            diameter_keywords = ['APERTURE', 'TELESCOP_DIAM', 'MIRROR_DIAM', 'PRIMARY_DIAM', 'DIAMETER', 'APTDIA', 'TELDIAM', 'MIRROR_D', 'PRIMARY_D']
            for keyword in diameter_keywords:
                if keyword in header:
                    try:
                        diameter = float(header[keyword])
                        break
                    except:
                        pass
            
            # If diameter not found in header, try to extract from telescope name
            if diameter is None and telescope:
                telescope_str = str(telescope).upper()
                # Look for diameter patterns in telescope name
                import re
                diameter_match = re.search(r'(\d+(?:\.\d+)?)\s*mm', telescope_str)
                if diameter_match:
                    diameter = float(diameter_match.group(1))
                else:
                    # Try to extract from telescope model names
                    diameter_match = re.search(r'(\d+(?:\.\d+)?)', telescope_str)
                    if diameter_match:
                        potential_diameter = float(diameter_match.group(1))
                        # Check if it's a reasonable telescope diameter (50mm to 2000mm)
                        if 50 <= potential_diameter <= 2000:
                            diameter = potential_diameter
            
            # If not found, try to deduce from telescope name
            if diameter is None:
                telescope_str = str(telescope).upper()
                
                # First, try to match with our database
                for telescope_name, characteristics in TELESCOPES_DATABASE.items():
                    if telescope_name.upper() in telescope_str:
                        diameter = characteristics['diameter_mm']
                        break
                
                # If still not found, try common patterns
                if diameter is None:
                    # Ritchey-Chrétien (RC)
                    if 'RC8' in telescope_str or '8"' in telescope_str:
                        diameter = 203.2  # mm
                    elif 'RC6' in telescope_str or '6"' in telescope_str:
                        diameter = 152.4  # mm
                    elif 'RC10' in telescope_str or '10"' in telescope_str:
                        diameter = 254.0  # mm
                    elif 'RC12' in telescope_str or '12"' in telescope_str:
                        diameter = 304.8  # mm
                    elif 'RC14' in telescope_str or '14"' in telescope_str:
                        diameter = 355.6  # mm
                    elif 'RC16' in telescope_str or '16"' in telescope_str:
                        diameter = 406.4  # mm
                    # Takahashi
                    elif 'FSQ-85' in telescope_str or 'FSQ85' in telescope_str or 'FSQ-85EDP' in telescope_str or 'FSQ85EDP' in telescope_str:
                        diameter = 85.0  # mm
                elif 'FSQ-106' in telescope_str or 'FSQ106' in telescope_str:
                    diameter = 106.0  # mm
                elif 'FSQ-130' in telescope_str or 'FSQ130' in telescope_str:
                    diameter = 130.0  # mm
                elif 'TOA-130' in telescope_str:
                    diameter = 130.0  # mm
                elif 'TOA-150' in telescope_str:
                    diameter = 150.0  # mm
                elif 'TOA-160' in telescope_str:
                    diameter = 160.0  # mm
                elif 'TSA-102' in telescope_str:
                    diameter = 102.0  # mm
                elif 'TSA-120' in telescope_str:
                    diameter = 120.0  # mm
                elif 'Epsilon-130' in telescope_str:
                    diameter = 130.0  # mm
                elif 'Epsilon-160' in telescope_str:
                    diameter = 160.0  # mm
                elif 'Epsilon-180' in telescope_str:
                    diameter = 180.0  # mm
                # PlaneWave
                elif 'CDK12' in telescope_str or '12"' in telescope_str:
                    diameter = 304.8  # mm
                elif 'CDK14' in telescope_str or '14"' in telescope_str:
                    diameter = 355.6  # mm
                elif 'CDK16' in telescope_str or '16"' in telescope_str:
                    diameter = 406.4  # mm
                elif 'CDK17' in telescope_str or '17"' in telescope_str:
                    diameter = 431.8  # mm
                elif 'CDK20' in telescope_str or '20"' in telescope_str:
                    diameter = 508.0  # mm
                elif 'CDK24' in telescope_str or '24"' in telescope_str:
                    diameter = 609.6  # mm
                elif 'L-350' in telescope_str:
                    diameter = 350.0  # mm
                elif 'L-500' in telescope_str:
                    diameter = 500.0  # mm
                elif 'L-600' in telescope_str:
                    diameter = 600.0  # mm
                # Celestron
                elif 'C8' in telescope_str or '8"' in telescope_str:
                    diameter = 203.2  # mm
                elif 'C9' in telescope_str or 'C9.25' in telescope_str or '9.25"' in telescope_str:
                    diameter = 235.0  # mm
                elif 'C11' in telescope_str or '11"' in telescope_str:
                    diameter = 279.4  # mm
                elif 'C14' in telescope_str or '14"' in telescope_str:
                    diameter = 355.6  # mm
                elif 'EDGEHD8' in telescope_str:
                    diameter = 203.2  # mm
                elif 'EDGEHD9.25' in telescope_str:
                    diameter = 235.0  # mm
                elif 'EDGEHD11' in telescope_str:
                    diameter = 279.4  # mm
                elif 'EDGEHD14' in telescope_str:
                    diameter = 355.6  # mm
                elif 'RASA8' in telescope_str:
                    diameter = 203.2  # mm
                elif 'RASA11' in telescope_str:
                    diameter = 279.4  # mm
                elif 'RASA14' in telescope_str:
                    diameter = 355.6  # mm
                elif 'STARIZON' in telescope_str:
                    diameter = 130.0  # mm
                elif 'STARIZON-130' in telescope_str:
                    diameter = 130.0  # mm
                elif 'STARIZON-150' in telescope_str:
                    diameter = 150.0  # mm
                elif 'STARIZON-180' in telescope_str:
                    diameter = 180.0  # mm
                # CFF (Classical Cassegrain)
                elif 'CFF160' in telescope_str:
                    diameter = 160.0  # mm
                elif 'CFF185' in telescope_str:
                    diameter = 185.0  # mm
                elif 'CFF200' in telescope_str:
                    diameter = 200.0  # mm
                elif 'CFF250' in telescope_str:
                    diameter = 250.0  # mm
                elif 'CFF300' in telescope_str:
                    diameter = 300.0  # mm
                elif 'CFF350' in telescope_str:
                    diameter = 350.0  # mm
                elif 'CFF400' in telescope_str:
                    diameter = 400.0  # mm
                elif 'CFF500' in telescope_str:
                    diameter = 500.0  # mm
                # TS-Optics Telescopes
                elif 'TS-APO65Q' in telescope_str or 'APO65Q' in telescope_str:
                    diameter = 65.0  # mm
                elif 'TS-APO80Q' in telescope_str or 'APO80Q' in telescope_str:
                    diameter = 80.0  # mm
                elif 'TS-APO102Q' in telescope_str or 'APO102Q' in telescope_str:
                    diameter = 102.0  # mm
                elif 'TS-APO115Q' in telescope_str or 'APO115Q' in telescope_str:
                    diameter = 115.0  # mm
                elif 'TS-APO130Q' in telescope_str or 'APO130Q' in telescope_str:
                    diameter = 130.0  # mm
                elif 'TS-APO140Q' in telescope_str or 'APO140Q' in telescope_str:
                    diameter = 140.0  # mm
                elif 'TS-APO150Q' in telescope_str or 'APO150Q' in telescope_str:
                    diameter = 150.0  # mm
                elif 'TS-APO160Q' in telescope_str or 'APO160Q' in telescope_str:
                    diameter = 160.0  # mm
                elif 'TS-APO180Q' in telescope_str or 'APO180Q' in telescope_str:
                    diameter = 180.0  # mm
                elif 'TS-APO200Q' in telescope_str or 'APO200Q' in telescope_str:
                    diameter = 200.0  # mm
                elif 'TS-APO250Q' in telescope_str or 'APO250Q' in telescope_str:
                    diameter = 250.0  # mm
                elif 'TS-APO300Q' in telescope_str or 'APO300Q' in telescope_str:
                    diameter = 300.0  # mm
                elif 'TS-APO350Q' in telescope_str or 'APO350Q' in telescope_str:
                    diameter = 350.0  # mm
                elif 'TS-APO400Q' in telescope_str or 'APO400Q' in telescope_str:
                    diameter = 400.0  # mm
                elif 'TS-APO500Q' in telescope_str or 'APO500Q' in telescope_str:
                    diameter = 500.0  # mm
                elif 'TS-APO600Q' in telescope_str or 'APO600Q' in telescope_str:
                    diameter = 600.0  # mm
                elif 'TS-APO700Q' in telescope_str or 'APO700Q' in telescope_str:
                    diameter = 700.0  # mm
                elif 'TS-APO800Q' in telescope_str or 'APO800Q' in telescope_str:
                    diameter = 800.0  # mm
                elif 'TS-APO900Q' in telescope_str or 'APO900Q' in telescope_str:
                    diameter = 900.0  # mm
                elif 'TS-APO1000Q' in telescope_str or 'APO1000Q' in telescope_str:
                    diameter = 1000.0  # mm
                # Askar Telescopes
                elif 'ASKAR-50PHQ' in telescope_str or '50PHQ' in telescope_str:
                    diameter = 50.0  # mm
                elif 'ASKAR-60PHQ' in telescope_str or '60PHQ' in telescope_str:
                    diameter = 60.0  # mm
                elif 'ASKAR-70PHQ' in telescope_str or '70PHQ' in telescope_str:
                    diameter = 70.0  # mm
                elif 'ASKAR-80PHQ' in telescope_str or '80PHQ' in telescope_str:
                    diameter = 80.0  # mm
                elif 'ASKAR-90PHQ' in telescope_str or '90PHQ' in telescope_str:
                    diameter = 90.0  # mm
                elif 'ASKAR-100PHQ' in telescope_str or '100PHQ' in telescope_str:
                    diameter = 100.0  # mm
                elif 'ASKAR-120PHQ' in telescope_str or '120PHQ' in telescope_str:
                    diameter = 120.0  # mm
                elif 'ASKAR-130PHQ' in telescope_str or '130PHQ' in telescope_str:
                    diameter = 130.0  # mm
                elif 'ASKAR-150PHQ' in telescope_str or '150PHQ' in telescope_str:
                    diameter = 150.0  # mm
                elif 'ASKAR-180PHQ' in telescope_str or '180PHQ' in telescope_str:
                    diameter = 180.0  # mm
                elif 'ASKAR-200PHQ' in telescope_str or '200PHQ' in telescope_str:
                    diameter = 200.0  # mm
                elif 'ASKAR-250PHQ' in telescope_str or '250PHQ' in telescope_str:
                    diameter = 250.0  # mm
                elif 'ASKAR-300PHQ' in telescope_str or '300PHQ' in telescope_str:
                    diameter = 300.0  # mm
                elif 'ASKAR-350PHQ' in telescope_str or '350PHQ' in telescope_str:
                    diameter = 350.0  # mm
                elif 'ASKAR-400PHQ' in telescope_str or '400PHQ' in telescope_str:
                    diameter = 400.0  # mm
                elif 'ASKAR-500PHQ' in telescope_str or '500PHQ' in telescope_str:
                    diameter = 500.0  # mm
                elif 'ASKAR-600PHQ' in telescope_str or '600PHQ' in telescope_str:
                    diameter = 600.0  # mm
                elif 'ASKAR-700PHQ' in telescope_str or '700PHQ' in telescope_str:
                    diameter = 700.0  # mm
                elif 'ASKAR-800PHQ' in telescope_str or '800PHQ' in telescope_str:
                    diameter = 800.0  # mm
                elif 'ASKAR-900PHQ' in telescope_str or '900PHQ' in telescope_str:
                    diameter = 900.0  # mm
                elif 'ASKAR-1000PHQ' in telescope_str or '1000PHQ' in telescope_str:
                    diameter = 1000.0  # mm
                # Askar FRA Series (f/5.5)
                elif 'ASKAR-FRA300' in telescope_str or 'FRA300' in telescope_str:
                    diameter = 300.0  # mm
                elif 'ASKAR-FRA400' in telescope_str or 'FRA400' in telescope_str:
                    diameter = 400.0  # mm
                elif 'ASKAR-FRA500' in telescope_str or 'FRA500' in telescope_str:
                    diameter = 500.0  # mm
                elif 'ASKAR-FRA600' in telescope_str or 'FRA600' in telescope_str:
                    diameter = 600.0  # mm
                elif 'ASKAR-FRA700' in telescope_str or 'FRA700' in telescope_str:
                    diameter = 700.0  # mm
                elif 'ASKAR-FRA800' in telescope_str or 'FRA800' in telescope_str:
                    diameter = 800.0  # mm
                else:
                    # Unrecognized telescope, use retrieval function
                    try:
                        telescope_characteristics = get_telescope_characteristics(telescope)
                        diameter = telescope_characteristics['diameter_mm']
                    except Exception as e:
                        print(f"Warning: Error getting telescope characteristics: {e}")
                        diameter = 200.0  # Default diameter
            
            # Extract focal length
            focal_length = None
            focal_keywords = ['FOCALLEN', 'FOCAL_LENGTH', 'FOCAL', 'FL']
            for keyword in focal_keywords:
                if keyword in header:
                    try:
                        focal_length = float(header[keyword])
                        break
                    except:
                        pass
            
            # If not found, try to deduce from telescope name
            if focal_length is None:
                telescope_str = str(telescope).upper()
                # Ritchey-Chrétien (RC)
                if 'RC8' in telescope_str:
                    focal_length = 1625.6  # mm (f/8)
                elif 'RC6' in telescope_str:
                    focal_length = 1219.2  # mm (f/8)
                elif 'RC10' in telescope_str:
                    focal_length = 2032.0  # mm (f/8)
                elif 'RC12' in telescope_str:
                    focal_length = 2438.4  # mm (f/8)
                elif 'RC14' in telescope_str:
                    focal_length = 2844.8  # mm (f/8)
                elif 'RC16' in telescope_str:
                    focal_length = 3251.2  # mm (f/8)
                # Takahashi
                elif 'FSQ-85' in telescope_str or 'FSQ85' in telescope_str or 'FSQ-85EDP' in telescope_str or 'FSQ85EDP' in telescope_str:
                    focal_length = 455.0  # mm (f/5.35)
                elif 'FSQ-106' in telescope_str or 'FSQ106' in telescope_str:
                    focal_length = 530.0  # mm (f/5)
                elif 'FSQ-130' in telescope_str or 'FSQ130' in telescope_str:
                    focal_length = 650.0  # mm (f/5)
                elif 'TOA-130' in telescope_str:
                    focal_length = 1000.0  # mm (f/7.7)
                elif 'TOA-150' in telescope_str:
                    focal_length = 1100.0  # mm (f/7.3)
                elif 'TOA-160' in telescope_str:
                    focal_length = 1200.0  # mm (f/7.5)
                elif 'TSA-102' in telescope_str:
                    focal_length = 816.0  # mm (f/8)
                elif 'TSA-120' in telescope_str:
                    focal_length = 900.0  # mm (f/7.5)
                elif 'Epsilon-130' in telescope_str:
                    focal_length = 430.0  # mm (f/3.3)
                elif 'Epsilon-160' in telescope_str:
                    focal_length = 530.0  # mm (f/3.3)
                elif 'Epsilon-180' in telescope_str:
                    focal_length = 600.0  # mm (f/3.3)
                # PlaneWave
                elif 'CDK12' in telescope_str:
                    focal_length = 2438.4  # mm (f/8)
                elif 'CDK14' in telescope_str:
                    focal_length = 2844.8  # mm (f/8)
                elif 'CDK16' in telescope_str:
                    focal_length = 3251.2  # mm (f/8)
                elif 'CDK17' in telescope_str:
                    focal_length = 3454.4  # mm (f/8)
                elif 'CDK20' in telescope_str:
                    focal_length = 4064.0  # mm (f/8)
                elif 'CDK24' in telescope_str:
                    focal_length = 4876.8  # mm (f/8)
                elif 'L-350' in telescope_str:
                    focal_length = 2450.0  # mm (f/7)
                elif 'L-500' in telescope_str:
                    focal_length = 3500.0  # mm (f/7)
                elif 'L-600' in telescope_str:
                    focal_length = 4200.0  # mm (f/7)
                # Celestron
                elif 'C8' in telescope_str:
                    focal_length = 2032.0  # mm (f/10)
                elif 'C9' in telescope_str or 'C9.25' in telescope_str:
                    focal_length = 2350.0  # mm (f/10)
                elif 'C11' in telescope_str:
                    focal_length = 2794.0  # mm (f/10)
                elif 'C14' in telescope_str:
                    focal_length = 3910.0  # mm (f/11)
                elif 'EDGEHD8' in telescope_str:
                    focal_length = 2032.0  # mm (f/10)
                elif 'EDGEHD9.25' in telescope_str:
                    focal_length = 2350.0  # mm (f/10)
                elif 'EDGEHD11' in telescope_str:
                    focal_length = 2794.0  # mm (f/10)
                elif 'EDGEHD14' in telescope_str:
                    focal_length = 3910.0  # mm (f/11)
                elif 'RASA8' in telescope_str:
                    focal_length = 400.0  # mm (f/2)
                elif 'RASA11' in telescope_str:
                    focal_length = 620.0  # mm (f/2.2)
                elif 'RASA14' in telescope_str:
                    focal_length = 780.0  # mm (f/2.2)
                elif 'STARIZON' in telescope_str:
                    focal_length = 650.0  # mm (f/5)
                elif 'STARIZON-130' in telescope_str:
                    focal_length = 650.0  # mm (f/5)
                elif 'STARIZON-150' in telescope_str:
                    focal_length = 750.0  # mm (f/5)
                elif 'STARIZON-180' in telescope_str:
                    focal_length = 900.0  # mm (f/5)
                # CFF (Classical Cassegrain)
                elif 'CFF160' in telescope_str:
                    focal_length = 1280.0  # mm (f/8)
                elif 'CFF185' in telescope_str:
                    focal_length = 1480.0  # mm (f/8)
                elif 'CFF200' in telescope_str:
                    focal_length = 1600.0  # mm (f/8)
                elif 'CFF250' in telescope_str:
                    focal_length = 2000.0  # mm (f/8)
                elif 'CFF300' in telescope_str:
                    focal_length = 2400.0  # mm (f/8)
                elif 'CFF350' in telescope_str:
                    focal_length = 2800.0  # mm (f/8)
                elif 'CFF400' in telescope_str:
                    focal_length = 3200.0  # mm (f/8)
                elif 'CFF500' in telescope_str:
                    focal_length = 4000.0  # mm (f/8)
                # TS-Optics Telescopes (f/6.5 for most)
                elif 'TS-APO65Q' in telescope_str or 'APO65Q' in telescope_str:
                    focal_length = 422.5  # mm (f/6.5)
                elif 'TS-APO80Q' in telescope_str or 'APO80Q' in telescope_str:
                    focal_length = 520.0  # mm (f/6.5)
                elif 'TS-APO102Q' in telescope_str or 'APO102Q' in telescope_str:
                    focal_length = 663.0  # mm (f/6.5)
                elif 'TS-APO115Q' in telescope_str or 'APO115Q' in telescope_str:
                    focal_length = 747.5  # mm (f/6.5)
                elif 'TS-APO130Q' in telescope_str or 'APO130Q' in telescope_str:
                    focal_length = 845.0  # mm (f/6.5)
                elif 'TS-APO140Q' in telescope_str or 'APO140Q' in telescope_str:
                    focal_length = 910.0  # mm (f/6.5)
                elif 'TS-APO150Q' in telescope_str or 'APO150Q' in telescope_str:
                    focal_length = 975.0  # mm (f/6.5)
                elif 'TS-APO160Q' in telescope_str or 'APO160Q' in telescope_str:
                    focal_length = 1040.0  # mm (f/6.5)
                elif 'TS-APO180Q' in telescope_str or 'APO180Q' in telescope_str:
                    focal_length = 1170.0  # mm (f/6.5)
                elif 'TS-APO200Q' in telescope_str or 'APO200Q' in telescope_str:
                    focal_length = 1300.0  # mm (f/6.5)
                elif 'TS-APO250Q' in telescope_str or 'APO250Q' in telescope_str:
                    focal_length = 1625.0  # mm (f/6.5)
                elif 'TS-APO300Q' in telescope_str or 'APO300Q' in telescope_str:
                    focal_length = 1950.0  # mm (f/6.5)
                elif 'TS-APO350Q' in telescope_str or 'APO350Q' in telescope_str:
                    focal_length = 2275.0  # mm (f/6.5)
                elif 'TS-APO400Q' in telescope_str or 'APO400Q' in telescope_str:
                    focal_length = 2600.0  # mm (f/6.5)
                elif 'TS-APO500Q' in telescope_str or 'APO500Q' in telescope_str:
                    focal_length = 3250.0  # mm (f/6.5)
                elif 'TS-APO600Q' in telescope_str or 'APO600Q' in telescope_str:
                    focal_length = 3900.0  # mm (f/6.5)
                elif 'TS-APO700Q' in telescope_str or 'APO700Q' in telescope_str:
                    focal_length = 4550.0  # mm (f/6.5)
                elif 'TS-APO800Q' in telescope_str or 'APO800Q' in telescope_str:
                    focal_length = 5200.0  # mm (f/6.5)
                elif 'TS-APO900Q' in telescope_str or 'APO900Q' in telescope_str:
                    focal_length = 5850.0  # mm (f/6.5)
                elif 'TS-APO1000Q' in telescope_str or 'APO1000Q' in telescope_str:
                    focal_length = 6500.0  # mm (f/6.5)
                # Askar Telescopes (f/5.6 for most)
                elif 'ASKAR-50PHQ' in telescope_str or '50PHQ' in telescope_str:
                    focal_length = 280.0  # mm (f/5.6)
                elif 'ASKAR-60PHQ' in telescope_str or '60PHQ' in telescope_str:
                    focal_length = 336.0  # mm (f/5.6)
                elif 'ASKAR-70PHQ' in telescope_str or '70PHQ' in telescope_str:
                    focal_length = 392.0  # mm (f/5.6)
                elif 'ASKAR-80PHQ' in telescope_str or '80PHQ' in telescope_str:
                    focal_length = 448.0  # mm (f/5.6)
                elif 'ASKAR-90PHQ' in telescope_str or '90PHQ' in telescope_str:
                    focal_length = 504.0  # mm (f/5.6)
                elif 'ASKAR-100PHQ' in telescope_str or '100PHQ' in telescope_str:
                    focal_length = 560.0  # mm (f/5.6)
                elif 'ASKAR-120PHQ' in telescope_str or '120PHQ' in telescope_str:
                    focal_length = 672.0  # mm (f/5.6)
                elif 'ASKAR-130PHQ' in telescope_str or '130PHQ' in telescope_str:
                    focal_length = 728.0  # mm (f/5.6)
                elif 'ASKAR-150PHQ' in telescope_str or '150PHQ' in telescope_str:
                    focal_length = 840.0  # mm (f/5.6)
                elif 'ASKAR-180PHQ' in telescope_str or '180PHQ' in telescope_str:
                    focal_length = 1008.0  # mm (f/5.6)
                elif 'ASKAR-200PHQ' in telescope_str or '200PHQ' in telescope_str:
                    focal_length = 1120.0  # mm (f/5.6)
                elif 'ASKAR-250PHQ' in telescope_str or '250PHQ' in telescope_str:
                    focal_length = 1400.0  # mm (f/5.6)
                elif 'ASKAR-300PHQ' in telescope_str or '300PHQ' in telescope_str:
                    focal_length = 1680.0  # mm (f/5.6)
                elif 'ASKAR-350PHQ' in telescope_str or '350PHQ' in telescope_str:
                    focal_length = 1960.0  # mm (f/5.6)
                elif 'ASKAR-400PHQ' in telescope_str or '400PHQ' in telescope_str:
                    focal_length = 2240.0  # mm (f/5.6)
                elif 'ASKAR-500PHQ' in telescope_str or '500PHQ' in telescope_str:
                    focal_length = 2800.0  # mm (f/5.6)
                elif 'ASKAR-600PHQ' in telescope_str or '600PHQ' in telescope_str:
                    focal_length = 3360.0  # mm (f/5.6)
                elif 'ASKAR-700PHQ' in telescope_str or '700PHQ' in telescope_str:
                    focal_length = 3920.0  # mm (f/5.6)
                elif 'ASKAR-800PHQ' in telescope_str or '800PHQ' in telescope_str:
                    focal_length = 4480.0  # mm (f/5.6)
                elif 'ASKAR-900PHQ' in telescope_str or '900PHQ' in telescope_str:
                    focal_length = 5040.0  # mm (f/5.6)
                elif 'ASKAR-1000PHQ' in telescope_str or '1000PHQ' in telescope_str:
                    focal_length = 5600.0  # mm (f/5.6)
                # Askar FRA Series (f/5.5)
                elif 'ASKAR-FRA300' in telescope_str or 'FRA300' in telescope_str:
                    focal_length = 1650.0  # mm (f/5.5)
                elif 'ASKAR-FRA400' in telescope_str or 'FRA400' in telescope_str:
                    focal_length = 2200.0  # mm (f/5.5)
                elif 'ASKAR-FRA500' in telescope_str or 'FRA500' in telescope_str:
                    focal_length = 2750.0  # mm (f/5.5)
                elif 'ASKAR-FRA600' in telescope_str or 'FRA600' in telescope_str:
                    focal_length = 3300.0  # mm (f/5.5)
                elif 'ASKAR-FRA700' in telescope_str or 'FRA700' in telescope_str:
                    focal_length = 3850.0  # mm (f/5.5)
                elif 'ASKAR-FRA800' in telescope_str or 'FRA800' in telescope_str:
                    focal_length = 4400.0  # mm (f/5.5)
                else:
                    # Focal length not recognized, use telescope if already requested or ask
                    if 'telescope_characteristics' in locals():
                        focal_length = telescope_characteristics['focal_length_mm']
                    else:
                        telescope_characteristics = get_telescope_characteristics(telescope)
                        focal_length = telescope_characteristics['focal_length_mm']
            
            # Calculate aperture (f-number)
            f_number = focal_length / diameter if diameter and diameter > 0 else 8.0
            
            # Additional information
            additional_info = {
                'date_obs': header.get('DATE-OBS', header.get('DATE', 'Unknown')),
                'instrument': instrument,
                'telescope': telescope,
                'diameter_mm': diameter,
                'focal_length_mm': focal_length,
                'f_number': f_number,
                'gain': header.get('GAIN', None),
                'temperature': header.get('CCD-TEMP', None),
                'binning': f"{header.get('XBINNING', 1)}x{header.get('YBINNING', 1)}",
                'pixel_scale': header.get('PIXSCALE', None)
            }
            
            # Calculate photons from ADU data (intelligent sampling)
            photons_info = None
            advanced_snr_info = None
            
            if image_type == 'LIGHT' and filter_found and ADU_ANALYSIS_ENABLED and should_analyze_adu:
                # Automatically detect sensor from FITS header
                detected_sensor = detect_sensor_from_fits_header(file_path)
                
                # Use detected sensor or instrument one
                sensor_name = detected_sensor if detected_sensor else instrument
                
                # Get sensor characteristics
                try:
                    sensor_characteristics = get_sensor_characteristics(sensor_name)
                except Exception as e:
                    print(f"Warning: Error getting sensor characteristics: {e}")
                    sensor_characteristics = SENSORS_DATABASE['default']
                
                # Search for calibration files
                bias_files, dark_files = find_calibration_files(BIAS_DARK_PATH, exposure_time, sensor_characteristics.get('gain'), sensor_name)
                bias_path = bias_files[0] if bias_files else None
                dark_path = dark_files[0] if dark_files else None
                
                # Advanced SNR calculation with calibration
                advanced_snr_info = calculate_advanced_snr(file_path, sensor_characteristics, dark_path, bias_path)
                
                # Photon analysis disabled - using theoretical calculation only
            elif image_type == 'LIGHT' and filter_found and not ADU_ANALYSIS_ENABLED:
                # Fast mode: no ADU calculation, no repetitive message
                pass
            elif image_type == 'LIGHT' and filter_found and ADU_ANALYSIS_ENABLED and not should_analyze_adu:
                # Fast mode: no ADU calculation, no repetitive message
                pass
            else:
                # Skip files (fast mode)
                pass
            
            # Extract observation date
            observation_date = extract_observation_date(file_path, additional_info)
            
            return {
                'type': image_type,
                'filter': filter_found,
                'exposure_time': exposure_time,
                'target': target_found,
                'ra': ra,
                'dec': dec,
                'observation_date': observation_date,
                'info': additional_info,
                'adu_photons': photons_info,
                'advanced_snr': advanced_snr_info
            }
            
    except Exception as e:
        # Handle specific FITS file issues
        if "Header missing END card" in str(e):
            print(f"⚠️  Skipping {file_path.name}: Corrupted FITS header (missing END card)")
        elif "non-ASCII characters" in str(e):
            print(f"⚠️  Skipping {file_path.name}: Non-ASCII characters in header")
        elif "null bytes" in str(e):
            print(f"⚠️  Skipping {file_path.name}: Non-compliant FITS header (null bytes)")
        else:
            print(f"Error reading {file_path.name}: {e}")
        return None

def process_file_info(info, file, data_by_target, global_data, is_adu_sample=False):
    """
    Processes FITS file information and adds it to collected data
    """
    if not info or not info['exposure_time'] or info['type'] in ['FLAT', 'BIAS', 'DARK'] or 'FLATWIZARD' in str(info['info']['instrument']).upper():
        # Silently skip files with missing information or calibration files
        return
    
    # Ensure filter is set (use appropriate default based on context)
    if not info['filter'] or info['filter'] == 'Unknown':
        # Check if this is a color camera file (Bayer matrix detection)
        if info.get('is_color', False):
            info['filter'] = 'RGB'  # One Shot Color
        else:
            # Check if this is likely a color camera file based on filename/path
            filename_lower = str(file).lower()
            osc_indicators = ['color', 'colour', 'osc', 'one shot', 'oneshot', 'rgb', 'camera', 'cam']
            if any(indicator in filename_lower for indicator in osc_indicators):
                info['filter'] = 'RGB'  # One Shot Color
            else:
                info['filter'] = 'L'  # Default to Luminance for monochrome cameras
    
    # Determine target with case-insensitive normalization
    target = info['target'] or file.parent.name
    if not target or target == 'Unknown':
        target = file.parent.name
    
    # Normalize target name (case-insensitive, remove extra spaces)
    target = normalize_target_name(target)
    
    # Add to target data (check for duplicates)
    file_data = {
        'name': file.name,
        'path': str(file),
        'info': info
    }
    
    # Check if file already exists to avoid duplication
    existing_files = [f.get('name', '') for f in data_by_target[target]['files']]
    if file.name not in existing_files:
        data_by_target[target]['files'].append(file_data)
    else:
        return  # Skip processing this duplicate file
    
    # Group by observation date
    obs_date = info['observation_date']
    if 'files_by_date' not in data_by_target[target]:
        data_by_target[target]['files_by_date'] = {}
    
    if obs_date not in data_by_target[target]['files_by_date']:
        data_by_target[target]['files_by_date'][obs_date] = {
            'files': [],
            'time_by_filter': {},
            'exposure_details': {},  # New: detailed exposure times per filter
            'total_time': 0
        }
    
    # Add file to date group (only for LIGHT files)
    if info['type'] == 'LIGHT':
        # Don't add file_data to files_by_date['files'] to avoid duplication
        # The file is already in data_by_target[target]['files']
        
        # Add to time_by_filter for this date
        filter_name = info['filter']
        if filter_name not in data_by_target[target]['files_by_date'][obs_date]['time_by_filter']:
            data_by_target[target]['files_by_date'][obs_date]['time_by_filter'][filter_name] = []
        
        data_by_target[target]['files_by_date'][obs_date]['time_by_filter'][filter_name].append(info['exposure_time'])
        data_by_target[target]['files_by_date'][obs_date]['total_time'] += info['exposure_time']
        
        
        # Store detailed exposure information
        if filter_name not in data_by_target[target]['files_by_date'][obs_date]['exposure_details']:
            data_by_target[target]['files_by_date'][obs_date]['exposure_details'][filter_name] = {}
        
        exposure_time = info['exposure_time']
        if exposure_time not in data_by_target[target]['files_by_date'][obs_date]['exposure_details'][filter_name]:
            data_by_target[target]['files_by_date'][obs_date]['exposure_details'][filter_name][exposure_time] = 0
        
        data_by_target[target]['files_by_date'][obs_date]['exposure_details'][filter_name][exposure_time] += 1
    
    # Only include LIGHT files in statistics (exclude bias, dark, flat)
    if info['type'] == 'LIGHT':
        # Add to global time_by_filter for graph generation
        filter_name = info['filter']
        if filter_name not in data_by_target[target]['time_by_filter']:
            data_by_target[target]['time_by_filter'][filter_name] = []
        data_by_target[target]['time_by_filter'][filter_name].append(info['exposure_time'])
        
        if info['info']['instrument'] != 'Unknown':
            data_by_target[target]['instruments'].add(info['info']['instrument'])
        if info['info']['telescope'] != 'Unknown':
            data_by_target[target]['telescopes'].add(info['info']['telescope'])
        
        data_by_target[target]['dates'].add(info['info']['date_obs'])
        data_by_target[target]['apertures'].add(info['info']['f_number'])
        data_by_target[target]['diameters'].add(info['info']['diameter_mm'])
        data_by_target[target]['focal_lengths'].add(info['info']['focal_length_mm'])
        
        if info['ra'] and info['dec']:
            data_by_target[target]['coordinates'].append((info['ra'], info['dec']))
    
    # Calculate received light for LIGHT images
    if info['type'] == 'LIGHT' and info['filter'] in FILTERS_INFO:
        # Collect ADU samples for statistical analysis
        if is_adu_sample and info.get('adu_photons') and info['adu_photons']:
            data_by_target[target]['adu_samples'][info['filter']].append({
                'file': file.name,
                'adu_photons': info['adu_photons']['total_photons'],
                'exposure_time': info['exposure_time'],
                'adu_stats': info['adu_photons']
            })
            data_by_target[target]['adu_counter_by_filter'][info['filter']] += 1
            print(f"   ADU sample collected: {info['adu_photons']['total_photons']:.2e} photons")
            print(f"   Sample {data_by_target[target]['adu_counter_by_filter'][info['filter']]}/{ADU_SAMPLE_PER_FILTER} for filter {info['filter']}")
        
        # Light calculation removed (photon analysis disabled)
        # Store basic exposure information instead
        light = {
            'exposure_time': info['exposure_time'],
            'filter': info['filter'],
            'diameter_mm': info['info'].get('diameter_mm', 200.0),
            'source': 'basic_info'
        }
        
        data_by_target[target]['received_light'][info['filter']].append(light)
    
    # Update global data
    global_data['found_targets'].add(target)
    global_data['used_instruments'].add(info['info']['instrument'])
    global_data['used_telescopes'].add(info['info']['telescope'])
    global_data['total_time'] += info['exposure_time']
    
    if info['ra'] and info['dec']:
        global_data['sky_regions'].append((info['ra'], info['dec']))
    
    print(f"✅ {target} - {file.name}")
    print(f"   📷 {info['type']} | 🎨 {info['filter']} | ⏱️  {format_time(info['exposure_time'])}")
    print(f"   🔧 {info['info']['instrument']} | 🔭 {info['info']['telescope']} | 📏 f/{info['info']['f_number']:.1f}")

def _process_file_phase1(file):
    """Process single file for Phase 1 (parallelizable)"""
    try:
        basic_info = extract_fits_header_info_fast(file)
        if basic_info and basic_info['exposure_time'] and basic_info['type'] != 'FLAT' and 'FLATWIZARD' not in str(basic_info['info']['instrument']).upper():
            # Ensure filter is set
            if not basic_info['filter'] or basic_info['filter'] == 'Unknown':
                if basic_info.get('is_color', False):
                    basic_info['filter'] = 'OSC'
                else:
                    # Check filename for OSC indicators
                    filename_lower = str(file).lower()
                    osc_indicators = ['color', 'colour', 'osc', 'one shot', 'oneshot', 'rgb', 'camera', 'cam']
                    if any(indicator in filename_lower for indicator in osc_indicators):
                        basic_info['filter'] = 'OSC'
                    else:
                        basic_info['filter'] = 'L'
            
            # Get target name
            target = basic_info['target'] or file.parent.name
            if not target or target == 'Unknown':
                target = file.parent.name
            
            # Normalize target name
            target = normalize_target_name(target)
            
            return {
                'target': target,
                'filter': basic_info['filter'],
                'file': file,
                'info': basic_info
            }
    except Exception as e:
        return None
    return None

def _process_file(task):
    """Process single file for Phase 2 (parallelizable)"""
    file_str, do_adu = task
    try:
        p = Path(file_str)
        # Remove debug print to avoid interfering with tqdm
        info = extract_fits_header_info(p, should_analyze_adu=do_adu)
        return (file_str, info, do_adu)
    except Exception as e:
        # Handle specific FITS file issues silently to avoid spam
        if "Header missing END card" in str(e) or "non-ASCII characters" in str(e) or "null bytes" in str(e):
            # Skip problematic files silently
            return (file_str, None, do_adu)
        else:
            # For other errors, return None to indicate failure
            return (file_str, None, do_adu)

def analyze_folder_recursive(root_folder, workers=1):
    """Recursively analyzes all subfolders (parallelizable via workers)"""
    global ADU_ANALYSIS_ENABLED, ADU_SAMPLE_PER_FILTER, FAST_ANALYSIS
    
    # Handle None workers (fallback to auto-detection)
    if workers is None:
        import multiprocessing
        try:
            workers = multiprocessing.cpu_count()
            if workers <= 0:
                workers = 1  # Fallback for invalid CPU count
        except (OSError, NotImplementedError):
            # Fallback for systems where CPU count detection fails
            workers = 1
    
    
    if not ASTROPY_AVAILABLE:
        print("ERROR: Astropy is not installed. Cannot continue.")
        return None
    
    folder_path = Path(root_folder)
    if not folder_path.exists():
        print(f"ERROR: Folder '{root_folder}' does not exist.")
        return None
    
    # Data structure to organize results
    data_by_target = defaultdict(lambda: {
        'files': [],
        'time_by_filter': defaultdict(list),
        'time_by_type': defaultdict(list),
        'instruments': set(),
        'telescopes': set(),
        'dates': set(),
        'coordinates': [],
        'apertures': set(),
        'diameters': set(),
        'focal_lengths': set(),
        'received_light': defaultdict(list),
        'adu_samples': defaultdict(list),  # To store ADU samples by filter
        'adu_counter_by_filter': defaultdict(int)  # To count ADU samples by filter
    })
    
    global_data = {
        'total_files': 0,
        'found_targets': set(),
        'used_instruments': set(),
        'used_telescopes': set(),
        'sky_regions': [],
        'total_time': 0,
        'total_light': 0
    }
    
    # Recursive traversal (case-insensitive on Windows; avoid duplicates)
    # Use pathlib for better cross-platform compatibility
    fits_files = list(folder_path.rglob("*.fit")) + list(folder_path.rglob("*.fits"))
    # Deduplicate by lowercase absolute path to avoid double-processing
    _seen_paths = set()
    _unique_files = []
    for _p in fits_files:
        _key = str(_p).lower()
        if _key not in _seen_paths:
            _seen_paths.add(_key)
            _unique_files.append(_p)
    fits_files = _unique_files
    
    print(f"COMPLETE ASTROPHOTOGRAPHY ANALYSIS")
    print("=" * 80)
    print(f"Root folder: {root_folder}")
    print(f"FITS files found: {len(fits_files)}")
    print("-" * 80)
    # Display analysis mode once
    if not ADU_ANALYSIS_ENABLED:
        pass
    elif ADU_SAMPLE_PER_FILTER == float('inf'):
        print("COMPLETE mode: Fast analysis of all files")
        print("Photon calculation: Theoretical (fast mode)")
    else:
        print(f"PARTIAL mode: ADU analysis of {ADU_SAMPLE_PER_FILTER} samples per filter")
        print("Analysis: Basic exposure information only (photon calculation disabled)")
        print("Optimizations: 1% pixel sampling")
    print("-" * 80)
    
    # First pass: collect all valid files and organize by target/filter
    files_by_target_filter = defaultdict(lambda: defaultdict(list))
    
    print("📋 Phase 1: Collecting valid files...")
    
    # Add progress bar for Phase 1
    if TQDM_AVAILABLE:
        fits_files_progress = create_enhanced_progress_bar(
            fits_files, 
            total=len(fits_files), 
            desc="📋 Collecting valid files",
            unit="file"
        )
    else:
        fits_files_progress = fits_files
    
    # Process files in batches to reduce memory pressure
    # Adaptive batch size based on system resources
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        memory_available = True
    except (ImportError, OSError, AttributeError):
        # Fallback: estimate memory based on workers and platform
        import platform
        system = platform.system().lower()
        if system == "windows":
            memory_gb = (workers * 2) if workers else 4  # Conservative estimate for Windows
        elif system == "darwin":  # macOS
            memory_gb = (workers * 4) if workers else 8  # macOS typically has more memory
        else:  # Linux and others
            memory_gb = (workers * 3) if workers else 6  # Linux estimate
        memory_available = False
    
    cpu_count = workers if workers else 1
    
    if memory_gb >= 32 and cpu_count >= 16:
        batch_size = 500  # High-end systems
    elif memory_gb >= 16 and cpu_count >= 8:
        batch_size = 300  # Mid-range systems
    elif memory_gb >= 8 and cpu_count >= 4:
        batch_size = 200  # Entry-level systems
    else:
        batch_size = 100  # Low-end systems
    
    processed_count = 0
    
    # Use parallel processing for Phase 1 if workers > 1
    if workers and workers > 1:
        print(f"🧵 Phase 1: Parallel processing with {workers} workers")
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = (ex.submit(_process_file_phase1, file) for file in fits_files)
            
            if TQDM_AVAILABLE:
                results = create_enhanced_progress_bar(
                    (f.result() for f in as_completed(list(futures))), 
                    total=len(fits_files), 
                    desc="📋 Collecting valid files"
                )
            else:
                results = (f.result() for f in as_completed(list(futures)))
            
            for result in results:
                if result:
                    files_by_target_filter[result['target']][result['filter']].append(result['file'])
                    global_data['total_files'] += 1
                    processed_count += 1
                    
                    # Periodic memory cleanup
                    if processed_count % batch_size == 0:
                        import gc
                        gc.collect()
    else:
        # Sequential processing (original method)
        for file in fits_files_progress:
            global_data['total_files'] += 1
            processed_count += 1
            
            # Extract basic info to identify target and filter (optimized for Phase 1)
            basic_info = extract_fits_header_info_fast(file)
            if basic_info and basic_info['exposure_time'] and basic_info['type'] != 'FLAT' and 'FLATWIZARD' not in str(basic_info['info']['instrument']).upper():
                # Ensure filter is set (use appropriate default based on context)
                if not basic_info['filter'] or basic_info['filter'] == 'Unknown':
                    # Check if this is a color camera file (Bayer matrix detection)
                    if basic_info.get('is_color', False):
                        basic_info['filter'] = 'OSC'  # One Shot Color
                    else:
                        # Check if this is likely a color camera file based on filename/path
                        filename_lower = str(file).lower()
                        osc_indicators = ['color', 'colour', 'osc', 'one shot', 'oneshot', 'rgb', 'camera', 'cam']
                        if any(indicator in filename_lower for indicator in osc_indicators):
                            basic_info['filter'] = 'OSC'  # One Shot Color
                        else:
                            basic_info['filter'] = 'L'  # Default to Luminance for monochrome cameras
                target = basic_info['target'] or file.parent.name
                if not target or target == 'Unknown':
                    target = file.parent.name
                
                # Normalize target name (case-insensitive, remove extra spaces)
                target = normalize_target_name(target)
                
                # Store file for this filter of this target
                files_by_target_filter[target][basic_info['filter']].append(file)
            
            # Periodic memory cleanup to prevent slowdown
            if processed_count % batch_size == 0:
                import gc
                gc.collect()
    
    # Second pass: file processing

    def _consume_results(results_iter):
        for file_str, info, do_adu in results_iter:
            if info:
                process_file_info(info, Path(file_str), data_by_target, global_data, is_adu_sample=bool(do_adu))

    if ADU_ANALYSIS_ENABLED:
        print("🎲 Phase 2: Random selection and analysis...")
        import random

        tasks = []
        for target, filters in files_by_target_filter.items():
            for filter_name, files in filters.items():
                if ADU_SAMPLE_PER_FILTER != float('inf'):
                    nb_files = min(ADU_SAMPLE_PER_FILTER, len(files))
                    adu_files = random.sample(files, nb_files) if nb_files > 0 else []
                    non_adu_files = [f for f in files if f not in adu_files]
                    print(f"   🎲 {target} - {filter_name}: {len(files)} files, {len(adu_files)} randomly selected for ADU")
                else:
                    adu_files = files
                    non_adu_files = []
                    print(f"   🎲 {target} - {filter_name}: All {len(files)} files selected for ADU")

                tasks.extend([(str(f), True) for f in adu_files])
                tasks.extend([(str(f), False) for f in non_adu_files])
        
        adu_tasks = sum(1 for _, do_adu in tasks if do_adu)
        non_adu_tasks = sum(1 for _, do_adu in tasks if not do_adu)
        print(f"   📊 Analysis breakdown:")
        print(f"      ⚡ Fast processing: {len(tasks)} files")
        print(f"      📁 Total: {len(tasks)} files")
        
        # Print detailed progress information
        print_progress_info(adu_tasks, non_adu_tasks, len(tasks))

        if workers and workers > 1:
            print(f"🧵 Parallel execution with {workers} workers")
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = (ex.submit(_process_file, t) for t in tasks)
            if TQDM_AVAILABLE:
                progress_bar = create_enhanced_progress_bar(
                    (f.result() for f in as_completed(list(futures))), 
                    total=len(tasks), 
                    desc=f"🔄 Processing {len(tasks)} files"
                )
                _consume_results(progress_bar)
            else:
                _consume_results((f.result() for f in as_completed(list(futures))))
        else:
            print("🧵 Sequential execution (1 worker)")
            if TQDM_AVAILABLE:
                progress_bar = create_enhanced_progress_bar(
                    (_process_file(t) for t in tasks), 
                    total=len(tasks), 
                    desc=f"🔄 Processing {len(tasks)} files"
                )
                _consume_results(progress_bar)
            else:
                _consume_results(_process_file(t) for t in tasks)
    else:
        print("⚡ Phase 2: Fast analysis (no ADU)...")
        tasks = [(str(f), False) for f in fits_files]
        
        # Print progress information for fast mode
        print_progress_info(0, len(tasks), len(tasks))
        
        if workers and workers > 1:
            print(f"🧵 Parallel execution with {workers} workers")
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = (ex.submit(_process_file, t) for t in tasks)
                if TQDM_AVAILABLE:
                    _consume_results(create_enhanced_progress_bar(
                        (f.result() for f in as_completed(list(futures))), 
                        total=len(tasks), 
                        desc="⚡ Fast processing files"
                    ))
                else:
                    _consume_results((f.result() for f in as_completed(list(futures))))
        else:
            print("🧵 Sequential execution (1 worker)")
            if TQDM_AVAILABLE:
                _consume_results(create_enhanced_progress_bar(
                    (_process_file(t) for t in tasks), 
                    total=len(tasks), 
                    desc="⚡ Fast processing files"
                ))
            else:
                _consume_results(_process_file(t) for t in tasks)
    
    # Print completion message
    print_progress_completion()
    
    # Third pass: ADU statistical analysis and extrapolation
    if ADU_ANALYSIS_ENABLED and any(data_by_target[target]['adu_samples'] for target in data_by_target):
        print("📊 Phase 3: ADU statistical analysis and extrapolation...")
        data_by_target = calculate_adu_statistics_by_filter(data_by_target)
    
    # Convert sets to lists for JSON serialization
    for target_data in data_by_target.values():
        target_data['instruments'] = list(target_data['instruments'])
        target_data['telescopes'] = list(target_data['telescopes'])
        target_data['dates'] = list(target_data['dates'])
        target_data['apertures'] = list(target_data['apertures'])
        target_data['diameters'] = list(target_data['diameters'])
        target_data['focal_lengths'] = list(target_data['focal_lengths'])
    
    global_data['found_targets'] = list(global_data['found_targets'])
    global_data['used_instruments'] = list(global_data['used_instruments'])
    global_data['used_telescopes'] = list(global_data['used_telescopes'])
    
    return data_by_target, global_data

def analyze_fits_files(folder_path, output_folder, num_workers=1):
    """Analyzes all FITS files in the folder and subfolders"""
    print(f"\n🔍 ANALYZING FITS FILES")
    print("=" * 80)
    print(f"📁 Folder: {folder_path}")
    print(f"👥 Workers: {num_workers}")
    
    # Find all FITS files recursively
    fits_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.fits', '.fit')):
                fits_files.append(os.path.join(root, file))
    
    if not fits_files:
        print("❌ No FITS files found")
        return {}, {}
    
    print(f"📊 Found {len(fits_files)} FITS files")
    
    # Initialize data structures
    data_by_target = {}
    global_data = {
        'total_files': 0,
        'found_targets': set(),
        'used_instruments': set(),
        'used_telescopes': set(),
        'total_time': 0
    }
    
    # Count files by type
    file_type_counts = {'LIGHT': 0, 'FLAT': 0, 'DARK': 0, 'BIAS': 0, 'OTHER': 0}
    
    # Process files
    print(f"\n📋 PROCESSING FILES")
    print("-" * 60)
    
    for i, file_path in enumerate(fits_files, 1):
        print(f"📄 [{i}/{len(fits_files)}] {os.path.basename(file_path)}")
        
        try:
            # Extract header information
            header_info = analyze_fits_header(file_path)
            
            if not header_info:
                print(f"   ⚠️  Could not read header")
                continue
            
            # Check image type - only process LIGHT files
            image_type = header_info.get('image_type', 'LIGHT')
            
            # Count file types
            if image_type in file_type_counts:
                file_type_counts[image_type] += 1
            else:
                file_type_counts['OTHER'] += 1
            
            if image_type not in ['LIGHT', 'SCIENCE', 'OBJECT']:
                print(f"   ⏭️  Skipping {image_type} file")
                continue
            
            # Extract target name
            target = header_info.get('object', 'Unknown')
            if not target or target == 'Unknown':
                # Try to extract from filename
                filename = os.path.basename(file_path)
                target = filename.split('_')[0] if '_' in filename else 'Unknown'
            
            # Extract filter
            filter_name = header_info.get('filter')
            if not filter_name:
                filter_name, _ = extract_filter_from_filename(file_path)
                if not filter_name:
                    # Check if this is a color camera file (Bayer matrix detection)
                    if header_info.get('is_color', False):
                        filter_name = 'RGB'  # One Shot Color
                    else:
                        # Check if this is likely a color camera file based on filename/path
                        filename_lower = str(file_path).lower()
                        osc_indicators = ['color', 'colour', 'osc', 'one shot', 'oneshot', 'rgb', 'camera', 'cam']
                        if any(indicator in filename_lower for indicator in osc_indicators):
                            filter_name = 'RGB'  # One Shot Color
                        else:
                            filter_name = 'L'  # Default to Luminance for monochrome cameras
            
            # Extract exposure time
            exposure_time = header_info.get('exposure_time', 0)
            
            # Extract instrument and telescope
            instrument = header_info.get('instrument', 'Unknown')
            telescope = header_info.get('telescope', 'Unknown')
            
            # Initialize target data if not exists
            if target not in data_by_target:
                data_by_target[target] = {
                    'files': [],
                    'time_by_filter': defaultdict(list),
                    'received_light': defaultdict(list),
                    'adu_samples': defaultdict(list),
                    'instruments': set(),
                    'telescopes': set(),
                    'dates': [],
                    'apertures': [],
                    'diameters': [],
                    'focal_lengths': []
                }
            
            # Add file data
            data_by_target[target]['files'].append(file_path)
            data_by_target[target]['time_by_filter'][filter_name].append(exposure_time)
            data_by_target[target]['instruments'].add(instrument)
            data_by_target[target]['telescopes'].add(telescope)
            
            # Extract date
            date_obs = header_info.get('date_obs', '')
            if date_obs:
                data_by_target[target]['dates'].append(date_obs)
            
            # Extract telescope characteristics
            diameter = header_info.get('diameter')
            if diameter:
                print(f"   ✅ Diameter detected: {diameter}mm")
                data_by_target[target]['diameters'].append(diameter)
                data_by_target[target]['apertures'].append(diameter / 1000)  # Convert to meters
                data_by_target[target]['focal_lengths'].append(diameter * 8)  # Assume f/8
            else:
                print(f"   ⚠️  No diameter detected in header")
                
                # Try to get telescope characteristics from database using telescope name
                telescope_name = header_info.get('telescope', '')
                if telescope_name and telescope_name != 'Unknown':
                    print(f"   🔍 Looking up telescope '{telescope_name}' in database...")
                    telescope_characteristics = get_telescope_characteristics(telescope_name)
                    
                    if telescope_characteristics and telescope_characteristics != TELESCOPES_DATABASE['default']:
                        diameter = telescope_characteristics['diameter_mm']
                        focal_length = telescope_characteristics['focal_length_mm']
                        f_number = telescope_characteristics['f_number']
                        
                        print(f"   ✅ Found in database: {diameter}mm, {focal_length}mm, f/{f_number}")
                        
                        data_by_target[target]['diameters'].append(diameter)
                        data_by_target[target]['apertures'].append(f_number)
                        data_by_target[target]['focal_lengths'].append(focal_length)
                    else:
                        print(f"   ⚠️  Telescope '{telescope_name}' not found in database")
                else:
                    print(f"   ⚠️  No telescope name available for database lookup")
            
            # Calculate theoretical light quantity
            # Light calculation removed (photon analysis disabled)
            # Store basic exposure information instead
            light_data = {
                'exposure_time': exposure_time,
                'filter': filter_name,
                'diameter_mm': diameter if diameter else 200.0,
                'source': 'basic_info'
            }
            
            data_by_target[target]['received_light'][filter_name].append(light_data)
            print(f"   ✅ Basic exposure info stored: {exposure_time}s exposure, {filter_name} filter")
            
            # Update global data
            global_data['total_files'] += 1
            global_data['found_targets'].add(target)
            global_data['used_instruments'].add(instrument)
            global_data['used_telescopes'].add(telescope)
            global_data['total_time'] += exposure_time
            
            print(f"   ✅ Target: {target}, Filter: {filter_name}, Time: {format_time(exposure_time)}")
            
        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            continue
    
    # Convert sets to lists for JSON serialization
    for target_data in data_by_target.values():
        target_data['instruments'] = list(target_data['instruments'])
        target_data['telescopes'] = list(target_data['telescopes'])
    
    global_data['found_targets'] = list(global_data['found_targets'])
    global_data['used_instruments'] = list(global_data['used_instruments'])
    global_data['used_telescopes'] = list(global_data['used_telescopes'])
    
    print(f"\n✅ Analysis completed!")
    print(f"📊 Targets found: {len(data_by_target)}")
    print(f"📊 Total LIGHT files processed: {global_data['total_files']}")
    print(f"⏰ Total observation time: {format_time(global_data['total_time'])}")
    
    # Display file type statistics
    print(f"\n📋 FILE TYPE STATISTICS:")
    print(f"   🔆 LIGHT files: {file_type_counts['LIGHT']}")
    print(f"   🔧 FLAT files: {file_type_counts['FLAT']}")
    print(f"   🌑 DARK files: {file_type_counts['DARK']}")
    print(f"   ⚡ BIAS files: {file_type_counts['BIAS']}")
    if file_type_counts['OTHER'] > 0:
        print(f"   ❓ OTHER files: {file_type_counts['OTHER']}")
    
    return data_by_target, global_data

def is_calibration_target(target_name):
    """Check if target name indicates calibration files"""
    target_upper = target_name.upper()
    calibration_keywords = ['BIAS', 'DARK', 'FLAT', 'CALIBRATION', 'CAL']
    return any(keyword in target_upper for keyword in calibration_keywords)

def display_target_statistics(data_by_target):
    """Displays detailed statistics by target"""
    # Defensive initializations to avoid NameError if plotting context is missing
    try:
        import matplotlib.pyplot as plt  # Ensure matplotlib is available in this scope
    except Exception:
        plt = None
    if 'fig' not in globals():
        try:
            fig = plt.figure(figsize=(16, 9)) if plt else None
        except Exception:
            fig = None
        globals()['fig'] = fig
    if 'ax3' not in globals():
        globals()['ax3'] = None
    if 'top_targets' not in globals():
        globals()['top_targets'] = []
    if 'filter_times' not in globals():
        globals()['filter_times'] = {}
    print(f"\nDETAILED STATISTICS")
    print("=" * 100)
    
    # Sort targets alphabetically with improved astronomical object handling
    sorted_targets = sorted(data_by_target.items(), key=lambda x: get_astronomical_sort_key(x[0]))
    
    for target, data in sorted_targets:
        if not data['files']:
            continue
        
        # Skip calibration targets
        if is_calibration_target(target):
            continue
            
        print(f"\n{target}")
        print("-" * 80)
        
        # General statistics
        total_files = len(data['files'])
        
        # Calculate total time from files_by_date to avoid duplication
        total_time = 0
        if 'files_by_date' in data:
            for date_data in data['files_by_date'].values():
                total_time += date_data['total_time']
        else:
            # Fallback to time_by_filter if files_by_date not available
            total_time = sum(sum(times) for times in data['time_by_filter'].values())
        
        print(f"📊 Files: {total_files}")
        print(f"⏰ Total time: {format_time(total_time)} ({total_time/3600:.2f}h)")
        print(f"🔭 Telescopes: {', '.join(data['telescopes'])}")
        
        # Display mosaic panel information if this is a mosaic
        if 'panels' in data and data['panels']:
            print(f"🧩 Mosaic panels: {len(data['panels'])}")
            for panel_num, panel_info in sorted(data['panels'].items(), key=lambda x: int(x[0])):
                panel_files = len(panel_info['files'])
                panel_time = panel_info['total_time']
                print(f"   Panel {panel_num}: {panel_files} files, {format_time(panel_time)} ({panel_time/3600:.2f}h)")
                print(f"   Original name: {panel_info['original_name']}")
        
        # Display period - use dates from files if available, otherwise show message
        if data['dates']:
            print(f"📅 Period: {min(data['dates'])} to {max(data['dates'])}")
        else:
            # Try to get dates from files
            file_dates = []
            for file_info in data['files']:
                if 'info' in file_info and 'date_obs' in file_info['info']:
                    file_dates.append(file_info['info']['date_obs'])
            if file_dates:
                print(f"📅 Period: {min(file_dates)} to {max(file_dates)}")
            else:
                print(f"📅 Period: Not available")
        
        # Display telescope characteristics if available
        if data['apertures'] or data['diameters'] or data['focal_lengths']:
            print(f"\n🔭 TELESCOPE CHARACTERISTICS:")
            
            if data['apertures']:
                # Filter out None values before sorting
                valid_apertures = [a for a in data['apertures'] if a is not None]
                if valid_apertures:
                    unique_apertures = sorted(set(valid_apertures))
                    print(f"   📏 Apertures: {', '.join([f'f/{a:.1f}' for a in unique_apertures])}")
            
            if data['diameters']:
                # Filter out None values before sorting
                valid_diameters = [d for d in data['diameters'] if d is not None]
                if valid_diameters:
                    unique_diameters = sorted(set(valid_diameters))
                    print(f"   🔍 Diameters: {', '.join([f'{d}mm' for d in unique_diameters])}")
            
            if data['focal_lengths']:
                # Filter out None values before sorting
                valid_focal_lengths = [f for f in data['focal_lengths'] if f is not None]
                if valid_focal_lengths:
                    unique_focal_lengths = sorted(set(valid_focal_lengths))
                    print(f"   📐 Focal lengths: {', '.join([f'{f}mm' for f in unique_focal_lengths])}")
        else:
            print(f"\n⚠️  No telescope characteristics detected in FITS headers")
        
        # Statistics by filter - ordered by specific sequence
        print(f"\nFILTER DISTRIBUTION:")
        
        # Aggregate filter data from files_by_date
        aggregated_filters = {}
        if 'files_by_date' in data:
            for date_data in data['files_by_date'].values():
                for filter_name, time_list in date_data['time_by_filter'].items():
                    if filter_name not in aggregated_filters:
                        aggregated_filters[filter_name] = []
                    aggregated_filters[filter_name].extend(time_list)
        else:
            # Fallback to time_by_filter if files_by_date not available
            aggregated_filters = data['time_by_filter']
        
        # Define the specific order for filters
        filter_order = ['L', 'R', 'G', 'B', 'SII', 'Ha', 'OIII']
        
        # First, add filters in the specified order
        for filter_name in filter_order:
            if filter_name in aggregated_filters:
                time_list = aggregated_filters[filter_name]
                total_time = sum(time_list)
                nb_images = len(time_list)
                average_time = total_time / nb_images
                
                # Get filter info with fallback for unknown filters
                if filter_name in FILTERS_INFO:
                    filter_display_name = FILTERS_INFO[filter_name]['name']
                else:
                    filter_display_name = f"Unknown Filter ({filter_name})"
                
                print(f"   {filter_name} ({filter_display_name}):")
                print(f"     📸 {nb_images} images | ⏱️  {format_time(total_time)} | 📊 {format_time(average_time)}/image")
        
        # Then add any remaining filters not in the specified order
        for filter_name in sorted(aggregated_filters.keys()):
            if filter_name not in filter_order:
                time_list = aggregated_filters[filter_name]
                total_time = sum(time_list)
                nb_images = len(time_list)
                average_time = total_time / nb_images
                
                # Get filter info with fallback for unknown filters
                if filter_name in FILTERS_INFO:
                    filter_display_name = FILTERS_INFO[filter_name]['name']
                else:
                    filter_display_name = f"Unknown Filter ({filter_name})"
                
                print(f"   {filter_name} ({filter_display_name}):")
                print(f"     📸 {nb_images} images | ⏱️  {format_time(total_time)} | 📊 {format_time(average_time)}/image")
                
                # Basic exposure information
                if filter_name in data['received_light']:
                    exposure_times = [l['exposure_time'] for l in data['received_light'][filter_name]]
                    total_exposure = sum(exposure_times)
                    avg_exposure = total_exposure / len(exposure_times) if exposure_times else 0
                    
                    print(f"     📸 Total exposure: {format_time(total_exposure)} | 📊 Average: {format_time(avg_exposure)}/image")
            
            # Basic exposure information (photon analysis removed)
            if filter_name in data['received_light']:
                exposure_times = [l['exposure_time'] for l in data['received_light'][filter_name]]
                total_exposure = sum(exposure_times)
                avg_exposure = total_exposure / len(exposure_times) if exposure_times else 0
                
                print(f"     📸 Total exposure: {format_time(total_exposure)} | 📊 Average: {format_time(avg_exposure)}/image")
                
                # ADU and photon analysis removed

def find_latex_executable():
    """Find LaTeX executable across different platforms with comprehensive Linux support"""
    import shutil
    import platform
    import glob
    import os
    
    # Common LaTeX executable names
    latex_names = ['pdflatex', 'latex']
    
    # Try to find LaTeX in PATH first
    for name in latex_names:
        if shutil.which(name):
            return name
    
    # Platform-specific paths
    system = platform.system().lower()
    
    if system == 'windows':
        # Windows paths
        common_paths = [
            r'C:\texlive\*\bin\windows\pdflatex.exe',
            r'C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe',
            r'C:\Program Files (x86)\MiKTeX\miktex\bin\pdflatex.exe',
            r'C:\texlive\*\bin\windows\latex.exe'
        ]
    elif system == 'darwin':  # macOS - comprehensive paths for all Mac variants
        # Get Mac-specific paths
        mac_paths = get_mac_variants_paths()
        common_paths = []
        
        # Add all Mac variant LaTeX paths
        for variant, paths in mac_paths.items():
            common_paths.extend(paths['latex_paths'])
        
        # Additional macOS-specific paths
        additional_mac_paths = [
            # MacTeX installations
            '/Library/TeX/texbin/pdflatex',
            '/Library/TeX/texbin/latex',
            '/usr/local/texlive/*/bin/*/pdflatex',
            '/usr/local/texlive/*/bin/*/latex',
            '/usr/texbin/pdflatex',
            '/usr/texbin/latex',
            
            # Homebrew paths (both Intel and Apple Silicon)
            '/opt/homebrew/bin/pdflatex',  # Apple Silicon
            '/usr/local/bin/pdflatex',     # Intel
            '/opt/homebrew/bin/latex',
            '/usr/local/bin/latex',
            '/opt/homebrew/opt/texlive/bin/pdflatex',
            '/usr/local/opt/texlive/bin/pdflatex',
            
            # MacPorts paths
            '/opt/local/bin/pdflatex',
            '/opt/local/bin/latex',
            '/opt/local/share/texmf/bin/pdflatex',
            '/opt/local/share/texmf/bin/latex',
            
            # Xcode Command Line Tools
            '/Library/Developer/CommandLineTools/usr/bin/pdflatex',
            '/Applications/Xcode.app/Contents/Developer/usr/bin/pdflatex',
            
            # Custom installations
            '/usr/local/share/texmf/bin/pdflatex',
            '/usr/local/share/texmf/bin/latex',
            
            # Conda installations on Mac
            '/opt/conda/bin/pdflatex',
            '/usr/local/conda/bin/pdflatex',
            '/opt/miniconda3/bin/pdflatex',
            '/usr/local/miniconda3/bin/pdflatex'
        ]
        
        common_paths.extend(additional_mac_paths)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in common_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        common_paths = unique_paths
    else:  # Linux and others - comprehensive paths for all distributions
        # Get distribution-specific paths
        distro_paths = get_linux_distribution_paths()
        common_paths = []
        
        # Add all distribution-specific LaTeX paths
        for distro, paths in distro_paths.items():
            common_paths.extend(paths['latex_paths'])
        
        # Additional Linux-specific paths
        additional_linux_paths = [
            # TeX Live installations
            '/usr/local/texlive/*/bin/*/pdflatex',
            '/opt/texlive/*/bin/*/pdflatex',
            '/usr/share/texlive/bin/*/pdflatex',
            '/usr/share/texlive/bin/x86_64-linux/pdflatex',
            '/usr/share/texlive/bin/i386-linux/pdflatex',
            '/usr/share/texlive/bin/amd64-linux/pdflatex',
            
            # Snap packages
            '/snap/bin/pdflatex',
            '/snap/texlive/current/bin/pdflatex',
            
            # AppImage installations
            '/opt/texlive/*/bin/*/pdflatex',
            
            # Custom installations
            '/opt/tex/bin/pdflatex',
            '/usr/local/bin/pdflatex',
            '/usr/bin/pdflatex',
            
            # Container-specific paths
            '/usr/share/texmf/bin/pdflatex',
            '/var/lib/texmf/bin/pdflatex',
            
            # Homebrew on Linux
            '/home/linuxbrew/.linuxbrew/bin/pdflatex',
            '/opt/homebrew/bin/pdflatex',
            
            # Conda installations
            '/opt/conda/bin/pdflatex',
            '/usr/local/conda/bin/pdflatex'
        ]
        
        common_paths.extend(additional_linux_paths)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in common_paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        common_paths = unique_paths
    
    # Check common paths
    for pattern in common_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    # Final fallback: try to find any pdflatex in common directories
    if system == 'linux':
        fallback_dirs = [
            '/usr/bin', '/usr/local/bin', '/opt/bin',
            '/usr/share/bin', '/usr/local/share/bin'
        ]
        
        for directory in fallback_dirs:
            if os.path.exists(directory):
                for latex_name in latex_names:
                    latex_path = os.path.join(directory, latex_name)
                    if os.path.exists(latex_path) and os.access(latex_path, os.X_OK):
                        return latex_path
    
    # LaTeX not found - display installation suggestions
    display_latex_installation_suggestions()
    return None

def get_required_latex_packages():
    """Get list of required LaTeX packages for the report"""
    return [
        'amsmath',      # Mathematical typesetting
        'amsfonts',     # Mathematical fonts
        'amssymb',      # Mathematical symbols
        'geometry',     # Page layout
        'graphicx',     # Graphics inclusion
        'booktabs',     # Professional tables
        'longtable',    # Multi-page tables
        'hyperref',     # Hyperlinks
        'xcolor',       # Colors
        'tikz',         # Vector graphics
        'pgf',          # TikZ backend
        'siunitx',      # SI units
        'float',        # Float positioning
        'needspace',    # Space control
        'translations', # Internationalization
        'array',        # Table extensions
        'url',          # URL formatting
        'rerunfilecheck' # File checking
    ]

def get_required_python_packages():
    """Get list of required Python packages"""
    return [
        'matplotlib',   # Plotting and graphics
        'numpy',        # Numerical computing
        'pandas',       # Data manipulation
        'reportlab',    # PDF generation
        'tqdm',         # Progress bars
        'astropy',      # Astronomical calculations
        'pillow',       # Image processing
        'scipy',        # Scientific computing
        'requests'      # HTTP requests
    ]

def check_python_packages():
    """Check which Python packages are missing and provide installation instructions"""
    required_packages = get_required_python_packages()
    missing_packages = []
    available_packages = []
    
    for package in required_packages:
        try:
            if package == 'pillow':
                import PIL
            else:
                __import__(package)
            available_packages.append(package)
        except ImportError:
            missing_packages.append(package)
    
    return {
        'missing': missing_packages,
        'available': available_packages,
        'all_available': len(missing_packages) == 0
    }

def get_mac_variants_paths():
    """Get comprehensive Python and LaTeX paths for all Mac variants"""
    return {
        # Intel Mac paths
        'intel': {
            'python_paths': [
                '/usr/bin/python3', '/usr/bin/python',
                '/usr/local/bin/python3', '/usr/local/bin/python',
                '/opt/homebrew/bin/python3',  # Homebrew on Intel
                '/usr/local/opt/python@3.*/bin/python3',
                '/usr/local/Cellar/python@3.*/bin/python3',
                '/opt/local/bin/python3',  # MacPorts
                '/opt/local/bin/python',
                '/Library/Frameworks/Python.framework/Versions/*/bin/python3',
                '/Applications/Python 3.*/bin/python3'
            ],
            'pip_paths': [
                '/usr/bin/pip3', '/usr/bin/pip',
                '/usr/local/bin/pip3', '/usr/local/bin/pip',
                '/opt/homebrew/bin/pip3',
                '/usr/local/opt/python@3.*/bin/pip3',
                '/usr/local/Cellar/python@3.*/bin/pip3',
                '/opt/local/bin/pip3',  # MacPorts
                '/opt/local/bin/pip',
                '/Library/Frameworks/Python.framework/Versions/*/bin/pip3',
                '/Applications/Python 3.*/bin/pip3'
            ],
            'latex_paths': [
                '/usr/bin/pdflatex', '/usr/bin/latex',
                '/usr/local/bin/pdflatex', '/usr/local/bin/latex',
                '/Library/TeX/texbin/pdflatex',  # MacTeX
                '/usr/local/texlive/*/bin/*/pdflatex',
                '/opt/homebrew/bin/pdflatex',  # Homebrew LaTeX
                '/opt/local/bin/pdflatex',  # MacPorts LaTeX
                '/opt/local/bin/latex',
                '/usr/local/opt/texlive/bin/pdflatex'
            ]
        },
        # Apple Silicon (M1/M2/M3) Mac paths
        'apple_silicon': {
            'python_paths': [
                '/usr/bin/python3', '/usr/bin/python',
                '/opt/homebrew/bin/python3',  # Homebrew on Apple Silicon
                '/opt/homebrew/bin/python',
                '/usr/local/bin/python3', '/usr/local/bin/python',
                '/opt/homebrew/opt/python@3.*/bin/python3',
                '/opt/homebrew/Cellar/python@3.*/bin/python3',
                '/opt/local/bin/python3',  # MacPorts
                '/opt/local/bin/python',
                '/Library/Frameworks/Python.framework/Versions/*/bin/python3',
                '/Applications/Python 3.*/bin/python3',
                '/System/Library/Frameworks/Python.framework/Versions/*/bin/python3'
            ],
            'pip_paths': [
                '/usr/bin/pip3', '/usr/bin/pip',
                '/opt/homebrew/bin/pip3',  # Homebrew on Apple Silicon
                '/opt/homebrew/bin/pip',
                '/usr/local/bin/pip3', '/usr/local/bin/pip',
                '/opt/homebrew/opt/python@3.*/bin/pip3',
                '/opt/homebrew/Cellar/python@3.*/bin/pip3',
                '/opt/local/bin/pip3',  # MacPorts
                '/opt/local/bin/pip',
                '/Library/Frameworks/Python.framework/Versions/*/bin/pip3',
                '/Applications/Python 3.*/bin/pip3',
                '/System/Library/Frameworks/Python.framework/Versions/*/bin/pip3'
            ],
            'latex_paths': [
                '/usr/bin/pdflatex', '/usr/bin/latex',
                '/Library/TeX/texbin/pdflatex',  # MacTeX
                '/Library/TeX/texbin/latex',
                '/opt/homebrew/bin/pdflatex',  # Homebrew LaTeX on Apple Silicon
                '/opt/homebrew/bin/latex',
                '/usr/local/bin/pdflatex', '/usr/local/bin/latex',
                '/usr/local/texlive/*/bin/*/pdflatex',
                '/opt/local/bin/pdflatex',  # MacPorts LaTeX
                '/opt/local/bin/latex',
                '/opt/homebrew/opt/texlive/bin/pdflatex'
            ]
        },
        # macOS with Xcode Command Line Tools
        'xcode_tools': {
            'python_paths': [
                '/usr/bin/python3', '/usr/bin/python',
                '/Applications/Xcode.app/Contents/Developer/usr/bin/python3',
                '/Library/Developer/CommandLineTools/usr/bin/python3',
                '/usr/local/bin/python3', '/usr/local/bin/python'
            ],
            'pip_paths': [
                '/usr/bin/pip3', '/usr/bin/pip',
                '/usr/local/bin/pip3', '/usr/local/bin/pip'
            ],
            'latex_paths': [
                '/usr/bin/pdflatex', '/usr/bin/latex',
                '/Library/TeX/texbin/pdflatex',
                '/Library/TeX/texbin/latex'
            ]
        },
        # macOS with Homebrew
        'homebrew': {
            'python_paths': [
                '/opt/homebrew/bin/python3',  # Apple Silicon
                '/usr/local/bin/python3',     # Intel
                '/opt/homebrew/bin/python',
                '/usr/local/bin/python',
                '/opt/homebrew/opt/python@3.*/bin/python3',
                '/usr/local/opt/python@3.*/bin/python3',
                '/opt/homebrew/Cellar/python@3.*/bin/python3',
                '/usr/local/Cellar/python@3.*/bin/python3'
            ],
            'pip_paths': [
                '/opt/homebrew/bin/pip3',     # Apple Silicon
                '/usr/local/bin/pip3',       # Intel
                '/opt/homebrew/bin/pip',
                '/usr/local/bin/pip',
                '/opt/homebrew/opt/python@3.*/bin/pip3',
                '/usr/local/opt/python@3.*/bin/pip3',
                '/opt/homebrew/Cellar/python@3.*/bin/pip3',
                '/usr/local/Cellar/python@3.*/bin/pip3'
            ],
            'latex_paths': [
                '/opt/homebrew/bin/pdflatex',  # Apple Silicon
                '/usr/local/bin/pdflatex',     # Intel
                '/opt/homebrew/bin/latex',
                '/usr/local/bin/latex',
                '/opt/homebrew/opt/texlive/bin/pdflatex',
                '/usr/local/opt/texlive/bin/pdflatex'
            ]
        },
        # macOS with MacPorts
        'macports': {
            'python_paths': [
                '/opt/local/bin/python3', '/opt/local/bin/python',
                '/opt/local/bin/python3.*',
                '/opt/local/Library/Frameworks/Python.framework/Versions/*/bin/python3'
            ],
            'pip_paths': [
                '/opt/local/bin/pip3', '/opt/local/bin/pip',
                '/opt/local/bin/pip3.*',
                '/opt/local/Library/Frameworks/Python.framework/Versions/*/bin/pip3'
            ],
            'latex_paths': [
                '/opt/local/bin/pdflatex', '/opt/local/bin/latex',
                '/opt/local/share/texmf/bin/pdflatex',
                '/opt/local/share/texmf/bin/latex'
            ]
        },
        # macOS with MacTeX
        'mactex': {
            'python_paths': [
                '/usr/bin/python3', '/usr/bin/python',
                '/usr/local/bin/python3', '/usr/local/bin/python'
            ],
            'pip_paths': [
                '/usr/bin/pip3', '/usr/bin/pip',
                '/usr/local/bin/pip3', '/usr/local/bin/pip'
            ],
            'latex_paths': [
                '/Library/TeX/texbin/pdflatex',
                '/Library/TeX/texbin/latex',
                '/usr/local/texlive/*/bin/*/pdflatex',
                '/usr/local/texlive/*/bin/*/latex',
                '/usr/texbin/pdflatex',
                '/usr/texbin/latex'
            ]
        }
    }

def get_linux_distribution_paths():
    """Get comprehensive Python and LaTeX paths for all major Linux distributions"""
    return {
        # Ubuntu/Debian paths
        'ubuntu': {
            'python_paths': [
                '/usr/bin/python3', '/usr/bin/python',
                '/usr/local/bin/python3', '/usr/local/bin/python',
                '/opt/python/bin/python3', '/opt/python/bin/python'
            ],
            'pip_paths': [
                '/usr/bin/pip3', '/usr/bin/pip',
                '/usr/local/bin/pip3', '/usr/local/bin/pip',
                '/opt/python/bin/pip3', '/opt/python/bin/pip'
            ],
            'latex_paths': [
                '/usr/bin/pdflatex', '/usr/bin/latex',
                '/usr/local/bin/pdflatex', '/usr/local/bin/latex',
                '/usr/share/texlive/bin/x86_64-linux/pdflatex',
                '/usr/share/texlive/bin/x86_64-linux/latex'
            ]
        },
        # Fedora/CentOS/RHEL paths
        'fedora': {
            'python_paths': [
                '/usr/bin/python3', '/usr/bin/python',
                '/usr/local/bin/python3', '/usr/local/bin/python',
                '/opt/python/bin/python3', '/opt/python/bin/python'
            ],
            'pip_paths': [
                '/usr/bin/pip3', '/usr/bin/pip',
                '/usr/local/bin/pip3', '/usr/local/bin/pip',
                '/opt/python/bin/pip3', '/opt/python/bin/pip'
            ],
            'latex_paths': [
                '/usr/bin/pdflatex', '/usr/bin/latex',
                '/usr/local/bin/pdflatex', '/usr/local/bin/latex',
                '/usr/share/texlive/bin/x86_64-linux/pdflatex',
                '/usr/share/texlive/bin/x86_64-linux/latex'
            ]
        },
        # Arch/Manjaro paths
        'arch': {
            'python_paths': [
                '/usr/bin/python3', '/usr/bin/python',
                '/usr/local/bin/python3', '/usr/local/bin/python',
                '/opt/python/bin/python3', '/opt/python/bin/python'
            ],
            'pip_paths': [
                '/usr/bin/pip3', '/usr/bin/pip',
                '/usr/local/bin/pip3', '/usr/local/bin/pip',
                '/opt/python/bin/pip3', '/opt/python/bin/pip'
            ],
            'latex_paths': [
                '/usr/bin/pdflatex', '/usr/bin/latex',
                '/usr/local/bin/pdflatex', '/usr/local/bin/latex',
                '/usr/share/texmf-dist/bin/pdflatex',
                '/usr/share/texmf-dist/bin/latex',
                '/var/lib/texmf/bin/pdflatex',
                '/var/lib/texmf/bin/latex'
            ]
        },
        # openSUSE paths
        'opensuse': {
            'python_paths': [
                '/usr/bin/python3', '/usr/bin/python',
                '/usr/local/bin/python3', '/usr/local/bin/python',
                '/opt/python/bin/python3', '/opt/python/bin/python'
            ],
            'pip_paths': [
                '/usr/bin/pip3', '/usr/bin/pip',
                '/usr/local/bin/pip3', '/usr/local/bin/pip',
                '/opt/python/bin/pip3', '/opt/python/bin/pip'
            ],
            'latex_paths': [
                '/usr/bin/pdflatex', '/usr/bin/latex',
                '/usr/local/bin/pdflatex', '/usr/local/bin/latex',
                '/usr/share/texlive/bin/x86_64-linux/pdflatex',
                '/usr/share/texlive/bin/x86_64-linux/latex'
            ]
        },
        # Gentoo paths
        'gentoo': {
            'python_paths': [
                '/usr/bin/python3', '/usr/bin/python',
                '/usr/local/bin/python3', '/usr/local/bin/python',
                '/opt/python/bin/python3', '/opt/python/bin/python'
            ],
            'pip_paths': [
                '/usr/bin/pip3', '/usr/bin/pip',
                '/usr/local/bin/pip3', '/usr/local/bin/pip',
                '/opt/python/bin/pip3', '/opt/python/bin/pip'
            ],
            'latex_paths': [
                '/usr/bin/pdflatex', '/usr/bin/latex',
                '/usr/local/bin/pdflatex', '/usr/local/bin/latex',
                '/usr/share/texlive/bin/x86_64-linux/pdflatex',
                '/usr/share/texlive/bin/x86_64-linux/latex'
            ]
        },
        # Alpine paths
        'alpine': {
            'python_paths': [
                '/usr/bin/python3', '/usr/bin/python',
                '/usr/local/bin/python3', '/usr/local/bin/python'
            ],
            'pip_paths': [
                '/usr/bin/pip3', '/usr/bin/pip',
                '/usr/local/bin/pip3', '/usr/local/bin/pip'
            ],
            'latex_paths': [
                '/usr/bin/pdflatex', '/usr/bin/latex',
                '/usr/local/bin/pdflatex', '/usr/local/bin/latex'
            ]
        }
    }

def find_pip_executable():
    """Find the correct pip executable for the current Python installation"""
    import sys
    import os
    import subprocess
    import shutil
    import platform
    
    # Get platform information
    system = platform.system().lower()
    is_linux = system == 'linux'
    
    # Try different pip locations with comprehensive Linux paths
    pip_candidates = []
    
    # 1. Direct pip commands (most common)
    pip_candidates.extend(['pip', 'pip3'])
    
    # 2. Python module approach (most reliable)
    pip_candidates.extend([
        f'{sys.executable} -m pip',
        f'python -m pip',
        f'python3 -m pip'
    ])
    
    # 3. Platform-specific paths
    if is_linux:
        # Get distribution-specific paths
        distro_paths = get_linux_distribution_paths()
        
        # Try to find python in PATH first
        for python_cmd in ['python3', 'python']:
            python_path = shutil.which(python_cmd)
            if python_path:
                pip_candidates.append(f'{python_path} -m pip')
        
        # Try all distribution-specific paths
        for distro, paths in distro_paths.items():
            for python_path in paths['python_paths']:
                if os.path.exists(python_path):
                    pip_candidates.append(f'{python_path} -m pip')
            
            for pip_path in paths['pip_paths']:
                if os.path.exists(pip_path):
                    pip_candidates.append(pip_path)
    elif system == 'darwin':  # macOS
        # Get Mac-specific paths
        mac_paths = get_mac_variants_paths()
        
        # Try to find python in PATH first
        for python_cmd in ['python3', 'python']:
            python_path = shutil.which(python_cmd)
            if python_path:
                pip_candidates.append(f'{python_path} -m pip')
        
        # Try all Mac variant paths
        for variant, paths in mac_paths.items():
            for python_path in paths['python_paths']:
                if os.path.exists(python_path):
                    pip_candidates.append(f'{python_path} -m pip')
            
            for pip_path in paths['pip_paths']:
                if os.path.exists(pip_path):
                    pip_candidates.append(pip_path)
    
    # 4. Additional common Linux locations
    if is_linux:
        additional_paths = [
            '/snap/bin/python3',  # Snap packages
            '/snap/bin/pip3',
            '/home/linuxbrew/.linuxbrew/bin/python3',  # Homebrew on Linux
            '/home/linuxbrew/.linuxbrew/bin/pip3',
            '/opt/homebrew/bin/python3',  # Homebrew on ARM Linux
            '/opt/homebrew/bin/pip3'
        ]
        
        for path in additional_paths:
            if os.path.exists(path):
                if 'python' in path:
                    pip_candidates.append(f'{path} -m pip')
                else:
                    pip_candidates.append(path)
    
    # Test each candidate
    for pip_cmd in pip_candidates:
        try:
            if ' -m ' in pip_cmd:
                # For python -m pip commands
                cmd_parts = pip_cmd.split()
                result = subprocess.run(cmd_parts + ['--version'], 
                                      capture_output=True, text=True, timeout=10)
            else:
                # For direct pip commands
                result = subprocess.run([pip_cmd, '--version'], 
                                      capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return pip_cmd
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
    
    return None

def diagnose_linux_distribution_issues(platform_info):
    """Diagnose and provide solutions for Linux distribution-specific issues"""
    if not platform_info['is_linux']:
        return
    
    distro = platform_info['linux_distro']
    print(f"\n🔧 DIAGNOSTIC {distro.upper()} - Chemins Python et LaTeX")
    print("=" * 60)
    
    python_paths = platform_info['python_paths']
    
    # Check Python executable
    if python_paths['python_executable']:
        print(f"✅ Python exécutable: {python_paths['python_executable']}")
    else:
        print("❌ Python exécutable non trouvé")
    
    # Check Python3 in PATH
    if python_paths['python3_in_path']:
        print(f"✅ Python3 dans PATH: {python_paths['python3_in_path']}")
    else:
        print("❌ Python3 non trouvé dans PATH")
    
    # Check pip3 in PATH
    if python_paths['pip3_in_path']:
        print(f"✅ Pip3 dans PATH: {python_paths['pip3_in_path']}")
    else:
        print("❌ Pip3 non trouvé dans PATH")
    
    # Get distribution-specific paths
    distro_paths = get_linux_distribution_paths()
    if distro in distro_paths:
        paths = distro_paths[distro]
    else:
        # Use Ubuntu paths as default for unknown distributions
        paths = distro_paths['ubuntu']
    
    # Check Python paths
    print(f"\n🔍 Vérification des chemins Python pour {distro}:")
    for path in paths['python_paths']:
        import os
        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
    
    # Check pip paths
    print(f"\n🔍 Vérification des chemins Pip pour {distro}:")
    for path in paths['pip_paths']:
        import os
        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
    
    # Check LaTeX paths
    print(f"\n🔍 Vérification des chemins LaTeX pour {distro}:")
    for path in paths['latex_paths']:
        import os
        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
    
    # Provide distribution-specific solutions
    print(f"\n💡 Solutions recommandées pour {distro.upper()}:")
    
    if distro in ['ubuntu', 'debian']:
        print("   Python: sudo apt update && sudo apt install python3 python3-pip")
        print("   LaTeX: sudo apt install texlive-full")
        print("   Packages: sudo apt install python3-astropy python3-matplotlib python3-pillow")
    elif distro in ['fedora', 'centos', 'rhel']:
        print("   Python: sudo dnf install python3 python3-pip")
        print("   LaTeX: sudo dnf install texlive-scheme-full")
        print("   Packages: sudo dnf install python3-astropy python3-matplotlib python3-pillow")
    elif distro in ['arch', 'manjaro']:
        print("   Python: sudo pacman -S python python-pip")
        print("   LaTeX: sudo pacman -S texlive-most texlive-lang")
        print("   Packages: sudo pacman -S python-astropy python-matplotlib python-pillow")
    elif distro in ['opensuse', 'suse']:
        print("   Python: sudo zypper install python3 python3-pip")
        print("   LaTeX: sudo zypper install texlive")
        print("   Packages: sudo zypper install python3-astropy python3-matplotlib python3-pillow")
    elif distro == 'gentoo':
        print("   Python: emerge dev-lang/python")
        print("   LaTeX: emerge app-text/texlive")
        print("   Packages: emerge dev-python/astropy dev-python/matplotlib dev-python/pillow")
    elif distro == 'alpine':
        print("   Python: apk add python3 py3-pip")
        print("   LaTeX: apk add texlive")
        print("   Packages: apk add py3-astropy py3-matplotlib py3-pillow")
    else:
        print("   Python: Install python3 and python3-pip using your package manager")
        print("   LaTeX: Install texlive or texlive-full using your package manager")
        print("   Packages: Install python3-astropy, python3-matplotlib, python3-pillow")

def diagnose_mac_variants_issues(platform_info):
    """Diagnose and provide solutions for Mac variant-specific issues"""
    if not platform_info['is_macos']:
        return
    
    mac_variant = platform_info['mac_variant']
    print(f"\n🍎 DIAGNOSTIC MAC - {mac_variant.upper()} - Chemins Python et LaTeX")
    print("=" * 70)
    
    python_paths = platform_info['python_paths']
    
    # Check Python executable
    if python_paths['python_executable']:
        print(f"✅ Python exécutable: {python_paths['python_executable']}")
    else:
        print("❌ Python exécutable non trouvé")
    
    # Check Python3 in PATH
    if python_paths['python3_in_path']:
        print(f"✅ Python3 dans PATH: {python_paths['python3_in_path']}")
    else:
        print("❌ Python3 non trouvé dans PATH")
    
    # Check pip3 in PATH
    if python_paths['pip3_in_path']:
        print(f"✅ Pip3 dans PATH: {python_paths['pip3_in_path']}")
    else:
        print("❌ Pip3 non trouvé dans PATH")
    
    # Get Mac-specific paths
    mac_paths = get_mac_variants_paths()
    
    # Determine which variant to check
    variant_to_check = 'apple_silicon' if platform_info['is_apple_silicon'] else 'intel'
    if platform_info['has_homebrew']:
        variant_to_check = 'homebrew'
    elif platform_info['has_macports']:
        variant_to_check = 'macports'
    elif platform_info['has_mactex']:
        variant_to_check = 'mactex'
    
    if variant_to_check in mac_paths:
        paths = mac_paths[variant_to_check]
    else:
        # Use Apple Silicon as default for unknown variants
        paths = mac_paths['apple_silicon']
    
    # Check Python paths
    print(f"\n🔍 Vérification des chemins Python pour {mac_variant}:")
    for path in paths['python_paths']:
        import os
        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
    
    # Check pip paths
    print(f"\n🔍 Vérification des chemins Pip pour {mac_variant}:")
    for path in paths['pip_paths']:
        import os
        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
    
    # Check LaTeX paths
    print(f"\n🔍 Vérification des chemins LaTeX pour {mac_variant}:")
    for path in paths['latex_paths']:
        import os
        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
    
    # Provide Mac-specific solutions
    print(f"\n💡 Solutions recommandées pour {mac_variant.upper()}:")
    
    if platform_info['is_apple_silicon']:
        print("   🍎 Apple Silicon (M1/M2/M3) Mac:")
        print("   Python: brew install python")
        print("   LaTeX: brew install --cask mactex")
        print("   Packages: brew install python-astropy python-matplotlib python-pillow")
    elif platform_info['is_intel_mac']:
        print("   🍎 Intel Mac:")
        print("   Python: brew install python")
        print("   LaTeX: brew install --cask mactex")
        print("   Packages: brew install python-astropy python-matplotlib python-pillow")
    
    if platform_info['has_homebrew']:
        print("   🍺 Homebrew détecté:")
        print("   Python: brew install python")
        print("   LaTeX: brew install --cask mactex")
        print("   Packages: brew install python-astropy python-matplotlib python-pillow")
    elif platform_info['has_macports']:
        print("   🍺 MacPorts détecté:")
        print("   Python: sudo port install python3 py3-pip")
        print("   LaTeX: sudo port install texlive")
        print("   Packages: sudo port install py3-astropy py3-matplotlib py3-pillow")
    else:
        print("   📦 Installation recommandée:")
        print("   1. Installer Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("   2. Installer Python: brew install python")
        print("   3. Installer LaTeX: brew install --cask mactex")
        print("   4. Installer packages: brew install python-astropy python-matplotlib python-pillow")

def diagnose_manjaro_python_issues(platform_info):
    """Legacy function - now redirects to comprehensive Linux diagnostic"""
    diagnose_linux_distribution_issues(platform_info)

def install_python_packages_automatically():
    """Attempt to install missing Python packages automatically"""
    package_status = check_python_packages()
    
    if package_status['all_available']:
        print("✅ All required Python packages are available!")
        return True
    
    missing = package_status['missing']
    print(f"⚠️  Missing Python packages: {', '.join(missing)}")
    print("   🤖 Attempting automatic installation...")
    
    # Find working pip executable
    pip_cmd = find_pip_executable()
    if not pip_cmd:
        print("   ❌ Could not find pip executable")
        print("   💡 Solutions:")
        print("      1. Use: python -m pip install <package>")
        print("      2. Or install pip: python -m ensurepip --upgrade")
        print("      3. Or use Anaconda: https://www.anaconda.com/download")
        return False
    
    print(f"   🔍 Found pip: {pip_cmd}")
    
    try:
        import subprocess
        import sys
        import platform
        
        # Try to install missing packages with multiple strategies
        failed_packages = []
        is_windows = platform.system().lower() == 'windows'
        
        for package in missing:
            print(f"   📦 Installing {package}...")
            success = False
            
            # Strategy 1: Try pre-compiled wheels first (especially for Windows)
            if is_windows:
                try:
                    if ' -m ' in pip_cmd:
                        cmd_parts = pip_cmd.split() + ['install', package, '--only-binary=all', '--upgrade', '--quiet']
                    else:
                        cmd_parts = [pip_cmd, 'install', package, '--only-binary=all', '--upgrade', '--quiet']
                    
                    result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        print(f"   ✅ {package} installed successfully (pre-compiled)")
                        success = True
                    else:
                        print(f"   ⚠️  Pre-compiled version failed, trying other methods...")
                except Exception as e:
                    print(f"   ⚠️  Pre-compiled installation failed: {e}")
            
            # Strategy 2: Try regular pip install
            if not success:
                try:
                    if ' -m ' in pip_cmd:
                        cmd_parts = pip_cmd.split() + ['install', package, '--upgrade', '--quiet']
                    else:
                        cmd_parts = [pip_cmd, 'install', package, '--upgrade', '--quiet']
                    
                    result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        print(f"   ✅ {package} installed successfully")
                        success = True
                    else:
                        print(f"   ❌ Failed to install {package}")
                        if 'compiler' in result.stderr.lower() or 'build' in result.stderr.lower():
                            print(f"   💡 {package} requires compilation - missing C++ compiler")
                        
                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Timeout installing {package}")
                except Exception as e:
                    print(f"   ❌ Error installing {package}: {e}")
            
            if not success:
                failed_packages.append(package)
        
        # Re-check packages after installation
        print("\n   🔍 Re-checking packages...")
        new_status = check_python_packages()
        
        if new_status['all_available']:
            print("✅ All packages are now available!")
            return True
        else:
            still_missing = new_status['missing']
            print(f"⚠️  Still missing: {', '.join(still_missing)}")
            
            # Provide comprehensive solutions for failed packages
            print("\n   🔧 Alternative solutions for failed packages:")
            print("      📦 For packages requiring compilation (matplotlib, pandas, scipy):")
            print()
            print("      🚀 QUICK SOLUTION - Use Anaconda/Miniconda:")
            print("         1. Download Anaconda: https://www.anaconda.com/download")
            print("         2. Install Anaconda")
            print("         3. Run: conda install matplotlib pandas scipy")
            print("         4. Run: pip install reportlab tqdm astropy pillow")
            print()
            print("      🔧 ALTERNATIVE - Manual installation:")
            print("         1. Install Visual Studio Build Tools:")
            print("            https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            print("         2. Restart command prompt")
            print("         3. Run: pip install matplotlib pandas scipy")
            print()
            print("      🎯 SPECIFIC COMMANDS for your system:")
            print("         # Try pre-compiled versions:")
            print("         pip install --only-binary=all matplotlib pandas scipy")
            print("         pip install reportlab tqdm astropy pillow")
            print()
            print("         # Or use conda for scientific packages:")
            print("         conda install matplotlib pandas scipy")
            print("         pip install reportlab tqdm astropy pillow")
            print()
            print("      💡 TIP: Anaconda is the easiest solution for Windows!")
            
            return False
            
    except ImportError:
        print("   ❌ subprocess module not available for automatic installation")
        return False
    except Exception as e:
        print(f"   ❌ Error during automatic installation: {e}")
        return False

def suggest_python_installation():
    """Suggest Python package installation commands"""
    package_status = check_python_packages()
    
    if package_status['all_available']:
        print("✅ All required Python packages are available!")
        return True
    
    missing = package_status['missing']
    print(f"⚠️  Missing Python packages: {', '.join(missing)}")
    
    # Ask user if they want automatic installation
    try:
        response = input("   🤖 Would you like to install missing packages automatically? (y/n): ").lower().strip()
        if response in ['y', 'yes', 'oui', 'o']:
            if install_python_packages_automatically():
                return True
    except (KeyboardInterrupt, EOFError):
        print("\n   ⏹️  Installation cancelled by user")
    
    # Show manual installation instructions with correct pip commands
    print("   🐍 Manual installation commands:")
    print("      # Try these commands in order:")
    print(f"      python -m pip install {' '.join(missing)}")
    print()
    print("   📦 Or install all at once:")
    print("      python -m pip install matplotlib numpy pandas reportlab tqdm astropy pillow scipy")
    print()
    print("   🔧 Alternative installation methods:")
    print("      # Using conda (if available):")
    print("      conda install matplotlib numpy pandas scipy astropy")
    print("      python -m pip install reportlab tqdm pillow")
    print()
    print("      # Using pip with requirements file:")
    print("      python -m pip install -r requirements.txt")
    print()
    print("   💡 If 'python -m pip' doesn't work, try:")
    print("      py -m pip install <package>")
    print("      or")
    print("      python3 -m pip install <package>")
    print()
    
    return False

def get_platform_latex_instructions():
    """Get platform-specific LaTeX installation instructions with package details"""
    import platform
    
    system = platform.system().lower()
    required_packages = get_required_latex_packages()
    
    if system == 'windows':
        return {
            'name': 'Windows',
            'distributions': [
                {
                    'name': 'MiKTeX (Recommandé)',
                    'url': 'https://miktex.org/download',
                    'install_cmd': 'Télécharger et exécuter l\'installateur',
                    'package_install': 'MiKTeX Console → Packages → Install missing packages automatically',
                    'description': 'Distribution LaTeX optimisée pour Windows avec installation automatique des packages'
                },
                {
                    'name': 'TeX Live',
                    'url': 'https://www.tug.org/texlive/acquire-netinst.html',
                    'install_cmd': 'Télécharger et exécuter l\'installateur',
                    'package_install': 'tlmgr install <package_name>',
                    'description': 'Distribution LaTeX complète et multiplateforme'
                },
                {
                    'name': 'TeXstudio (Éditeur)',
                    'url': 'https://www.texstudio.org/',
                    'install_cmd': 'Télécharger et installer',
                    'package_install': 'Nécessite MiKTeX ou TeX Live',
                    'description': 'Éditeur LaTeX moderne avec interface graphique'
                }
            ],
            'package_managers': [
                {
                    'name': 'Chocolatey',
                    'install_cmd': 'choco install miktex',
                    'url': 'https://chocolatey.org/',
                    'description': 'Gestionnaire de packages pour Windows'
                },
                {
                    'name': 'Winget',
                    'install_cmd': 'winget install MiKTeX.MiKTeX',
                    'url': 'https://docs.microsoft.com/en-us/windows/package-manager/',
                    'description': 'Gestionnaire de packages natif Windows 10/11'
                }
            ],
            'alternatives': [
                {
                    'name': 'Overleaf (En ligne)',
                    'url': 'https://www.overleaf.com/',
                    'description': 'Éditeur LaTeX en ligne, aucune installation requise'
                },
                {
                    'name': 'Typst (Moderne)',
                    'url': 'https://typst.app/',
                    'description': 'Alternative moderne à LaTeX avec syntaxe simplifiée'
                }
            ]
        }
    elif system == 'darwin':
        return {
            'name': 'macOS',
            'distributions': [
                {
                    'name': 'MacTeX (Recommandé)',
                    'url': 'https://www.tug.org/mactex/',
                    'install_cmd': 'Télécharger et exécuter l\'installateur',
                    'package_install': 'TeX Live Utility → Install packages',
                    'description': 'Distribution LaTeX complète pour macOS'
                },
                {
                    'name': 'BasicTeX',
                    'url': 'https://www.tug.org/mactex/morepackages.html',
                    'install_cmd': 'Télécharger BasicTeX (plus léger)',
                    'package_install': 'tlmgr install <package_name>',
                    'description': 'Version allégée de MacTeX'
                }
            ],
            'package_managers': [
                {
                    'name': 'Homebrew',
                    'install_cmd': 'brew install --cask mactex',
                    'url': 'https://brew.sh/',
                    'description': 'Gestionnaire de packages pour macOS'
                },
                {
                    'name': 'MacPorts',
                    'install_cmd': 'sudo port install texlive',
                    'url': 'https://www.macports.org/',
                    'description': 'Alternative à Homebrew'
                }
            ],
            'alternatives': [
                {
                    'name': 'Overleaf (En ligne)',
                    'url': 'https://www.overleaf.com/',
                    'description': 'Éditeur LaTeX en ligne'
                },
                {
                    'name': 'Typst (Moderne)',
                    'url': 'https://typst.app/',
                    'description': 'Alternative moderne à LaTeX'
                }
            ]
        }
    else:  # Linux and others
        return {
            'name': 'Linux',
            'distributions': [
                {
                    'name': 'TeX Live (Ubuntu/Debian)',
                    'install_cmd': 'sudo apt install texlive-full texlive-latex-extra',
                    'package_install': 'sudo apt install texlive-latex-extra',
                    'description': 'Distribution complète via apt'
                },
                {
                    'name': 'TeX Live (Fedora/RHEL)',
                    'install_cmd': 'sudo dnf install texlive-scheme-full',
                    'package_install': 'sudo dnf install texlive-latex-extra',
                    'description': 'Distribution complète via dnf'
                },
                {
                    'name': 'TeX Live (Arch Linux)',
                    'install_cmd': 'sudo pacman -S texlive-most texlive-lang',
                    'package_install': 'sudo pacman -S texlive-latexextra',
                    'description': 'Distribution complète via pacman'
                },
                {
                    'name': 'TeX Live (openSUSE)',
                    'install_cmd': 'sudo zypper install texlive',
                    'package_install': 'sudo zypper install texlive-latex',
                    'description': 'Distribution via zypper'
                },
                {
                    'name': 'TeX Live (Gentoo)',
                    'install_cmd': 'emerge app-text/texlive',
                    'package_install': 'emerge app-text/texlive-latex',
                    'description': 'Distribution via emerge'
                }
            ],
            'package_managers': [
                {
                    'name': 'Snap',
                    'install_cmd': 'sudo snap install texlive',
                    'url': 'https://snapcraft.io/',
                    'description': 'Packages universels Linux'
                },
                {
                    'name': 'Flatpak',
                    'install_cmd': 'flatpak install flathub org.texlive.TeXLive',
                    'url': 'https://flatpak.org/',
                    'description': 'Packages universels Linux'
                }
            ],
            'manual_install': {
                'url': 'https://www.tug.org/texlive/',
                'instructions': 'Télécharger et exécuter le script install-tl',
                'description': 'Installation manuelle de TeX Live'
            },
            'alternatives': [
                {
                    'name': 'Overleaf (En ligne)',
                    'url': 'https://www.overleaf.com/',
                    'description': 'Éditeur LaTeX en ligne'
                },
                {
                    'name': 'Typst (Moderne)',
                    'url': 'https://typst.app/',
                    'description': 'Alternative moderne à LaTeX'
                }
            ]
        }

def display_latex_installation_suggestions():
    """Display LaTeX installation suggestions for the current platform"""
    instructions = get_platform_latex_instructions()
    
    print(f"\n{'='*80}")
    print(f"📄 DISTRIBUTION LaTeX MANQUANTE - {instructions['name'].upper()}")
    print(f"{'='*80}")
    print(f"❌ Aucune distribution LaTeX détectée sur votre système {instructions['name']}")
    print(f"📋 Voici les options recommandées pour installer LaTeX :")
    print()
    
    # Distributions principales
    print("🎯 DISTRIBUTIONS RECOMMANDÉES :")
    print("-" * 50)
    for i, dist in enumerate(instructions['distributions'], 1):
        print(f"{i}. {dist['name']}")
        print(f"   📝 Description: {dist['description']}")
        print(f"   🔗 Lien: {dist['url']}")
        print(f"   ⚙️  Installation: {dist['install_cmd']}")
        if 'package_install' in dist:
            print(f"   📦 Packages: {dist['package_install']}")
        print()
    
    # Gestionnaires de packages
    if 'package_managers' in instructions:
        print("📦 GESTIONNAIRES DE PACKAGES :")
        print("-" * 50)
        for pm in instructions['package_managers']:
            print(f"• {pm['name']}")
            print(f"  📝 Description: {pm['description']}")
            print(f"  🔗 Lien: {pm['url']}")
            print(f"  ⚙️  Installation: {pm['install_cmd']}")
            print()
    
    # Installation manuelle
    if 'manual_install' in instructions:
        print("🔧 INSTALLATION MANUELLE :")
        print("-" * 50)
        print(f"• {instructions['manual_install']['description']}")
        print(f"  🔗 Lien: {instructions['manual_install']['url']}")
        print(f"  ⚙️  Instructions: {instructions['manual_install']['instructions']}")
        print()
    
    # Alternatives
    if 'alternatives' in instructions:
        print("🌐 ALTERNATIVES (Pas d'installation requise) :")
        print("-" * 50)
        for alt in instructions['alternatives']:
            print(f"• {alt['name']}")
            print(f"  📝 Description: {alt['description']}")
            print(f"  🔗 Lien: {alt['url']}")
            print()
    
    print("💡 CONSEIL : Pour une installation rapide, nous recommandons :")
    if instructions['name'] == 'Windows':
        print("   1. Télécharger MiKTeX depuis https://miktex.org/download")
        print("   2. Ou utiliser : winget install MiKTeX.MiKTeX")
    elif instructions['name'] == 'macOS':
        print("   1. Télécharger MacTeX depuis https://www.tug.org/mactex/")
        print("   2. Ou utiliser : brew install --cask mactex")
    else:  # Linux
        print("   1. Utiliser votre gestionnaire de packages :")
        print("      • Ubuntu/Debian: sudo apt install texlive-full")
        print("      • Fedora: sudo dnf install texlive-scheme-full")
        print("      • Arch: sudo pacman -S texlive-most")
    
    print(f"\n{'='*80}")
    print("⚠️  Après installation, relancez ce programme pour générer les rapports LaTeX")
    print(f"{'='*80}\n")

def cleanup_latex_temp_files(output_folder):
    """Clean up LaTeX temporary files after compilation"""
    import os
    
    # LaTeX temporary file extensions to clean
    temp_extensions = ['.aux', '.log', '.out', '.synctex.gz', '.toc', '.lof', '.lot', 
                       '.fls', '.fdb_latexmk', '.bbl', '.blg', '.nav', '.snm', '.vrb']
    
    cleaned_files = []
    
    try:
        print(f"🔍 Recherche de fichiers temporaires dans: {output_folder}")
        
        # Check if the folder exists (try both relative and absolute paths)
        if not os.path.exists(output_folder):
            # Try to find the folder in current directory
            current_dir = os.getcwd()
            potential_path = os.path.join(current_dir, output_folder)
            if os.path.exists(potential_path):
                output_folder = potential_path
                print(f"   📁 Dossier trouvé dans le répertoire courant: {output_folder}")
            else:
                # Maybe we're already in the output folder, try current directory
                if os.path.basename(current_dir) == output_folder:
                    output_folder = current_dir
                    print(f"   📁 Nous sommes déjà dans le dossier de sortie: {output_folder}")
                else:
                    print(f"   ❌ Dossier n'existe pas: {output_folder}")
                    print(f"   🔍 Répertoire courant: {current_dir}")
                    return
        
        # Get all files in the directory
        all_files = os.listdir(output_folder)
        print(f"   📁 Fichiers dans le dossier: {len(all_files)}")
        
        # Look for files with temporary extensions
        for filename in all_files:
            file_path = os.path.join(output_folder, filename)
            
            # Check if file has a temporary extension
            should_delete = False
            for ext in temp_extensions:
                if filename.endswith(ext):
                    should_delete = True
                    break
            
            if should_delete:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        cleaned_files.append(filename)
                        print(f"   ✅ Supprimé: {filename}")
                except (OSError, PermissionError) as e:
                    print(f"   ❌ Impossible de supprimer {filename}: {e}")
        
        if cleaned_files:
            print(f"🧹 Nettoyage terminé: {len(cleaned_files)} fichier(s) temporaire(s) supprimé(s)")
        else:
            print("🧹 Aucun fichier temporaire LaTeX trouvé")
            
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage: {e}")

def generate_latex_report(data_by_target, global_data, output_folder):
    """Generates LaTeX report"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  LaTeX report generation requires matplotlib")
        return
    
    import os
    import shutil
    report_path = os.path.join(output_folder, "astronomical_analysis_report.tex")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\\documentclass[12pt]{article}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage[T1]{fontenc}\n")
        f.write("\\usepackage{geometry}\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{longtable}\n")
        f.write("\\usepackage{amsmath}\n")
        f.write("\\usepackage{siunitx}\n")
        f.write("\\usepackage{hyperref}\n")
        f.write("\\usepackage{needspace}\n")
        f.write("\\usepackage{float}\n")
        f.write("\\usepackage{tikz}\n")
        f.write("\\usepackage{xcolor}\n")
        f.write("\\geometry{margin=2.5cm}\n")
        f.write("\\title{Astronomical Analysis Report}\n")
        f.write("\\author{Generated by Astronomical Analysis Program}\n")
        f.write("\\date{\\today}\n")
        f.write("\\begin{document}\n")

        # Draw the physics-accurate triple-atom logo (S II, Hα, O III) directly in TikZ above the title
        f.write("\\begin{center}\n")
        f.write("\\vspace*{-0.5cm}\n")
        f.write(r"""
\begin{tikzpicture}[scale=0.8]
% Pastel/soft hues closer to real emission colors
\definecolor{rouge}{RGB}{200,70,70}   % S II soft red
\definecolor{rougeH}{RGB}{210,60,60}  % H-alpha soft red
\definecolor{cyanO}{RGB}{80,170,185}  % O III soft cyan
\definecolor{bleu}{RGB}{0,0,255}
\definecolor{vert}{RGB}{0,128,0}
\definecolor{orange}{RGB}{255,165,0}
\definecolor{darkgreen}{RGB}{0,100,0}

% --- S II realistic (left): forbidden lines [S II] 6716/6731 Å ---
\begin{scope}[shift={(-6,0)}]
  \node at (0,3.0) {\textbf{\normalsize S II}};
  \node at (0,2.4) {\small Sulfur ion (S$^+$)};
  % Energy levels
  \draw[rouge,opacity=0.9,line width=1pt] (-2,1.4) -- (2,1.4); % ^2D5/2
  \node[anchor=west,rouge] at (2.1,1.4) {$\,\,{}^{2}\,\mathrm{D}_{5/2}$};
  \draw[rouge,opacity=0.9,line width=0.8pt] (-2,0.9) -- (2,0.9); % ^2D3/2
  \node[anchor=west,rouge] at (2.1,0.9) {$\,\,{}^{2}\,\mathrm{D}_{3/2}$};
  \draw[black,line width=0.8pt] (-2,-0.2) -- (2,-0.2); % ^4S3/2
  \node[anchor=west] at (2.1,-0.2) {$\,\,{}^{4}\,\mathrm{S}_{3/2}$};
  % Transitions and labels
  \draw[rouge,opacity=0.9,line width=1pt,->] (-0.45,1.4) -- (-0.45,-0.2);
  \draw[rouge,opacity=0.9,line width=0.6pt,dashed,->] (0.85,0.9) -- (0.85,-0.2);
  \node[rouge] at (-1.1,1.55) {\small\textbf{[S II]}};
  \draw[rouge,opacity=0.9,line width=0.6pt,->] (-0.45,-0.2) -- (1.8,-0.2);
  \node[rouge,fill=white,inner sep=1pt] at (1.0,-0.02) {\tiny 671.6 nm};
  \draw[rouge,opacity=0.9,line width=0.5pt,dashed,->] (0.85,-0.2) -- (1.8,-0.2);
  \node[rouge,fill=white,inner sep=1pt] at (1.05,-0.38) {\tiny 673.1 nm};
  \node at (0,-1.8) {\scriptsize $3\mathrm{s}^{2}3\mathrm{p}^{3}: {}^{2}\,\mathrm{D} \rightarrow {}^{4}\,\mathrm{S}$};
\end{scope}

% --- H\alpha (center): n=3 -> n=2, horizontal-level style ---
\begin{scope}
  \node at (0,3.0) {\textbf{\normalsize H$\alpha$}};
  \node at (0,2.4) {\small Hydrogen atom};
  \draw[rougeH,opacity=0.9,line width=1pt] (-2,1.2) -- (2,1.2);
  \node[anchor=west,rougeH] at (2.1,1.2) {$\,\,\mathit{n}=3$};
  \draw[black,line width=0.8pt] (-2,-0.2) -- (2,-0.2);
  \node[anchor=west] at (2.1,-0.2) {$\,\,\mathit{n}=2$};
  \draw[rougeH,opacity=0.9,line width=1pt,->] (-0.3,1.2) -- (-0.3,-0.2);
  \node[rougeH] at (-1.0,1.45) {\small\textbf{H$\alpha$}};
  \draw[rougeH,opacity=0.9,line width=0.8pt,->] (-0.3,-0.2) -- (1.8,-0.2);
  \node[rougeH,fill=white,inner sep=1pt] at (0.9,-0.05) {\tiny 656.3 nm};
  \node at (0,-1.8) {\scriptsize Balmer: $\mathit{n}=3 \to \mathit{n}=2$};
\end{scope}

% --- O III realistic (right): forbidden lines ---
\begin{scope}[shift={(6,0)}]
  \node at (0,3.0) {\textbf{\normalsize O III}};
  \node at (0,2.4) {\small Doubly ionized oxygen (O$^{2+}$)};
  \draw[cyanO,opacity=0.9,line width=1pt] (-2,1.2) -- (2,1.2);
  \node[anchor=west,cyanO] at (2.1,1.2) {$\,\,{}^{1}\,\mathrm{D}_{2}$};
  \draw[black,line width=0.8pt] (-2,-0.4) -- (2,-0.4);
  \node[anchor=west] at (2.1,-0.4) {$\,\,{}^{3}\,\mathrm{P}_{2}$};
  \draw[black,line width=0.6pt] (-2,-0.8) -- (2,-0.8);
  \node[anchor=west] at (2.1,-0.8) {$\,\,{}^{3}\,\mathrm{P}_{1}$};
  \draw[cyanO,opacity=0.9,line width=1pt,->] (-0.45,1.2) -- (-0.45,-0.4);
  \node[cyanO] at (1.25,1.45) {\small\textbf{[O III]}};
  \draw[cyanO,opacity=0.9,line width=0.6pt,->] (-0.45,-0.4) -- (1.8,-0.4);
  \node[cyanO,fill=white,inner sep=1pt] at (1.12,-0.18) {\tiny 500.7 nm};
  \draw[cyanO,opacity=0.9,line width=0.5pt,dashed,->] (0.45,1.2) -- (0.45,-0.8);
  \draw[cyanO,opacity=0.9,line width=0.5pt,dashed,->] (0.45,-0.8) -- (1.8,-0.8);
  \node[cyanO,fill=white,inner sep=1pt] at (1.12,-0.58) {\tiny 495.9 nm};
  \node at (0,-1.8) {\scriptsize $2\mathrm{p}^{2}: {}^{1}\,\mathrm{D}_{2} \rightarrow {}^{3}\,\mathrm{P}_{2}, {}^{3}\,\mathrm{P}_{1}$};
\end{scope}
\end{tikzpicture}
""")
        f.write("\\vspace{1.2cm}\n")
        f.write("\\Huge \\textbf{Astronomical Analysis}\n")
        f.write("\\vspace{1cm}\n")
        f.write("\\end{center}\n")
        
        # Add photon propagation equation
        f.write("\\vspace{0.5cm}\n")
        f.write("\\begin{center}\n")
        f.write("\\textbf{Photon Propagation Equation in Vacuum:}\n")
        f.write("\\end{center}\n")
        f.write("\\vspace{0.3cm}\n")
        f.write("\\begin{center}\n")
        f.write("\\begin{equation}\n")
        f.write("\\nabla^2 \\vec{E} - \\frac{1}{c^2} \\frac{\\partial^2 \\vec{E}}{\\partial t^2} = \\nabla^2 \\vec{B} - \\frac{1}{c^2} \\frac{\\partial^2 \\vec{B}}{\\partial t^2}\n")
        f.write("\\end{equation}\n")
        f.write("\\end{center}\n")
        f.write("\\vspace{0.3cm}\n")
        f.write("\\begin{center}\n")
        f.write("where $c = 299\\,792\\,458$ m/s is the speed of light in vacuum\n")
        f.write("\\end{center}\n")
        f.write("\\vspace{0.5cm}\n")
        
        # Global summary
        f.write("\\section{Global Summary}\n")
        f.write(f"Total files analyzed: {global_data['total_files']}\n\n")
        f.write(f"Targets found: {len(global_data['found_targets'])}\n\n")
        f.write(f"Telescopes used: {len(global_data['used_telescopes'])}\n\n")
        f.write(f"Total observation time: {format_time_with_details(global_data['total_time'])}\n\n")
        
        # Start Targets Summary on page 2
        f.write("\\newpage\n")
        
        # Add targets summary table
        f.write("\\subsection{Targets Summary}\n")
        f.write("\\begin{longtable}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Target & Time (hours) & Telescope & Files \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endfirsthead\n")
        f.write("\\toprule\n")
        f.write("Target & Time (hours) & Telescope & Files \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endhead\n")
        f.write("\\bottomrule\n")
        f.write("\\endfoot\n")
        f.write("\\bottomrule\n")
        f.write("\\endlastfoot\n")
        
        # Sort targets alphabetically
        target_summary = []
        sorted_targets = sorted(data_by_target.items(), key=lambda x: x[0].lower())
        for target, data in sorted_targets:
            if not data['files']:
                continue
            
            # Skip calibration targets
            if is_calibration_target(target):
                continue
            
            # Use files_by_date to calculate total time (same as detailed section)
            files_by_date = group_files_by_date(data)
            total_time = 0
            for date_data in files_by_date.values():
                total_time += date_data['total_time']
            total_time_hours = total_time / 3600  # Convert to hours
            
            # Get telescope (use first one if multiple)
            telescope = list(data['telescopes'])[0] if data['telescopes'] else 'Unknown'
            
            total_files = len(data['files'])
            
            target_summary.append((target, total_time_hours, telescope, total_files))
        
        # Sort target_summary alphabetically by target name (case-insensitive, improved for astronomical objects)
        target_summary.sort(key=lambda x: get_astronomical_sort_key(x[0]))
        
        # Add rows to table
        for target, time_hours, telescope, files in target_summary:
            f.write(f"{format_target_name_for_latex(target)} & {time_hours:.1f} & {escape_latex(telescope)} & {files} \\\\\n")
        
        f.write("\\end{longtable}\n\n")
        
        # Target details - sort targets alphabetically
        target_count = 0
        
        sorted_targets = sorted(data_by_target.items(), key=lambda x: get_astronomical_sort_key(x[0]))
        for target, data in sorted_targets:
            if not data['files']:
                continue
            
            # Skip calibration targets
            if is_calibration_target(target):
                continue
            
            # Add page break before each target (except the first one)
            if target_count > 0:
                f.write("\\newpage\n")
            
            f.write(f"\\section{{{format_target_name_for_latex(target)}}}\n")
            target_count += 1
            
            # Global summary for this target
            total_files = len(data['files'])
            files_by_date = group_files_by_date(data)
            num_nights = len(files_by_date)
            
            # Calculate total time from files_by_date to avoid duplication
            total_time = 0
            for date_data in files_by_date.values():
                total_time += date_data['total_time']
            
            f.write("\\subsection{Global Summary}\n")
            f.write(f"Files: {total_files}\n\n")
            f.write(f"Total time: {format_time_with_details(total_time)}\n\n")
            f.write(f"Observation nights: {num_nights}\n\n")
            f.write(f"Telescopes: {', '.join(data['telescopes'])}\n\n")
            
            # Check if this is a mosaic target
            if 'panels' in data and data['panels']:
                f.write("\\subsection{Mosaic Panels}\n")
                f.write("This target is composed of multiple mosaic panels:\n\n")
                f.write("\\begin{itemize}\n")
                for panel_num, panel_info in sorted(data['panels'].items(), key=lambda x: int(x[0])):
                    panel_name = panel_info['original_name']
                    panel_files = len(panel_info['files'])
                    panel_time = panel_info['total_time']
                    f.write(f"\\item Panel {panel_num}: {escape_latex(panel_name)} ({panel_files} files, {format_time_with_details(panel_time)})\n")
                f.write("\\end{itemize}\n\n")
            
            # Group files by telescope and instrument combination
            telescope_instrument_groups = {}
            
            # Use files_by_date structure (same as individual analysis) to avoid duplication
            files_by_date = group_files_by_date(data)
            
            # Process each file individually to get correct telescope/instrument associations
            for file_data in data['files']:
                file_info = file_data['info']
                telescope = file_info['info']['telescope']
                instrument = file_info['info']['instrument']
                
                # Skip if both are Unknown (likely not a LIGHT file)
                if telescope == 'Unknown' and instrument == 'Unknown':
                    continue
                
                key = f"{telescope} + {instrument}"
                
                if key not in telescope_instrument_groups:
                    telescope_instrument_groups[key] = {
                        'telescope': telescope,
                        'instrument': instrument,
                        'time_by_filter': {},
                        'total_time': 0
                    }
                
                # Add this file's data
                filter_name = file_info['filter']
                exposure_time = file_info['exposure_time']
                
                if filter_name not in telescope_instrument_groups[key]['time_by_filter']:
                    telescope_instrument_groups[key]['time_by_filter'][filter_name] = []
                
                telescope_instrument_groups[key]['time_by_filter'][filter_name].append(exposure_time)
                telescope_instrument_groups[key]['total_time'] += exposure_time
            
            # Create sections for each telescope/instrument combination
            if not telescope_instrument_groups:
                f.write("\\subsection{Telescope/Instrument Data}\n")
                f.write("No valid telescope/instrument combinations found for this target.\n\n")
            else:
                for group_key, group_data in telescope_instrument_groups.items():
                    telescope = group_data['telescope']
                    instrument = group_data['instrument']
                    group_time = group_data['total_time']
                    group_time_by_filter = group_data['time_by_filter']
                    
                    # Calculate total files from time_by_filter
                    total_group_files = sum(len(time_list) for time_list in group_time_by_filter.values())
                    
                    f.write(f"\\subsection{{{escape_latex(telescope)} + {escape_latex(instrument)}}}\n")
                    f.write(f"Files: {total_group_files}\n\n")
                    f.write(f"Total time: {format_time(group_time)}\n\n")
                    
                    
                    # Filter details for this telescope/instrument combination
                    f.write("\\subsubsection{Filter Distribution}\n")
                    f.write("\\begin{tabular}{lccc}\n")
                    f.write("\\toprule\n")
                    f.write("Filter & Images & Total Time & Average Time \\\\\n")
                    f.write("\\midrule\n")
                    
                    # Define the specific order for filters
                    filter_order = ['L', 'R', 'G', 'B', 'SII', 'Ha', 'OIII']
                    
                    # First, add filters in the specified order
                    for filter_name in filter_order:
                        if filter_name in group_time_by_filter:
                            time_list = group_time_by_filter[filter_name]
                            total_time = sum(time_list)
                            nb_images = len(time_list)
                            average_time = total_time / nb_images
                            
                            f.write(f"{convert_filter_name_to_greek_latex(filter_name)} & {nb_images} & {format_time_hours_minutes(total_time)} & {format_time_hours_minutes(average_time)} \\\\\n")
                    
                    # Then add any remaining filters not in the specified order
                    for filter_name in sorted(group_time_by_filter.keys()):
                        if filter_name not in filter_order:
                            time_list = group_time_by_filter[filter_name]
                            total_time = sum(time_list)
                            nb_images = len(time_list)
                            average_time = total_time / nb_images
                            
                            f.write(f"{convert_filter_name_to_greek_latex(filter_name)} & {nb_images} & {format_time_hours_minutes(total_time)} & {format_time_hours_minutes(average_time)} \\\\\n")
                    
                    f.write("\\bottomrule\n")
                    f.write("\\end{tabular}\n\n")
                    
                    # Night-by-night observation details for this telescope/instrument
                    f.write("\\subsubsection{Observation Details by Night}\n")
                    f.write("\\needspace{3cm}\n")
                    
                    # Filter files for this specific telescope/instrument combination
                    filtered_files = []
                    for file_data in data['files']:
                        file_info = file_data['info']
                        file_telescope = file_info['info']['telescope']
                        file_instrument = file_info['info']['instrument']
                        
                        if file_telescope == telescope and file_instrument == instrument:
                            filtered_files.append(file_data)
                    
                    # Group filtered files by date
                    # Use the original files_by_date structure that contains exposure_details
                    files_by_date = data['files_by_date']
                    
                    # Add total_files and filters fields to each date
                    for date_str, date_data in files_by_date.items():
                        date_data['total_files'] = len(date_data['files'])
                        
                        # Calculate filter statistics from time_by_filter
                        date_data['filters'] = {}
                        for filter_name, time_list in date_data['time_by_filter'].items():
                            total_time = sum(time_list)
                            count = len(time_list)
                            date_data['filters'][filter_name] = {
                                'count': count,
                                'time': total_time
                            }
                    
                    # Sort dates chronologically
                    sorted_dates = sorted(files_by_date.keys())
                    
                    if len(sorted_dates) == 1:
                        # Only one date, show with the date
                        date_str = sorted_dates[0]
                        night_data = files_by_date[date_str]
                        display_date = format_night_display(date_str)
                        f.write(f"\\paragraph{{{display_date}}}\n")
                        f.write(f"Total observation time: {format_time(night_data['total_time'])}\n")
                        f.write(f"Number of images: {night_data['total_files']}\n\n")
                        
                        # Filter breakdown for all observations
                        if night_data['filters']:
                            f.write("\n\\begin{tabular}{lccc}\n")
                            f.write("\\toprule\n")
                            f.write("Filter & Images & Total Time & Average Time \\\\\n")
                            f.write("\\midrule\n")
                            
                            # Use the same filter order
                            filter_order = ['L', 'R', 'G', 'B', 'SII', 'Ha', 'OIII']
                            
                            # First, add filters in the specified order
                            for filter_name in filter_order:
                                if filter_name in night_data['filters']:
                                    stats = night_data['filters'][filter_name]
                                    avg_time = stats['time'] / stats['count'] if stats['count'] > 0 else 0
                                    f.write(f"{convert_filter_name_to_greek_latex(filter_name)} & {stats['count']} & {format_time(stats['time'])} & {format_time(avg_time)} \\\\\n")
                            
                            # Then add any remaining filters not in the specified order
                            for filter_name in sorted(night_data['filters'].keys()):
                                if filter_name not in filter_order:
                                    stats = night_data['filters'][filter_name]
                                    avg_time = stats['time'] / stats['count'] if stats['count'] > 0 else 0
                                    f.write(f"{convert_filter_name_to_greek_latex(filter_name)} & {stats['count']} & {format_time(stats['time'])} & {format_time(avg_time)} \\\\\n")
                            
                            f.write("\\bottomrule\n")
                            f.write("\\end{tabular}\n\n")
                        else:
                            f.write("No filter information available.\n\n")
                    else:
                        # Multiple dates, show each night separately
                        for date_str in sorted_dates:
                            night_data = files_by_date[date_str]
                            # Format night display with readable format
                            display_date = format_night_display(date_str)
                            
                            # Check if we need a page break (only if not enough space)
                            # Reserve space for the entire night section to avoid page breaks
                            f.write("\\needspace{8cm}\n")
                            f.write(f"\\paragraph{{{display_date}}}\n")
                            f.write(f"Total observation time: {format_time(night_data['total_time'])}\n")
                            f.write(f"Number of images: {night_data['total_files']}\n\n")
                            
                            
                            # Filter breakdown for this night
                            if night_data['filters']:
                                f.write("\\begin{table}[H]\n")
                                f.write("\\begin{tabular}{lccc}\n")
                                f.write("\\toprule\n")
                                f.write("Filter & Images & Total Time & Average Time \\\\\n")
                                f.write("\\midrule\n")
                                
                                # Use the same filter order
                                filter_order = ['L', 'R', 'G', 'B', 'SII', 'Ha', 'OIII']
                                
                                # First, add filters in the specified order
                                for filter_name in filter_order:
                                    if filter_name in night_data['filters']:
                                        stats = night_data['filters'][filter_name]
                                        avg_time = stats['time'] / stats['count'] if stats['count'] > 0 else 0
                                        f.write(f"{convert_filter_name_to_greek_latex(filter_name)} & {stats['count']} & {format_time(stats['time'])} & {format_time(avg_time)} \\\\\n")
                                
                                # Then add any remaining filters not in the specified order
                                for filter_name in sorted(night_data['filters'].keys()):
                                    if filter_name not in filter_order:
                                        stats = night_data['filters'][filter_name]
                                        avg_time = stats['time'] / stats['count'] if stats['count'] > 0 else 0
                                        f.write(f"{convert_filter_name_to_greek_latex(filter_name)} & {stats['count']} & {format_time(stats['time'])} & {format_time(avg_time)} \\\\\n")
                                
                                f.write("\\bottomrule\n")
                                f.write("\\end{tabular}\n")
                                f.write("\\end{table}\n\n")
                                
                                # Add detailed exposure breakdown for this night
                                if 'exposure_details' in night_data and night_data['exposure_details']:
                                    f.write("\\needspace{3cm}\n")
                                    f.write("\\textbf{Detailed Exposure Times by Filter:}\n\n")
                                    
                                    # Create a compact horizontal table
                                    f.write("\\begin{table}[H]\n")
                                    f.write("\\small\n")
                                    f.write("\\begin{tabular}{l|")
                                    
                                    # Count total columns needed (all unique exposure times)
                                    all_exposure_times = set()
                                    for filter_details in night_data['exposure_details'].values():
                                        all_exposure_times.update(filter_details.keys())
                                    sorted_exposure_times = sorted(all_exposure_times)
                                    
                                    # Add column headers for each exposure time
                                    for _ in sorted_exposure_times:
                                        f.write("c|")
                                    f.write("}\n")
                                    f.write("\\toprule\n")
                                    
                                    # Header row with exposure times
                                    f.write("Filter")
                                    for exp_time in sorted_exposure_times:
                                        f.write(f" & {exp_time:.0f}s")
                                    f.write(" \\\\\n")
                                    f.write("\\midrule\n")
                                    
                                    # Use the same filter order
                                    filter_order = ['L', 'R', 'G', 'B', 'SII', 'Ha', 'OIII']
                                    
                                    # First, add filters in the specified order
                                    for filter_name in filter_order:
                                        if filter_name in night_data['exposure_details']:
                                            f.write(convert_filter_name_to_greek_latex(filter_name))
                                            for exp_time in sorted_exposure_times:
                                                count = night_data['exposure_details'][filter_name].get(exp_time, 0)
                                                f.write(f" & {count if count > 0 else '-'}")
                                            f.write(" \\\\\n")
                                    
                                    # Then add any remaining filters not in the specified order
                                    for filter_name in sorted(night_data['exposure_details'].keys()):
                                        if filter_name not in filter_order:
                                            f.write(convert_filter_name_to_greek_latex(filter_name))
                                            for exp_time in sorted_exposure_times:
                                                count = night_data['exposure_details'][filter_name].get(exp_time, 0)
                                                f.write(f" & {count if count > 0 else '-'}")
                                            f.write(" \\\\\n")
                                    
                                    f.write("\\bottomrule\n")
                                    f.write("\\end{tabular}\n")
                                    f.write("\\end{table}\n\n")
                            else:
                                f.write("No filter information available for this night.\n\n")
            
            
        
        f.write("\\end{document}\n")
    
    print(f"LaTeX report generated: {report_path}")
    
    # Try to compile LaTeX to PDF
    try:
        import subprocess
        
        # Find LaTeX executable
        latex_exe = find_latex_executable()
        
        if not latex_exe:
            platform_info = get_platform_latex_instructions()
            required_latex_packages = get_required_latex_packages()
            required_python_packages = get_required_python_packages()
            
            print(f"⚠️  LaTeX not found on this {platform_info['name']} system.")
            print("   📄 LaTeX report file generated: astronomical_analysis_report.tex")
            print("   🌐 You can compile it online with Overleaf:")
            print("      https://www.overleaf.com/")
            print("   📋 Overleaf Instructions:")
            print("      1. Go to https://www.overleaf.com/")
            print("      2. Create a new project")
            print("      3. Upload the .tex file from your output folder")
            print("      4. Click 'Compile' to generate the PDF")
            print(f"   💡 Or install LaTeX locally on {platform_info['name']}:")
            print()
            
            # Show distribution options
            if 'distributions' in platform_info:
                print("   📦 LaTeX Distributions:")
                for dist in platform_info['distributions']:
                    print(f"      • {dist['name']}")
                    print(f"        URL: {dist['url']}")
                    print(f"        Install: {dist['install_cmd']}")
                    if 'package_install' in dist:
                        print(f"        Packages: {dist['package_install']}")
                    print()
            
            # Show package managers
            if 'package_managers' in platform_info:
                print("   📦 Package Managers:")
                for pm in platform_info['package_managers']:
                    print(f"      • {pm['name']}: {pm['install_cmd']}")
                print()
            elif 'package_manager' in platform_info:
                pm = platform_info['package_manager']
                print(f"   📦 Package Manager: {pm['name']}")
                print(f"      Install: {pm['install_cmd']}")
                print()
            
            # Show required LaTeX packages
            print("   📋 Required LaTeX Packages:")
            print("      The following packages are needed for the report:")
            for i, pkg in enumerate(required_latex_packages, 1):
                print(f"      {i:2d}. {pkg}")
            print()
            
            # Show required Python packages
            print("   🐍 Required Python Packages:")
            print("      Install with: pip install <package_name>")
            for i, pkg in enumerate(required_python_packages, 1):
                print(f"      {i:2d}. {pkg}")
            print()
            
            # Show installation commands
            if platform_info['name'] == 'Windows':
                print("   🚀 Quick Installation Commands:")
                print("      # Using Chocolatey (if installed):")
                print("      choco install miktex")
                print("      # Or download MiKTeX from: https://miktex.org/download")
                print()
            elif platform_info['name'] == 'macOS':
                print("   🚀 Quick Installation Commands:")
                print("      # Using Homebrew (if installed):")
                print("      brew install --cask mactex")
                print("      # Or download MacTeX from: https://www.tug.org/mactex/")
                print()
            else:  # Linux
                print("   🚀 Quick Installation Commands:")
                print("      # Ubuntu/Debian:")
                print("      sudo apt update && sudo apt install texlive-full texlive-latex-extra")
                print("      # Fedora:")
                print("      sudo dnf install texlive-scheme-full")
                print("      # Arch Linux:")
                print("      sudo pacman -S texlive-most texlive-lang")
                print()
            
            return
        
        print(f"📄 Found LaTeX: {latex_exe}")
        print("📄 Compiling LaTeX to PDF...")
        
        # Change to output directory
        original_dir = os.getcwd()
        os.chdir(output_folder)
        
        try:
            # Run pdflatex (try multiple times for cross-references)
            result = subprocess.run([latex_exe, '-interaction=nonstopmode', 'astronomical_analysis_report.tex'], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ PDF generated successfully: astronomical_analysis_report.pdf")
                
                # Run pdflatex again for cross-references
                subprocess.run([latex_exe, '-interaction=nonstopmode', 'astronomical_analysis_report.tex'], 
                              capture_output=True, text=True, timeout=60)
                print("✅ PDF cross-references updated")
            else:
                print("⚠️  LaTeX compilation failed:")
                print(f"   Error: {result.stderr}")
                print("   📄 LaTeX report file available: astronomical_analysis_report.tex")
                print("   🌐 You can compile it online with Overleaf:")
                print("      https://www.overleaf.com/")
                
        except subprocess.TimeoutExpired:
            print("⚠️  LaTeX compilation timed out")
            print("   📄 LaTeX report file available: astronomical_analysis_report.tex")
            print("   🌐 You can compile it online with Overleaf:")
            print("      https://www.overleaf.com/")
        except FileNotFoundError:
            print("⚠️  LaTeX executable not found in PATH")
            print("   📄 LaTeX report file available: astronomical_analysis_report.tex")
            print("   🌐 You can compile it online with Overleaf:")
            print("      https://www.overleaf.com/")
            
            # Try alternative PDF generation
            if REPORTLAB_AVAILABLE:
                print("🔄 Attempting PDF generation without LaTeX...")
                generate_pdf_report_without_latex(data_by_target, global_data, output_folder)
            else:
                print("💡 To generate PDF without LaTeX, install reportlab: pip install reportlab")
                
        except Exception as e:
            print(f"⚠️  LaTeX compilation error: {e}")
            
            # Try alternative PDF generation
            if REPORTLAB_AVAILABLE:
                print("🔄 Attempting PDF generation without LaTeX...")
                generate_pdf_report_without_latex(data_by_target, global_data, output_folder)
            else:
                print("💡 To generate PDF without LaTeX, install reportlab: pip install reportlab")
            
        finally:
            # Clean up LaTeX temporary files
            cleanup_latex_temp_files(output_folder)
            # Return to original directory
            os.chdir(original_dir)
            
    except ImportError:
        print("⚠️  subprocess module not available for LaTeX compilation")
    except Exception as e:
        print(f"⚠️  Error during LaTeX compilation: {e}")

def generate_pdf_report_without_latex(data_by_target, global_data, output_folder):
    """Generates a PDF report without LaTeX using reportlab"""
    if not REPORTLAB_AVAILABLE:
        print("⚠️  PDF generation without LaTeX requires reportlab")
        print("   Install with: pip install reportlab")
        return
    
    print(f"\nGENERATING PDF REPORT (without LaTeX)")
    print("=" * 80)
    
    # Create PDF file
    pdf_path = os.path.join(output_folder, "astronomical_analysis_report.pdf")
    
    try:
        # Create document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12,
            textColor=colors.darkblue
        )
        normal_style = styles['Normal']
        
        # Title
        story.append(Paragraph("Astronomical Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Global Summary
        story.append(Paragraph("Global Summary", heading_style))
        
        # Global statistics
        total_files = global_data['total_files']
        total_targets = len([t for t in data_by_target.keys() if not is_calibration_target(t)])
        total_time_hours = global_data['total_time'] / 3600
        
        global_text = f"""
        <b>Total files analyzed:</b> {total_files}<br/>
        <b>Targets found:</b> {total_targets}<br/>
        <b>Total observation time:</b> {format_time_with_details(global_data['total_time'])}<br/>
        <b>Telescopes used:</b> {len(global_data['used_telescopes'])}<br/>
        """
        story.append(Paragraph(global_text, normal_style))
        story.append(Spacer(1, 20))
        
        # Targets Summary Table
        story.append(Paragraph("Targets Summary", heading_style))
        
        # Prepare targets data
        target_data = []
        for target, data in data_by_target.items():
            if not data['files'] or is_calibration_target(target):
                continue
            
            # Calculate total time
            files_by_date = group_files_by_date(data)
            total_time = 0
            for date_data in files_by_date.values():
                total_time += date_data['total_time']
            total_time_hours = total_time / 3600
            
            # Get telescope
            telescope = list(data['telescopes'])[0] if data['telescopes'] else 'Unknown'
            total_files = len(data['files'])
            
            target_data.append([target, f"{total_time_hours:.1f}", telescope, str(total_files)])
        
        # Sort by observation time
        target_data.sort(key=lambda x: float(x[1]), reverse=True)
        
        # Create table
        table_data = [['Target', 'Time (h)', 'Telescope', 'Files']] + target_data
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(PageBreak())
        
        # Individual target details
        for target, data in data_by_target.items():
            if not data['files'] or is_calibration_target(target):
                continue
            
            story.append(Paragraph(target, heading_style))
            
            # Target summary
            total_files = len(data['files'])
            files_by_date = group_files_by_date(data)
            total_time = 0
            for date_data in files_by_date.values():
                total_time += date_data['total_time']
            
            target_summary = f"""
            <b>Files:</b> {total_files}<br/>
            <b>Total time:</b> {format_time_with_details(total_time)}<br/>
            <b>Observation nights:</b> {len(files_by_date)}<br/>
            <b>Telescopes:</b> {', '.join(data['telescopes'])}<br/>
            """
            story.append(Paragraph(target_summary, normal_style))
            
            # Filter distribution
            story.append(Paragraph("Filter Distribution", heading_style))
            
            # Aggregate filter data
            filter_data = []
            for date_data in files_by_date.values():
                for filter_name, time_list in date_data['time_by_filter'].items():
                    if filter_name not in [row[0] for row in filter_data]:
                        filter_data.append([filter_name, 0, 0])
                    
                    # Find existing row and update
                    for row in filter_data:
                        if row[0] == filter_name:
                            row[1] += len(time_list)
                            row[2] += sum(time_list)
                            break
            
            # Sort by filter name
            filter_data.sort(key=lambda x: x[0])
            
            # Create filter table
            filter_table_data = [['Filter', 'Images', 'Total Time', 'Average Time']]
            for filter_name, count, total_time in filter_data:
                avg_time = total_time / count if count > 0 else 0
                filter_table_data.append([
                    filter_name,
                    str(count),
                    format_time(total_time),
                    format_time(avg_time)
                ])
            
            filter_table = Table(filter_table_data)
            filter_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(filter_table)
            story.append(PageBreak())
        
        # Build PDF
        doc.build(story)
        print(f"✅ PDF report generated: {pdf_path}")
        
    except Exception as e:
        print(f"❌ Error generating PDF report: {e}")

def get_season_from_date(date_str):
    """Determines the season (Northern Hemisphere) from a date string.
    Accepts formats: YYYY-MM-DD, YYYY/MM/DD, YYYY-MM-DDTHH:MM:SS, YYYY-MM-DD HH:MM:SS.
    Uses meteorological seasons for robustness: Spring=Mar-May, Summer=Jun-Aug, Autumn=Sep-Nov, Winter=Dec-Feb.
    """
    from datetime import datetime
    if not isinstance(date_str, str):
        return 'Unknown'
    candidates = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S'
    ]
    date_obj = None
    for fmt in candidates:
        try:
            date_obj = datetime.strptime(date_str[:19], fmt)
            break
        except Exception:
            continue
    if date_obj is None:
        # Try to extract just the date part before 'T' or space
        basic = date_str.split('T')[0].split(' ')[0].replace('/', '-')
        try:
            date_obj = datetime.strptime(basic, '%Y-%m-%d')
        except Exception:
            return 'Unknown'
    m = date_obj.month
    if m in (3, 4, 5):
        return 'Spring'
    if m in (6, 7, 8):
        return 'Summer'
    if m in (9, 10, 11):
        return 'Autumn'
    return 'Winter'

def generate_graphs(data_by_target, global_data, output_folder):
    """Generates combined analysis graphs in a single PNG file"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Graph generation requires matplotlib")
        return
    
    print(f"\nGENERATING COMBINED GRAPHS")
    print("=" * 80)
    
    # Set style
    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Time distribution by filter (top left)
    print("📊 Generating time distribution graph...")
    ax1 = plt.subplot(2, 3, 1)
    
    filter_times = {}
    for target, data in data_by_target.items():
        # Skip calibration targets
        if is_calibration_target(target):
            continue
            
        # Use files_by_date instead of time_by_filter
        if 'files_by_date' in data:
            for date_data in data['files_by_date'].values():
                for filter_name, time_list in date_data['time_by_filter'].items():
                    if filter_name not in filter_times:
                        filter_times[filter_name] = []
                    filter_times[filter_name].extend(time_list)
        else:
            # Fallback to time_by_filter if files_by_date not available
            for filter_name, time_list in data['time_by_filter'].items():
                if filter_name not in filter_times:
                    filter_times[filter_name] = []
                filter_times[filter_name].extend(time_list)
    
    if filter_times:
        # Desired display order
        desired_order = ['L', 'R', 'G', 'B', 'SII', 'HA', 'OIII']
        # Normalize keys for ordering (map variants to canonical)
        def canonical(f):
            u = f.upper()
            if u in ('LUM', 'L'): return 'L'
            if u in ('H-ALPHA', 'HA'): return 'HA'
            if u in ('O3', 'OIII'): return 'OIII'
            if u in ('S2', 'SII'): return 'SII'
            return u
        # Build ordered list of filters present according to desired order, then append others
        present = list(filter_times.keys())
        ordered = [f for key in desired_order for f in present if canonical(f) == key]
        others = [f for f in present if f not in ordered]
        filters = ordered + others
        times = [sum(filter_times[f]) for f in filters]

        # Consistent colors per filter
        filter_colors = {
            'HA': '#d62728', 'Ha': '#d62728', 'H-ALPHA': '#d62728',
            'OIII': '#1f77b4', 'O3': '#1f77b4',
            'SII': '#9467bd', 'S2': '#9467bd',
            'L': '#7f7f7f', 'LUM': '#7f7f7f',
            'R': '#e41a1c',
            'G': '#4daf4a',
            'B': '#377eb8',
            'RGB': '#ff7f00',
            'OSC': '#ff7f00'
        }

        # Convert filter names to Greek characters for display
        display_filters = [convert_filter_name_to_greek_matplotlib(f) for f in filters]
        bar_colors = [filter_colors.get(f, '#87CEFA') for f in filters]

        # Convert times from seconds to hours for display
        times_hours = [t / 3600 for t in times]
        bars = ax1.bar(display_filters, times_hours, color=bar_colors, edgecolor='navy', alpha=0.85)
        ax1.set_xlabel('Filter')
        ax1.set_ylabel('Total Time (hours)')
        ax1.set_title('Total Observation Time by Filter')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time_hours in zip(bars, times_hours):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_hours:.1f}h', ha='center', va='bottom')
    
    # 2. Average exposure time by filter (top center)
    print("📊 Generating average exposure time by filter...")
    ax2 = plt.subplot(2, 3, 2)
    
    # Collect exposure time data by filter
    filter_exposure_times = {}
    for target, data in data_by_target.items():
        # Skip calibration targets
        if is_calibration_target(target):
            continue
            
        if 'files_by_date' in data:
            for date_data in data['files_by_date'].values():
                for filter_name, time_list in date_data['time_by_filter'].items():
                    if filter_name not in filter_exposure_times:
                        filter_exposure_times[filter_name] = []
                    filter_exposure_times[filter_name].extend(time_list)
        else:
            # Fallback to time_by_filter if files_by_date not available
            for filter_name, time_list in data['time_by_filter'].items():
                if filter_name not in filter_exposure_times:
                    filter_exposure_times[filter_name] = []
                filter_exposure_times[filter_name].extend(time_list)
    
    if filter_exposure_times:
        # Desired display order
        desired_order = ['L', 'R', 'G', 'B', 'SII', 'HA', 'OIII']
        def canonical(f):
            u = f.upper()
            if u in ('LUM', 'L'): return 'L'
            if u in ('H-ALPHA', 'HA'): return 'HA'
            if u in ('O3', 'OIII'): return 'OIII'
            if u in ('S2', 'SII'): return 'SII'
            return u
        present = list(filter_exposure_times.keys())
        ordered = [f for key in desired_order for f in present if canonical(f) == key]
        others = [f for f in present if f not in ordered]
        filters = ordered + others
        avg_exposure_times = [sum(filter_exposure_times[f]) / len(filter_exposure_times[f]) for f in filters]

        # Consistent colors per filter
        filter_colors = {
            'HA': '#d62728', 'Ha': '#d62728', 'H-ALPHA': '#d62728',
            'OIII': '#1f77b4', 'O3': '#1f77b4',
            'SII': '#9467bd', 'S2': '#9467bd',
            'L': '#7f7f7f', 'LUM': '#7f7f7f',
            'R': '#e41a1c',
            'G': '#4daf4a',
            'B': '#377eb8',
            'RGB': '#ff7f00',
            'OSC': '#ff7f00'
        }

        # Convert filter names to Greek characters for display
        display_filters = [convert_filter_name_to_greek_matplotlib(f) for f in filters]
        bar_colors = [filter_colors.get(f, '#F08080') for f in filters]

        bars = ax2.bar(display_filters, avg_exposure_times, color=bar_colors, edgecolor='darkred', alpha=0.85)
        ax2.set_xlabel('Filter')
        ax2.set_ylabel('Average Exposure Time (seconds)')
        ax2.set_title('Average Exposure Time by Filter')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars, avg_exposure_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.1f}s', ha='center', va='bottom')
    else:
        ax2.set_xlabel('Filter')
        ax2.set_ylabel('Average Exposure Time (seconds)')
        ax2.set_title('Average Exposure Time by Filter (No Data)')
        ax2.text(0.5, 0.5, 'No exposure data available', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=12, color='red')
    
    # 3. Target time comparison (top right)
    print("📊 Generating target time comparison...")
    ax3 = plt.subplot(2, 3, 3)
    
    # Limit to top 10 targets by observation time to keep graph readable
    target_time_pairs = []
    for target, data in data_by_target.items():
        # Skip calibration targets
        if is_calibration_target(target):
            continue
            
        # Use files_by_date instead of time_by_filter
        if 'files_by_date' in data:
            total_time = 0
            for date_data in data['files_by_date'].values():
                total_time += date_data['total_time']
        else:
            # Fallback to time_by_filter if files_by_date not available
            total_time = sum(sum(times) for times in data['time_by_filter'].values())
        target_time_pairs.append((target, total_time))
    
    # Sort by time (descending) and take top 10
    target_time_pairs.sort(key=lambda x: x[1], reverse=True)
    top_targets = target_time_pairs[:10]
    
    if top_targets:
        targets = [pair[0] for pair in top_targets]
        target_times = [pair[1] for pair in top_targets]
        
        # Truncate long target names for better display
        display_targets = []
        for target in targets:
            if len(target) > 20:
                display_targets.append(target[:17] + '...')
            else:
                display_targets.append(target)
        
        bars = ax3.bar(display_targets, target_times, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        ax3.set_xlabel('Target')
        ax3.set_ylabel('Total Time (seconds)')
        ax3.set_title('Top 10 Targets by Observation Time')
        ax3.tick_params(axis='x', rotation=90)
        
        # Add extra margin at the top to prevent annotation from masking bars
        y_max = max(target_times) if target_times else 0
        ax3.set_ylim(0, y_max * 1.15)  # Add 15% margin at the top
        
        # Adjust layout to prevent label overlap
        plt.tight_layout()
        
        # Add value labels on bars
        for bar, time in zip(bars, target_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                   f'{format_time_hours_minutes(time)}', ha='center', va='bottom')
        
        # Add note if there are more targets
        if len(data_by_target) > 10:
            # Position the annotation at the bottom to avoid masking bars
            ax3.text(0.02, 0.02, f'Showing top 10 of {len(data_by_target)} targets', 
                    transform=ax3.transAxes, fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    else:
        ax3.set_xlabel('Target')
        ax3.set_ylabel('Total Time (seconds)')
        ax3.set_title('Observation Time by Target (No Data)')
        ax3.text(0.5, 0.5, 'No target data available', transform=ax3.transAxes, 
                ha='center', va='center', fontsize=12, color='red')
    
    # 4. Target files comparison (bottom left)
    print("📊 Generating target files comparison...")
    ax4 = plt.subplot(2, 3, 4)
    
    # Collect all targets with their file counts and sort by files (descending)
    target_file_pairs = []
    for target, data in data_by_target.items():
        # Skip calibration targets
        if is_calibration_target(target):
            continue
        total_files = len(data['files'])
        target_file_pairs.append((target, total_files))
    
    # Sort by file count (descending) and take top 10
    target_file_pairs.sort(key=lambda x: x[1], reverse=True)
    top_file_targets = target_file_pairs[:10]
    
    if top_file_targets:
        targets = [pair[0] for pair in top_file_targets]
        target_files = [pair[1] for pair in top_file_targets]
        
        # Truncate long target names for better display
        display_targets = []
        for target in targets:
            if len(target) > 20:
                display_targets.append(target[:17] + '...')
            else:
                display_targets.append(target)
        
        bars = ax4.bar(display_targets, target_files, color='orange', edgecolor='darkorange', alpha=0.7)
        ax4.set_xlabel('Target')
        ax4.set_ylabel('Number of Files')
        ax4.set_title('Top 10 Targets by Number of Files')
        ax4.tick_params(axis='x', rotation=90)
        
        # Add extra margin at the top to prevent annotation from masking bars
        y_max = max(target_files) if target_files else 0
        ax4.set_ylim(0, y_max * 1.15)  # Add 15% margin at the top
        
        # Adjust layout to prevent label overlap
        plt.tight_layout()
        
        # Add value labels on bars
        for bar, files in zip(bars, target_files):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                   f'{files}', ha='center', va='bottom')
        
        # Add note if there are more targets
        if len(data_by_target) > 10:
            # Position the annotation at the bottom to avoid masking bars
            ax4.text(0.02, 0.02, f'Showing top 10 of {len(data_by_target)} targets', 
                    transform=ax4.transAxes, fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    else:
        ax4.set_xlabel('Target')
        ax4.set_ylabel('Number of Files')
        ax4.set_title('Number of Files by Target (No Data)')
        ax4.text(0.5, 0.5, 'No target data available', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=12, color='red')
    
    # 5. Observation hours by season (bottom center)
    print("📊 Generating observation hours by season...")
    ax5 = plt.subplot(2, 3, 5)
    
    # Collect observation data by season
    season_data = {}
    for target, data in data_by_target.items():
        # Skip calibration targets
        if is_calibration_target(target):
            continue
            
        if 'files_by_date' in data:
            for date_str, date_data in data['files_by_date'].items():
                season = get_season_from_date(date_str)
                if season not in season_data:
                    season_data[season] = 0
                season_data[season] += date_data['total_time']
        else:
            # Fallback: try to extract dates from other sources
            if 'dates' in data and data['dates']:
                for date_str in data['dates']:
                    season = get_season_from_date(date_str)
                    if season not in season_data:
                        season_data[season] = 0
                    # Estimate time per date (total time / number of dates)
                    estimated_time_per_date = data['total_time'] / len(data['dates']) if data['dates'] else 0
                    season_data[season] += estimated_time_per_date
    
    if season_data:
        seasons = list(season_data.keys())
        # Sort seasons in chronological order
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        seasons = [s for s in season_order if s in seasons] + [s for s in seasons if s not in season_order]
        
        times = [season_data[s] for s in seasons]
        hours = [t / 3600 for t in times]  # Convert to hours
        
        # Color mapping for seasons
        season_colors = {
            'Spring': '#90EE90',  # Light green
            'Summer': '#FFD700',  # Gold
            'Autumn': '#FF8C00',  # Dark orange
            'Winter': '#87CEEB',  # Sky blue
            'Unknown': '#D3D3D3'  # Light gray
        }
        
        colors = [season_colors.get(season, '#D3D3D3') for season in seasons]
        
        bars = ax5.bar(seasons, hours, color=colors, edgecolor='black', alpha=0.8)
        ax5.set_xlabel('Season')
        ax5.set_ylabel('Observation Hours')
        ax5.set_title('Observation Hours by Season')
        ax5.tick_params(axis='x', rotation=45)
        
        # Add extra margin at the top to prevent annotation from masking bars
        y_max = max(hours) if hours else 0
        ax5.set_ylim(0, y_max * 1.15)  # Add 15% margin at the top
        
        # Add value labels on bars
        for bar, hours_val in zip(bars, hours):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                   f'{hours_val:.1f}h', ha='center', va='bottom')
        
        # Add total hours annotation
        total_hours = sum(hours)
        # Position the annotation at the bottom to avoid masking bars
        ax5.text(0.02, 0.02, f'Total: {total_hours:.1f} hours', 
                transform=ax5.transAxes, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    else:
        ax5.set_xlabel('Season')
        ax5.set_ylabel('Observation Hours')
        ax5.set_title('Observation Hours by Season (No Data)')
        ax5.text(0.5, 0.5, 'No seasonal data available', transform=ax5.transAxes, 
                ha='center', va='center', fontsize=12, color='red')
    
    # 6. Global statistics summary (bottom right)
    print("📊 Generating global statistics summary...")
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate global statistics
    total_files = global_data['total_files']
    total_time = global_data['total_time']
    total_targets = len(data_by_target)
    # Calculate total filters from files_by_date
    all_filters = set()
    for target, data in data_by_target.items():
        # Skip calibration targets
        if is_calibration_target(target):
            continue
            
        if 'files_by_date' in data:
            for date_data in data['files_by_date'].values():
                all_filters.update(date_data['time_by_filter'].keys())
        else:
            all_filters.update(data['time_by_filter'].keys())
    total_filters = len(all_filters)
    
    # Calculate filter statistics
    filter_stats = {}
    for target, data in data_by_target.items():
        # Skip calibration targets
        if is_calibration_target(target):
            continue
            
        if 'files_by_date' in data:
            for date_data in data['files_by_date'].values():
                for filter_name, time_list in date_data['time_by_filter'].items():
                    if filter_name not in filter_stats:
                        filter_stats[filter_name] = {'files': 0, 'time': 0}
                    filter_stats[filter_name]['files'] += len(time_list)
                    filter_stats[filter_name]['time'] += sum(time_list)
        else:
            for filter_name, time_list in data['time_by_filter'].items():
                if filter_name not in filter_stats:
                    filter_stats[filter_name] = {'files': 0, 'time': 0}
                filter_stats[filter_name]['files'] += len(time_list)
                filter_stats[filter_name]['time'] += sum(time_list)
    
    # Create summary text
    summary_text = f"""
GLOBAL STATISTICS SUMMARY

Total Files: {total_files}
Total Time: {format_time_hours_minutes(total_time)}
Targets: {total_targets}
Filters: {total_filters}

FILTER DETAILS:
"""
    
    # Add filter details in specific order
    filter_order = ['L', 'R', 'G', 'B', 'SII', 'Ha', 'OIII']
    for filter_name in filter_order:
        if filter_name in filter_stats:
            stats = filter_stats[filter_name]
            summary_text += f"• {filter_name}: {stats['files']} files, {format_time_hours_minutes(stats['time'])}\n"
    
    # Add any remaining filters not in the specified order
    for filter_name in sorted(filter_stats.keys()):
        if filter_name not in filter_order:
            stats = filter_stats[filter_name]
            summary_text += f"• {filter_name}: {stats['files']} files, {format_time_hours_minutes(stats['time'])}\n"
    
    summary_text += "\nTARGETS:\n"
    for target in targets:
        data = data_by_target[target]
        target_time = sum(sum(times) for times in data['time_by_filter'].values())
        target_files = len(data['files'])
        summary_text += f"• {target}: {target_files} files, {format_time_hours_minutes(target_time)}\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Add main title
    fig.suptitle('Astronomical Analysis - Complete Statistics', fontsize=16, fontweight='bold')
    
    # Adjust layout and save - give more space at bottom for vertical labels
    plt.subplots_adjust(top=0.93, bottom=0.15, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
    
    # Save the combined graph in both PNG and SVG formats
    combined_graph_png_path = os.path.join(output_folder, 'astronomical_analysis_complete.png')
    combined_graph_svg_path = os.path.join(output_folder, 'astronomical_analysis_complete.svg')
    
    # Save as PNG (high resolution)
    plt.savefig(combined_graph_png_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ PNG graph saved: {combined_graph_png_path}")
    
    # Save as SVG (vector format)
    plt.savefig(combined_graph_svg_path, format='svg', bbox_inches='tight')
    print(f"   ✅ SVG graph saved: {combined_graph_svg_path}")
    
    plt.close()
    
    print("✅ All graphs generated successfully in both PNG and SVG formats")

def compress_output_folder(output_folder):
    """Compresses the output folder to a ZIP file"""
    try:
        zip_path = output_folder + ".zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_folder)
                    zipf.write(file_path, arcname)
        
        print(f"📦 Output folder compressed to: {zip_path}")
        return zip_path
        
    except Exception as e:
        print(f"❌ Error compressing output folder: {e}")
        return None

def save_config(config):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"💾 Configuration saved to: {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def load_config():
    """Load configuration from file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(config):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving configuration: {e}")

def get_sensor_info(camera_model):
    """Get sensor information from database"""
    return SENSORS_DATABASE.get(camera_model, SENSORS_DATABASE['default'])

def get_telescope_info(telescope_model):
    """Get telescope information from database"""
    return TELESCOPES_DATABASE.get(telescope_model, TELESCOPES_DATABASE['default'])

def extract_filter_from_filename(filename):
    """Extract filter information from filename"""
    filename_upper = filename.upper()
    
    # Direct filter matches (including substrings like H-Alpha, Light, Clear etc.)
    for filter_code, info in FILTERS_INFO.items():
        if filter_code.upper() in filename_upper:
            return filter_code, info
    
        # Pattern matching for common naming conventions - Enhanced for all variants
        patterns = {
            # H-Alpha variants - comprehensive coverage
            r'\bH[-_\s]?A(LPHA)?\b': 'HA',
            r'\bH[-_\s]?A\b': 'HA',  # H-A, H_A, H A
            r'\bHA\b': 'HA',  # HA, ha, Ha
            r'\bHALPHA\b': 'HA',  # HALPHA, halpha, Halpha
            r'\bHYDROGEN[-_\s]?ALPHA\b': 'HA',  # HYDROGEN ALPHA, HYDROGEN-ALPHA
            r'\bH[-_\s]?ALPHA\b': 'HA',  # H-ALPHA, H_ALPHA, H ALPHA
            
            # H-Beta variants
            r'\bH[-_\s]?B(ETA)?\b': 'HBETA',
            r'\bH[-_\s]?B\b': 'HBETA',  # H-B, H_B, H B
            r'\bHB\b': 'HBETA',  # HB, hb, Hb
            r'\bHBETA\b': 'HBETA',  # HBETA, hbeta, Hbeta
            r'\bHYDROGEN[-_\s]?BETA\b': 'HBETA',  # HYDROGEN BETA, HYDROGEN-BETA
            
            # Support for Greek characters in filter names
            r'\bHα\b': 'HA',
            r'\bHβ\b': 'HBETA',
            r'\bHγ\b': 'HGAMMA',
            r'\bHδ\b': 'HDELTA',
            r'\bHε\b': 'HEPSILON',
            r'\bHζ\b': 'HZETA',
            r'\bHη\b': 'HETA',
            r'\bHθ\b': 'HTHETA',
            r'\bHι\b': 'HIOTA',
            r'\bHκ\b': 'HKAPPA',
            r'\bHλ\b': 'HLAMBDA',
            r'\bHμ\b': 'HMU',
            r'\bHν\b': 'HNU',
            r'\bHξ\b': 'HXI',
            r'\bHο\b': 'HOMICRON',
            r'\bHπ\b': 'HPI',
            r'\bHρ\b': 'HRHO',
            r'\bHσ\b': 'HSIGMA',
            r'\bHτ\b': 'HTAU',
            r'\bHυ\b': 'HUPSILON',
            r'\bHφ\b': 'HPHI',
            r'\bHχ\b': 'HCHI',
            r'\bHψ\b': 'HPSI',
            r'\bHω\b': 'HOMEGA',
            
            # OIII variants - comprehensive coverage
            r'\bOIII\b': 'OIII',
            r'\bO3\b': 'OIII',  # O3, o3
            r'\bO[-_\s]?3\b': 'OIII',  # O-3, O_3, O 3
            r'\bOXYGEN[-_\s]?III\b': 'OIII',  # OXYGEN III, OXYGEN-III
            r'\bOXYGEN[-_\s]?3\b': 'OIII',  # OXYGEN 3, OXYGEN-3
            
            # SII variants - comprehensive coverage  
            r'\bSII\b': 'SII',
            r'\bS2\b': 'SII',  # S2, s2
            r'\bS[-_\s]?2\b': 'SII',  # S-2, S_2, S 2
            r'\bSULFUR[-_\s]?II\b': 'SII',  # SULFUR II, SULFUR-II
            r'\bSULFUR[-_\s]?2\b': 'SII',  # SULFUR 2, SULFUR-2
            
            # NII variants
            r'\bNII\b': 'NII',
            r'\bN2\b': 'NII',  # N2, n2
            r'\bN[-_\s]?2\b': 'NII',  # N-2, N_2, N 2
            r'\bNITROGEN[-_\s]?II\b': 'NII',  # NITROGEN II, NITROGEN-II
            r'\bNITROGEN[-_\s]?2\b': 'NII',  # NITROGEN 2, NITROGEN-2
            
            # HEII variants
            r'\bHEII\b': 'HEII',
            r'\bHE[-_\s]?II\b': 'HEII',  # HE-II, HE_II, HE II
            r'\bHE[-_\s]?2\b': 'HEII',  # HE-2, HE_2, HE 2
            r'\bHELIUM[-_\s]?II\b': 'HEII',  # HELIUM II, HELIUM-II
            r'\bHELIUM[-_\s]?2\b': 'HEII',  # HELIUM 2, HELIUM-2
            
            # Luminance variants - comprehensive coverage
            r'\bLUM(INANCE)?\b': 'LUM',
            r'\bLUMINANCE\b': 'LUM',  # LUMINANCE, luminance, Luminance
            r'\bLUM\b': 'LUM',  # LUM, lum, Lum
            r'\bLIGHT\b': 'LUM',  # LIGHT, light, Light
            r'\bL\b': 'L',  # L, l (single letter)
            
            # Clear filter
            r'\bCLEAR\b': 'CLEAR',
            
            # Other filters
            r'\bIR[-_\s]?CUT\b': 'IRCUT',
            r'\bUV[-_\s]?IR\b': 'UVIR',
            r'\bARGON\b': 'ARGON',
            r'\bARIII\b': 'ARIII',
            r'\bARIV\b': 'ARIV',
            r'\bARV\b': 'ARV',
            r'\bNEON\b': 'NEON',
            r'\bKRYPTON\b|\bKR\b': 'KRYPTON',
            r'\bXENON\b|\bXE\b': 'XENON',
            r'\bHE[-_\s]?I\b|\bHELIUM\b': 'HEI',
            r'\bSODIUM\b|\bNA\b': 'SODIUM',
            r'\bPOTASSIUM\b|\bK\b': 'K',
            r'\bCA[-_\s]?K\b': 'CAK',
            r'\bCA[-_\s]?H\b': 'CAH',
            r'\bOI[-_\s]?5577\b': 'OI_5577',
            r'\bOI[-_\s]?6300\b': 'OI_6300',
            r'\bOI[-_\s]?6364\b': 'OI_6364',
            r'\bSIII[-_\s]?9531\b|\bS3[-_\s]?9531\b': 'SIII_9531',
            r'\bCH4\b|\bMETHANE\b': 'CH4',
            r'\bRGB\b|\bOSC\b|\bCOLOR\b': 'RGB',
            
            # Broadband filters - comprehensive coverage
            r'\bU\b': 'U',
            r'\bV\b': 'V',
            r'\bR\b': 'R',
            r'\bI\b': 'I',
            r'\bRC\b': 'RC',
            r'\bIC\b': 'IC',
            r'\bG\b': 'G',
            r'\bB\b': 'B',
            
            # RGB variants - comprehensive coverage
            r'\bGREEN\b': 'G',  # GREEN, green, Green
            r'\bRED\b': 'R',    # RED, red, Red
            r'\bBLUE\b': 'B',   # BLUE, blue, Blue
            
            # Light pollution and multiband
            r'\bCLS\b': 'CLS',
            r'\bUHC\b': 'UHC',
            r'\bL[-_\s]?PRO\b|\bLPRO\b': 'LPRO',
            r'\bL[-_\s]?E[Nn]HANCE\b|\bLENHANCE\b': 'LEHNANCE',
            r'\bL[-_\s]?E[Xx]TREME\b|\bLEXTREME\b': 'LEXTREME',
            r'\bL[-_\s]?ULTIMATE\b|\bLULTIMATE\b': 'LULTIMATE',
            r'\bIDAS\b.*\bLPS\b': 'IDAS_LPS',
            r'\bIDAS\b.*\bD1\b': 'IDAS_LPS_D1',
            r'\bIDAS\b.*\bD2\b': 'IDAS_LPS_D2',
            r'\bIDAS\b.*\bNBZ\b|\bNBZ\b': 'NBZ',
            r'\bTRI[-_\s]?BAND\b': 'TRIBAND',
            r'\bQUAD[-_\s]?BAND\b': 'QUAD_BAND',
            
            # Sloan aliases
            r'\bU[_-]?SDSS|U[_-]?SLOAN\b': 'U_SDSS',
            r'\bG[_-]?SDSS|G[_-]?SLOAN\b': 'G_SDSS',
            r'\bR[_-]?SDSS|R[_-]?SLOAN\b': 'R_SDSS',
            r'\bI[_-]?SDSS|I[_-]?SLOAN\b': 'I_SDSS',
            r'\bZ[_-]?SDSS|Z[_-]?SLOAN\b': 'Z_SDSS',
            
            # Filter aliases - improved patterns for better detection
            r'\bLUMINANCE\b': 'L',
            r'\bLUM\b': 'L',
            r'\bHALPHA\b': 'HA',
            r'\bH[-_\s]?ALPHA\b': 'HA',
            r'\bH\b': 'HA'  # Single H defaults to H-Alpha
        }
        
        # Additional patterns for common filter naming in folder structures
        # These patterns are more flexible and don't require word boundaries
        additional_patterns = {
            r'LUMINANCE': 'L',
            r'LUM': 'L', 
            r'GREEN': 'G',
            r'RED': 'R',
            r'BLUE': 'B',
            r'LIGHT': 'L',  # Often used as luminance
            r'CLEAR': 'CLEAR'
        }
    
    for pattern, filter_code in patterns.items():
        if re.search(pattern, filename_upper):
            return filter_code, FILTERS_INFO[filter_code]
    
    # Try additional patterns for common filter naming in folder structures
    for pattern, filter_code in additional_patterns.items():
        if pattern in filename_upper:
            return filter_code, FILTERS_INFO[filter_code]
    
    # Width extraction (e.g., Ha_3nm, OIII5nm)
    width_match = re.search(r'(\d{1,2})\s*nm', filename_upper)
    if width_match:
        nm = float(width_match.group(1))
        # Try to infer base filter
        for hint, code in [('HA','HA'), ('OIII','OIII'), ('O3','O3'), ('SII','SII'), ('S2','S2'), ('HB','HBETA'), ('HEII','HEII'), ('NII','NII')]:
            if hint in filename_upper:
                base = FILTERS_INFO.get(code)
                if base:
                    # Override width with detected nm
                    return code, {**base, 'width': nm}
    return None, None

def analyze_fits_header(file_path):
    """Analyze FITS header to extract metadata"""
    try:
        if not ASTROPY_AVAILABLE:
            return {}
            
        with open_fits_for_data(file_path) as hdul:
            header = hdul[0].header
            
            metadata = {}
            
            # Basic information
            if 'EXPTIME' in header:
                metadata['exposure_time'] = float(header['EXPTIME'])
            elif 'EXPOSURE' in header:
                metadata['exposure_time'] = float(header['EXPOSURE'])
            
            if 'GAIN' in header:
                metadata['gain'] = float(header['GAIN'])
            elif 'EGAIN' in header:
                metadata['gain'] = float(header['EGAIN'])
            
            if 'TEMPERATURE' in header:
                metadata['temperature'] = float(header['TEMPERATURE'])
            elif 'CCD-TEMP' in header:
                metadata['temperature'] = float(header['CCD-TEMP'])
            
            # Normalize image type; accept variants like "Light", "Light Frame", etc.
            image_type_raw = None
            if 'IMAGETYP' in header:
                image_type_raw = str(header['IMAGETYP'])
            elif 'IMTYPE' in header:
                image_type_raw = str(header['IMTYPE'])
            
            if image_type_raw:
                t = image_type_raw.strip().upper()
                if 'LIGHT' in t:
                    metadata['image_type'] = 'LIGHT'
                elif 'DARK' in t:
                    metadata['image_type'] = 'DARK'
                elif 'BIAS' in t or 'OFFSET' in t:
                    metadata['image_type'] = 'BIAS'
                else:
                    metadata['image_type'] = t
            else:
                # Fallback from filename: if not containing dark/bias, assume LIGHT
                name_upper = file_path.name.upper()
                if ('DARK' in name_upper) or ('BIAS' in name_upper) or ('OFFSET' in name_upper):
                    if 'DARK' in name_upper:
                        metadata['image_type'] = 'DARK'
                    else:
                        metadata['image_type'] = 'BIAS'
                else:
                    metadata['image_type'] = 'LIGHT'
            
            if 'FILTER' in header:
                metadata['filter'] = header['FILTER']
            
            # Detect Bayer pattern to flag OSC/color cameras
            bayer_keys = ['BAYERPAT', 'BAYERPATN', 'BAYERPATTERN', 'COLORTYP', 'COLORSPACE']
            bayer_value = None
            for k in bayer_keys:
                if k in header:
                    try:
                        bayer_value = str(header[k]).strip().upper()
                        break
                    except Exception:
                        pass
            
            # Check if the bayer value is a valid Bayer pattern
            valid_bayer_patterns = [
                'BGGR', 'RGGB', 'GRBG', 'GBRG',  # Standard Bayer patterns
                'BG', 'RG', 'GR', 'GB',          # 2-letter patterns
                'BAYER', 'COLOR', 'RGB',         # Generic color indicators
                'CFA', 'COLORFILTER',            # Color filter array indicators
                '1', 'TRUE', 'YES', 'ON'        # Boolean indicators
            ]
            
            if bayer_value and any(pattern in bayer_value for pattern in valid_bayer_patterns):
                metadata['is_color'] = True
                metadata['bayer_pattern'] = bayer_value
                # If no explicit filter provided, mark as RGB/OSC so files are not dropped
                if 'filter' not in metadata or not str(metadata['filter']).strip():
                    metadata['filter'] = 'RGB'
            
            if 'INSTRUME' in header:
                metadata['instrument'] = header['INSTRUME']
            
            if 'TELESCOP' in header:
                metadata['telescope'] = normalize_telescope_name(header['TELESCOP'])
            
            # Object name with fallback from filename if missing/empty
            obj = None
            if 'OBJECT' in header:
                obj = str(header['OBJECT']).strip()
            if not obj:
                # Derive from filename: take leading words before first underscore or date pattern
                base = file_path.stem  # filename without extension
                # remove trailing timestamp-like suffixes: _YYYY-MM-DD or _YYYYMMDD etc.
                base_clean = re.split(r"_\d{4}[-_]?(\d{2})?[-_]?(\d{2})?", base)[0]
                # replace separators with space
                base_clean = re.sub(r"[_-]+", " ", base_clean).strip()
                obj = base_clean if base_clean else 'Unknown'
            
            # Normalize object name (case-insensitive, remove extra spaces)
            if obj and obj != 'Unknown':
                obj = normalize_target_name(obj)
            metadata['object'] = obj
            
            if 'DATE-OBS' in header:
                metadata['date_obs'] = header['DATE-OBS']
            
            if 'SITELAT' in header:
                metadata['latitude'] = float(header['SITELAT'])
            
            if 'SITELONG' in header:
                metadata['longitude'] = float(header['SITELONG'])
            
            # Fallbacks from filename if missing
            base_upper = file_path.stem.upper()
            # Instrument/camera
            if 'instrument' not in metadata or not str(metadata['instrument']).strip():
                for cam in SENSORS_DATABASE.keys():
                    if cam == 'default':
                        continue
                    if cam.upper() in base_upper:
                        metadata['instrument'] = cam
                        break
            # Telescope
            if 'telescope' not in metadata or not str(metadata['telescope']).strip():
                for tel in TELESCOPES_DATABASE.keys():
                    if tel == 'default':
                        continue
                    if tel.upper() in base_upper:
                        metadata['telescope'] = normalize_telescope_name(tel)
                        break
            # Filter from filename if missing
            if 'filter' not in metadata or not str(metadata['filter']).strip():
                filt_code, filt_info = extract_filter_from_filename(file_path.name)
                if filt_code:
                    metadata['filter'] = filt_code
            
            # Ensure essential fields for inclusion
            if 'image_type' not in metadata or not str(metadata['image_type']).strip():
                metadata['image_type'] = 'LIGHT'
            if 'filter' not in metadata or not str(metadata['filter']).strip():
                # Try to detect filter from filename more aggressively
                filename_upper = file_path.name.upper()
                if 'LUMINANCE' in filename_upper or 'LUM' in filename_upper:
                    metadata['filter'] = 'L'
                elif 'RED' in filename_upper:
                    metadata['filter'] = 'R'
                elif 'GREEN' in filename_upper:
                    metadata['filter'] = 'G'
                elif 'BLUE' in filename_upper:
                    metadata['filter'] = 'B'
                elif 'HALPHA' in filename_upper or 'H-ALPHA' in filename_upper:
                    metadata['filter'] = 'HA'
                elif 'OIII' in filename_upper or 'O3' in filename_upper:
                    metadata['filter'] = 'OIII'
                elif 'SII' in filename_upper or 'S2' in filename_upper:
                    metadata['filter'] = 'SII'
                else:
                    metadata['filter'] = 'RGB' if metadata.get('is_color') else 'L'  # Default to L for unknown filters
            if 'object' not in metadata or not str(metadata['object']).strip():
                # Reuse previous filename-based object derivation
                base = file_path.stem
                base_clean = re.split(r"_\d{4}[-_]?(\d{2})?[-_]?(\d{2})?", base)[0]
                base_clean = re.sub(r"[_-]+", " ", base_clean).strip()
                if base_clean:
                    # Normalize object name (case-insensitive, remove extra spaces)
                    metadata['object'] = normalize_target_name(base_clean)
                else:
                    metadata['object'] = 'Unknown'

            return metadata
            
    except Exception as e:
        print(f"Error analyzing FITS header for {file_path}: {e}")
        return {}

# Placeholder for the complete implementation
# This is a simplified version - the full implementation would need to be translated
# from the original French code which is much more complex


def smart_title_case(text):
    """
    Convert text to title case while respecting apostrophes
    """
    if not text:
        return text
    
    # Split by spaces
    words = text.split()
    result = []
    
    for word in words:
        if not word:
            continue
            
        # Handle words with apostrophes
        if "'" in word:
            # Split by apostrophe and capitalize each part
            parts = word.split("'")
            capitalized_parts = []
            for i, part in enumerate(parts):
                if part:  # Only capitalize if part is not empty
                    if i == 0:  # First part
                        capitalized_parts.append(part.capitalize())
                    else:  # Parts after apostrophe (like 's, 't, etc.)
                        capitalized_parts.append(part.lower())
            result.append("'".join(capitalized_parts))
        else:
            # Regular word - just capitalize first letter
            result.append(word.capitalize())
    
    return ' '.join(result)

def normalize_telescope_name(telescope_name):
    """
    Normalize telescope name, replacing technical descriptions with 'Unknown'
    """
    if not telescope_name or telescope_name.strip() == '' or telescope_name.strip() == 'Unknown':
        return 'Unknown'
    
    telescope = telescope_name.strip()
    
    # Check for technical descriptions that should be replaced with 'Unknown'
    technical_indicators = [
        '->', '->', 'driver', 'connected', 'through', 'for telescope', 'telescope connected',
        'driver for', 'connected through', 'telescope driver', 'driver connected',
        'ACP->', 'ACP->Driver', 'Driver for', 'Connected through', 'Telescope connected',
        'TELESCOPE CONNECTED', 'DRIVER FOR', 'CONNECTED THROUGH', 'ACP->DRIVER'
    ]
    
    telescope_upper = telescope.upper()
    for indicator in technical_indicators:
        if indicator.upper() in telescope_upper:
            return 'Unknown'
    
    # Check for very long names (more than 50 characters) that might break table formatting
    if len(telescope) > 50:
        return 'Unknown'
    
    # Check for names that look like technical descriptions - but be more selective
    # Only flag obvious technical descriptions, not normal telescope names with common characters
    problematic_chars = ['->', '(', ')', '[', ']', '{', '}', '|', '\\', '/', ':', ';', '=', '*', '&', '%', '$', '#', '@', '!', '?', '<', '>']
    
    # Allow common telescope naming patterns like "AUS-2", "FSQ-106ED", etc.
    # Only flag if the name contains multiple problematic characters or looks like a technical description
    problematic_count = sum(1 for char in telescope if char in problematic_chars)
    
    # If it has too many problematic characters or looks like a technical description, mark as Unknown
    if problematic_count > 2 or any(phrase in telescope_upper for phrase in ['DRIVER', 'CONNECTED', 'THROUGH', 'FOR TELESCOPE']):
        return 'Unknown'
    
    # If it looks like a normal telescope name, return it as is
    return telescope


def get_platform_info():
    """Get detailed platform information for compatibility"""
    import platform
    import sys
    import os
    import shutil
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # Detect architecture
    if 'arm' in machine or 'aarch64' in machine:
        arch = "ARM"
    elif 'x86_64' in machine or 'amd64' in machine:
        arch = "x64"
    elif 'x86' in machine or 'i386' in machine or 'i686' in machine:
        arch = "x86"
    else:
        arch = "Unknown"
    
    # Detect Linux distribution
    linux_distro = "Unknown"
    if system == 'linux':
        try:
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if line.startswith('ID='):
                        linux_distro = line.split('=')[1].strip().strip('"')
                        break
        except (FileNotFoundError, OSError):
            try:
                with open('/etc/issue', 'r') as f:
                    content = f.read().lower()
                    if 'manjaro' in content:
                        linux_distro = 'manjaro'
                    elif 'arch' in content:
                        linux_distro = 'arch'
                    elif 'ubuntu' in content:
                        linux_distro = 'ubuntu'
                    elif 'debian' in content:
                        linux_distro = 'debian'
            except (FileNotFoundError, OSError):
                pass
    
    # Detect Mac variants
    mac_variant = "Unknown"
    if system == 'darwin':
        try:
            # Detect Apple Silicon vs Intel
            import subprocess
            result = subprocess.run(['uname', '-m'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                arch = result.stdout.strip()
                if arch == 'arm64':
                    mac_variant = 'apple_silicon'
                elif arch == 'x86_64':
                    mac_variant = 'intel'
            
            # Detect package managers
            if shutil.which('brew'):
                if mac_variant == 'apple_silicon':
                    mac_variant = 'homebrew_apple_silicon'
                else:
                    mac_variant = 'homebrew_intel'
            elif shutil.which('port'):
                mac_variant = 'macports'
            
            # Detect MacTeX
            if os.path.exists('/Library/TeX/texbin/pdflatex'):
                if mac_variant == 'Unknown':
                    mac_variant = 'mactex'
                else:
                    mac_variant += '_mactex'
            
            # Detect Xcode Command Line Tools
            if os.path.exists('/Library/Developer/CommandLineTools/usr/bin/python3'):
                if mac_variant == 'Unknown':
                    mac_variant = 'xcode_tools'
                else:
                    mac_variant += '_xcode'
                    
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # Fallback detection
            if os.path.exists('/opt/homebrew/bin/python3'):
                mac_variant = 'apple_silicon'
            elif os.path.exists('/usr/local/bin/python3'):
                mac_variant = 'intel'
            else:
                mac_variant = 'unknown'
    
    # Detect Python and pip paths
    python_paths = {
        'python_executable': sys.executable,
        'python_in_path': shutil.which('python'),
        'python3_in_path': shutil.which('python3'),
        'pip_in_path': shutil.which('pip'),
        'pip3_in_path': shutil.which('pip3')
    }
    
    return {
        'system': system,
        'machine': machine,
        'architecture': arch,
        'python_version': python_version,
        'linux_distro': linux_distro,
        'mac_variant': mac_variant,
        'is_windows': system == 'windows',
        'is_linux': system == 'linux',
        'is_macos': system == 'darwin',
        'is_manjaro': linux_distro == 'manjaro',
        'is_apple_silicon': mac_variant in ['apple_silicon', 'homebrew_apple_silicon'],
        'is_intel_mac': mac_variant in ['intel', 'homebrew_intel'],
        'has_homebrew': 'homebrew' in mac_variant,
        'has_macports': 'macports' in mac_variant,
        'has_mactex': 'mactex' in mac_variant,
        'python_paths': python_paths
    }

def main():
    """Main function"""
    global ADU_ANALYSIS_ENABLED, ADU_SAMPLE_PER_FILTER, FAST_ANALYSIS, BIAS_DARK_PATH, DEFAULT_REGION_SIZE
    
    # Get platform information
    platform_info = get_platform_info()
    
    # Start timing
    import time
    start_time = time.time()
    
    print("TELESCOPE ASTROPHOTOGRAPHY ANALYZER")
    print("=" * 80)
    
    # Display platform information for compatibility
    print(f"🖥️  Plateforme: {platform_info['system'].title()} ({platform_info['architecture']})")
    if platform_info['is_linux'] and platform_info['linux_distro'] != 'Unknown':
        print(f"🐧 Distribution: {platform_info['linux_distro'].title()}")
    elif platform_info['is_macos'] and platform_info['mac_variant'] != 'Unknown':
        print(f"🍎 Mac Variant: {platform_info['mac_variant'].title()}")
        if platform_info['is_apple_silicon']:
            print(f"🍎 Type: Apple Silicon (M1/M2/M3)")
        elif platform_info['is_intel_mac']:
            print(f"🍎 Type: Intel Mac")
        if platform_info['has_homebrew']:
            print(f"🍺 Homebrew: Détecté")
        if platform_info['has_macports']:
            print(f"🍺 MacPorts: Détecté")
        if platform_info['has_mactex']:
            print(f"📄 MacTeX: Détecté")
    print(f"🐍 Python: {platform_info['python_version']}")
    print(f"🔧 Architecture: {platform_info['machine']}")
    
    # Show Python path information for debugging on Linux and Mac
    if platform_info['is_linux'] or platform_info['is_macos']:
        print(f"🔍 Python détecté: {platform_info['python_paths']['python_executable']}")
        if platform_info['python_paths']['python3_in_path']:
            print(f"🔍 Python3 dans PATH: {platform_info['python_paths']['python3_in_path']}")
        if platform_info['python_paths']['pip3_in_path']:
            print(f"🔍 Pip3 dans PATH: {platform_info['python_paths']['pip3_in_path']}")
        
        # Show LaTeX detection
        latex_exe = find_latex_executable()
        if latex_exe:
            print(f"🔍 LaTeX détecté: {latex_exe}")
        else:
            print("⚠️  LaTeX non détecté - rapport PDF non disponible")
    
    print("-" * 80)
    
    # Parse arguments first
    args = parse_args()
    
    # Auto-detect CPU cores if workers not specified
    if args.workers is None:
        import multiprocessing
        
        # Get system information with robust error handling
        try:
            cpu_count = multiprocessing.cpu_count()
            if cpu_count <= 0:
                cpu_count = 1  # Fallback for invalid CPU count
        except (OSError, NotImplementedError):
            # Fallback for systems where CPU count detection fails
            cpu_count = 1
        
        # Try to get memory info, fallback to CPU-based estimation
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            memory_available = True
        except (ImportError, OSError, AttributeError):
            # Fallback: estimate memory based on CPU count and platform
            import platform
            system = platform.system().lower()
            if system == "windows":
                memory_gb = cpu_count * 2  # Conservative estimate for Windows
            elif system == "darwin":  # macOS
                memory_gb = cpu_count * 4  # macOS typically has more memory
            else:  # Linux and others
                memory_gb = cpu_count * 3  # Linux estimate
            memory_available = False
        
        # Intelligent worker selection based on system capabilities
        if cpu_count >= 16 and memory_gb >= 16:
            # High-end system: use all cores
            args.workers = cpu_count
            system_type = "High-end"
        elif cpu_count >= 8 and memory_gb >= 8:
            # Mid-range system: use 75% of cores to leave resources for other tasks
            args.workers = max(4, int(cpu_count * 0.75))
            system_type = "Mid-range"
        elif cpu_count >= 4 and memory_gb >= 4:
            # Entry-level system: use 50% of cores
            args.workers = max(2, int(cpu_count * 0.5))
            system_type = "Entry-level"
        else:
            # Low-end system: use 1-2 cores maximum
            args.workers = min(2, cpu_count)
            system_type = "Low-end"
        
        memory_info = f"{memory_gb:.1f}GB RAM" if memory_available else "RAM (estimated)"
        print(f"🧵 Auto-detected {cpu_count} CPU cores, {memory_info} ({system_type} system)")
        print(f"   📊 Optimal workers: {args.workers} (auto-optimized for your system)")
        
        # Additional recommendations
        if system_type == "Low-end":
            print("   💡 Tip: Consider using --workers 1 for better stability on low-end systems")
        elif system_type == "High-end":
            print("   💡 Tip: Your system can handle maximum performance with all cores")
        elif not memory_available:
            print("   💡 Tip: Install psutil for more accurate memory detection: pip install psutil")
    
    # Check Python packages at startup
    print("🔍 Checking Python packages...")
    
    # Run platform-specific diagnostic if needed
    if platform_info['is_linux']:
        diagnose_linux_distribution_issues(platform_info)
    elif platform_info['is_macos']:
        diagnose_mac_variants_issues(platform_info)
    
    if args.auto_install:
        print("🤖 Auto-install mode enabled")
        if not install_python_packages_automatically():
            print("⚠️  Some packages could not be installed automatically")
            print("   Please install them manually or run without --auto-install")
            if platform_info['is_manjaro']:
                print("\n💡 Solutions spécifiques à Manjaro:")
                print("   sudo pacman -S python python-pip")
                print("   sudo pacman -S python-astropy python-matplotlib python-pillow")
                print("   sudo pacman -S python-requests python-tqdm")
    else:
        suggest_python_installation()
    print()
    print("Recursive FITS folder analysis")
    print("Target separation and detailed statistics")
    print("Observed sky regions analysis")
    print("=" * 80)
    
    if not ASTROPY_AVAILABLE:
        print("ERROR: Astropy is not installed. Cannot continue.")
        print("   Install with: pip install astropy")
        return
    
    # CLI Args
    args = parse_args()
    
    # Random seed
    if args.seed is not None:
        try:
            np.random.seed(args.seed)
            print(f"Random seed set: {args.seed}")
        except Exception as e:
            print(f"WARNING: Cannot set seed: {e}")

    # Load existing configuration
    print("Loading configuration...")
    loaded_config = load_config()

    # Analysis folder (cross-platform path handling)
    script_dir = Path(__file__).resolve().parent
    if args.folder:
        # Handle both relative and absolute paths
        folder = Path(args.folder).resolve()
    else:
        folder = script_dir
    print(f"Analysis folder: {folder}")
    if not folder.exists():
        print(f"ERROR: Analysis folder does not exist: {folder}")
        if platform_info['is_windows']:
            print("💡 Windows: Enclose paths in quotes, e.g. --folder \"C:\\Path\\To\\Your\\Folder\"")
        elif platform_info['is_linux']:
            print("💡 Linux: Use forward slashes, e.g. --folder \"/home/username/astro\"")
        elif platform_info['is_macos']:
            print("💡 macOS: Use forward slashes, e.g. --folder \"/Users/username/astro\"")
        return
    if not folder.is_dir():
        print(f"ERROR: Analysis folder is not a directory: {folder}")
        return
    

    # SNR region size
    DEFAULT_REGION_SIZE = max(16, int(args.region_size))

    # ADU Mode - Force Mode 1 only
    global ADU_ANALYSIS_ENABLED, FAST_ANALYSIS, ADU_SAMPLE_PER_FILTER
    
    # Force Mode 1: Fast analysis (no ADU analysis)
    ADU_ANALYSIS_ENABLED = False
    FAST_ANALYSIS = True
    ADU_SAMPLE_PER_FILTER = 3
    
    print(f"\nSTARTING ANALYSIS")
    print("=" * 80)
    
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"astronomical_analysis_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Output folder: {output_folder}")
    
    try:
        # Analyze FITS files
        print("🔍 Starting FITS analysis...")
        try:
            data_by_target, global_data = analyze_folder_recursive(str(folder), args.workers)
        except Exception as e:
            print(f"❌ Error during FITS analysis: {e}")
            import traceback
            traceback.print_exc()
            return
        
        if not data_by_target:
            print("❌ No FITS files found or analysis failed")
            return
        
        # Group normalized targets (e.g., LMC and lmc)
        print("🔗 Grouping normalized targets...")
        original_count = len(data_by_target)
        data_by_target = group_normalized_targets(data_by_target)
        normalized_count = len(data_by_target)
        
        if original_count != normalized_count:
            print(f"   📊 Target normalization: {original_count} → {normalized_count} targets")
            # Show which targets were grouped
            for target_name, target_data in data_by_target.items():
                if 'original_names' in target_data and len(target_data['original_names']) > 1:
                    print(f"   🔗 Grouped: {', '.join(target_data['original_names'])} → {target_name}")
        
        # Group mosaic panels
        print("🔗 Grouping mosaic panels...")
        mosaic_original_count = len(data_by_target)
        data_by_target = group_mosaic_panels(data_by_target)
        grouped_count = len(data_by_target)
        
        if mosaic_original_count != grouped_count:
            print(f"   📊 Mosaic grouping: {mosaic_original_count} → {grouped_count} targets")
            mosaic_targets = [name for name, data in data_by_target.items() if 'panels' in data]
            if mosaic_targets:
                print(f"   🧩 Mosaic targets found: {', '.join(mosaic_targets)}")
        
        # Display results
        display_target_statistics(data_by_target)
        
        # Generate outputs
        if not args.no_graphs:
            generate_graphs(data_by_target, global_data, output_folder)
        
        if not args.no_latex:
            generate_latex_report(data_by_target, global_data, output_folder)
        
        if args.export_csv:
            export_csv(data_by_target, global_data, output_folder)
        
        if args.zip_output:
            compress_output_folder(output_folder)
        
        # Calculate execution time and performance statistics
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get total files processed
        total_files = global_data.get('total_files', 0)
        
        # Calculate files per second
        if total_time > 0:
            files_per_second = total_files / total_time
        else:
            files_per_second = 0
        
        # Format time display
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved in: {output_folder}")
        
        # Enhanced performance statistics with more prominent execution time
        print(f"\n" + "=" * 80)
        print(f"⏱️  TEMPS TOTAL D'EXÉCUTION DU PROGRAMME: {time_str}")
        print(f"=" * 80)
        
        print(f"\n📊 STATISTIQUES DE PERFORMANCE DÉTAILLÉES")
        print(f"=" * 50)
        print(f"📁 Total fichiers FITS traités: {total_files:,}")
        print(f"⚡ Vitesse moyenne de traitement: {files_per_second:.2f} fichiers/seconde")
        print(f"📈 Efficacité: {files_per_second:.1f} fichiers par seconde")
        
        # Enhanced performance rating with more details
        if files_per_second >= 1000:
            rating = "🚀 Excellent"
            rating_desc = "Performance exceptionnelle"
        elif files_per_second >= 500:
            rating = "⚡ Très Bon"
            rating_desc = "Performance très satisfaisante"
        elif files_per_second >= 100:
            rating = "✅ Bon"
            rating_desc = "Performance correcte"
        else:
            rating = "🐌 Lent"
            rating_desc = "Performance lente"
        
        print(f"🏆 Évaluation de performance: {rating} - {rating_desc}")
        print(f"📊 Résumé: {total_files:,} fichiers traités en {time_str} ({files_per_second:.2f} fichiers/seconde)")
        print(f"=" * 50)
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Check if the error is related to missing Python packages
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['no module named', 'import', 'module not found', 'package', 'pip']):
            print("\n" + "="*80)
            print("🔧 PYTHON PACKAGE ERROR DETECTED")
            print("="*80)
            print("The error appears to be related to missing Python packages.")
            print("This can happen due to:")
            print("• Incompatible Python version (some packages don't support newer versions)")
            print("• Missing required packages")
            print("• Package installation failure")
            print("\n💡 SOLUTIONS:")
            print("1. Try running with --auto-install flag:")
            print("   python script.py --auto-install")
            print("2. Install packages manually:")
            print("   python -m pip install matplotlib numpy pandas reportlab tqdm astropy pillow scipy requests")
            print("3. If using Python 3.13+, try with Python 3.12:")
            print("   • Download Python 3.12 from https://www.python.org/downloads/")
            print("   • Or use a virtual environment with Python 3.12")
            print("4. Check your Python installation and pip configuration")
            print("="*80)

def smart_title_case(text):
    """
    Convert text to title case while respecting apostrophes
    """
    if not text:
        return text
    
    # Split by spaces
    words = text.split()
    result = []
    
    for word in words:
        if not word:
            continue
            
        # Handle words with apostrophes
        if "'" in word:
            # Split by apostrophe and capitalize each part
            parts = word.split("'")
            capitalized_parts = []
            for i, part in enumerate(parts):
                if part:  # Only capitalize if part is not empty
                    if i == 0:  # First part
                        capitalized_parts.append(part.capitalize())
                    else:  # Parts after apostrophe (like 's, 't, etc.)
                        capitalized_parts.append(part.lower())
            result.append("'".join(capitalized_parts))
        else:
            # Regular word - just capitalize first letter
            result.append(word.capitalize())
    
    return ' '.join(result)

def normalize_target_name(target_name):
    """
    Normalize target name with special handling for astronomical catalogs and object names
    """
    if not target_name or target_name.strip() == '' or target_name.strip() == 'Unknown':
        return 'Unknown'
    
    target = target_name.strip()
    target_upper = target.upper()

    # Preserve Solar System object names exactly (avoid catalog normalization like "M oon")
    solar_system_objects = {
        # Star
        'SUN': 'Sun', 'SOL': 'Sun',
        # Planets
        'MERCURY': 'Mercury', 'VENUS': 'Venus', 'EARTH': 'Earth',
        'MARS': 'Mars', 'JUPITER': 'Jupiter', 'SATURN': 'Saturn',
        'URANUS': 'Uranus', 'NEPTUNE': 'Neptune',
        # Earth Moon
        'MOON': 'Moon', 'LUNA': 'Moon',
        # Dwarf planets
        'PLUTO': 'Pluto', 'CERES': 'Ceres', 'ERIS': 'Eris',
        'HAUMEA': 'Haumea', 'MAKEMAKE': 'Makemake',
        # Mars moons
        'PHOBOS': 'Phobos', 'DEIMOS': 'Deimos',
        # Jupiter moons (Galilean + common)
        'IO': 'Io', 'EUROPA': 'Europa', 'GANYMEDE': 'Ganymede', 'CALLISTO': 'Callisto',
        'AMALTHEA': 'Amalthea', 'THEBE': 'Thebe', 'METIS': 'Metis', 'ADRASTEA': 'Adrastea',
        'HIMALIA': 'Himalia', 'ELARA': 'Elara', 'LEDA': 'Leda', 'LYSITHEA': 'Lysithea',
        'ANANKE': 'Ananke', 'CARME': 'Carme', 'PASIPHAE': 'Pasiphae', 'SINOPE': 'Sinope',
        # Saturn moons (major)
        'TITAN': 'Titan', 'RHEA': 'Rhea', 'IAPETUS': 'Iapetus', 'DIONE': 'Dione',
        'TETHYS': 'Tethys', 'ENCELADUS': 'Enceladus', 'MIMAS': 'Mimas', 'HYPERION': 'Hyperion', 'PHOEBE': 'Phoebe',
        'ATLAS': 'Atlas', 'PROMETHEUS': 'Prometheus', 'PANDORA': 'Pandora', 'JANUS': 'Janus', 'EPIMETHEUS': 'Epimetheus',
        'HELENE': 'Helene', 'TELESTO': 'Telesto', 'CALYPSO': 'Calypso',
        # Uranus moons (major)
        'MIRANDA': 'Miranda', 'ARIEL': 'Ariel', 'UMBRIEL': 'Umbriel', 'TITANIA': 'Titania', 'OBERON': 'Oberon',
        'PUCK': 'Puck',
        # Neptune moons
        'TRITON': 'Triton', 'NEREID': 'Nereid', 'PROTEUS': 'Proteus', 'LARISSA': 'Larissa', 'DESPINA': 'Despina',
        'GALATEA': 'Galatea', 'THALASSA': 'Thalassa', 'NAIAD': 'Naiad',
        # Pluto system
        'CHARON': 'Charon', 'PLUTO I': 'Charon', 'PLUTO 1': 'Charon',
        'HYDRA': 'Hydra', 'NIX': 'Nix', 'KERBEROS': 'Kerberos', 'STYX': 'Styx',
        'PLUTO II': 'Hydra', 'PLUTO 2': 'Hydra',
        'PLUTO III': 'Nix', 'PLUTO 3': 'Nix',
        'PLUTO IV': 'Kerberos', 'PLUTO 4': 'Kerberos',
        'PLUTO V': 'Styx', 'PLUTO 5': 'Styx',
        # Roman numeral aliases for major moons by primary
        # Earth
        'EARTH I': 'Moon', 'EARTH 1': 'Moon',
        # Mars
        'MARS I': 'Phobos', 'MARS 1': 'Phobos',
        'MARS II': 'Deimos', 'MARS 2': 'Deimos',
        # Jupiter (Galilean)
        'JUPITER I': 'Io', 'JUPITER 1': 'Io',
        'JUPITER II': 'Europa', 'JUPITER 2': 'Europa',
        'JUPITER III': 'Ganymede', 'JUPITER 3': 'Ganymede',
        'JUPITER IV': 'Callisto', 'JUPITER 4': 'Callisto',
        # Saturn (classics)
        'SATURN I': 'Mimas', 'SATURN 1': 'Mimas',
        'SATURN II': 'Enceladus', 'SATURN 2': 'Enceladus',
        'SATURN III': 'Tethys', 'SATURN 3': 'Tethys',
        'SATURN IV': 'Dione', 'SATURN 4': 'Dione',
        'SATURN V': 'Rhea', 'SATURN 5': 'Rhea',
        'SATURN VI': 'Titan', 'SATURN 6': 'Titan',
        'SATURN VII': 'Hyperion', 'SATURN 7': 'Hyperion',
        'SATURN VIII': 'Iapetus', 'SATURN 8': 'Iapetus',
        'SATURN IX': 'Phoebe', 'SATURN 9': 'Phoebe',
        # Uranus (classics)
        'URANUS I': 'Ariel', 'URANUS 1': 'Ariel',
        'URANUS II': 'Umbriel', 'URANUS 2': 'Umbriel',
        'URANUS III': 'Titania', 'URANUS 3': 'Titania',
        'URANUS IV': 'Oberon', 'URANUS 4': 'Oberon',
        'URANUS V': 'Miranda', 'URANUS 5': 'Miranda',
        # Neptune (classics)
        'NEPTUNE I': 'Triton', 'NEPTUNE 1': 'Triton',
        'NEPTUNE II': 'Nereid', 'NEPTUNE 2': 'Nereid',
        # Dwarf planet moons
        'DYSNOMIA': 'Dysnomia',  # Eris
        'HIIAKA': "Hi'iaka", "HII'AKA": "Hi'iaka", 'HII-AKA': "Hi'iaka", 'HII IAKA': "Hi'iaka", 'HIIIAKA': "Hi'iaka",
        'NAMAKA': 'Namaka',      # Haumea
        # Notable asteroids
        'VESTA': 'Vesta', 'PALLAS': 'Pallas', 'HYGEIA': 'Hygiea', 'EROS': 'Eros',
        'ITOKAWA': 'Itokawa', 'BENNU': 'Bennu', 'RYUGU': 'Ryugu', 'PSYCHE': 'Psyche',
        # Famous comets (common names)
        'HALLEY': 'Halley', 'ENCKE': 'Encke', 'HALE BOPP': 'Hale-Bopp', 'HALE-BOPP': 'Hale-Bopp',
        'SHOEMAKER LEVY 9': 'Shoemaker-Levy 9', 'SHOEMAKER-LEVY 9': 'Shoemaker-Levy 9',
        '67P': '67P/Churyumov–Gerasimenko', 'CHURYUMOV GERASIMENKO': '67P/Churyumov–Gerasimenko',
        'C2020 F3': 'C/2020 F3 (NEOWISE)', 'NEOWISE': 'C/2020 F3 (NEOWISE)',
        # Additional famous comets and aliases
        '1P': '1P/Halley', '1P/HALLEY': '1P/Halley',
        '2P': '2P/Encke', '2P/ENCKE': '2P/Encke',
        '9P': '9P/Tempel 1', '9P/TEMPEL 1': '9P/Tempel 1', 'TEMPEL 1': '9P/Tempel 1',
        '12P': '12P/Pons-Brooks', '12P/PONS BROOKS': '12P/Pons-Brooks', 'PONS BROOKS': '12P/Pons-Brooks', 'PONS-BROOKS': '12P/Pons-Brooks',
        '19P': '19P/Borrelly', '19P/BORRELLY': '19P/Borrelly', 'BORRELLY': '19P/Borrelly',
        '21P': '21P/Giacobini-Zinner', '21P/GIACOBINI ZINNER': '21P/Giacobini-Zinner', 'GIACOBINI ZINNER': '21P/Giacobini-Zinner', 'GIACOBINI-ZINNER': '21P/Giacobini-Zinner',
        '26P': '26P/Grigg-Skjellerup', '26P/GRIGG SKJELLERUP': '26P/Grigg-Skjellerup', 'GRIGG SKJELLERUP': '26P/Grigg-Skjellerup', 'GRIGG-SKJELLERUP': '26P/Grigg-Skjellerup',
        '55P': '55P/Tempel-Tuttle', '55P/TEMPEL TUTTLE': '55P/Tempel-Tuttle', 'TEMPEL TUTTLE': '55P/Tempel-Tuttle', 'TEMPEL-TUTTLE': '55P/Tempel-Tuttle',
        '67P': '67P/Churyumov–Gerasimenko', '67P/CHURYUMOV GERASIMENKO': '67P/Churyumov–Gerasimenko', 'CHURYUMOV GERASIMENKO': '67P/Churyumov–Gerasimenko', 'CHURYUMOV-GERASIMENKO': '67P/Churyumov–Gerasimenko',
        '73P': '73P/Schwassmann–Wachmann', '73P/SCHWASSMANN WACHMANN': '73P/Schwassmann–Wachmann', 'SCHWASSMANN WACHMANN': '73P/Schwassmann–Wachmann', 'SCHWASSMANN-WACHMANN': '73P/Schwassmann–Wachmann',
        '81P': '81P/Wild 2', '81P/WILD 2': '81P/Wild 2', 'WILD 2': '81P/Wild 2',
        '96P': '96P/Machholz', '96P/MACHHOLZ': '96P/Machholz', 'MACHHOLZ': '96P/Machholz',
        '103P': '103P/Hartley', '103P/HARTLEY': '103P/Hartley', 'HARTLEY': '103P/Hartley',
        '109P': '109P/Swift-Tuttle', '109P/SWIFT TUTTLE': '109P/Swift-Tuttle', 'SWIFT TUTTLE': '109P/Swift-Tuttle', 'SWIFT-TUTTLE': '109P/Swift-Tuttle',
        'ISON': 'C/2012 S1 (ISON)', 'C2012 S1': 'C/2012 S1 (ISON)', 'C/2012 S1': 'C/2012 S1 (ISON)',
        # Hale-Bopp (MPC designation)
        'C1995 O1': 'C/1995 O1 (Hale-Bopp)', 'C/1995 O1': 'C/1995 O1 (Hale-Bopp)',
        'PANSTARRS': 'C/2011 L4 (PANSTARRS)', 'C2011 L4': 'C/2011 L4 (PANSTARRS)', 'C/2011 L4': 'C/2011 L4 (PANSTARRS)',
        'SIDING SPRING': 'C/2013 A1 (Siding Spring)', 'C2013 A1': 'C/2013 A1 (Siding Spring)', 'C/2013 A1': 'C/2013 A1 (Siding Spring)',
        'CATALINA': 'C/2013 US10 (Catalina)', 'C2013 US10': 'C/2013 US10 (Catalina)', 'C/2013 US10': 'C/2013 US10 (Catalina)',
        'MCNAUGHT': 'C/2006 P1 (McNaught)', 'C2006 P1': 'C/2006 P1 (McNaught)', 'C/2006 P1': 'C/2006 P1 (McNaught)',
        'NEAT': 'C/2001 Q4 (NEAT)', 'C2001 Q4': 'C/2001 Q4 (NEAT)', 'C/2001 Q4': 'C/2001 Q4 (NEAT)',
        'HYAKUTAKE': 'C/1996 B2 (Hyakutake)', 'C1996 B2': 'C/1996 B2 (Hyakutake)', 'C/1996 B2': 'C/1996 B2 (Hyakutake)',
        'OUMUAMUA': "1I/'Oumuamua", "1I": "1I/'Oumuamua", "1I/‘OUMUAMUA": "1I/'Oumuamua",
        '2I': '2I/Borisov', 'BORISOV': '2I/Borisov'
    }
    # Preserve well-known artificial satellites/spacecraft as-is
    artificial_space_objects = {
        'JWST': 'JWST', 'JAMES WEBB': 'JWST', 'JAMES WEBB SPACE TELESCOPE': 'JWST',
        'HUBBLE': 'Hubble', 'HST': 'Hubble',
        'ISS': 'ISS', 'INTERNATIONAL SPACE STATION': 'ISS',
        'TIANGONG': 'Tiangong', 'CHINESE SPACE STATION': 'Tiangong', 'CSS': 'Tiangong',
        'SKYLAB': 'Skylab', 'MIR': 'Mir',
        'SPUTNIK': 'Sputnik', 'SPUTNIK 1': 'Sputnik 1', 'SPUTNIK-1': 'Sputnik 1',
        'VOYAGER 1': 'Voyager 1', 'VOYAGER-1': 'Voyager 1', 'VOYAGER1': 'Voyager 1',
        'VOYAGER 2': 'Voyager 2', 'VOYAGER-2': 'Voyager 2', 'VOYAGER2': 'Voyager 2',
        'CASSINI': 'Cassini', 'JUNO': 'Juno', 'NEW HORIZONS': 'New Horizons',
        'ROSETTA': 'Rosetta', 'PHILAE': 'Philae',
    }
    if target_upper in artificial_space_objects:
        return artificial_space_objects[target_upper]
    if target_upper in solar_system_objects:
        return solar_system_objects[target_upper]
    
    # List of astronomical catalogs that should be in uppercase
    # Note: Single letters like 'C' are only catalogs if followed by a number
    catalogs = [
        'NGC', 'IC', 'ARP', 'M', 'MESSIER', 'BARNARD', 'LDN', 'LBN',
        'RCW', 'GUM', 'VDB', 'VAN DEN BERGH', 'LBN', 'LDN', 'LBN',
        'PK', 'PN', 'PLANETARY', 'OC', 'OPEN CLUSTER', 'GC', 'GLOBULAR', 'GC', 'GLOBULAR',
        'UGC', 'PGC', 'ESO', 'MCG', 'IRAS', '2MASS', 'WISE', 'SDSS', 'HIP', 'TYC',
        'HD', 'HR', 'SAO', 'BD', 'CD', 'CP', 'GJ', 'GL', 'GJ', 'GLIESE', 'LHS', 'LTT',
        'NLTT', 'LP', 'LPM', 'LTT', 'NLTT', 'LP', 'LPM', 'LTT', 'NLTT', 'LP', 'LPM'
    ]
    
    # Special catalogs that need custom handling (handled by specific functions)
    special_catalogs = ['SH2', 'SHARPLESS']
    
    # Special handling for single letter catalogs that need number validation
    single_letter_catalogs = ['B', 'C']
    
    # Check if target starts with a catalog name
    
    # Skip special catalogs that are handled by specific functions
    for special_catalog in special_catalogs:
        if target_upper.startswith(special_catalog):
            # Let the specific function handle this catalog
            break
    else:
        # Only process if it's not a special catalog
        for catalog in catalogs:
            if target_upper.startswith(catalog):
                # Extract the catalog part and number/name part
                catalog_part = catalog.upper()
                
                # Remove the catalog name from the beginning
                remaining = target[len(catalog):].strip()
                
                # For catalogues, extract only the number part (ignore additional words)
                import re
                if catalog in ['M', 'NGC', 'IC', 'ARP']:
                    # Extract only the number part (digits and letters like M 31A, NGC 211A)
                    number_match = re.match(r'^[_\s\-\.]*(\d+[A-Za-z]*)', remaining)
                    if number_match:
                        number_part = number_match.group(1)
                        return f"{catalog_part} {number_part}"
                    else:
                        # If no number found, try to extract any digits
                        digits_match = re.search(r'\d+', remaining)
                        if digits_match:
                            return f"{catalog_part} {digits_match.group()}"
                        else:
                            # No digits at all: this is not a catalog reference (e.g., "Moon")
                            # Return the original target unmodified to avoid outputs like "M oon"
                            return target
                else:
                    # For other catalogs, keep the full remaining part
                    return f"{catalog_part} {remaining}"
    
    # Check for single letter catalogs that need number validation
    for catalog in single_letter_catalogs:
        if target_upper.startswith(catalog):
            # Check if single letter is followed by a number (like B144, C14, etc.)
            remaining = target[1:].strip()
            import re
            if re.match(r'^\d', remaining):
                # It's a catalog entry - normalize the number part
                catalog_part = catalog.upper()
                # Extract just the number part, removing any extra spaces
                number_match = re.match(r'^(\d+)', remaining)
                if number_match:
                    number_part = number_match.group(1)
                    return f"{catalog_part} {number_part}"
                else:
                    return f"{catalog_part} {remaining}"
            else:
                # Not a catalog entry, continue to normal processing
                break
    
    # Check for common astronomical abbreviations
    target_upper = target.upper()
    
    # Magellanic Clouds
    if target_upper in ['LMC', 'LARGE MAGELLANIC CLOUD', 'LARGE MAGELLANIC', 'ESO056-G115', 'ESO 056-G115']:
        return 'Large Magellanic Cloud (LMC)'
    elif target_upper in ['SMC', 'SMALL MAGELLANIC CLOUD', 'SMALL MAGELLANIC']:
        return 'Small Magellanic Cloud (SMC)'
    
    # Common astronomical objects with catalog numbers
    elif target_upper in ['MW', 'MILKY WAY', 'MILKY WAY GALAXY']:
        return 'Milky Way'
    elif target_upper in ['ANDROMEDA', 'ANDROMEDA GALAXY', 'ANDROMEDA NEBULA', 'M31', 'M 31', 'NGC224', 'NGC 224']:
        return 'M 31 (Andromeda Galaxy)'
    elif target_upper in ['TRIANGULUM', 'TRIANGULUM GALAXY', 'TRIANGULUM NEBULA', 'M33', 'M 33', 'NGC598', 'NGC 598']:
        return 'M 33 (Triangulum Galaxy)'
    elif target_upper in ['WHIRLPOOL', 'WHIRLPOOL GALAXY', 'M51', 'M 51', 'NGC5194', 'NGC 5194']:
        return 'M 51 (Whirlpool Galaxy)'
    elif target_upper in ['SOMBRERO', 'SOMBRERO GALAXY', 'M104', 'M 104', 'NGC4594', 'NGC 4594']:
        return 'M 104 (Sombrero Galaxy)'
    elif target_upper in ['PINWHEEL', 'PINWHEEL GALAXY', 'M101', 'M 101', 'NGC5457', 'NGC 5457']:
        return 'M 101 (Pinwheel Galaxy)'
    elif target_upper in ['BLACK EYE', 'BLACK EYE GALAXY', 'M64', 'M 64', 'NGC4826', 'NGC 4826']:
        return 'M 64 (Black Eye Galaxy)'
    elif target_upper in ['SUNFLOWER', 'SUNFLOWER GALAXY', 'M63', 'M 63', 'NGC5055', 'NGC 5055']:
        return 'M 63 (Sunflower Galaxy)'
    elif target_upper in ['CIGAR', 'CIGAR GALAXY', 'M82', 'M 82', 'NGC3034', 'NGC 3034']:
        return 'M 82 (Cigar Galaxy)'
    elif target_upper in ['BODES', 'BODES GALAXY', 'M81', 'M 81', 'NGC3031', 'NGC 3031']:
        return 'M 81 (Bode\'s Galaxy)'
    elif target_upper in ['CARTWHEEL', 'CARTWHEEL GALAXY']:
        return 'Cartwheel Galaxy'
    elif target_upper in ['ANTENNAE', 'ANTENNAE GALAXIES', 'NGC4038', 'NGC 4038', 'NGC4039', 'NGC 4039']:
        return 'NGC 4038/4039 (Antennae Galaxies)'
    elif target_upper in ['MICE', 'MICE GALAXIES', 'NGC4676', 'NGC 4676']:
        return 'NGC 4676 (Mice Galaxies)'
    elif target_upper in ['TADPOLE', 'TADPOLE GALAXY', 'UGC10214', 'UGC 10214']:
        return 'UGC 10214 (Tadpole Galaxy)'
    elif target_upper in ['BUTTERFLY', 'BUTTERFLY GALAXY']:
        return 'Butterfly Galaxy'
    
    # Messier objects with common names
    elif target_upper in ['CRAB', 'CRAB NEBULA', 'M1', 'M 1', 'NGC1952', 'NGC 1952']:
        return 'M 1 (Crab Nebula)'
    elif target_upper in ['ORION', 'ORION NEBULA', 'M42', 'M 42', 'NGC1976', 'NGC 1976']:
        return 'M 42 (Orion Nebula)'
    elif target_upper in ['PLEIADES', 'PLEIADES CLUSTER', 'M45', 'M 45']:
        return 'M 45 (Pleiades)'
    elif target_upper in ['BEEHIVE', 'BEEHIVE CLUSTER', 'M44', 'M 44', 'NGC2632', 'NGC 2632']:
        return 'M 44 (Beehive Cluster)'
    elif target_upper in ['LAGOON', 'LAGOON NEBULA', 'M8', 'M 8', 'NGC6523', 'NGC 6523']:
        return 'M 8 (Lagoon Nebula)'
    elif target_upper in ['TRIFID', 'TRIFID NEBULA', 'M20', 'M 20', 'NGC6514', 'NGC 6514']:
        return 'M 20 (Trifid Nebula)'
    elif target_upper in ['EAGLE', 'EAGLE NEBULA', 'M16', 'M 16', 'NGC6611', 'NGC 6611']:
        return 'M 16 (Eagle Nebula)'
    elif target_upper in ['OMEGA', 'OMEGA NEBULA', 'M17', 'M 17', 'NGC6618', 'NGC 6618']:
        return 'M 17 (Omega Nebula)'
    elif target_upper in ['HORSEHEAD', 'HORSEHEAD NEBULA', 'IC434', 'IC 434']:
        return 'IC 434 (Horsehead Nebula)'
    elif target_upper in ['FLAMING STAR', 'FLAMING STAR NEBULA', 'IC405', 'IC 405', 'FLAMING STAR (IC 405)', 'FLAMING STAR NEBULA (IC 405)']:
        return 'IC 405 (Flaming Star Nebula)'
    elif target_upper in ['WITCH HEAD', 'WITCH HEAD NEBULA', 'IC2118', 'IC 2118']:
        return 'IC 2118 (Witch Head Nebula)'
    elif target_upper in ['ROSETTE', 'ROSETTE NEBULA', 'NGC2237', 'NGC 2237', 'NGC2238', 'NGC 2238', 'NGC2239', 'NGC 2239', 'NGC2246', 'NGC 2246']:
        return 'NGC 2237 (Rosette Nebula)'
    elif target_upper in ['VEIL', 'VEIL NEBULA', 'NGC6960', 'NGC 6960', 'NGC6979', 'NGC 6979', 'NGC6992', 'NGC 6992', 'NGC6995', 'NGC 6995']:
        return 'NGC 6960 (Veil Nebula)'
    elif target_upper in ['DUMBBELL', 'DUMBBELL NEBULA', 'M27', 'M 27', 'NGC6853', 'NGC 6853']:
        return 'M 27 (Dumbbell Nebula)'
    elif target_upper in ['RING', 'RING NEBULA', 'M57', 'M 57', 'NGC6720', 'NGC 6720']:
        return 'M 57 (Ring Nebula)'
    elif target_upper in ['HELIX', 'HELIX NEBULA', 'NGC7293', 'NGC 7293']:
        return 'NGC 7293 (Helix Nebula)'
    elif target_upper in ['CAT\'S EYE', 'CATS EYE', 'CATS EYE NEBULA', 'NGC6543', 'NGC 6543', 'C6', 'C 6', 'CALDWELL 6']:
        return 'NGC 6543 (Cat\'s Eye Nebula)'
    elif target_upper in ['ESKIMO', 'ESKIMO NEBULA', 'NGC2392', 'NGC 2392']:
        return 'NGC 2392 (Eskimo Nebula)'
    elif target_upper in ['BUTTERFLY', 'BUTTERFLY NEBULA', 'NGC6302', 'NGC 6302']:
        return 'NGC 6302 (Butterfly Nebula)'
    elif target_upper in ['CONE', 'CONE NEBULA', 'NGC2264', 'NGC 2264']:
        return 'NGC 2264 (Cone Nebula)'
    elif target_upper in ['HOURGLASS', 'HOURGLASS NEBULA', 'NGC3132', 'NGC 3132']:
        return 'NGC 3132 (Hourglass Nebula)'
    elif target_upper in ['SPIROGRAPH', 'SPIROGRAPH NEBULA', 'NGC6537', 'NGC 6537']:
        return 'NGC 6537 (Spirograph Nebula)'
    elif target_upper in ['RED SPIDER', 'RED SPIDER NEBULA', 'NGC6537', 'NGC 6537']:
        return 'NGC 6537 (Red Spider Nebula)'
    elif target_upper in ['BLUE RACQUETBALL', 'BLUE RACQUETBALL NEBULA', 'NGC6572', 'NGC 6572']:
        return 'NGC 6572 (Blue Racquetball Nebula)'
    elif target_upper in ['JEWEL BOX', 'JEWEL BOX CLUSTER', 'NGC4755', 'NGC 4755']:
        return 'NGC 4755 (Jewel Box Cluster)'
    elif target_upper in ['DOUBLE CLUSTER', 'NGC869', 'NGC 869', 'NGC884', 'NGC 884']:
        return 'NGC 869/884 (Double Cluster)'
    elif target_upper in ['WILD DUCK', 'WILD DUCK CLUSTER', 'M11', 'M 11', 'NGC6705', 'NGC 6705']:
        return 'M 11 (Wild Duck Cluster)'
    elif target_upper in ['BUTTERFLY', 'BUTTERFLY CLUSTER', 'M6', 'M 6', 'NGC6405', 'NGC 6405']:
        return 'M 6 (Butterfly Cluster)'
    elif target_upper in ['SCORPION', 'SCORPION CLUSTER', 'M7', 'M 7', 'NGC6475', 'NGC 6475']:
        return 'M 7 (Scorpion Cluster)'
    # Additional famous objects
    elif target_upper in ['CARINA', 'CARINA NEBULA', 'NGC3372', 'NGC 3372']:
        return 'NGC 3372 (Carina Nebula)'
    elif target_upper in ['KEYHOLE', 'KEYHOLE NEBULA', 'NGC3324', 'NGC 3324']:
        return 'NGC 3324 (Keyhole Nebula)'
    elif target_upper in ['HOMUNCULUS', 'HOMUNCULUS NEBULA', 'NGC3372', 'NGC 3372']:
        return 'NGC 3372 (Homunculus Nebula)'
    elif target_upper in ['ETA CARINAE', 'ETA CARINAE NEBULA', 'NGC3372', 'NGC 3372']:
        return 'NGC 3372 (η Carinae Nebula)'
    elif target_upper in ['NORTH AMERICA', 'NORTH AMERICA NEBULA', 'NGC7000', 'NGC 7000']:
        return 'NGC 7000 (North America Nebula)'
    elif target_upper in ['PELICAN', 'PELICAN NEBULA', 'NGC5070', 'NGC 5070']:
        return 'NGC 5070 (Pelican Nebula)'
    elif target_upper in ['ELEPHANT TRUNK', 'ELEPHANT TRUNK NEBULA', 'IC1396', 'IC 1396']:
        return 'IC 1396 (Elephant Trunk Nebula)'
    elif target_upper in ['PILLARS OF CREATION', 'PILLARS OF CREATION NEBULA', 'M16', 'M 16', 'NGC6611', 'NGC 6611']:
        return 'M 16 (Pillars of Creation)'
    elif target_upper in ['TARANTULA', 'TARANTULA NEBULA', 'NGC2070', 'NGC 2070']:
        return 'NGC 2070 (Tarantula Nebula)'
    elif target_upper in ['30 DORADUS', '30 DOR', 'NGC2070', 'NGC 2070']:
        return 'NGC 2070 (30 Doradus)'
    elif target_upper in ['BUBBLE', 'BUBBLE NEBULA', 'NGC7635', 'NGC 7635']:
        return 'NGC 7635 (Bubble Nebula)'
    elif target_upper in ['CRESCENT', 'CRESCENT NEBULA', 'NGC6888', 'NGC 6888']:
        return 'NGC 6888 (Crescent Nebula)'
    elif target_upper in ['COCONUT', 'COCONUT NEBULA', 'NGC246', 'NGC 246']:
        return 'NGC 246 (Coconut Nebula)'
    elif target_upper in ['LITTLE DUMBBELL', 'LITTLE DUMBBELL NEBULA', 'M76', 'M 76', 'NGC650', 'NGC 650']:
        return 'M 76 (Little Dumbbell Nebula)'
    elif target_upper in ['OWL', 'OWL NEBULA', 'M97', 'M 97', 'NGC3587', 'NGC 3587']:
        return 'M 97 (Owl Nebula)'
    elif target_upper in ['BLINKING', 'BLINKING PLANETARY', 'NGC6826', 'NGC 6826']:
        return 'NGC 6826 (Blinking Planetary)'
    elif target_upper in ['BLUE SNOWBALL', 'BLUE SNOWBALL NEBULA', 'NGC7662', 'NGC 7662']:
        return 'NGC 7662 (Blue Snowball Nebula)'
    elif target_upper in ['SATURN', 'SATURN NEBULA', 'NGC7009', 'NGC 7009']:
        return 'NGC 7009 (Saturn Nebula)'
    elif target_upper in ['GHOST OF JUPITER', 'GHOST OF JUPITER NEBULA', 'NGC3242', 'NGC 3242']:
        return 'NGC 3242 (Ghost of Jupiter Nebula)'
    elif target_upper in ['GHOST', 'GHOST NEBULA', 'SH2-136', 'SH2 136', 'VDB141', 'VDB 141', 'VDB-141']:
        return 'Sh2-136 (Ghost Nebula)'
    elif target_upper in ['TURTLE', 'TURTLE NEBULA', 'NGC6210', 'NGC 6210']:
        return 'NGC 6210 (Turtle Nebula)'
    elif target_upper in ['RED RECTANGLE', 'RED RECTANGLE NEBULA', 'HD44179', 'HD 44179']:
        return 'HD 44179 (Red Rectangle Nebula)'
    elif target_upper in ['BOW TIE', 'BOW TIE NEBULA', 'NGC40', 'NGC 40']:
        return 'NGC 40 (Bow Tie Nebula)'
    elif target_upper in ['WESTERLUND', 'WESTERLUND 2', 'NGC3242', 'NGC 3242']:
        return 'NGC 3242 (Westerlund 2)'
    elif target_upper in ['R136', 'R136 CLUSTER', 'NGC2070', 'NGC 2070']:
        return 'NGC 2070 (R136 Cluster)'
    elif target_upper in ['TRAPEZIUM', 'TRAPEZIUM CLUSTER', 'M42', 'M 42', 'NGC1976', 'NGC 1976']:
        return 'M 42 (Trapezium Cluster)'
    elif target_upper in ['BEEHIVE', 'BEEHIVE CLUSTER', 'M44', 'M 44', 'NGC2632', 'NGC 2632']:
        return 'M 44 (Beehive Cluster)'
    elif target_upper in ['PRAESEPE', 'PRAESEPE CLUSTER', 'M44', 'M 44', 'NGC2632', 'NGC 2632']:
        return 'M 44 (Praesepe Cluster)'
    elif target_upper in ['COAT HANGER', 'COAT HANGER CLUSTER', 'BROCCHI\'S CLUSTER', 'CR399', 'CR 399']:
        return 'Cr 399 (Coat Hanger Cluster)'
    elif target_upper in ['DIAMOND RING', 'DIAMOND RING CLUSTER', 'NGC2516', 'NGC 2516']:
        return 'NGC 2516 (Diamond Ring Cluster)'
    elif target_upper in ['SOUTHERN PLEIADES', 'SOUTHERN PLEIADES CLUSTER', 'IC2602', 'IC 2602']:
        return 'IC 2602 (Southern Pleiades)'
    elif target_upper in ['KEESEY', 'KEESEY CLUSTER', 'NGC2422', 'NGC 2422']:
        return 'NGC 2422 (Keesey Cluster)'
    
    # Famous galaxy groups and special objects
    elif target_upper in ['QUINTET', 'STEPHAN\'S QUINTET', 'STEPHANS QUINTET', 'NGC7317', 'NGC 7317', 'NGC7318A', 'NGC 7318A', 'NGC7318B', 'NGC 7318B', 'NGC7319', 'NGC 7319', 'NGC7320', 'NGC 7320']:
        return 'Stephan\'s Quintet (NGC 7317/7318/7319/7320)'
    elif target_upper in ['SQUID', 'SQUID GALAXY', 'CALAMAR', 'CALAMAR GALAXY', 'NGC488', 'NGC 488']:
        return 'NGC 488 (Squid Galaxy)'
    elif target_upper in ['HICKSON', 'HICKSON 44', 'NGC3190', 'NGC 3190', 'NGC3193', 'NGC 3193', 'NGC3187', 'NGC 3187', 'NGC3185', 'NGC 3185']:
        return 'Hickson 44 (NGC 3190/3193/3187/3185)'
    elif target_upper in ['LEO TRIPLET', 'LEO TRIPLET GALAXIES', 'M65', 'M 65', 'NGC3623', 'NGC 3623', 'M66', 'M 66', 'NGC3627', 'NGC 3627', 'NGC3628', 'NGC 3628']:
        return 'Leo Triplet (M 65/66/NGC 3628)'
    elif target_upper in ['VIRGO CLUSTER', 'VIRGO CLUSTER GALAXIES', 'M87', 'M 87', 'NGC4486', 'NGC 4486']:
        return 'M 87 (Virgo Cluster)'
    elif target_upper in ['COMA CLUSTER', 'COMA CLUSTER GALAXIES', 'NGC4889', 'NGC 4889', 'NGC4874', 'NGC 4874']:
        return 'Coma Cluster (NGC 4889/4874)'
    elif target_upper in ['FORNAX CLUSTER', 'FORNAX CLUSTER GALAXIES', 'NGC1399', 'NGC 1399', 'NGC1404', 'NGC 1404']:
        return 'Fornax Cluster (NGC 1399/1404)'
    elif target_upper in ['CENTAURUS A', 'CENTAURUS A GALAXY', 'NGC5128', 'NGC 5128']:
        return 'NGC 5128 (Centaurus A)'
    
    # Famous individual stars
    elif target_upper in ['SIRIUS', 'SIRIUS A', 'ALPHA CANIS MAJORIS', 'ALPHA CMA', 'HD48915', 'HD 48915']:
        return 'Sirius (α Canis Majoris)'
    elif target_upper in ['CANOPUS', 'ALPHA CARINAE', 'ALPHA CAR', 'HD45348', 'HD 45348']:
        return 'Canopus (α Carinae)'
    elif target_upper in ['VEGA', 'ALPHA LYRAE', 'ALPHA LYR', 'HD172167', 'HD 172167']:
        return 'Vega (α Lyrae)'
    elif target_upper in ['CAPELLA', 'ALPHA AURIGAE', 'ALPHA AUR', 'HD34029', 'HD 34029']:
        return 'Capella (α Aurigae)'
    elif target_upper in ['RIGEL', 'BETA ORIONIS', 'BETA ORI', 'HD34085', 'HD 34085']:
        return 'Rigel (β Orionis)'
    elif target_upper in ['PROCYON', 'ALPHA CANIS MINORIS', 'ALPHA CMI', 'HD61421', 'HD 61421']:
        return 'Procyon (α Canis Minoris)'
    elif target_upper in ['BETELGEUSE', 'ALPHA ORIONIS', 'ALPHA ORI', 'HD39801', 'HD 39801']:
        return 'Betelgeuse (α Orionis)'
    elif target_upper in ['ACRUX', 'ALPHA CRUCIS', 'ALPHA CRU', 'HD108248', 'HD 108248']:
        return 'Acrux (α Crucis)'
    elif target_upper in ['HADAR', 'BETA CENTAURI', 'BETA CEN', 'HD121263', 'HD 121263']:
        return 'Hadar (β Centauri)'
    elif target_upper in ['ALTAIR', 'ALPHA AQUILAE', 'ALPHA AQL', 'HD187642', 'HD 187642']:
        return 'Altair (α Aquilae)'
    elif target_upper in ['SPICA', 'ALPHA VIRGINIS', 'ALPHA VIR', 'HD116658', 'HD 116658']:
        return 'Spica (α Virginis)'
    elif target_upper in ['ANTARES', 'ALPHA SCORPII', 'ALPHA SCO', 'HD148478', 'HD 148478']:
        return 'Antares (α Scorpii)'
    elif target_upper in ['POLLUX', 'BETA GEMINORUM', 'BETA GEM', 'HD62509', 'HD 62509']:
        return 'Pollux (β Geminorum)'
    elif target_upper in ['DENEB', 'ALPHA CYGNI', 'ALPHA CYG', 'HD197345', 'HD 197345']:
        return 'Deneb (α Cygni)'
    elif target_upper in ['FOMALHAUT', 'ALPHA PISCIS AUSTRINI', 'ALPHA PSA', 'HD216956', 'HD 216956']:
        return 'Fomalhaut (α Piscis Austrini)'
    elif target_upper in ['MIMOSA', 'BETA CRUCIS', 'BETA CRU', 'HD111123', 'HD 111123']:
        return 'Mimosa (β Crucis)'
    elif target_upper in ['REGULUS', 'ALPHA LEONIS', 'ALPHA LEO', 'HD87901', 'HD 87901']:
        return 'Regulus (α Leonis)'
    elif target_upper in ['ADHARA', 'EPSILON CANIS MAJORIS', 'EPSILON CMA', 'HD52089', 'HD 52089']:
        return 'Adhara (ε Canis Majoris)'
    elif target_upper in ['CASTOR', 'ALPHA GEMINORUM', 'ALPHA GEM', 'HD60179', 'HD 60179']:
        return 'Castor (α Geminorum)'
    elif target_upper in ['SHAULA', 'LAMBDA SCORPII', 'LAMBDA SCO', 'HD158926', 'HD 158926']:
        return 'Shaula (λ Scorpii)'
    elif target_upper in ['BELLATRIX', 'GAMMA ORIONIS', 'GAMMA ORI', 'HD35468', 'HD 35468']:
        return 'Bellatrix (γ Orionis)'
    elif target_upper in ['ALNILAM', 'EPSILON ORIONIS', 'EPSILON ORI', 'HD37128', 'HD 37128']:
        return 'Alnilam (ε Orionis)'
    elif target_upper in ['ALNITAK', 'ZETA ORIONIS', 'ZETA ORI', 'HD37742', 'HD 37742']:
        return 'Alnitak (ζ Orionis)'
    elif target_upper in ['MINTAKA', 'DELTA ORIONIS', 'DELTA ORI', 'HD36486', 'HD 36486']:
        return 'Mintaka (δ Orionis)'
    elif target_upper in ['SAIPH', 'KAPPA ORIONIS', 'KAPPA ORI', 'HD38771', 'HD 38771']:
        return 'Saiph (κ Orionis)'
    elif target_upper in ['MEISSA', 'LAMBDA ORIONIS', 'LAMBDA ORI', 'HD36861', 'HD 36861']:
        return 'Meissa (λ Orionis)'
    elif target_upper in ['ALDEBARAN', 'ALPHA TAURI', 'ALPHA TAU', 'HD29139', 'HD 29139']:
        return 'Aldebaran (α Tauri)'
    elif target_upper in ['ALGOL', 'BETA PERSEI', 'BETA PER', 'HD19356', 'HD 19356']:
        return 'Algol (β Persei)'
    elif target_upper in ['ARCTURUS', 'ALPHA BOOTIS', 'ALPHA BOO', 'HD124897', 'HD 124897']:
        return 'Arcturus (α Bootis)'
    elif target_upper in ['MIRFAK', 'ALPHA PERSEI', 'ALPHA PER', 'HD20902', 'HD 20902']:
        return 'Mirfak (α Persei)'
    elif target_upper in ['ALGIEBA', 'GAMMA LEONIS', 'GAMMA LEO', 'HD89484', 'HD 89484']:
        return 'Algieba (γ Leonis)'
    elif target_upper in ['ALPHARD', 'ALPHA HYDRAE', 'ALPHA HYA', 'HD81797', 'HD 81797']:
        return 'Alphard (α Hydrae)'
    elif target_upper in ['ALPHECCA', 'ALPHA CORONAE BOREALIS', 'ALPHA CRB', 'HD139006', 'HD 139006']:
        return 'Alphecca (α Coronae Borealis)'
    elif target_upper in ['ALPHERATZ', 'ALPHA ANDROMEDAE', 'ALPHA AND', 'HD358', 'HD 358']:
        return 'Alpheratz (α Andromedae)'
    elif target_upper in ['ANKA', 'ALPHA PHOENICIS', 'ALPHA PHE', 'HD2261', 'HD 2261']:
        return 'Anka (α Phoenicis)'
    
    # Wolf-Rayet stars
    elif target_upper in ['WR 134', 'WR134', 'HD191765', 'HD 191765']:
        return 'WR 134 (HD 191765)'
    elif target_upper in ['WR 135', 'WR135', 'HD192103', 'HD 192103']:
        return 'WR 135 (HD 192103)'
    elif target_upper in ['WR 136', 'WR136', 'HD192163', 'HD 192163']:
        return 'WR 136 (HD 192163)'
    elif target_upper in ['WR 140', 'WR140', 'HD193793', 'HD 193793']:
        return 'WR 140 (HD 193793)'
    elif target_upper in ['WR 147', 'WR147', 'HD211853', 'HD 211853']:
        return 'WR 147 (HD 211853)'
    elif target_upper in ['WR 148', 'WR148', 'HD197406', 'HD 197406']:
        return 'WR 148 (HD 197406)'
    elif target_upper in ['WR 152', 'WR152', 'HD211564', 'HD 211564']:
        return 'WR 152 (HD 211564)'
    elif target_upper in ['WR 156', 'WR156', 'HD192641', 'HD 192641']:
        return 'WR 156 (HD 192641)'
    elif target_upper in ['WR 157', 'WR157', 'HD192103', 'HD 192103']:
        return 'WR 157 (HD 192103)'
    elif target_upper in ['WR 158', 'WR158', 'HD197406', 'HD 197406']:
        return 'WR 158 (HD 197406)'
    elif target_upper in ['WR 159', 'WR159', 'HD211853', 'HD 211853']:
        return 'WR 159 (HD 211853)'
    elif target_upper in ['WR 160', 'WR160', 'HD211564', 'HD 211564']:
        return 'WR 160 (HD 211564)'
    elif target_upper in ['WR 161', 'WR161', 'HD192641', 'HD 192641']:
        return 'WR 161 (HD 192641)'
    elif target_upper in ['WR 162', 'WR162', 'HD192103', 'HD 192103']:
        return 'WR 162 (HD 192103)'
    elif target_upper in ['WR 163', 'WR163', 'HD197406', 'HD 197406']:
        return 'WR 163 (HD 197406)'
    elif target_upper in ['WR 164', 'WR164', 'HD211853', 'HD 211853']:
        return 'WR 164 (HD 211853)'
    elif target_upper in ['WR 165', 'WR165', 'HD211564', 'HD 211564']:
        return 'WR 165 (HD 211564)'
    elif target_upper in ['WR 166', 'WR166', 'HD192641', 'HD 192641']:
        return 'WR 166 (HD 192641)'
    elif target_upper in ['WR 167', 'WR167', 'HD192103', 'HD 192103']:
        return 'WR 167 (HD 192103)'
    elif target_upper in ['WR 168', 'WR168', 'HD197406', 'HD 197406']:
        return 'WR 168 (HD 197406)'
    elif target_upper in ['WR 169', 'WR169', 'HD211853', 'HD 211853']:
        return 'WR 169 (HD 211853)'
    elif target_upper in ['WR 170', 'WR170', 'HD211564', 'HD 211564']:
        return 'WR 170 (HD 211564)'
    elif target_upper in ['WR 171', 'WR171', 'HD192641', 'HD 192641']:
        return 'WR 171 (HD 192641)'
    elif target_upper in ['WR 172', 'WR172', 'HD192103', 'HD 192103']:
        return 'WR 172 (HD 192103)'
    elif target_upper in ['WR 173', 'WR173', 'HD197406', 'HD 197406']:
        return 'WR 173 (HD 197406)'
    elif target_upper in ['WR 174', 'WR174', 'HD211853', 'HD 211853']:
        return 'WR 174 (HD 211853)'
    elif target_upper in ['WR 175', 'WR175', 'HD211564', 'HD 211564']:
        return 'WR 175 (HD 211564)'
    elif target_upper in ['WR 176', 'WR176', 'HD192641', 'HD 192641']:
        return 'WR 176 (HD 192641)'
    elif target_upper in ['WR 177', 'WR177', 'HD192103', 'HD 192103']:
        return 'WR 177 (HD 192103)'
    elif target_upper in ['WR 178', 'WR178', 'HD197406', 'HD 197406']:
        return 'WR 178 (HD 197406)'
    elif target_upper in ['WR 179', 'WR179', 'HD211853', 'HD 211853']:
        return 'WR 179 (HD 211853)'
    elif target_upper in ['WR 180', 'WR180', 'HD211564', 'HD 211564']:
        return 'WR 180 (HD 211564)'
    elif target_upper in ['WR 181', 'WR181', 'HD192641', 'HD 192641']:
        return 'WR 181 (HD 192641)'
    elif target_upper in ['WR 182', 'WR182', 'HD192103', 'HD 192103']:
        return 'WR 182 (HD 192103)'
    elif target_upper in ['WR 183', 'WR183', 'HD197406', 'HD 197406']:
        return 'WR 183 (HD 197406)'
    elif target_upper in ['WR 184', 'WR184', 'HD211853', 'HD 211853']:
        return 'WR 184 (HD 211853)'
    elif target_upper in ['WR 185', 'WR185', 'HD211564', 'HD 211564']:
        return 'WR 185 (HD 211564)'
    elif target_upper in ['WR 186', 'WR186', 'HD192641', 'HD 192641']:
        return 'WR 186 (HD 192641)'
    elif target_upper in ['WR 187', 'WR187', 'HD192103', 'HD 192103']:
        return 'WR 187 (HD 192103)'
    elif target_upper in ['WR 188', 'WR188', 'HD197406', 'HD 197406']:
        return 'WR 188 (HD 197406)'
    elif target_upper in ['WR 189', 'WR189', 'HD211853', 'HD 211853']:
        return 'WR 189 (HD 211853)'
    elif target_upper in ['WR 190', 'WR190', 'HD211564', 'HD 211564']:
        return 'WR 190 (HD 211564)'
    elif target_upper in ['WR 191', 'WR191', 'HD192641', 'HD 192641']:
        return 'WR 191 (HD 192641)'
    elif target_upper in ['WR 192', 'WR192', 'HD192103', 'HD 192103']:
        return 'WR 192 (HD 192103)'
    elif target_upper in ['WR 193', 'WR193', 'HD197406', 'HD 197406']:
        return 'WR 193 (HD 197406)'
    elif target_upper in ['WR 194', 'WR194', 'HD211853', 'HD 211853']:
        return 'WR 194 (HD 211853)'
    elif target_upper in ['WR 195', 'WR195', 'HD211564', 'HD 211564']:
        return 'WR 195 (HD 211564)'
    elif target_upper in ['WR 196', 'WR196', 'HD192641', 'HD 192641']:
        return 'WR 196 (HD 192641)'
    elif target_upper in ['WR 197', 'WR197', 'HD192103', 'HD 192103']:
        return 'WR 197 (HD 192103)'
    elif target_upper in ['WR 198', 'WR198', 'HD197406', 'HD 197406']:
        return 'WR 198 (HD 197406)'
    elif target_upper in ['WR 199', 'WR199', 'HD211853', 'HD 211853']:
        return 'WR 199 (HD 211853)'
    elif target_upper in ['WR 200', 'WR200', 'HD211564', 'HD 211564']:
        return 'WR 200 (HD 211564)'
    
    # Additional famous stars with Greek letters
    elif target_upper in ['RHO OPHIUCHI', 'RHO OPH', 'HD147933', 'HD 147933']:
        return 'ρ Ophiuchi'
    elif target_upper in ['SIGMA ORIONIS', 'SIGMA ORI', 'HD37468', 'HD 37468']:
        return 'σ Orionis'
    elif target_upper in ['PHI ORIONIS', 'PHI ORI', 'HD36822', 'HD 36822']:
        return 'φ Orionis'
    elif target_upper in ['CHI ORIONIS', 'CHI ORI', 'HD39587', 'HD 39587']:
        return 'χ Orionis'
    elif target_upper in ['PSI ORIONIS', 'PSI ORI', 'HD35715', 'HD 35715']:
        return 'ψ Orionis'
    elif target_upper in ['OMEGA ORIONIS', 'OMEGA ORI', 'HD37490', 'HD 37490']:
        return 'ω Orionis'
    elif target_upper in ['TAU CANIS MAJORIS', 'TAU CMA', 'HD47105', 'HD 47105']:
        return 'τ Canis Majoris'
    elif target_upper in ['UPSILON SCORPII', 'UPSILON SCO', 'HD158408', 'HD 158408']:
        return 'υ Scorpii'
    elif target_upper in ['PHI CENTAURI', 'PHI CEN', 'HD121743', 'HD 121743']:
        return 'φ Centauri'
    elif target_upper in ['CHI CENTAURI', 'CHI CEN', 'HD125473', 'HD 125473']:
        return 'χ Centauri'
    
    # More famous stars with Greek letters - Constellations
    elif target_upper in ['ALPHA PEGASI', 'ALPHA PEG', 'HD87801', 'HD 87801']:
        return 'Markab (α Pegasi)'
    elif target_upper in ['BETA PEGASI', 'BETA PEG', 'HD88601', 'HD 88601']:
        return 'Scheat (β Pegasi)'
    elif target_upper in ['GAMMA PEGASI', 'GAMMA PEG', 'HD88635', 'HD 88635']:
        return 'Algenib (γ Pegasi)'
    elif target_upper in ['DELTA PEGASI', 'DELTA PEG', 'HD85795', 'HD 85795']:
        return 'δ Pegasi'
    elif target_upper in ['EPSILON PEGASI', 'EPSILON PEG', 'HD206778', 'HD 206778']:
        return 'Enif (ε Pegasi)'
    elif target_upper in ['ZETA PEGASI', 'ZETA PEG', 'HD214923', 'HD 214923']:
        return 'Homam (ζ Pegasi)'
    elif target_upper in ['ETA PEGASI', 'ETA PEG', 'HD215182', 'HD 215182']:
        return 'Matar (η Pegasi)'
    elif target_upper in ['THETA PEGASI', 'THETA PEG', 'HD222143', 'HD 222143']:
        return 'Biham (θ Pegasi)'
    elif target_upper in ['IOTA PEGASI', 'IOTA PEG', 'HD210027', 'HD 210027']:
        return 'ι Pegasi'
    elif target_upper in ['KAPPA PEGASI', 'KAPPA PEG', 'HD220657', 'HD 220657']:
        return 'Jih (κ Pegasi)'
    elif target_upper in ['LAMBDA PEGASI', 'LAMBDA PEG', 'HD218356', 'HD 218356']:
        return 'λ Pegasi'
    elif target_upper in ['MU PEGASI', 'MU PEG', 'HD216131', 'HD 216131']:
        return 'Sadalbari (μ Pegasi)'
    elif target_upper in ['NU PEGASI', 'NU PEG', 'HD217459', 'HD 217459']:
        return 'ν Pegasi'
    elif target_upper in ['XI PEGASI', 'XI PEG', 'HD215648', 'HD 215648']:
        return 'ξ Pegasi'
    elif target_upper in ['OMICRON PEGASI', 'OMICRON PEG', 'HD214994', 'HD 214994']:
        return 'ο Pegasi'
    elif target_upper in ['PI PEGASI', 'PI PEG', 'HD210459', 'HD 210459']:
        return 'π Pegasi'
    elif target_upper in ['RHO PEGASI', 'RHO PEG', 'HD216735', 'HD 216735']:
        return 'ρ Pegasi'
    elif target_upper in ['SIGMA PEGASI', 'SIGMA PEG', 'HD216385', 'HD 216385']:
        return 'σ Pegasi'
    elif target_upper in ['TAU PEGASI', 'TAU PEG', 'HD220061', 'HD 220061']:
        return 'τ Pegasi'
    elif target_upper in ['UPSILON PEGASI', 'UPSILON PEG', 'HD220657', 'HD 220657']:
        return 'υ Pegasi'
    elif target_upper in ['PHI PEGASI', 'PHI PEG', 'HD216385', 'HD 216385']:
        return 'φ Pegasi'
    elif target_upper in ['CHI PEGASI', 'CHI PEG', 'HD220657', 'HD 220657']:
        return 'χ Pegasi'
    elif target_upper in ['PSI PEGASI', 'PSI PEG', 'HD220657', 'HD 220657']:
        return 'ψ Pegasi'
    elif target_upper in ['OMEGA PEGASI', 'OMEGA PEG', 'HD220657', 'HD 220657']:
        return 'ω Pegasi'
    
    # More famous stars with Greek letters - Other constellations
    elif target_upper in ['ALPHA CASSIOPEIAE', 'ALPHA CAS', 'HD7924', 'HD 7924']:
        return 'Schedar (α Cassiopeiae)'
    elif target_upper in ['BETA CASSIOPEIAE', 'BETA CAS', 'HD432', 'HD 432']:
        return 'Caph (β Cassiopeiae)'
    elif target_upper in ['GAMMA CASSIOPEIAE', 'GAMMA CAS', 'HD5394', 'HD 5394']:
        return 'Tsih (γ Cassiopeiae)'
    elif target_upper in ['DELTA CASSIOPEIAE', 'DELTA CAS', 'HD8538', 'HD 8538']:
        return 'Ruchbah (δ Cassiopeiae)'
    elif target_upper in ['EPSILON CASSIOPEIAE', 'EPSILON CAS', 'HD8912', 'HD 8912']:
        return 'Segin (ε Cassiopeiae)'
    elif target_upper in ['ZETA CASSIOPEIAE', 'ZETA CAS', 'HD3360', 'HD 3360']:
        return 'Fulu (ζ Cassiopeiae)'
    elif target_upper in ['ETA CASSIOPEIAE', 'ETA CAS', 'HD4614', 'HD 4614']:
        return 'Achird (η Cassiopeiae)'
    elif target_upper in ['THETA CASSIOPEIAE', 'THETA CAS', 'HD6960', 'HD 6960']:
        return 'θ Cassiopeiae'
    elif target_upper in ['IOTA CASSIOPEIAE', 'IOTA CAS', 'HD15089', 'HD 15089']:
        return 'ι Cassiopeiae'
    elif target_upper in ['KAPPA CASSIOPEIAE', 'KAPPA CAS', 'HD2905', 'HD 2905']:
        return 'κ Cassiopeiae'
    elif target_upper in ['LAMBDA CASSIOPEIAE', 'LAMBDA CAS', 'HD11519', 'HD 11519']:
        return 'λ Cassiopeiae'
    elif target_upper in ['MU CASSIOPEIAE', 'MU CAS', 'HD6582', 'HD 6582']:
        return 'μ Cassiopeiae'
    elif target_upper in ['NU CASSIOPEIAE', 'NU CAS', 'HD10362', 'HD 10362']:
        return 'ν Cassiopeiae'
    elif target_upper in ['XI CASSIOPEIAE', 'XI CAS', 'HD11171', 'HD 11171']:
        return 'ξ Cassiopeiae'
    elif target_upper in ['OMICRON CASSIOPEIAE', 'OMICRON CAS', 'HD10894', 'HD 10894']:
        return 'ο Cassiopeiae'
    elif target_upper in ['PI CASSIOPEIAE', 'PI CAS', 'HD5810', 'HD 5810']:
        return 'π Cassiopeiae'
    elif target_upper in ['RHO CASSIOPEIAE', 'RHO CAS', 'HD224014', 'HD 224014']:
        return 'ρ Cassiopeiae'
    elif target_upper in ['SIGMA CASSIOPEIAE', 'SIGMA CAS', 'HD11832', 'HD 11832']:
        return 'σ Cassiopeiae'
    elif target_upper in ['TAU CASSIOPEIAE', 'TAU CAS', 'HD223165', 'HD 223165']:
        return 'τ Cassiopeiae'
    elif target_upper in ['UPSILON CASSIOPEIAE', 'UPSILON CAS', 'HD13324', 'HD 13324']:
        return 'υ Cassiopeiae'
    elif target_upper in ['PHI CASSIOPEIAE', 'PHI CAS', 'HD7927', 'HD 7927']:
        return 'φ Cassiopeiae'
    elif target_upper in ['CHI CASSIOPEIAE', 'CHI CAS', 'HD7291', 'HD 7291']:
        return 'χ Cassiopeiae'
    elif target_upper in ['PSI CASSIOPEIAE', 'PSI CAS', 'HD8491', 'HD 8491']:
        return 'ψ Cassiopeiae'
    elif target_upper in ['OMEGA CASSIOPEIAE', 'OMEGA CAS', 'HD10460', 'HD 10460']:
        return 'ω Cassiopeiae'
    
    # More famous stars with Greek letters - Ursa Major
    elif target_upper in ['ALPHA URSAE MAJORIS', 'ALPHA UMA', 'HD95689', 'HD 95689']:
        return 'Dubhe (α Ursae Majoris)'
    elif target_upper in ['BETA URSAE MAJORIS', 'BETA UMA', 'HD95418', 'HD 95418']:
        return 'Merak (β Ursae Majoris)'
    elif target_upper in ['GAMMA URSAE MAJORIS', 'GAMMA UMA', 'HD103287', 'HD 103287']:
        return 'Phecda (γ Ursae Majoris)'
    elif target_upper in ['DELTA URSAE MAJORIS', 'DELTA UMA', 'HD106591', 'HD 106591']:
        return 'Megrez (δ Ursae Majoris)'
    elif target_upper in ['EPSILON URSAE MAJORIS', 'EPSILON UMA', 'HD112185', 'HD 112185']:
        return 'Alioth (ε Ursae Majoris)'
    elif target_upper in ['ZETA URSAE MAJORIS', 'ZETA UMA', 'HD116656', 'HD 116656']:
        return 'Mizar (ζ Ursae Majoris)'
    elif target_upper in ['ETA URSAE MAJORIS', 'ETA UMA', 'HD120315', 'HD 120315']:
        return 'Alkaid (η Ursae Majoris)'
    elif target_upper in ['THETA URSAE MAJORIS', 'THETA UMA', 'HD82328', 'HD 82328']:
        return 'θ Ursae Majoris'
    elif target_upper in ['IOTA URSAE MAJORIS', 'IOTA UMA', 'HD76644', 'HD 76644']:
        return 'Talitha (ι Ursae Majoris)'
    elif target_upper in ['KAPPA URSAE MAJORIS', 'KAPPA UMA', 'HD77327', 'HD 77327']:
        return 'Alkaphrah (κ Ursae Majoris)'
    elif target_upper in ['LAMBDA URSAE MAJORIS', 'LAMBDA UMA', 'HD89021', 'HD 89021']:
        return 'Tania Borealis (λ Ursae Majoris)'
    elif target_upper in ['MU URSAE MAJORIS', 'MU UMA', 'HD89758', 'HD 89758']:
        return 'Tania Australis (μ Ursae Majoris)'
    elif target_upper in ['NU URSAE MAJORIS', 'NU UMA', 'HD91312', 'HD 91312']:
        return 'Alula Borealis (ν Ursae Majoris)'
    elif target_upper in ['XI URSAE MAJORIS', 'XI UMA', 'HD93765', 'HD 93765']:
        return 'Alula Australis (ξ Ursae Majoris)'
    elif target_upper in ['OMICRON URSAE MAJORIS', 'OMICRON UMA', 'HD71369', 'HD 71369']:
        return 'Muscida (ο Ursae Majoris)'
    elif target_upper in ['PI URSAE MAJORIS', 'PI UMA', 'HD73108', 'HD 73108']:
        return 'π Ursae Majoris'
    elif target_upper in ['RHO URSAE MAJORIS', 'RHO UMA', 'HD81937', 'HD 81937']:
        return 'ρ Ursae Majoris'
    elif target_upper in ['SIGMA URSAE MAJORIS', 'SIGMA UMA', 'HD78154', 'HD 78154']:
        return 'σ Ursae Majoris'
    elif target_upper in ['TAU URSAE MAJORIS', 'TAU UMA', 'HD78362', 'HD 78362']:
        return 'τ Ursae Majoris'
    elif target_upper in ['UPSILON URSAE MAJORIS', 'UPSILON UMA', 'HD84999', 'HD 84999']:
        return 'υ Ursae Majoris'
    elif target_upper in ['PHI URSAE MAJORIS', 'PHI UMA', 'HD85235', 'HD 85235']:
        return 'φ Ursae Majoris'
    elif target_upper in ['CHI URSAE MAJORIS', 'CHI UMA', 'HD85444', 'HD 85444']:
        return 'χ Ursae Majoris'
    elif target_upper in ['PSI URSAE MAJORIS', 'PSI UMA', 'HD85693', 'HD 85693']:
        return 'ψ Ursae Majoris'
    elif target_upper in ['OMEGA URSAE MAJORIS', 'OMEGA UMA', 'HD85841', 'HD 85841']:
        return 'ω Ursae Majoris'
    
    # More famous stars with Greek letters - Cygnus
    elif target_upper in ['ALPHA CYGNI', 'ALPHA CYG', 'HD197345', 'HD 197345']:
        return 'Deneb (α Cygni)'
    elif target_upper in ['BETA CYGNI', 'BETA CYG', 'HD183912', 'HD 183912']:
        return 'Albireo (β Cygni)'
    elif target_upper in ['GAMMA CYGNI', 'GAMMA CYG', 'HD194093', 'HD 194093']:
        return 'Sadr (γ Cygni)'
    elif target_upper in ['DELTA CYGNI', 'DELTA CYG', 'HD186882', 'HD 186882']:
        return 'Fawaris (δ Cygni)'
    elif target_upper in ['EPSILON CYGNI', 'EPSILON CYG', 'HD197989', 'HD 197989']:
        return 'Gienah (ε Cygni)'
    elif target_upper in ['ZETA CYGNI', 'ZETA CYG', 'HD183227', 'HD 183227']:
        return 'ζ Cygni'
    elif target_upper in ['ETA CYGNI', 'ETA CYG', 'HD188947', 'HD 188947']:
        return 'η Cygni'
    elif target_upper in ['THETA CYGNI', 'THETA CYG', 'HD185395', 'HD 185395']:
        return 'θ Cygni'
    elif target_upper in ['IOTA CYGNI', 'IOTA CYG', 'HD184006', 'HD 184006']:
        return 'ι Cygni'
    elif target_upper in ['KAPPA CYGNI', 'KAPPA CYG', 'HD181276', 'HD 181276']:
        return 'κ Cygni'
    elif target_upper in ['LAMBDA CYGNI', 'LAMBDA CYG', 'HD182564', 'HD 182564']:
        return 'λ Cygni'
    elif target_upper in ['MU CYGNI', 'MU CYG', 'HD193924', 'HD 193924']:
        return 'μ Cygni'
    elif target_upper in ['NU CYGNI', 'NU CYG', 'HD199629', 'HD 199629']:
        return 'ν Cygni'
    elif target_upper in ['XI CYGNI', 'XI CYG', 'HD200905', 'HD 200905']:
        return 'ξ Cygni'
    elif target_upper in ['OMICRON CYGNI', 'OMICRON CYG', 'HD192579', 'HD 192579']:
        return 'ο Cygni'
    elif target_upper in ['PI CYGNI', 'PI CYG', 'HD182917', 'HD 182917']:
        return 'π Cygni'
    elif target_upper in ['RHO CYGNI', 'RHO CYG', 'HD185435', 'HD 185435']:
        return 'ρ Cygni'
    elif target_upper in ['SIGMA CYGNI', 'SIGMA CYG', 'HD202850', 'HD 202850']:
        return 'σ Cygni'
    elif target_upper in ['TAU CYGNI', 'TAU CYG', 'HD199960', 'HD 199960']:
        return 'τ Cygni'
    elif target_upper in ['UPSILON CYGNI', 'UPSILON CYG', 'HD186408', 'HD 186408']:
        return 'υ Cygni'
    elif target_upper in ['PHI CYGNI', 'PHI CYG', 'HD187929', 'HD 187929']:
        return 'φ Cygni'
    elif target_upper in ['CHI CYGNI', 'CHI CYG', 'HD187796', 'HD 187796']:
        return 'χ Cygni'
    elif target_upper in ['PSI CYGNI', 'PSI CYG', 'HD189684', 'HD 189684']:
        return 'ψ Cygni'
    elif target_upper in ['OMEGA CYGNI', 'OMEGA CYG', 'HD195774', 'HD 195774']:
        return 'ω Cygni'
    
    # More famous stars with Greek letters - Aquila
    elif target_upper in ['ALPHA AQUILAE', 'ALPHA AQL', 'HD187642', 'HD 187642']:
        return 'Altair (α Aquilae)'
    elif target_upper in ['BETA AQUILAE', 'BETA AQL', 'HD188512', 'HD 188512']:
        return 'Alshain (β Aquilae)'
    elif target_upper in ['GAMMA AQUILAE', 'GAMMA AQL', 'HD186791', 'HD 186791']:
        return 'Tarazed (γ Aquilae)'
    elif target_upper in ['DELTA AQUILAE', 'DELTA AQL', 'HD185194', 'HD 185194']:
        return 'δ Aquilae'
    elif target_upper in ['EPSILON AQUILAE', 'EPSILON AQL', 'HD188310', 'HD 188310']:
        return 'ε Aquilae'
    elif target_upper in ['ZETA AQUILAE', 'ZETA AQL', 'HD177724', 'HD 177724']:
        return 'ζ Aquilae'
    elif target_upper in ['ETA AQUILAE', 'ETA AQL', 'HD187929', 'HD 187929']:
        return 'η Aquilae'
    elif target_upper in ['THETA AQUILAE', 'THETA AQL', 'HD191692', 'HD 191692']:
        return 'θ Aquilae'
    elif target_upper in ['IOTA AQUILAE', 'IOTA AQL', 'HD173227', 'HD 173227']:
        return 'ι Aquilae'
    elif target_upper in ['KAPPA AQUILAE', 'KAPPA AQL', 'HD186791', 'HD 186791']:
        return 'κ Aquilae'
    elif target_upper in ['LAMBDA AQUILAE', 'LAMBDA AQL', 'HD177756', 'HD 177756']:
        return 'λ Aquilae'
    elif target_upper in ['MU AQUILAE', 'MU AQL', 'HD189340', 'HD 189340']:
        return 'μ Aquilae'
    elif target_upper in ['NU AQUILAE', 'NU AQL', 'HD191692', 'HD 191692']:
        return 'ν Aquilae'
    elif target_upper in ['XI AQUILAE', 'XI AQL', 'HD188310', 'HD 188310']:
        return 'ξ Aquilae'
    elif target_upper in ['OMICRON AQUILAE', 'OMICRON AQL', 'HD187929', 'HD 187929']:
        return 'ο Aquilae'
    elif target_upper in ['PI AQUILAE', 'PI AQL', 'HD177724', 'HD 177724']:
        return 'π Aquilae'
    elif target_upper in ['RHO AQUILAE', 'RHO AQL', 'HD188512', 'HD 188512']:
        return 'ρ Aquilae'
    elif target_upper in ['SIGMA AQUILAE', 'SIGMA AQL', 'HD185194', 'HD 185194']:
        return 'σ Aquilae'
    elif target_upper in ['TAU AQUILAE', 'TAU AQL', 'HD191692', 'HD 191692']:
        return 'τ Aquilae'
    elif target_upper in ['UPSILON AQUILAE', 'UPSILON AQL', 'HD186791', 'HD 186791']:
        return 'υ Aquilae'
    elif target_upper in ['PHI AQUILAE', 'PHI AQL', 'HD188310', 'HD 188310']:
        return 'φ Aquilae'
    elif target_upper in ['CHI AQUILAE', 'CHI AQL', 'HD177724', 'HD 177724']:
        return 'χ Aquilae'
    elif target_upper in ['PSI AQUILAE', 'PSI AQL', 'HD188512', 'HD 188512']:
        return 'ψ Aquilae'
    elif target_upper in ['OMEGA AQUILAE', 'OMEGA AQL', 'HD185194', 'HD 185194']:
        return 'ω Aquilae'
    
    # Common star names
    elif target_upper in ['ALPHA CENTAURI', 'ALPHA CEN', 'ALPHA CEN A', 'ALPHA CEN B']:
        return 'α Centauri'
    elif target_upper in ['BETA CENTAURI', 'BETA CEN']:
        return 'β Centauri'
    elif target_upper in ['GAMMA CENTAURI', 'GAMMA CEN']:
        return 'γ Centauri'
    elif target_upper in ['OMEGA CENTAURI', 'OMEGA CEN']:
        return 'ω Centauri'
    
    # Additional objects that were missing from the original list
    elif target_upper in ['ELEPHANT TRUNK', 'ELEPHANT TRUNK NEBULA', 'IC1396', 'IC 1396']:
        return 'IC 1396 (Elephant Trunk Nebula)'
    elif target_upper in ['HEART', 'HEART NEBULA', 'IC1805', 'IC 1805']:
        return 'IC 1805 (Heart Nebula)'
    elif target_upper in ['CALIFORNIA', 'CALIFORNIA NEBULA', 'NGC1499', 'NGC 1499']:
        return 'NGC 1499 (California Nebula)'
    elif target_upper in ['WESTERN VEIL', 'WESTERN VEIL NEBULA', 'NGC6960', 'NGC 6960']:
        return 'NGC 6960 (Western Veil Nebula)'
    elif target_upper in ['NORTH AMERICA', 'NORTH AMERICA NEBULA', 'NGC7000', 'NGC 7000']:
        return 'NGC 7000 (North America Nebula)'
    
    # For non-catalog targets, normalize separators and use Title Case
    import re
    # Replace all separators (underscores, hyphens, dots) with spaces
    normalized = re.sub(r'[_\s\-\.]+', ' ', target)
    # Remove extra spaces and convert to Title Case
    normalized = ' '.join(normalized.split())
    
    # Custom title case that respects apostrophes
    normalized = smart_title_case(normalized)
    
    return normalized

def normalize_telescope_name(telescope_name):
    """
    Normalize telescope name, replacing technical descriptions with 'Unknown'
    """
    if not telescope_name or telescope_name.strip() == '' or telescope_name.strip() == 'Unknown':
        return 'Unknown'
    
    telescope = telescope_name.strip()
    
    # Check for technical descriptions that should be replaced with 'Unknown'
    technical_indicators = [
        '->', '->', 'driver', 'connected', 'through', 'for telescope', 'telescope connected',
        'driver for', 'connected through', 'telescope driver', 'driver connected',
        'ACP->', 'ACP->Driver', 'Driver for', 'Connected through', 'Telescope connected',
        'TELESCOPE CONNECTED', 'DRIVER FOR', 'CONNECTED THROUGH', 'ACP->DRIVER'
    ]
    
    telescope_upper = telescope.upper()
    for indicator in technical_indicators:
        if indicator.upper() in telescope_upper:
            return 'Unknown'
    
    # Check for very long names (more than 50 characters) that might break table formatting
    if len(telescope) > 50:
        return 'Unknown'
    
    # Check for names that look like technical descriptions - but be more selective
    # Only flag obvious technical descriptions, not normal telescope names with common characters
    problematic_chars = ['->', '(', ')', '[', ']', '{', '}', '|', '\\', '/', ':', ';', '=', '*', '&', '%', '$', '#', '@', '!', '?', '<', '>']
    
    # Allow common telescope naming patterns like "AUS-2", "FSQ-106ED", etc.
    # Only flag if the name contains multiple problematic characters or looks like a technical description
    problematic_count = sum(1 for char in telescope if char in problematic_chars)
    
    # If it has too many problematic characters or looks like a technical description, mark as Unknown
    if problematic_count > 2 or any(phrase in telescope_upper for phrase in ['DRIVER', 'CONNECTED', 'THROUGH', 'FOR TELESCOPE']):
        return 'Unknown'
    
    return telescope



if __name__ == "__main__":
    main()