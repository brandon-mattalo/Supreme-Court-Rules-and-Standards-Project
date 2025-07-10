# ASAP Library Installation Notes

## About ASAP
ASAP (Active Sampling for Pairwise Comparisons) is a library for efficiently conducting pairwise comparison experiments using active sampling methods.

## Installation Instructions

The ASAP library is not available via pip. To install it:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gfxdisp/asap.git
   cd asap/python
   ```

2. **Copy the Python files to your project:**
   Copy `asap_cpu.py` and `asap_gpu.py` to your project's `utils/` directory.

3. **Install dependencies:**
   The ASAP library requires NumPy, which is already in our requirements.txt.

## Fallback Implementation

If ASAP is not available, the system will fall back to a simple active sampling implementation that:
- Uses random pair selection with preference for least-compared pairs
- Calculates scores using simple win ratios
- Still implements the Spearman correlation convergence check

## Usage in the Project

The ASAP integration is handled through the `utils/asap_integration.py` module, which provides:
- `ASAPSampler` class for managing active sampling
- `estimate_required_comparisons()` function for cost estimation
- Automatic fallback when ASAP is not available

## Cost Estimation

With active sampling, the estimated number of comparisons is O(n log n) instead of O(n²) for full pairwise comparison, providing significant efficiency gains for large datasets.

## Files Modified

- `utils/asap_integration.py` - ASAP wrapper and integration
- `utils/case_management.py` - Updated cost estimation
- `pages/dashboard.py` - Updated database schema
- `requirements.txt` - Added scipy dependency