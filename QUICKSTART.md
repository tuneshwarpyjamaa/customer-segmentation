# Quick Start Guide

Get up and running with the Customer Segmentation Project in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 2GB RAM minimum (4GB recommended)
- 500MB free disk space

## Installation Steps

### 1. Clone or Download the Project

```bash
# Option A: Clone from GitHub
git clone https://github.com/yourusername/customer-segmentation-project.git
cd customer-segmentation-project

# Option B: Download and extract the ZIP file
# Then navigate to the extracted folder
```

### 2. Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages (~100MB).

### 4. Download the Dataset

**Option A: Automated Download (Requires Kaggle API)**

```bash
# Set up Kaggle API credentials first:
# 1. Go to https://www.kaggle.com/account
# 2. Create API token (downloads kaggle.json)
# 3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Then run:
python scripts/download_data.py
```

**Option B: Manual Download**

1. Visit [Online Retail Dataset on Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail)
2. Click "Download" button
3. Extract `OnlineRetail.xlsx`
4. Place it in `data/raw/` directory

### 5. Run the Analysis

```bash
python src/main.py
```

This will:
- Load and clean the data
- Calculate RFM metrics
- Segment customers
- Generate visualizations
- Save results to `outputs/` directory

Expected runtime: ~10 seconds for 541k records

### 6. View Results

After running, you'll find:

**Reports** (in `outputs/reports/`):
- `rfm_customer_segments.csv` - Complete RFM data with segments
- `segment_summary.csv` - Summary statistics per segment
- `high_value_customers.csv` - Top 100 customers

**Visualizations** (in `outputs/figures/`):
- `segment_distribution.html` - Interactive pie chart
- `segment_revenue.html` - Revenue by segment
- `rfm_3d_scatter.html` - 3D visualization
- And more!

### 7. Launch the Dashboard (Optional)

```bash
python src/dashboard.py
```

Then open your browser to: `http://localhost:8050`

The dashboard provides:
- Interactive visualizations
- Real-time filtering
- Segment comparison
- Customer insights

Press `Ctrl+C` to stop the dashboard server.

## Quick Commands Reference

```bash
# Run main analysis
python src/main.py

# Launch dashboard
python src/dashboard.py

# Run tests
pytest tests/

# Open Jupyter notebook
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Project Structure Overview

```
customer-segmentation-project/
├── data/
│   ├── raw/              # Place OnlineRetail.xlsx here
│   └── processed/        # Cleaned data outputs
├── src/
│   ├── main.py          # Main analysis script
│   ├── dashboard.py     # Interactive dashboard
│   ├── data_loader.py   # Data loading utilities
│   ├── data_cleaner.py  # Data cleaning
│   ├── rfm_analysis.py  # RFM calculation
│   └── visualizations.py # Chart generation
├── outputs/
│   ├── reports/         # CSV reports
│   └── figures/         # Visualizations
├── notebooks/           # Jupyter notebooks
├── tests/              # Unit tests
└── scripts/            # Utility scripts
```

## Understanding the Output

### Customer Segments

1. **Champions** - Your best customers (High R, F, M)
2. **Loyal** - Consistent purchasers
3. **Potential Loyalist** - Recent customers with growth potential
4. **New Customers** - Just joined, need nurturing
5. **At Risk** - Declining engagement, need attention
6. **Lost** - Haven't purchased in a long time

### Key Metrics

- **Recency**: Days since last purchase (lower is better)
- **Frequency**: Number of purchases (higher is better)
- **Monetary**: Total spending (higher is better)

### RFM Scores

Each customer gets scores from 1-5 for R, F, and M:
- 5 = Best (top 20%)
- 1 = Worst (bottom 20%)

## Common Issues & Solutions

### Issue: "Data file not found"
**Solution**: Make sure `OnlineRetail.xlsx` is in `data/raw/` directory

### Issue: "Module not found"
**Solution**: Make sure you've installed dependencies: `pip install -r requirements.txt`

### Issue: "Out of memory"
**Solution**: The project is optimized for low memory. If still having issues:
- Close other applications
- Use a machine with more RAM
- Reduce chunk_size in data_loader.py

### Issue: "Dashboard won't load"
**Solution**: 
- Make sure port 8050 is not in use
- Check all dependencies are installed
- Try: `python src/dashboard.py --port 8051`

## Next Steps

1. **Review the Output**
   - Check `outputs/reports/` for CSV files
   - Open HTML visualizations in browser

2. **Customize the Analysis**
   - Edit parameters in `src/main.py`
   - Adjust segment definitions in `src/rfm_analysis.py`
   - Modify visualizations in `src/visualizations.py`

3. **Explore the Code**
   - Read `DOCUMENTATION.md` for technical details
   - Check `notebooks/exploratory_analysis.ipynb` for interactive exploration
   - Review `tests/` for examples

4. **Contribute**
   - See `CONTRIBUTING.md` for guidelines
   - Submit issues or pull requests on GitHub

## Getting Help

- **Documentation**: Read `DOCUMENTATION.md` for detailed information
- **Issues**: Check existing [GitHub Issues](https://github.com/yourusername/customer-segmentation-project/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/yourusername/customer-segmentation-project/discussions)

## Performance Tips

1. **For Faster Processing**:
   - Use SSD instead of HDD
   - Increase RAM if possible
   - Close unnecessary applications

2. **For Larger Datasets**:
   - Adjust `chunk_size` in data_loader.py
   - Consider using database backend
   - Use sampling for exploratory analysis

3. **For Better Visualizations**:
   - Increase figure DPI in visualizations.py
   - Customize color schemes
   - Add more chart types as needed

## Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Dataset downloaded and in correct location
- [ ] Analysis runs without errors
- [ ] Results generated in outputs/ directory
- [ ] Dashboard accessible at localhost:8050

**Congratulations! You're ready to start analyzing customer segments!** 🎉

---

**Need more help?** Open an issue or check the full documentation.
