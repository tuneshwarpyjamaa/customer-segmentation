# Technical Documentation

## Project Architecture

### Overview

This project implements a customer segmentation system using RFM (Recency, Frequency, Monetary) analysis. The architecture is designed for scalability, maintainability, and memory efficiency.

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  Raw Data → Cleaning → RFM Calculation → Segmentation       │
│     ↓          ↓             ↓                ↓              │
│  Validation  Transform    Analysis      Classification       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Output Layer                                │
├─────────────────────────────────────────────────────────────┤
│  CSV Reports | Interactive Dashboard | Visualizations        │
└─────────────────────────────────────────────────────────────┘
```

## Module Documentation

### 1. data_loader.py

**Purpose**: Efficiently load large datasets with memory optimization.

**Key Classes**:
- `DataLoader`: Handles data loading with memory optimization

**Key Functions**:
- `load_data_optimized()`: Main function for loading data
- `optimize_dtypes()`: Reduces memory footprint by optimizing data types

**Memory Optimization Techniques**:
1. **Type Optimization**: Converts int64/float64 to smaller types (int8, int16, float32)
2. **Categorical Encoding**: Uses category dtype for low-cardinality strings
3. **Chunked Processing**: Processes large files in manageable chunks
4. **Garbage Collection**: Explicit memory cleanup after operations

**Example Usage**:
```python
from data_loader import load_data_optimized

df = load_data_optimized('data/raw/OnlineRetail.xlsx', file_format='excel')
```

### 2. data_cleaner.py

**Purpose**: Clean and preprocess transaction data.

**Key Classes**:
- `DataCleaner`: Implements data cleaning pipeline

**Data Quality Issues Addressed**:
1. Missing CustomerID values
2. Cancelled orders (InvoiceNo starting with 'C')
3. Invalid quantities (≤ 0)
4. Invalid prices (≤ 0)
5. Outliers in quantity, price, and total amount
6. Duplicate records

**Cleaning Pipeline**:
```
Input Data
    ↓
Remove Missing CustomerID
    ↓
Remove Cancelled Orders
    ↓
Remove Invalid Quantities/Prices
    ↓
Handle Outliers (IQR method)
    ↓
Create Derived Features
    ↓
Clean Output
```

**Example Usage**:
```python
from data_cleaner import clean_data

cleaned_df = clean_data(df, remove_outliers=True)
```

### 3. rfm_analysis.py

**Purpose**: Calculate RFM metrics and segment customers.

**Key Classes**:
- `RFMAnalyzer`: Core RFM calculation and segmentation logic

**RFM Metrics**:

1. **Recency (R)**: Days since last purchase
   - Calculation: `reference_date - last_purchase_date`
   - Lower is better (more recent customers)

2. **Frequency (F)**: Number of unique transactions
   - Calculation: `count(distinct InvoiceNo)`
   - Higher is better (more engaged customers)

3. **Monetary (M)**: Total spending
   - Calculation: `sum(TotalAmount)`
   - Higher is better (more valuable customers)

**Scoring System**:
- Each metric is divided into 5 quintiles (1-5)
- Scores are assigned based on quantile ranges
- Combined RFM score: concatenation of R, F, M scores (e.g., "555")

**Segmentation Logic**:

| Segment | Characteristics | R Score | F Score | M Score |
|---------|----------------|---------|---------|---------|
| Champions | Best customers | 4-5 | 4-5 | 4-5 |
| Loyal | Regular buyers | 3-5 | 3-5 | 3-5 |
| Potential Loyalist | Recent, growing | 3-5 | 1-3 | 1-3 |
| New Customers | Very recent | 4-5 | ≤2 | ≤2 |
| At Risk | Declining engagement | ≤2 | 3-5 | 3-5 |
| Lost | Churned | ≤2 | ≤2 | ≤2 |

**Example Usage**:
```python
from rfm_analysis import calculate_rfm, segment_customers

rfm = calculate_rfm(cleaned_df)
rfm_segmented = segment_customers(rfm, n_quantiles=5)
```

### 4. visualizations.py

**Purpose**: Generate insightful visualizations.

**Key Classes**:
- `RFMVisualizer`: Creates interactive and static plots

**Visualization Types**:
1. Segment distribution (pie chart)
2. Revenue by segment (bar chart)
3. 3D RFM scatter plot
4. Correlation heatmap
5. Time series trends
6. Top products analysis

**Technology Stack**:
- Plotly: Interactive visualizations
- Matplotlib/Seaborn: Static plots
- HTML export for sharing

**Example Usage**:
```python
from visualizations import create_all_visualizations

create_all_visualizations(df, rfm_segmented)
```

### 5. dashboard.py

**Purpose**: Interactive web dashboard for exploration.

**Technology**: Dash (Flask-based)

**Features**:
- Real-time filtering
- Interactive charts
- Segment comparison
- Export capabilities

**Components**:
1. Header with key metrics
2. Segment distribution visualization
3. Revenue analysis
4. 3D RFM scatter plot
5. Customer trends over time
6. Top customers table

**Running the Dashboard**:
```bash
python src/dashboard.py
# Visit: http://localhost:8050
```

## Performance Optimization

### Memory Usage

**Baseline**: ~400MB for 541k records
**Optimized**: ~150MB (62.5% reduction)

**Optimization Strategies**:
1. Data type optimization (int64 → int8/int16)
2. Categorical encoding for strings
3. float64 → float32 conversion
4. Chunked processing for large files
5. Explicit garbage collection

### Processing Speed

**Typical Performance** (on 541k records):
- Data loading: ~3 seconds
- Data cleaning: ~2 seconds
- RFM calculation: ~1 second
- Segmentation: <1 second
- Visualization generation: ~2 seconds

**Total**: ~8-10 seconds end-to-end

### Scalability Considerations

**Current Capacity**: Up to 1M records on 8GB RAM

**For Larger Datasets**:
1. Use database backend (PostgreSQL, SQLite)
2. Implement parallel processing
3. Use Dask for distributed computing
4. Stream processing for real-time updates

## API Reference

### Data Loading

```python
load_data_optimized(
    filepath: str,
    file_format: str = 'excel',
    chunk_size: int = 50000
) -> pd.DataFrame
```

### Data Cleaning

```python
clean_data(
    df: pd.DataFrame,
    remove_outliers: bool = True
) -> pd.DataFrame
```

### RFM Calculation

```python
calculate_rfm(
    df: pd.DataFrame,
    customer_col: str = 'CustomerID',
    date_col: str = 'InvoiceDate',
    amount_col: str = 'TotalAmount',
    invoice_col: str = 'InvoiceNo'
) -> pd.DataFrame
```

### Customer Segmentation

```python
segment_customers(
    rfm: pd.DataFrame,
    n_quantiles: int = 5
) -> pd.DataFrame
```

## Testing

### Test Coverage

- Unit tests for RFM calculation
- Integration tests for pipeline
- Edge case handling
- Performance benchmarks

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_rfm_analysis.py -v
```

## Deployment

### Local Deployment

```bash
# Clone repository
git clone https://github.com/yourusername/customer-segmentation-project.git

# Install dependencies
pip install -r requirements.txt

# Run analysis
python src/main.py
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

CMD ["python", "src/main.py"]
```

### Cloud Deployment (AWS)

1. **EC2 Instance**: Run dashboard on server
2. **S3 Storage**: Store large datasets
3. **Lambda**: Scheduled RFM calculations
4. **RDS**: Database backend for transactions

## Troubleshooting

### Common Issues

**Issue**: Out of memory error
**Solution**: Reduce chunk_size or use database backend

**Issue**: Slow performance
**Solution**: Enable multiprocessing or use sampling

**Issue**: Dashboard not loading
**Solution**: Check port 8050 is available, install all dependencies

## Future Enhancements

1. **Real-time Processing**: Stream processing with Kafka
2. **Machine Learning**: Predictive CLV models
3. **A/B Testing**: Campaign effectiveness tracking
4. **API Service**: RESTful API for integration
5. **Multi-language Support**: Internationalization
6. **Cloud Native**: Kubernetes deployment

## References

1. [RFM Analysis Overview](https://en.wikipedia.org/wiki/RFM_(customer_value))
2. [Pandas Optimization](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
3. [Dash Documentation](https://dash.plotly.com/)
4. [Plotly Visualization](https://plotly.com/python/)
