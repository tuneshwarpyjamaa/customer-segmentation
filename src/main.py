"""
Main Execution Script for Customer Segmentation Project

Orchestrates the entire analysis pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from data_loader import load_data_optimized, get_data_info
from data_cleaner import clean_data, get_data_quality_report
from rfm_analysis import (calculate_rfm, segment_customers, get_segment_summary,
                          identify_high_value_customers, calculate_customer_lifetime_value)
from visualizations import create_all_visualizations


def print_header(text: str) -> None:
    """Print formatted header."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def print_section(text: str) -> None:
    """Print formatted section header."""
    print("\n" + "-"*80)
    print(text)
    print("-"*80 + "\n")


def save_results(rfm_segmented: pd.DataFrame, segment_summary: pd.DataFrame,
                high_value_customers: pd.DataFrame, output_dir: Path) -> None:
    """
    Save analysis results to files.
    
    Args:
        rfm_segmented: RFM data with segments
        segment_summary: Segment summary statistics
        high_value_customers: Top customers
        output_dir: Output directory
    """
    print_section("SAVING RESULTS")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save RFM data
    rfm_output = output_dir / "rfm_customer_segments.csv"
    rfm_segmented.to_csv(rfm_output, index=False)
    print(f"✓ Saved RFM customer segments to: {rfm_output}")
    
    # Save segment summary
    summary_output = output_dir / "segment_summary.csv"
    segment_summary.to_csv(summary_output, index=False)
    print(f"✓ Saved segment summary to: {summary_output}")
    
    # Save high-value customers
    hvc_output = output_dir / "high_value_customers.csv"
    high_value_customers.to_csv(hvc_output, index=False)
    print(f"✓ Saved high-value customers to: {hvc_output}")
    
    print(f"\n📁 All results saved to: {output_dir}/")


def generate_insights(rfm_segmented: pd.DataFrame, segment_summary: pd.DataFrame) -> None:
    """
    Generate and print key business insights.
    
    Args:
        rfm_segmented: RFM data with segments
        segment_summary: Segment summary statistics
    """
    print_section("KEY BUSINESS INSIGHTS")
    
    # Top segment by customer count
    top_segment = segment_summary.loc[segment_summary['Customer_Count'].idxmax()]
    print(f"🎯 Largest Segment: {top_segment['Segment']}")
    print(f"   - Customers: {top_segment['Customer_Count']:,} ({top_segment['Customer_Percentage']:.1f}%)")
    print(f"   - Avg. Monetary Value: ${top_segment['Avg_Monetary']:,.2f}\n")
    
    # Top segment by revenue
    top_revenue_segment = segment_summary.loc[segment_summary['Total_Revenue'].idxmax()]
    print(f"💰 Highest Revenue Segment: {top_revenue_segment['Segment']}")
    print(f"   - Total Revenue: ${top_revenue_segment['Total_Revenue']:,.2f} ({top_revenue_segment['Revenue_Percentage']:.1f}%)")
    print(f"   - Customers: {top_revenue_segment['Customer_Count']:,}\n")
    
    # Champions analysis
    champions = rfm_segmented[rfm_segmented['Segment'] == 'Champions']
    if len(champions) > 0:
        print(f"🏆 Champions (Best Customers):")
        print(f"   - Count: {len(champions):,} ({len(champions)/len(rfm_segmented)*100:.1f}%)")
        print(f"   - Total Revenue: ${champions['Monetary'].sum():,.2f}")
        print(f"   - Avg. Purchase Frequency: {champions['Frequency'].mean():.1f} orders")
        print(f"   - Avg. Days Since Last Purchase: {champions['Recency'].mean():.0f} days\n")
    
    # At Risk customers
    at_risk = rfm_segmented[rfm_segmented['Segment'] == 'At Risk']
    if len(at_risk) > 0:
        print(f"⚠️  At Risk Customers (Need Re-engagement):")
        print(f"   - Count: {len(at_risk):,} ({len(at_risk)/len(rfm_segmented)*100:.1f}%)")
        print(f"   - Potential Revenue at Stake: ${at_risk['Monetary'].sum():,.2f}")
        print(f"   - Avg. Days Since Last Purchase: {at_risk['Recency'].mean():.0f} days\n")
    
    # Lost customers
    lost = rfm_segmented[rfm_segmented['Segment'].isin(['Lost', 'Hibernating'])]
    if len(lost) > 0:
        print(f"💤 Lost/Hibernating Customers:")
        print(f"   - Count: {len(lost):,} ({len(lost)/len(rfm_segmented)*100:.1f}%)")
        print(f"   - Historical Value: ${lost['Monetary'].sum():,.2f}")
        print(f"   - Consider win-back campaigns\n")
    
    # New customers
    new_customers = rfm_segmented[rfm_segmented['Segment'] == 'New Customers']
    if len(new_customers) > 0:
        print(f"✨ New Customers (Recent Acquisitions):")
        print(f"   - Count: {len(new_customers):,} ({len(new_customers)/len(rfm_segmented)*100:.1f}%)")
        print(f"   - Nurture with onboarding campaigns\n")


def print_recommendations(segment_summary: pd.DataFrame) -> None:
    """
    Print actionable marketing recommendations.
    
    Args:
        segment_summary: Segment summary statistics
    """
    print_section("RECOMMENDED ACTIONS BY SEGMENT")
    
    recommendations = {
        'Champions': '🏆 Reward loyalty, upsell premium products, make them brand ambassadors',
        'Loyal': '💙 Engage regularly, offer member benefits, ask for reviews',
        'Potential Loyalist': '🌟 Offer membership, recommend popular products, build relationship',
        'New Customers': '✨ Provide excellent onboarding, educate about products, build engagement',
        'Promising': '📈 Offer special deals to increase purchase frequency',
        'Need Attention': '📣 Reactivate with targeted campaigns and limited-time offers',
        'About to Sleep': '⏰ Share valuable content, recommend based on past purchases',
        'At Risk': '⚠️  Win back with personalized offers, surveys, and discounts',
        'Cannot Lose': '🚨 High-priority win-back campaigns, personal outreach',
        'Hibernating': '💤 Aggressive win-back campaigns, special discounts',
        'Lost': '🔄 Consider if worth pursuing, test low-cost re-engagement'
    }
    
    for segment in segment_summary['Segment'].values:
        if segment in recommendations:
            print(f"{recommendations[segment]}")


def main():
    """Main execution function."""
    start_time = time.time()
    
    # Print welcome message
    print_header("CUSTOMER SEGMENTATION ANALYSIS")
    print("A comprehensive RFM-based customer segmentation system")
    print("Optimized for large-scale retail transaction data\n")
    
    # Define paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "raw" / "OnlineRetail.xlsx"
    processed_data_path = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "reports"
    figures_dir = project_root / "outputs" / "figures"
    
    # Check if data file exists
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print("\n📥 Please download the dataset:")
        print("   1. Visit: https://www.kaggle.com/datasets/vijayuv/onlineretail")
        print("   2. Download 'OnlineRetail.xlsx'")
        print(f"   3. Place it in: {data_path.parent}/")
        print("\n   Or run: python scripts/download_data.py")
        sys.exit(1)
    
    # Step 1: Load Data
    print_section("STEP 1: LOADING DATA")
    df = load_data_optimized(str(data_path), file_format='excel')
    get_data_info(df)
    
    # Step 2: Clean Data
    print_section("STEP 2: CLEANING DATA")
    cleaned_df = clean_data(df, remove_outliers=True)
    
    # Save cleaned data
    cleaned_data_path = processed_data_path / "cleaned_data.csv"
    processed_data_path.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(cleaned_data_path, index=False)
    print(f"✓ Saved cleaned data to: {cleaned_data_path}")
    
    # Step 3: Calculate RFM Metrics
    print_section("STEP 3: CALCULATING RFM METRICS")
    rfm = calculate_rfm(cleaned_df)
    print(f"\n✓ RFM metrics calculated for {len(rfm):,} customers")
    
    # Step 4: Segment Customers
    print_section("STEP 4: SEGMENTING CUSTOMERS")
    rfm_segmented = segment_customers(rfm, n_quantiles=5)
    
    # Calculate CLV
    rfm_segmented = calculate_customer_lifetime_value(rfm_segmented)
    
    # Step 5: Generate Summary Statistics
    print_section("STEP 5: GENERATING SUMMARY STATISTICS")
    segment_summary = get_segment_summary(rfm_segmented)
    print("\nSegment Summary:")
    print(segment_summary.to_string(index=False))
    
    # Step 6: Identify High-Value Customers
    print_section("STEP 6: IDENTIFYING HIGH-VALUE CUSTOMERS")
    high_value_customers = identify_high_value_customers(rfm_segmented, top_n=100)
    print(f"\n✓ Identified top 100 high-value customers")
    print(f"   Total value: ${high_value_customers['Monetary'].sum():,.2f}")
    print(f"   Avg. value: ${high_value_customers['Monetary'].mean():,.2f}")
    
    # Step 7: Save Results
    save_results(rfm_segmented, segment_summary, high_value_customers, output_dir)
    
    # Step 8: Generate Visualizations
    print_section("STEP 8: GENERATING VISUALIZATIONS")
    create_all_visualizations(cleaned_df, rfm_segmented, output_dir=str(figures_dir))
    
    # Step 9: Generate Insights
    generate_insights(rfm_segmented, segment_summary)
    
    # Step 10: Print Recommendations
    print_recommendations(segment_summary)
    
    # Print completion message
    elapsed_time = time.time() - start_time
    print_header("ANALYSIS COMPLETE")
    print(f"⏱️  Total execution time: {elapsed_time:.2f} seconds")
    print(f"📊 Results saved to: {output_dir}/")
    print(f"📈 Visualizations saved to: {figures_dir}/")
    print("\n🚀 Next steps:")
    print("   1. Review the generated reports and visualizations")
    print("   2. Run the interactive dashboard: python src/dashboard.py")
    print("   3. Explore the Jupyter notebook: notebooks/exploratory_analysis.ipynb")
    print("\n✨ Happy analyzing!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
