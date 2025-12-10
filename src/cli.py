"""
CLI Script for DataMender
Demonstrates the complete DataMender workflow from profiling to cleaning
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.profiler import DataProfiler
from src.rule_discovery import RuleDiscovery
from src.data_cleaner import DataCleaner
from src.metrics import CleaningMetrics


def demo_workflow(data_file: str = "datasets/sample_rides_25k.csv", sample_size: int = 10000):
    """
    Complete demo workflow showing DataMender capabilities
    
    Args:
        data_file: Path to data file
        sample_size: Number of rows to sample (0 for all)
    """
    print("=" * 80)
    print("DataMender: Complete Workflow Demo")
    print("=" * 80)
    print()
    
    print("📊 Step 1: Profiling Dataset")
    print("-" * 80)
    profiler = DataProfiler(data_file)
    profiler.load_data(sample_size=sample_size if sample_size > 0 else None)
    profile = profiler.profile_all()
    
    print(f"✅ Profiled {profile['row_count']:,} rows, {profile['column_count']} columns")
    print(f"   File: {profile['file_name']}")
    print()
    
    print("   Column Summary:")
    for col in profile['columns'][:5]:
        null_pct = col.get('null_percentage', 0)
        print(f"   - {col['name']}: {col['dtype']}, {null_pct}% nulls")
    if len(profile['columns']) > 5:
        print(f"   ... and {len(profile['columns']) - 5} more columns")
    print()
    
    print("🔍 Step 2: Discovering Data Quality Rules")
    print("-" * 80)
    rule_discovery = RuleDiscovery(llm_provider="groq")
    
    try:
        rules = rule_discovery.discover_rules(profile, use_llm=True)
        print("✅ Using LLM-enhanced rule discovery")
    except:
        rules = rule_discovery.discover_rules(profile, use_llm=False)
        print("✅ Using heuristic rule discovery (LLM unavailable)")
    
    total_rules = sum(len(r) for r in rules.values())
    heuristic_count = sum(1 for rules_list in rules.values() 
                         for r in rules_list if r.get("source") == "heuristic")
    llm_count = total_rules - heuristic_count
    
    print(f"   Discovered {total_rules} rules:")
    print(f"   - {heuristic_count} heuristic rules")
    print(f"   - {llm_count} LLM-generated rules")
    print()
    
    print("   Sample Rules:")
    rule_count = 0
    for col_name, col_rules in rules.items():
        if rule_count >= 5:
            break
        for rule in col_rules[:2]:
            if rule_count >= 5:
                break
            source = "⚡" if rule.get("source") == "heuristic" else "🤖"
            print(f"   {source} {col_name}: {rule['type']} - {rule['description']}")
            rule_count += 1
    print()
    
    print("✅ Step 3: Selecting Rules to Apply")
    print("-" * 80)
    accepted_rules = {}
    for col_name, col_rules in rules.items():
        accepted = [r for r in col_rules if r.get("severity") in ["high", "medium"]]
        if accepted:
            accepted_rules[col_name] = accepted
    
    total_accepted = sum(len(r) for r in accepted_rules.values())
    print(f"   Accepted {total_accepted} rules (high/medium severity)")
    print()
    
    print("🧹 Step 4: Cleaning Data with Vectorized Operations")
    print("-" * 80)
    cleaner = DataCleaner(profiler.df)
    cleaner.apply_rules(accepted_rules)
    
    cleaning_stats = cleaner.get_cleaning_stats()
    print(f"✅ Cleaning complete!")
    print(f"   Original rows: {cleaning_stats['original_row_count']:,}")
    print(f"   Cleaned rows: {cleaning_stats['cleaned_row_count']:,}")
    print(f"   Rows removed: {cleaning_stats['rows_removed']:,} ({cleaning_stats['rows_removed_percentage']:.2f}%)")
    print(f"   Processing time: {cleaning_stats['processing_time_seconds']:.3f}s")
    print(f"   Speed: {cleaning_stats['rows_per_second']:,.0f} rows/second")
    print(f"   Rules applied: {cleaning_stats['rules_successful']}/{cleaning_stats['rules_applied']} successful")
    print()
    
    print("📊 Step 5: Re-Profiling Cleaned Data")
    print("-" * 80)
    cleaned_profiler = DataProfiler("")
    cleaned_profiler.df = cleaner.df
    cleaned_profile = cleaned_profiler.profile_all()
    
    print(f"✅ Re-profiled {cleaned_profile['row_count']:,} rows")
    print()
    
    print("📈 Step 6: Calculating Cleaning Metrics")
    print("-" * 80)
    metrics_calc = CleaningMetrics(profile, cleaned_profile, cleaning_stats)
    metrics = metrics_calc.calculate_metrics()
    
    summary = metrics["summary"]
    null_metrics = metrics["null_metrics"]
    anomaly_metrics = metrics["anomaly_metrics"]
    
    print("✅ Metrics Summary:")
    print(f"   Rows: {summary['original_rows']:,} → {summary['cleaned_rows']:,} "
          f"({summary['rows_removed_percentage']:.2f}% removed)")
    print(f"   Nulls removed: {null_metrics['nulls_removed']:,} "
          f"({null_metrics['null_reduction_percentage']:.2f}% reduction)")
    print(f"   Anomalies fixed: {anomaly_metrics['anomalies_fixed']}")
    print()
    
    print("💾 Step 7: Exporting Cleaned Data")
    print("-" * 80)
    input_path = Path(data_file)
    input_ext = input_path.suffix.lower()
    if input_ext == ".parquet":
        output_ext = "parquet"
        output_file = input_path.stem + "_cleaned.parquet"
    else:
        output_ext = "csv"
        output_file = input_path.stem + "_cleaned.csv"
    
    cleaner.export_cleaned_data(output_file, output_ext)
    file_size = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"✅ Exported to {output_file} ({file_size:.2f} MB)")
    print()
    
    print("=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)
    print(f"Workflow Summary:")
    print(f"  • Profiled {profile['row_count']:,} rows in {profile['column_count']} columns")
    print(f"  • Discovered {total_rules} data quality rules")
    print(f"  • Applied {total_accepted} rules to clean data")
    print(f"  • Removed {summary['rows_removed']:,} rows ({summary['rows_removed_percentage']:.2f}%)")
    print(f"  • Fixed {null_metrics['nulls_removed']:,} null values")
    print(f"  • Processed at {cleaning_stats['rows_per_second']:,.0f} rows/second")
    print(f"  • Exported cleaned data to {output_file}")
    print()
    print("Next Steps:")
    print("  1. Review cleaned data in the exported file")
    print("  2. Use Streamlit UI for interactive workflow: streamlit run src/app.py")
    print("  3. Export rules as YAML/JSON for reuse")
    print()


if __name__ == "__main__":
    import sys
    
    data_file = "datasets/sample_rides_25k.csv"
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    
    if not Path(data_file).exists():
        print(f"❌ Error: Data file '{data_file}' not found")
        print("   Please provide a valid CSV or Parquet file path")
        print("   Or generate sample data: python src/generate_datasets.py")
        sys.exit(1)
    
    sample_size = 10000
    if len(sys.argv) > 2:
        sample_size = int(sys.argv[2])
    
    demo_workflow(data_file, sample_size)

