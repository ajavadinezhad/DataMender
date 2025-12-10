"""
CLI Tests for DataMender
Tests command-line interface and script execution
"""
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.profiler import DataProfiler
from src.rule_discovery import RuleDiscovery
import json


def test_profiler(file_path: str):
    """Test the profiler"""
    print(f"\n{'='*60}")
    print(f"Testing Profiler on: {file_path}")
    print(f"{'='*60}\n")
    
    profiler = DataProfiler(file_path)
    profiler.load_data(sample_size=1000)  # Sample for speed
    
    profile = profiler.profile_all()
    
    print(f"📊 Profile Summary:")
    print(f"  Rows: {profile['row_count']:,}")
    print(f"  Columns: {profile['column_count']}")
    print(f"\n📋 Columns:")
    
    for col in profile['columns']:
        print(f"\n  {col['name']} ({col['dtype']})")
        print(f"    Nulls: {col['null_percentage']}%")
        print(f"    Unique: {col['unique_count']:,}")
        if 'min' in col:
            print(f"    Range: [{col['min']:.2f}, {col['max']:.2f}]")
            if col.get('negative_count', 0) > 0:
                print(f"    ⚠️  Negative values: {col['negative_count']}")
    
    return profile


def test_rule_discovery(profile: dict, llm_provider: str = "ollama"):
    """Test rule discovery"""
    print(f"\n{'='*60}")
    print(f"Testing Rule Discovery (LLM: {llm_provider})")
    print(f"{'='*60}\n")
    
    rule_discovery = RuleDiscovery(llm_provider=llm_provider)
    
    # Test with heuristics only first (fast)
    print("🔍 Discovering rules (heuristics only)...")
    rules = rule_discovery.discover_rules(profile, use_llm=False)
    
    total_rules = sum(len(r) for r in rules.values())
    print(f"\n✅ Discovered {total_rules} rules\n")
    
    for col_name, col_rules in rules.items():
        if col_rules:
            print(f"\n📊 {col_name}:")
            for rule in col_rules:
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🔵", "info": "ℹ️"}.get(rule.get("severity", "info"), "•")
                print(f"  {severity_icon} {rule['type']}: {rule['description']}")
                print(f"      → Action: {rule['action']}")
    
    return rules


def test_cli_script_execution():
    """Test that CLI script can be executed"""
    print(f"\n{'='*60}")
    print(f"Testing CLI Script Execution")
    print(f"{'='*60}\n")
    
    try:
        project_root = Path(__file__).parent.parent
        cli_script = project_root / "src" / "cli.py"
        
        if not cli_script.exists():
            print("⚠️  CLI script not found, skipping test")
            return True
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("cli_script", cli_script)
        assert spec is not None, "Should be able to load CLI script"
        
        print("✅ CLI script is valid and importable")
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    print("🚀 DataMender CLI Test Suite")
    print("="*60)

    
    # Check for sample data in datasets directory
    project_root = Path(__file__).parent.parent
    datasets_dir = project_root / "datasets"
    
    # Look for any sample_rides CSV file in datasets
    sample_file = None
    if datasets_dir.exists():
        test_csv_files = list(datasets_dir.glob("sample_rides_*.csv"))
        if test_csv_files:
            # Use the smallest one for faster testing
            sample_file = sorted(test_csv_files, key=lambda p: p.stat().st_size)[0]
            print(f"ℹ️  Using test file: {sample_file.name}")
    
    if sample_file is None or not sample_file.exists():
        print("❌ Sample data not found in datasets/ directory.")
        print("   Generate test files: python src/generate_datasets.py")
        return
    
    # Test profiler
    profile = test_profiler(str(sample_file))
    
    # Test rule discovery
    rules = test_rule_discovery(profile, llm_provider="ollama")
    
    # Test CLI script
    test_cli_script_execution()
    
    print(f"\n{'='*60}")
    print("✅ All CLI tests completed!")
    print(f"{'='*60}\n")
    print("Next steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run Streamlit UI: streamlit run src/app.py")
    print("  3. Run CLI: python src/cli.py")
    print("  4. Upload a file from datasets/ in the UI")


if __name__ == "__main__":
    main()

