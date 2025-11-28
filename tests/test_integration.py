#!/usr/bin/env python3
"""
Integration Tests for DataMender
Tests complete workflows: Profile → Rules → Clean → Metrics → Export
"""

import sys
import time
from pathlib import Path
import polars as pl

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.profiler import DataProfiler
from src.rule_discovery import RuleDiscovery
from src.data_cleaner import DataCleaner
from src.metrics import CleaningMetrics
from src.generate_sample_data import generate_ride_sharing_data


class IntegrationTestSuite:
    """Integration tests for complete workflows"""
    
    def __init__(self):
        self.test_results = []
        self.test_data_path = None
        
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({"test": test_name, "passed": passed, "message": message})
        print(f"{status} {test_name}: {message}")
    
    def test_complete_workflow(self):
        """Test complete workflow: Profile → Rules → Clean → Metrics"""
        print("\n" + "="*60)
        print("🔄 TEST: Complete Workflow")
        print("="*60)
        
        try:
            # Generate test data
            df = generate_ride_sharing_data(1000)
            import tempfile
            temp_path = tempfile.mktemp(suffix=".csv")
            df.write_csv(temp_path)
            
            # Step 1: Profile
            profiler = DataProfiler(temp_path)
            profiler.load_data()
            profile = profiler.profile_all()
            assert profile["row_count"] > 0, "Should have rows"
            
            # Step 2: Discover rules
            rule_discovery = RuleDiscovery(llm_provider="groq")
            rules = rule_discovery.discover_rules(profile, use_llm=False)
            assert len(rules) > 0, "Should discover rules"
            
            # Step 3: Clean
            cleaner = DataCleaner(profiler.df)
            cleaner.apply_rules(rules)
            stats = cleaner.get_cleaning_stats()
            assert stats["rules_applied"] > 0, "Should apply rules"
            
            # Step 4: Re-profile
            cleaned_profiler = DataProfiler("")
            cleaned_profiler.df = cleaner.df
            cleaned_profile = cleaned_profiler.profile_all()
            
            # Step 5: Metrics
            metrics_calc = CleaningMetrics(profile, cleaned_profile, stats)
            metrics = metrics_calc.calculate_metrics()
            assert "summary" in metrics, "Should have summary metrics"
            
            # Cleanup
            Path(temp_path).unlink()
            
            self.log_test("Complete Workflow", True, "All steps executed successfully")
            return True
            
        except Exception as e:
            self.log_test("Complete Workflow", False, f"Error: {str(e)}")
            return False
    
    def test_streamlit_workflow_simulation(self):
        """Test workflow as it would run in Streamlit UI"""
        print("\n" + "="*60)
        print("🖥️  TEST: Streamlit Workflow Simulation")
        print("="*60)
        
        try:
            # Simulate UI workflow
            df = generate_ride_sharing_data(500)
            import tempfile
            temp_path = tempfile.mktemp(suffix=".csv")
            df.write_csv(temp_path)
            
            # UI Step 1: Load and profile
            profiler = DataProfiler(temp_path)
            profiler.load_data(sample_size=500)
            profile = profiler.profile_all()
            
            # UI Step 2: Discover rules
            rule_discovery = RuleDiscovery(llm_provider="groq")
            rules = rule_discovery.discover_rules(profile, use_llm=False)
            
            # UI Step 3: Accept some rules (simulate user selection)
            accepted_rules = {}
            for col_name, col_rules in rules.items():
                # Accept high/medium severity rules
                accepted = [r for r in col_rules if r.get("severity") in ["high", "medium"]]
                if accepted:
                    accepted_rules[col_name] = accepted
            
            # UI Step 4: Clean data
            cleaner = DataCleaner(profiler.df)
            cleaner.apply_rules(accepted_rules)
            stats = cleaner.get_cleaning_stats()
            
            # UI Step 5: Calculate metrics
            cleaned_profiler = DataProfiler("")
            cleaned_profiler.df = cleaner.df
            cleaned_profile = cleaned_profiler.profile_all()
            
            metrics = CleaningMetrics(profile, cleaned_profile, stats)
            results = metrics.calculate_metrics()
            
            # UI Step 6: Export
            import tempfile
            export_path = tempfile.mktemp(suffix=".parquet")
            cleaner.export_cleaned_data(export_path, "parquet")
            assert Path(export_path).exists(), "Export file should exist"
            Path(export_path).unlink()
            Path(temp_path).unlink()
            
            self.log_test("Streamlit Workflow", True, "UI workflow simulation successful")
            return True
            
        except Exception as e:
            self.log_test("Streamlit Workflow", False, f"Error: {str(e)}")
            return False
    
    def test_incremental_cleaning(self):
        """Test incremental cleaning - applying rules on already cleaned data"""
        print("\n" + "="*60)
        print("🔄 TEST: Incremental Cleaning")
        print("="*60)
        
        try:
            # Generate test data with issues
            df = generate_ride_sharing_data(1000)
            
            # Step 1: Initial cleaning - remove negatives
            rules1 = {
                "driver_age": [
                    {"column": "driver_age", "type": "negative_check", "action": "drop_rows", "condition": "negative"}
                ]
            }
            
            cleaner1 = DataCleaner(df)
            cleaner1.apply_rules(rules1)
            
            initial_row_count = len(cleaner1.df)
            assert initial_row_count < len(df), "Should remove some rows"
            
            # Step 2: Incremental cleaning - apply additional rules on cleaned data
            rules2 = {
                "passenger_age": [
                    {"column": "passenger_age", "type": "negative_check", "action": "drop_rows", "condition": "negative"}
                ]
            }
            
            # Create new cleaner from already cleaned data
            cleaner2 = DataCleaner(cleaner1.df)
            cleaner2.apply_rules(rules2)
            
            final_row_count = len(cleaner2.df)
            assert final_row_count <= initial_row_count, "Should remove more rows"
            
            # Verify previous transformations are preserved
            # All driver_ages should still be non-negative (from first cleaning)
            numeric_cols = [col for col, dtype in zip(cleaner2.df.columns, cleaner2.df.dtypes) 
                          if dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]]
            
            if "driver_age" in numeric_cols:
                assert (cleaner2.df["driver_age"] > 0).all() or cleaner2.df["driver_age"].null_count() > 0, \
                    "Previous cleaning should be preserved"
            
            self.log_test("Incremental Cleaning", True, 
                         f"Cleaned from {len(df)} -> {initial_row_count} -> {final_row_count} rows")
            return True
            
        except Exception as e:
            self.log_test("Incremental Cleaning", False, f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_cumulative_statistics(self):
        """Test cumulative statistics tracking across multiple cleaning passes"""
        print("\n" + "="*60)
        print("📊 TEST: Cumulative Statistics")
        print("="*60)
        
        try:
            # Generate test data
            df = generate_ride_sharing_data(1000)
            original_row_count = len(df)
            
            # Simulate cumulative statistics tracking
            cumulative_rows_removed = 0
            cumulative_nulls_filled = 0
            cumulative_rows_modified = 0
            
            # First cleaning pass
            rules1 = {
                "driver_age": [
                    {"column": "driver_age", "type": "negative_check", "action": "drop_rows", "condition": "negative"}
                ]
            }
            
            cleaner1 = DataCleaner(df)
            cleaner1.apply_rules(rules1)
            stats1 = cleaner1.get_cleaning_stats()
            
            cumulative_rows_removed += stats1["rows_removed"]
            initial_removed = stats1["rows_removed"]
            
            # Second cleaning pass (incremental)
            rules2 = {
                "passenger_age": [
                    {"column": "passenger_age", "type": "negative_check", "action": "drop_rows", "condition": "negative"}
                ]
            }
            
            cleaner2 = DataCleaner(cleaner1.df)
            cleaner2.apply_rules(rules2)
            stats2 = cleaner2.get_cleaning_stats()
            
            cumulative_rows_removed += stats2["rows_removed"]
            
            # Verify cumulative statistics
            total_rows_removed = original_row_count - len(cleaner2.df)
            assert cumulative_rows_removed == total_rows_removed, \
                f"Cumulative should match total: {cumulative_rows_removed} == {total_rows_removed}"
            
            assert cumulative_rows_removed >= initial_removed, \
                "Cumulative should increase across passes"
            
            self.log_test("Cumulative Statistics", True, 
                         f"Cumulative rows removed: {cumulative_rows_removed}, Total removed: {total_rows_removed}")
            return True
            
        except Exception as e:
            self.log_test("Cumulative Statistics", False, f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 DataMender Integration Test Suite")
        print("="*60)
        
        self.test_complete_workflow()
        self.test_streamlit_workflow_simulation()
        self.test_incremental_cleaning()
        self.test_cumulative_statistics()
        
        # Summary
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        
        print("\n" + "="*60)
        print(f"📊 Results: {passed}/{total} tests passed")
        print("="*60)
        
        return passed == total


if __name__ == "__main__":
    suite = IntegrationTestSuite()
    success = suite.run_all_tests()
    sys.exit(0 if success else 1)

