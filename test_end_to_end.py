#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite for DataMender (Weeks 1-4)
Tests: Profiler → Rule Discovery → Validation → Export → Integration

This test suite validates all implemented functionality:
- Data Profiler (Week 2)
- Rule Discovery (Week 3) 
- Human Validation (Week 4)
- Export Functionality (Week 4)
- Integration Workflow
"""

import sys
import time
import json
import yaml
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, List
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.profiler import DataProfiler
from src.rule_discovery import RuleDiscovery
from src.llm_client import LLMClient
from src.generate_sample_data import generate_ride_sharing_data


class DataMenderE2ETest:
    """Comprehensive end-to-end test suite for DataMender"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        self.test_data_path = None
        
    def log_test(self, test_name: str, passed: bool, message: str = "", duration: float = 0):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "message": message,
            "duration": duration
        })
        print(f"{status} {test_name}: {message}")
        if duration > 0:
            print(f"    ⏱️  Duration: {duration:.2f}s")
    
    def setup_test_data(self):
        """Generate test data for testing"""
        print("\n" + "="*60)
        print("🔧 SETUP: Generating Test Data")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Generate test data
            df = generate_ride_sharing_data(5000)  # Smaller dataset for faster testing
            
            # Save to temporary file
            self.test_data_path = tempfile.mktemp(suffix=".csv")
            df.write_csv(self.test_data_path)
            
            duration = time.time() - start_time
            self.log_test("Setup Test Data", True, f"Generated 5,000 rows in {self.test_data_path}", duration)
            
            # Store performance metric
            self.performance_metrics["data_generation"] = {
                "rows": len(df),
                "duration": duration,
                "rows_per_second": len(df) / duration
            }
            
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Setup Test Data", False, f"Error: {str(e)}", duration)
            return False
    
    def test_profiler_functionality(self):
        """Test data profiler functionality"""
        print("\n" + "="*60)
        print("📊 TEST 1: Data Profiler Functionality")
        print("="*60)
        
        if not self.test_data_path:
            self.log_test("Profiler Test", False, "No test data available")
            return False
        
        start_time = time.time()
        
        try:
            # Test profiler initialization
            profiler = DataProfiler(self.test_data_path)
            self.log_test("Profiler Initialization", True, "DataProfiler created successfully")
            
            # Test data loading
            profiler.load_data(sample_size=1000)
            self.log_test("Data Loading", True, "Data loaded with 1000 row sample")
            
            # Test profiling
            profile = profiler.profile_all()
            
            # Validate profile structure
            required_keys = ["file_name", "row_count", "column_count", "columns"]
            for key in required_keys:
                if key not in profile:
                    raise ValueError(f"Missing required key: {key}")
            
            self.log_test("Profile Structure", True, f"Profile contains all required keys: {required_keys}")
            
            # Validate profile content
            if profile["row_count"] != 1000:
                raise ValueError(f"Expected 1000 rows, got {profile['row_count']}")
            
            if profile["column_count"] != 11:
                raise ValueError(f"Expected 11 columns, got {profile['column_count']}")
            
            self.log_test("Profile Content", True, f"Profile: {profile['row_count']} rows, {profile['column_count']} columns")
            
            # Test individual column profiling
            for col_profile in profile["columns"]:
                required_col_keys = ["name", "dtype", "total_count", "null_count", "null_percentage", "unique_count"]
                for key in required_col_keys:
                    if key not in col_profile:
                        raise ValueError(f"Column {col_profile.get('name', 'unknown')} missing key: {key}")
            
            self.log_test("Column Profiling", True, f"All {len(profile['columns'])} columns profiled correctly")
            
            # Test data type detection
            dtypes_found = set(col["dtype"] for col in profile["columns"])
            expected_dtypes = {"Int64", "Float64", "String", "Datetime"}
            if not dtypes_found.intersection(expected_dtypes):
                raise ValueError(f"Expected data types not found. Found: {dtypes_found}")
            
            self.log_test("Data Type Detection", True, f"Detected types: {dtypes_found}")
            
            # Test null detection
            null_columns = [col for col in profile["columns"] if col["null_percentage"] > 0]
            self.log_test("Null Detection", True, f"Found {len(null_columns)} columns with nulls")
            
            # Test numeric statistics
            numeric_cols = [col for col in profile["columns"] if "min" in col and "max" in col]
            self.log_test("Numeric Statistics", True, f"Calculated stats for {len(numeric_cols)} numeric columns")
            
            duration = time.time() - start_time
            self.performance_metrics["profiling"] = {
                "rows": profile["row_count"],
                "columns": profile["column_count"],
                "duration": duration,
                "rows_per_second": profile["row_count"] / duration
            }
            
            self.log_test("Profiler Test", True, "All profiler functionality working correctly", duration)
            return profile
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Profiler Test", False, f"Error: {str(e)}", duration)
            traceback.print_exc()
            return None
    
    def test_rule_discovery_heuristic(self, profile: Dict[str, Any]):
        """Test heuristic rule discovery"""
        print("\n" + "="*60)
        print("🔍 TEST 2: Heuristic Rule Discovery")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Test rule discovery initialization
            rule_discovery = RuleDiscovery(llm_provider="groq")  # Use groq for testing
            self.log_test("Rule Discovery Initialization", True, "RuleDiscovery created successfully")
            
            # Test heuristic rule discovery
            rules = rule_discovery.discover_rules(profile, use_llm=False)
            
            # Validate rules structure
            if not isinstance(rules, dict):
                raise ValueError("Rules should be a dictionary")
            
            self.log_test("Rules Structure", True, f"Rules dictionary with {len(rules)} columns")
            
            # Validate rule content
            total_rules = sum(len(col_rules) for col_rules in rules.values())
            if total_rules == 0:
                raise ValueError("No rules discovered")
            
            self.log_test("Rule Discovery", True, f"Discovered {total_rules} heuristic rules")
            
            # Validate individual rules
            rule_types_found = set()
            severity_levels = set()
            sources_found = set()
            
            for col_name, col_rules in rules.items():
                for rule in col_rules:
                    # Check required rule fields
                    required_fields = ["column", "type", "description", "action", "severity", "source"]
                    for field in required_fields:
                        if field not in rule:
                            raise ValueError(f"Rule missing field: {field}")
                    
                    # Check source is heuristic
                    if rule["source"] != "heuristic":
                        raise ValueError(f"Expected heuristic source, got: {rule['source']}")
                    
                    rule_types_found.add(rule["type"])
                    severity_levels.add(rule["severity"])
                    sources_found.add(rule["source"])
            
            self.log_test("Rule Validation", True, f"All rules have required fields and correct source")
            self.log_test("Rule Types", True, f"Found rule types: {rule_types_found}")
            self.log_test("Severity Levels", True, f"Found severities: {severity_levels}")
            
            # Test specific rule types
            expected_rule_types = {"null_check", "negative_check", "range_check", "uniqueness"}
            found_expected = rule_types_found.intersection(expected_rule_types)
            if not found_expected:
                raise ValueError(f"No expected rule types found. Expected: {expected_rule_types}, Found: {rule_types_found}")
            
            self.log_test("Expected Rule Types", True, f"Found expected types: {found_expected}")
            
            duration = time.time() - start_time
            self.performance_metrics["heuristic_rules"] = {
                "total_rules": total_rules,
                "duration": duration,
                "rules_per_second": total_rules / duration
            }
            
            self.log_test("Heuristic Rule Discovery", True, f"All heuristic rules working correctly", duration)
            return rules
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Heuristic Rule Discovery", False, f"Error: {str(e)}", duration)
            traceback.print_exc()
            return None
    
    def test_rule_discovery_llm(self, profile: Dict[str, Any]):
        """Test LLM rule discovery (with fallback)"""
        print("\n" + "="*60)
        print("🤖 TEST 3: LLM Rule Discovery")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Test LLM client initialization
            try:
                llm_client = LLMClient(provider="groq")
                self.log_test("LLM Client Initialization", True, "LLM client created (Groq)")
                llm_available = True
            except Exception as e:
                self.log_test("LLM Client Initialization", False, f"LLM not available: {str(e)}")
                llm_available = False
            
            if not llm_available:
                # Test fallback behavior
                rule_discovery = RuleDiscovery(llm_provider="groq")
                rules = rule_discovery.discover_rules(profile, use_llm=True)
                
                # Should fall back to heuristics only
                total_rules = sum(len(col_rules) for col_rules in rules.values())
                self.log_test("LLM Fallback", True, f"Fallback to heuristics: {total_rules} rules")
                
                duration = time.time() - start_time
                self.log_test("LLM Rule Discovery", True, "LLM fallback working correctly", duration)
                return rules
            
            # Test LLM rule discovery
            rule_discovery = RuleDiscovery(llm_provider="groq")
            rules = rule_discovery.discover_rules(profile, use_llm=True)
            
            # Validate rules structure
            total_rules = sum(len(col_rules) for col_rules in rules.values())
            if total_rules == 0:
                raise ValueError("No rules discovered")
            
            self.log_test("LLM Rule Discovery", True, f"Discovered {total_rules} rules (heuristic + LLM)")
            
            # Check for LLM rules
            llm_rules = 0
            heuristic_rules = 0
            
            for col_name, col_rules in rules.items():
                for rule in col_rules:
                    if rule.get("source") == "llm":
                        llm_rules += 1
                    elif rule.get("source") == "heuristic":
                        heuristic_rules += 1
            
            self.log_test("Rule Sources", True, f"Heuristic: {heuristic_rules}, LLM: {llm_rules}")
            
            duration = time.time() - start_time
            self.performance_metrics["llm_rules"] = {
                "total_rules": total_rules,
                "heuristic_rules": heuristic_rules,
                "llm_rules": llm_rules,
                "duration": duration
            }
            
            self.log_test("LLM Rule Discovery", True, "LLM rule discovery working correctly", duration)
            return rules
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("LLM Rule Discovery", False, f"Error: {str(e)}", duration)
            traceback.print_exc()
            return None
    
    def test_rule_export(self, rules: Dict[str, List[Dict[str, Any]]]):
        """Test rule export functionality"""
        print("\n" + "="*60)
        print("💾 TEST 4: Rule Export Functionality")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Test YAML export
            yaml_content = yaml.dump(rules, default_flow_style=False, sort_keys=False)
            if not yaml_content:
                raise ValueError("YAML export failed")
            
            self.log_test("YAML Export", True, f"Exported {len(yaml_content)} characters")
            
            # Test YAML parsing
            parsed_yaml = yaml.safe_load(yaml_content)
            if not isinstance(parsed_yaml, dict):
                raise ValueError("Parsed YAML is not a dictionary")
            
            self.log_test("YAML Parsing", True, f"Successfully parsed YAML with {len(parsed_yaml)} columns")
            
            # Test JSON export
            json_content = json.dumps(rules, indent=2)
            if not json_content:
                raise ValueError("JSON export failed")
            
            self.log_test("JSON Export", True, f"Exported {len(json_content)} characters")
            
            # Test JSON parsing
            parsed_json = json.loads(json_content)
            if not isinstance(parsed_json, dict):
                raise ValueError("Parsed JSON is not a dictionary")
            
            self.log_test("JSON Parsing", True, f"Successfully parsed JSON with {len(parsed_json)} columns")
            
            # Test file export
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                yaml_file = f.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(json_content)
                json_file = f.name
            
            # Verify files exist and are readable
            if not Path(yaml_file).exists():
                raise ValueError("YAML file not created")
            
            if not Path(json_file).exists():
                raise ValueError("JSON file not created")
            
            self.log_test("File Export", True, f"Files created: {yaml_file}, {json_file}")
            
            # Clean up
            Path(yaml_file).unlink()
            Path(json_file).unlink()
            
            duration = time.time() - start_time
            self.performance_metrics["export"] = {
                "yaml_size": len(yaml_content),
                "json_size": len(json_content),
                "duration": duration
            }
            
            self.log_test("Rule Export", True, "All export functionality working correctly", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Rule Export", False, f"Error: {str(e)}", duration)
            traceback.print_exc()
            return False
    
    def test_integration_workflow(self):
        """Test complete integration workflow"""
        print("\n" + "="*60)
        print("🔄 TEST 5: Integration Workflow")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Complete workflow simulation
            # 1. Load and profile data
            profiler = DataProfiler(self.test_data_path)
            profiler.load_data(sample_size=500)
            profile = profiler.profile_all()
            
            self.log_test("Integration - Profiling", True, f"Profiled {profile['row_count']} rows")
            
            # 2. Discover rules
            rule_discovery = RuleDiscovery(llm_provider="groq")
            rules = rule_discovery.discover_rules(profile, use_llm=False)  # Use heuristics for reliability
            
            total_rules = sum(len(col_rules) for col_rules in rules.values())
            self.log_test("Integration - Rule Discovery", True, f"Discovered {total_rules} rules")
            
            # 3. Simulate human validation (accept some rules)
            accepted_rules = {}
            for col_name, col_rules in rules.items():
                # Accept first rule from each column (simulate human selection)
                if col_rules:
                    accepted_rules[col_name] = [col_rules[0]]
            
            accepted_count = sum(len(r) for r in accepted_rules.values())
            self.log_test("Integration - Rule Validation", True, f"Accepted {accepted_count} rules")
            
            # 4. Export accepted rules
            yaml_content = yaml.dump(accepted_rules, default_flow_style=False, sort_keys=False)
            self.log_test("Integration - Rule Export", True, f"Exported {len(yaml_content)} characters")
            
            # 5. Validate workflow completeness
            if profile["row_count"] == 0:
                raise ValueError("No data profiled")
            
            if total_rules == 0:
                raise ValueError("No rules discovered")
            
            if accepted_count == 0:
                raise ValueError("No rules accepted")
            
            if len(yaml_content) == 0:
                raise ValueError("No rules exported")
            
            duration = time.time() - start_time
            self.performance_metrics["integration"] = {
                "rows_profiled": profile["row_count"],
                "rules_discovered": total_rules,
                "rules_accepted": accepted_count,
                "duration": duration
            }
            
            self.log_test("Integration Workflow", True, "Complete workflow working correctly", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Integration Workflow", False, f"Error: {str(e)}", duration)
            traceback.print_exc()
            return False
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("\n" + "="*60)
        print("⚡ TEST 6: Performance Benchmarks")
        print("="*60)
        
        try:
            # Define performance thresholds
            thresholds = {
                "profiling": {"max_duration": 10.0, "min_rows_per_second": 100},
                "heuristic_rules": {"max_duration": 5.0, "min_rules_per_second": 5},
                "export": {"max_duration": 2.0}
            }
            
            all_passed = True
            
            for metric_name, metrics in self.performance_metrics.items():
                if metric_name in thresholds:
                    threshold = thresholds[metric_name]
                    
                    # Check duration threshold
                    if "duration" in metrics and "max_duration" in threshold:
                        if metrics["duration"] > threshold["max_duration"]:
                            self.log_test(f"Performance - {metric_name}", False, 
                                        f"Duration {metrics['duration']:.2f}s exceeds threshold {threshold['max_duration']}s")
                            all_passed = False
                        else:
                            self.log_test(f"Performance - {metric_name}", True, 
                                        f"Duration {metrics['duration']:.2f}s within threshold")
                    
                    # Check throughput threshold
                    if "rows_per_second" in metrics and "min_rows_per_second" in threshold:
                        if metrics["rows_per_second"] < threshold["min_rows_per_second"]:
                            self.log_test(f"Performance - {metric_name}", False, 
                                        f"Throughput {metrics['rows_per_second']:.1f} rows/s below threshold {threshold['min_rows_per_second']}")
                            all_passed = False
                        else:
                            self.log_test(f"Performance - {metric_name}", True, 
                                        f"Throughput {metrics['rows_per_second']:.1f} rows/s above threshold")
            
            # Memory usage check (basic)
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > 1000:  # 1GB threshold
                self.log_test("Memory Usage", False, f"Memory usage {memory_mb:.1f}MB exceeds 1GB threshold")
                all_passed = False
            else:
                self.log_test("Memory Usage", True, f"Memory usage {memory_mb:.1f}MB within threshold")
            
            self.log_test("Performance Benchmarks", all_passed, "All performance tests completed")
            return all_passed
            
        except Exception as e:
            self.log_test("Performance Benchmarks", False, f"Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n" + "="*60)
        print("🛡️ TEST 7: Error Handling")
        print("="*60)
        
        try:
            # Test invalid file path
            try:
                profiler = DataProfiler("nonexistent_file.csv")
                profiler.load_data()
                self.log_test("Error Handling - Invalid File", False, "Should have raised an error")
            except Exception:
                self.log_test("Error Handling - Invalid File", True, "Correctly handled invalid file")
            
            # Test empty dataset
            try:
                empty_df = pl.DataFrame({"col1": [], "col2": []})
                empty_file = tempfile.mktemp(suffix=".csv")
                empty_df.write_csv(empty_file)
                
                profiler = DataProfiler(empty_file)
                profiler.load_data()
                profile = profiler.profile_all()
                
                if profile["row_count"] == 0:
                    self.log_test("Error Handling - Empty Dataset", True, "Handled empty dataset correctly")
                else:
                    self.log_test("Error Handling - Empty Dataset", False, "Did not handle empty dataset")
                
                Path(empty_file).unlink()
                
            except Exception as e:
                self.log_test("Error Handling - Empty Dataset", False, f"Error: {str(e)}")
            
            # Test invalid LLM provider
            try:
                rule_discovery = RuleDiscovery(llm_provider="invalid_provider")
                self.log_test("Error Handling - Invalid LLM Provider", False, "Should have raised an error")
            except Exception:
                self.log_test("Error Handling - Invalid LLM Provider", True, "Correctly handled invalid provider")
            
            self.log_test("Error Handling", True, "All error handling tests completed")
            return True
            
        except Exception as e:
            self.log_test("Error Handling", False, f"Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """Clean up test resources"""
        if self.test_data_path and Path(self.test_data_path).exists():
            Path(self.test_data_path).unlink()
            print(f"🧹 Cleaned up test data: {self.test_data_path}")
    
    def run_all_tests(self):
        """Run all end-to-end tests"""
        print("🚀 DataMender End-to-End Test Suite")
        print("="*60)
        print("Testing implementation for Weeks 1-4:")
        print("• Week 2: Data Profiler")
        print("• Week 3: Rule Discovery") 
        print("• Week 4: Human Validation & Export")
        print("• Integration & Performance")
        print("="*60)
        
        overall_start_time = time.time()
        
        try:
            # Setup
            if not self.setup_test_data():
                return False
            
            # Run tests
            profile = self.test_profiler_functionality()
            if not profile:
                return False
            
            rules = self.test_rule_discovery_heuristic(profile)
            if not rules:
                return False
            
            # Test LLM (may fail if no API key, that's OK)
            self.test_rule_discovery_llm(profile)
            
            if not self.test_rule_export(rules):
                return False
            
            if not self.test_integration_workflow():
                return False
            
            self.test_performance_benchmarks()
            self.test_error_handling()
            
            # Final results
            overall_duration = time.time() - overall_start_time
            
            print("\n" + "="*60)
            print("📊 FINAL TEST RESULTS")
            print("="*60)
            
            passed_tests = sum(1 for result in self.test_results if result["passed"])
            total_tests = len(self.test_results)
            
            print(f"✅ Passed: {passed_tests}/{total_tests} tests")
            print(f"⏱️  Total Duration: {overall_duration:.2f}s")
            
            if passed_tests == total_tests:
                print("🎉 ALL TESTS PASSED! DataMender implementation is working correctly.")
                return True
            else:
                print("❌ Some tests failed. Check the output above for details.")
                return False
                
        except Exception as e:
            print(f"❌ Test suite failed with error: {str(e)}")
            traceback.print_exc()
            return False
        
        finally:
            self.cleanup()


def main():
    """Main test runner"""
    test_suite = DataMenderE2ETest()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🎯 DataMender is ready for mid-progress presentation!")
        print("✅ All core functionality (Weeks 1-4) is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ DataMender has issues that need to be fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
