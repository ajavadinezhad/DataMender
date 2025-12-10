"""Re-Profiling & Metrics - Compare before/after cleaning"""
from typing import Dict, Any
from src.profiler import DataProfiler
from src.data_cleaner import DataCleaner


class CleaningMetrics:
    """Calculate metrics comparing original and cleaned data"""
    
    def __init__(self, original_profile: Dict[str, Any], cleaned_profile: Dict[str, Any], 
                 cleaning_stats: Dict[str, Any], applied_rules: list = None):
        """Initialize metrics calculator"""
        self.original_profile = original_profile
        self.cleaned_profile = cleaned_profile
        self.cleaning_stats = cleaning_stats
        self.applied_rules = applied_rules or []
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive cleaning metrics"""
        metrics = {
            "summary": self._calculate_summary(),
            "row_metrics": self._calculate_row_metrics(),
            "column_metrics": self._calculate_column_metrics(),
            "null_metrics": self._calculate_null_metrics(),
            "anomaly_metrics": self._calculate_anomaly_metrics(),
            "performance_metrics": self._calculate_performance_metrics()
        }
        
        return metrics
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate high-level summary metrics"""
        original_rows = self.original_profile["row_count"]
        cleaned_rows = self.cleaned_profile["row_count"]
        
        return {
            "original_rows": original_rows,
            "cleaned_rows": cleaned_rows,
            "rows_removed": original_rows - cleaned_rows,
            "rows_removed_percentage": round((original_rows - cleaned_rows) / original_rows * 100, 2) if original_rows > 0 else 0,
            "original_columns": self.original_profile["column_count"],
            "cleaned_columns": self.cleaned_profile["column_count"],
            "columns_removed": self.original_profile["column_count"] - self.cleaned_profile["column_count"]
        }
    
    def _calculate_row_metrics(self) -> Dict[str, Any]:
        """Calculate row-level metrics"""
        return {
            "rows_removed": self.cleaning_stats.get("rows_removed", 0),
            "rows_removed_percentage": self.cleaning_stats.get("rows_removed_percentage", 0),
            "rows_processed_per_second": self.cleaning_stats.get("rows_per_second", 0)
        }
    
    def _calculate_column_metrics(self) -> Dict[str, Any]:
        """Calculate column-level improvement metrics"""
        original_cols = {c["name"]: c for c in self.original_profile["columns"]}
        cleaned_cols = {c["name"]: c for c in self.cleaned_profile["columns"]}
        
        improvements = []
        for col_name, original_col in original_cols.items():
            if col_name not in cleaned_cols:
                continue
            
            cleaned_col = cleaned_cols[col_name]
            
            original_nulls = original_col.get("null_percentage", 0)
            cleaned_nulls = cleaned_col.get("null_percentage", 0)
            null_reduction = original_nulls - cleaned_nulls
            
            range_improvement = None
            if "min" in original_col and "min" in cleaned_col:
                original_range = original_col["max"] - original_col["min"]
                cleaned_range = cleaned_col["max"] - cleaned_col["min"]
                range_improvement = {
                    "original_range": original_range,
                    "cleaned_range": cleaned_range,
                    "range_reduction": original_range - cleaned_range
                }
            
            improvements.append({
                "column": col_name,
                "null_reduction_percentage": round(null_reduction, 2),
                "original_null_percentage": original_nulls,
                "cleaned_null_percentage": cleaned_nulls,
                "range_improvement": range_improvement
            })
        
        return {
            "columns_analyzed": len(improvements),
            "improvements": improvements
        }
    
    def _calculate_null_metrics(self) -> Dict[str, Any]:
        """Calculate null value reduction metrics"""
        original_total_nulls = sum(c.get("null_count", 0) for c in self.original_profile["columns"])
        cleaned_total_nulls = sum(c.get("null_count", 0) for c in self.cleaned_profile["columns"])
        
        original_total_cells = self.original_profile["row_count"] * self.original_profile["column_count"]
        cleaned_total_cells = self.cleaned_profile["row_count"] * self.cleaned_profile["column_count"]
        
        return {
            "original_total_nulls": original_total_nulls,
            "cleaned_total_nulls": cleaned_total_nulls,
            "nulls_removed": original_total_nulls - cleaned_total_nulls,
            "null_reduction_percentage": round((original_total_nulls - cleaned_total_nulls) / original_total_cells * 100, 2) if original_total_cells > 0 else 0,
            "columns_with_nulls_before": sum(1 for c in self.original_profile["columns"] if c.get("null_percentage", 0) > 0),
            "columns_with_nulls_after": sum(1 for c in self.cleaned_profile["columns"] if c.get("null_percentage", 0) > 0)
        }
    
    def _calculate_anomaly_metrics(self) -> Dict[str, Any]:
        """Calculate anomaly removal metrics"""
        original_cols = {c["name"]: c for c in self.original_profile["columns"]}
        cleaned_cols = {c["name"]: c for c in self.cleaned_profile["columns"]}
        
        anomalies_fixed = []
        
        for col_name, original_col in original_cols.items():
            if col_name not in cleaned_cols:
                continue
            
            cleaned_col = cleaned_cols[col_name]
            
            if "negative_count" in original_col:
                original_negatives = original_col.get("negative_count", 0)
                cleaned_negatives = cleaned_col.get("negative_count", 0)
                
                rule_applied_to_fix = False
                if self.applied_rules:
                    for rule_log in self.applied_rules:
                        rule = rule_log.get("rule", {})
                        rule_col = rule.get("column")
                        rule_action = rule.get("action")
                        rule_condition = rule.get("condition")
                        if (rule_col == col_name and 
                            rule_log.get("success", False)):
                            if rule_action == "abs_value":
                                rule_applied_to_fix = True
                                break
                            elif rule_action == "clip_range":
                                rule_min = rule.get("min")
                                if rule_min is not None and rule_min >= 0:
                                    rule_applied_to_fix = True
                                    break
                            elif rule_action == "drop_rows" and rule_condition in ["negative", "non_positive"]:
                                rule_applied_to_fix = True
                                break
                
                if rule_applied_to_fix:
                    fixed_count = (original_negatives - cleaned_negatives) if original_negatives > cleaned_negatives else 0
                    
                    anomalies_fixed.append({
                        "column": col_name,
                        "type": "negative_values",
                        "before": original_negatives,
                        "after": cleaned_negatives,
                            "fixed": fixed_count
                    })
            
            if "min" in original_col and "min" in cleaned_col:
                if (original_col["min"] < cleaned_col["min"] or 
                    original_col["max"] > cleaned_col["max"]):
                    range_rule_applied = False
                    rows_clipped = None
                    if self.applied_rules:
                        for rule_log in self.applied_rules:
                            rule = rule_log.get("rule", {})
                            if (rule.get("column") == col_name and 
                                rule.get("action") == "clip_range" and
                                rule_log.get("success", False)):
                                range_rule_applied = True
                                if rule_log.get("rows_clipped") is not None:
                                    rows_clipped = rule_log.get("rows_clipped")
                                break
                    
                    if range_rule_applied:
                        anomaly_detail = {
                            "column": col_name,
                            "type": "out_of_range",
                            "original_range": [original_col["min"], original_col["max"]],
                            "cleaned_range": [cleaned_col["min"], cleaned_col["max"]]
                        }
                        if rows_clipped is not None:
                            anomaly_detail["rows_affected"] = rows_clipped
                        anomalies_fixed.append(anomaly_detail)
        
        return {
            "anomalies_fixed": len(anomalies_fixed),
            "anomaly_details": anomalies_fixed
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        return {
            "processing_time_seconds": self.cleaning_stats.get("processing_time_seconds", 0),
            "rows_per_second": self.cleaning_stats.get("rows_per_second", 0),
            "rules_applied": self.cleaning_stats.get("rules_applied", 0),
            "rules_successful": self.cleaning_stats.get("rules_successful", 0),
            "rules_failed": self.cleaning_stats.get("rules_failed", 0)
        }
    
    def get_comparison_table(self) -> Dict[str, Any]:
        """Get before/after comparison for each column that has changed"""
        original_cols = {c["name"]: c for c in self.original_profile["columns"]}
        cleaned_cols = {c["name"]: c for c in self.cleaned_profile["columns"]}
        
        comparisons = []
        
        for col_name in set(list(original_cols.keys()) + list(cleaned_cols.keys())):
            original_col = original_cols.get(col_name, {})
            cleaned_col = cleaned_cols.get(col_name, {})
            
            has_changes = False
            
            orig_null = original_col.get("null_percentage", 0)
            clean_null = cleaned_col.get("null_percentage", 0)
            if abs(orig_null - clean_null) > 0.01:
                has_changes = True
            
            orig_unique = original_col.get("unique_count", 0)
            clean_unique = cleaned_col.get("unique_count", 0)
            if orig_unique != clean_unique:
                has_changes = True
            
            if "min" in original_col and "min" in cleaned_col:
                orig_min = original_col.get("min", 0)
                orig_max = original_col.get("max", 0)
                clean_min = cleaned_col.get("min", 0)
                clean_max = cleaned_col.get("max", 0)
                
                if abs(orig_min - clean_min) > 0.01 or abs(orig_max - clean_max) > 0.01:
                    has_changes = True
                
                orig_mean = original_col.get("mean", 0)
                clean_mean = cleaned_col.get("mean", 0)
                if abs(orig_mean - clean_mean) > 0.01:
                    has_changes = True
            
            if not has_changes:
                continue
            
            comparison = {
                "column": col_name,
                "before": {
                    "null_percentage": original_col.get("null_percentage", 0),
                    "unique_count": original_col.get("unique_count", 0),
                    "dtype": original_col.get("dtype", "unknown")
                },
                "after": {
                    "null_percentage": cleaned_col.get("null_percentage", 0),
                    "unique_count": cleaned_col.get("unique_count", 0),
                    "dtype": cleaned_col.get("dtype", "unknown")
                }
            }
            
            if "min" in original_col:
                comparison["before"]["min"] = original_col["min"]
                comparison["before"]["max"] = original_col["max"]
                comparison["before"]["mean"] = original_col.get("mean", 0)
            
            if "min" in cleaned_col:
                comparison["after"]["min"] = cleaned_col["min"]
                comparison["after"]["max"] = cleaned_col["max"]
                comparison["after"]["mean"] = cleaned_col.get("mean", 0)
            
            comparisons.append(comparison)
        
        return {
            "columns": comparisons
        }


def compare_profiles(original_profile: Dict[str, Any], cleaned_profile: Dict[str, Any],
                     cleaning_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to compare original and cleaned profiles"""
    metrics = CleaningMetrics(original_profile, cleaned_profile, cleaning_stats)
    return metrics.calculate_metrics()

