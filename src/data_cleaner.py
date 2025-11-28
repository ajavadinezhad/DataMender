"""Week 5: Batch Fix Engine - Vectorized data cleaning with Polars"""
import polars as pl
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from datetime import datetime


class DataCleaner:
    """Apply data quality rules to clean datasets using vectorized operations"""
    
    # Supported actions
    SUPPORTED_ACTIONS = {
        "clip_range", "fill_null", "drop_rows", "abs_value", 
        "treat_as_null", "mark_as_id", "cross_column_check"
    }
    
    @staticmethod
    def normalize_action(action: str, rule: Dict[str, Any]) -> str:
        """Normalize LLM-generated action names to supported actions"""
        action_lower = action.lower()
        
        if "reject" in action_lower or "drop" in action_lower or "remove" in action_lower:
            if "range" in action_lower or "outside" in action_lower:
                if "min" in rule or "max" in rule:
                    desc = rule.get("description", "").lower()
                    if "clip" in desc or "bound" in desc:
                        return "clip_range"
                    return "drop_rows"
                return "drop_rows"
            elif "negative" in action_lower or "non-positive" in action_lower or "nonpositive" in action_lower:
                return "drop_rows"
            else:
                return "drop_rows"
        
        elif "clip" in action_lower or "bound" in action_lower or "limit" in action_lower:
            return "clip_range"
        
        elif "fill" in action_lower or "replace" in action_lower:
            if "null" in action_lower or "missing" in action_lower:
                return "fill_null"
            return "fill_null"
        
        elif "absolute" in action_lower or "abs" in action_lower:
            return "abs_value"
        
        elif "treat" in action_lower and "null" in action_lower:
            return "treat_as_null"
        
        elif "mark" in action_lower and "id" in action_lower:
            return "mark_as_id"
        
        if action in DataCleaner.SUPPORTED_ACTIONS:
            return action
        
        return action
    
    def __init__(self, df: pl.DataFrame):
        """Initialize cleaner with a DataFrame (cloned to preserve original)"""
        self.df = df.clone()
        self.original_row_count = len(df)
        self.applied_rules: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def apply_rule(self, rule: Dict[str, Any], progress_callback=None) -> 'DataCleaner':
        """Apply a single rule to the dataset"""
        action = rule.get("action")
        column = rule.get("column")
        
        if not column or column not in self.df.columns:
            return self
        
        original_action = action
        action = self.normalize_action(action, rule)
        
        rule_copy = rule.copy()
        if action != original_action:
            rule_copy["original_action"] = original_action
            rule_copy["action"] = action
        
        rule_log = {
            "rule": rule_copy,
            "timestamp": datetime.now().isoformat(),
            "rows_before": len(self.df),
            "rows_after": None,
            "success": False,
            "error": None,
            "rows_clipped": None
        }
        
        try:
            if action == "clip_range":
                self._clip_range(column, rule)
                if "_rows_clipped" in rule:
                    rule_log["rows_clipped"] = rule["_rows_clipped"]
            elif action == "fill_null":
                self._fill_null(column, rule)
            elif action == "drop_rows":
                self._drop_rows(column, rule)
            elif action == "abs_value":
                self._abs_value(column)
            elif action == "treat_as_null":
                self._treat_as_null(column)
            elif action == "mark_as_id":
                pass
            elif action == "cross_column_check":
                self._cross_column_check(rule)
            else:
                rule_log["error"] = f"Unknown action: {action}"
                self.applied_rules.append(rule_log)
                return self
            
            rule_log["rows_after"] = len(self.df)
            rule_log["success"] = True
            self.applied_rules.append(rule_log)
            
        except Exception as e:
            rule_log["error"] = str(e)
            self.applied_rules.append(rule_log)
        
        return self
    
    def _clip_range(self, column: str, rule: Dict[str, Any]):
        """Clip numeric values to specified range"""
        min_val = rule.get("min")
        max_val = rule.get("max")
        
        if min_val is None and max_val is None:
            return
        
        col_data = self.df[column]
        
        if col_data.dtype not in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                                   pl.Float32, pl.Float64]:
            return
        
        rows_clipped = 0
        if min_val is not None:
            below_min = (col_data < min_val).sum()
            rows_clipped += below_min
        if max_val is not None:
            above_max = (col_data > max_val).sum()
            rows_clipped += above_max
        
        rule["_rows_clipped"] = int(rows_clipped)
        
        if min_val is not None and max_val is not None:
            self.df = self.df.with_columns(
                pl.col(column).clip(lower_bound=min_val, upper_bound=max_val)
            )
        elif min_val is not None:
            self.df = self.df.with_columns(
                pl.col(column).clip(lower_bound=min_val)
            )
        elif max_val is not None:
            self.df = self.df.with_columns(
                pl.col(column).clip(upper_bound=max_val)
            )
    
    def _fill_null(self, column: str, rule: Dict[str, Any]):
        """Fill null values with specified strategy"""
        strategy = rule.get("strategy", "mean")
        fill_value = rule.get("value")
        
        col_data = self.df[column]
        
        if strategy == "mean" and col_data.dtype in [pl.Float32, pl.Float64, 
                                                     pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            mean_val = col_data.mean()
            if mean_val is not None:
                self.df = self.df.with_columns(
                    pl.col(column).fill_null(mean_val)
                )
        elif strategy == "median" and col_data.dtype in [pl.Float32, pl.Float64,
                                                          pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            median_val = col_data.median()
            if median_val is not None:
                self.df = self.df.with_columns(
                    pl.col(column).fill_null(median_val)
                )
        elif strategy == "mode":
            mode_val = col_data.mode()
            if len(mode_val) > 0:
                self.df = self.df.with_columns(
                    pl.col(column).fill_null(mode_val[0])
                )
        elif strategy == "value" and fill_value is not None:
            self.df = self.df.with_columns(
                pl.col(column).fill_null(fill_value)
            )
        elif strategy == "forward_fill":
            self.df = self.df.with_columns(
                pl.col(column).forward_fill()
            )
        elif strategy == "backward_fill":
            self.df = self.df.with_columns(
                pl.col(column).backward_fill()
            )
        else:
            if col_data.dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                self.df = self.df.with_columns(
                    pl.col(column).fill_null(0)
                )
            elif col_data.dtype == pl.Utf8:
                self.df = self.df.with_columns(
                    pl.col(column).fill_null("")
                )
    
    def _drop_rows(self, column: str, rule: Dict[str, Any]):
        """Drop rows based on condition"""
        condition = rule.get("condition", "null")
        rule_type = rule.get("type", "").lower()
        description = rule.get("description", "").lower()
        
        col_data = self.df[column]
        
        if condition == "null" and "negative" in description:
            condition = "negative"
        elif condition == "null" and "non-positive" in description:
            condition = "negative"
        elif condition == "null" and "positive" in description and "non" in description:
            condition = "negative"
        elif condition == "null" and "range" in description and "outside" in description:
            if "min" in rule or "max" in rule:
                condition = "out_of_range"
        
        if condition == "null":
            self.df = self.df.filter(pl.col(column).is_not_null())
        elif condition == "zero":
            self.df = self.df.filter(pl.col(column) != 0)
        elif condition == "negative" or condition == "non_positive":
            if col_data.dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                self.df = self.df.filter(pl.col(column) > 0)
            else:
                pass
        elif condition == "out_of_range":
            min_val = rule.get("min")
            max_val = rule.get("max")
            if col_data.dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
                if min_val is not None and max_val is not None:
                    self.df = self.df.filter((pl.col(column) >= min_val) & (pl.col(column) <= max_val))
                elif min_val is not None:
                    self.df = self.df.filter(pl.col(column) >= min_val)
                elif max_val is not None:
                    self.df = self.df.filter(pl.col(column) <= max_val)
        elif condition == "empty_string":
            self.df = self.df.filter(pl.col(column) != "")
    
    def _abs_value(self, column: str):
        """Take absolute value of numeric column"""
        col_data = self.df[column]
        
        if col_data.dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            self.df = self.df.with_columns(
                pl.col(column).abs()
            )
    
    def _treat_as_null(self, column: str):
        """Convert empty strings to nulls"""
        if self.df[column].dtype == pl.Utf8:
            self.df = self.df.with_columns(
                pl.when(pl.col(column) == "")
                .then(None)
                .otherwise(pl.col(column))
                .alias(column)
            )
    
    def _cross_column_check(self, rule: Dict[str, Any]):
        """Apply cross-column validation (e.g., start_time < end_time)"""
        column1 = rule.get("column1")
        column2 = rule.get("column2")
        operator = rule.get("operator", "<")
        
        if not column1 or not column2:
            return
        if column1 not in self.df.columns or column2 not in self.df.columns:
            return
        
        if operator == "<":
            condition = pl.col(column1) < pl.col(column2)
        elif operator == ">":
            condition = pl.col(column1) > pl.col(column2)
        elif operator == "<=":
            condition = pl.col(column1) <= pl.col(column2)
        elif operator == ">=":
            condition = pl.col(column1) >= pl.col(column2)
        elif operator == "==":
            condition = pl.col(column1) == pl.col(column2)
        elif operator == "!=":
            condition = pl.col(column1) != pl.col(column2)
        else:
            return
        
        self.df = self.df.filter(
            condition | pl.col(column1).is_null() | pl.col(column2).is_null()
        )
    
    def apply_rules(self, rules: Dict[str, List[Dict[str, Any]]], 
                    chunk_size: int = None, progress_callback=None) -> 'DataCleaner':
        """Apply multiple rules organized by column (supports chunked processing)"""
        if chunk_size and len(self.df) > chunk_size:
            return self._apply_rules_chunked(rules, chunk_size)
        
        all_rules = []
        for col_name, col_rules in rules.items():
            for rule in col_rules:
                all_rules.append(rule)
        
        severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        all_rules.sort(key=lambda r: severity_order.get(r.get("severity", "info"), 3))
        
        for idx, rule in enumerate(all_rules):
            if progress_callback:
                progress_callback(idx + 1, len(all_rules), rule)
            self.apply_rule(rule, progress_callback)
        
        return self
    
    def _apply_rules_chunked(self, rules: Dict[str, List[Dict[str, Any]]], 
                            chunk_size: int) -> 'DataCleaner':
        """Apply rules in chunks for memory efficiency"""
        total_rows = len(self.df)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        cleaned_chunks = []
        chunk_cleaners = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            chunk = self.df.slice(start_idx, chunk_size)
            
            chunk_cleaner = DataCleaner(chunk)
            chunk_cleaner.apply_rules(rules, chunk_size=None)
            
            cleaned_chunks.append(chunk_cleaner.df)
            chunk_cleaners.append(chunk_cleaner)
        
        self.df = pl.concat(cleaned_chunks)
        
        all_chunk_rules = []
        for chunk_cleaner in chunk_cleaners:
            all_chunk_rules.extend(chunk_cleaner.applied_rules)
        
        seen_rules = {}
        for rule_log in all_chunk_rules:
            rule = rule_log.get("rule", {})
            rule_key = f"{rule.get('column', 'unknown')}_{rule.get('type', 'unknown')}_{rule.get('action', 'unknown')}"
            
            if rule_key not in seen_rules:
                seen_rules[rule_key] = rule_log.copy()
                seen_rules[rule_key]["rows_before"] = total_rows
                seen_rules[rule_key]["rows_after"] = len(self.df)
            else:
                if rule_log.get("success") and not seen_rules[rule_key].get("success"):
                    seen_rules[rule_key] = rule_log.copy()
                    seen_rules[rule_key]["rows_before"] = total_rows
                    seen_rules[rule_key]["rows_after"] = len(self.df)
        
        self.applied_rules = list(seen_rules.values())
        
        from datetime import datetime
        self.applied_rules.insert(0, {
            "rule": {"column": "all", "type": "chunked_processing", 
                    "description": f"Processed {num_chunks} chunks of {chunk_size:,} rows each",
                    "action": "chunked_processing"},
            "timestamp": datetime.now().isoformat(),
            "rows_before": total_rows,
            "rows_after": len(self.df),
            "success": True,
            "error": None
        })
        
        return self
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get statistics about the cleaning process"""
        end_time = time.time()
        processing_time = end_time - self.start_time
        
        rows_removed = self.original_row_count - len(self.df)
        rows_removed_pct = (rows_removed / self.original_row_count * 100) if self.original_row_count > 0 else 0
        
        successful_rules = sum(1 for r in self.applied_rules if r.get("success", False))
        failed_rules = len(self.applied_rules) - successful_rules
        
        return {
            "original_row_count": self.original_row_count,
            "cleaned_row_count": len(self.df),
            "rows_removed": rows_removed,
            "rows_removed_percentage": round(rows_removed_pct, 2),
            "rules_applied": len(self.applied_rules),
            "rules_successful": successful_rules,
            "rules_failed": failed_rules,
            "processing_time_seconds": round(processing_time, 3),
            "rows_per_second": round(len(self.df) / processing_time, 0) if processing_time > 0 else 0
        }
    
    def get_applied_rules_log(self) -> List[Dict[str, Any]]:
        """Get log of all applied rules"""
        return self.applied_rules
    
    def export_cleaned_data(self, output_path: str, format: str = "parquet", 
                           chunk_size: int = None) -> str:
        """Export cleaned data to file (supports chunked export)"""
        path = Path(output_path)
        
        if chunk_size and len(self.df) > chunk_size:
            return self._export_chunked(path, format, chunk_size)
        
        if format.lower() == "parquet":
            self.df.write_parquet(path)
        elif format.lower() == "csv":
            self.df.write_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'")
        
        return str(path)
    
    def _export_chunked(self, output_path: Path, format: str, chunk_size: int) -> str:
        """Export data in chunks for memory efficiency"""
        total_rows = len(self.df)
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            chunk = self.df.slice(start_idx, chunk_size)
            chunks.append(chunk)
        
        combined = pl.concat(chunks)
        
        if format.lower() == "parquet":
            combined.write_parquet(output_path)
        elif format.lower() == "csv":
            combined.write_csv(output_path)
        
        return str(output_path)


def clean_data(df: pl.DataFrame, rules: Dict[str, List[Dict[str, Any]]]) -> DataCleaner:
    """Convenience function to clean data with rules"""
    cleaner = DataCleaner(df)
    cleaner.apply_rules(rules)
    return cleaner

