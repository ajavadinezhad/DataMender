"""Week 3: Rule Discovery - LLM-based and heuristic rule generation"""
from typing import Dict, Any, List
import json
from src.llm_client import LLMClient


class RuleDiscovery:
    """Discover data quality rules using LLM and heuristics"""
    
    def __init__(self, llm_provider: str = "ollama"):
        """
        Initialize rule discovery
        
        Args:
            llm_provider: "openai" or "ollama"
        """
        self.llm = LLMClient(provider=llm_provider)
    
    def universal_checks(self, column_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply universal sanity checks (heuristics)
        
        Args:
            column_profile: Column profile from profiler
            
        Returns:
            List of suggested rules
        """
        rules = []
        col_name = column_profile["name"]
        dtype = column_profile["dtype"]
        
        # Null check
        if column_profile["null_percentage"] > 0:
            rules.append({
                "column": col_name,
                "type": "null_check",
                "description": f"Column has {column_profile['null_percentage']}% null values",
                "action": "fill_null" if column_profile["null_percentage"] < 50 else "drop_rows",
                "severity": "high" if column_profile["null_percentage"] > 20 else "medium",
                "source": "heuristic"
            })
        
        # Numeric checks
        if "min" in column_profile:
            # Negative values check
            if column_profile.get("negative_count", 0) > 0:
                # Check if column name suggests positive values
                if any(word in col_name.lower() for word in ['age', 'price', 'cost', 'amount', 'distance', 'duration', 'count', 'quantity']):
                    rules.append({
                        "column": col_name,
                        "type": "negative_check",
                        "description": f"Column '{col_name}' has {column_profile['negative_count']} negative values (likely invalid)",
                        "action": "abs_value",
                        "severity": "high",
                        "source": "heuristic"
                    })
            
            # Range checks
            min_val = column_profile["min"]
            max_val = column_profile["max"]
            
            # Age-like columns
            if 'age' in col_name.lower():
                if min_val < 0 or max_val > 150:
                    rules.append({
                        "column": col_name,
                        "type": "range_check",
                        "description": f"Age values outside reasonable range (0-150)",
                        "action": "clip_range",
                        "min": 0,
                        "max": 150,
                        "severity": "high",
                        "source": "heuristic"
                    })
            
            # Percentage-like columns
            if any(word in col_name.lower() for word in ['percent', 'rate', 'ratio']):
                if min_val < 0 or max_val > 100:
                    rules.append({
                        "column": col_name,
                        "type": "range_check",
                        "description": f"Percentage values outside 0-100 range",
                        "action": "clip_range",
                        "min": 0,
                        "max": 100,
                        "severity": "high",
                        "source": "heuristic"
                    })
        
        # String checks
        if dtype == "Utf8":
            # Empty string check
            rules.append({
                "column": col_name,
                "type": "empty_string_check",
                "description": "Check for empty strings",
                "action": "treat_as_null",
                "severity": "low",
                "source": "heuristic"
            })
        
        # Uniqueness check
        if column_profile["unique_count"] == column_profile["total_count"]:
            rules.append({
                "column": col_name,
                "type": "uniqueness",
                "description": f"Column appears to be unique (potential ID)",
                "action": "mark_as_id",
                "severity": "info",
                "source": "heuristic"
            })
        
        return rules
    
    def llm_suggest_rules(self, column_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM to suggest rules based on column profile
        
        Args:
            column_profile: Column profile from profiler
            
        Returns:
            List of LLM-suggested rules
        """
        system_prompt = """You are a data quality expert. Given a column profile, suggest data validation rules.
Return rules as a JSON array with this structure:
[
  {
    "column": "column_name",
    "type": "rule_type",
    "description": "what's wrong",
    "action": "suggested_fix",
    "severity": "high|medium|low"
  }
]

Focus on:
- Valid ranges for numeric data
- Monotonicity (e.g., timestamps should increase)
- Cross-column relationships
- Business logic constraints

Only return the JSON array, no other text."""

        user_prompt = f"""Column Profile:
{json.dumps(column_profile, indent=2)}

Suggest data quality rules for this column. Consider:
- Column name meaning
- Data type
- Statistical distribution
- Common domain constraints

Return only valid JSON array."""

        try:
            response = self.llm.generate(user_prompt, system_prompt)
            
            # Extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Remove code blocks
                lines = response.split("\n")
                response = "\n".join([l for l in lines if not l.startswith("```")])
            
            rules = json.loads(response)
            
            # Mark all LLM rules with source
            if isinstance(rules, list):
                for rule in rules:
                    rule["source"] = "llm"
                return rules
            return []
        except Exception as e:
            print(f"LLM suggestion failed: {e}")
            return []
    
    def llm_suggest_rules_batch(self, columns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Suggest rules for multiple columns in one LLM call (much faster!)
        
        Args:
            columns: List of column profiles
            
        Returns:
            Dictionary mapping column names to lists of rules
        """
        if not self.llm:
            return {}
        
        # Build compact summary of all columns
        columns_summary = []
        for col in columns:
            summary = {
                "name": col["name"],
                "type": col["dtype"],
                "null_pct": col["null_percentage"],
                "unique": col["unique_count"]
            }
            
            # Add type-specific info
            if "min" in col:
                summary["range"] = f"{col['min']:.2f} to {col['max']:.2f}"
            if "unique_values" in col:
                summary["values"] = col["unique_values"][:5]
            
            columns_summary.append(summary)
        
        system_prompt = """You are a data quality expert. Given multiple column profiles, suggest validation rules for columns that need them.
Focus on columns with potential issues. Return ONLY a JSON object mapping column names to arrays of rules.

Example format:
{
  "column1": [
    {"type": "range_check", "description": "...", "action": "...", "severity": "high"},
    {"type": "format_check", "description": "...", "action": "...", "severity": "medium"}
  ],
  "column2": [...]
}

Skip columns that look fine. Return {} if no issues found."""

        user_prompt = f"""Analyze these columns and suggest validation rules:

{json.dumps(columns_summary, indent=2)}

Return only JSON object mapping column names to rule arrays."""

        try:
            response = self.llm.generate(user_prompt, system_prompt)
            print(f"[DEBUG] LLM Response (first 500 chars): {response[:500]}")
            
            # Extract JSON from response - be robust about it
            response = response.strip()
            
            # Remove markdown code blocks
            if "```" in response:
                lines = response.split("\n")
                response = "\n".join([l for l in lines if not l.startswith("```")])
                response = response.strip()
            
            # Find JSON object boundaries (first { to last })
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start >= 0 and end > start:
                response = response[start:end]
            else:
                print(f"[ERROR] No JSON object found in response")
                return {}
            
            print(f"[DEBUG] Cleaned response (first 500 chars): {response[:500]}")
            rules_dict = json.loads(response)
            print(f"[DEBUG] Parsed {len(rules_dict)} columns with rules")
            
            # Mark all LLM rules with source
            if isinstance(rules_dict, dict):
                for col_name, rules in rules_dict.items():
                    if isinstance(rules, list):
                        for rule in rules:
                            rule["column"] = col_name  # Ensure column is set
                            rule["source"] = "llm"
                return rules_dict
            return {}
        except Exception as e:
            print(f"[ERROR] Batch LLM suggestion failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def discover_rules(self, profile: Dict[str, Any], use_llm: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover all rules for dataset profile
        
        Args:
            profile: Full dataset profile
            use_llm: Whether to use LLM for suggestions
            
        Returns:
            Dictionary mapping column names to lists of rules
        """
        all_rules = {}
        
        # First pass: heuristic rules for all columns
        for column_profile in profile["columns"]:
            col_name = column_profile["name"]
            rules = self.universal_checks(column_profile)
            all_rules[col_name] = rules
        
        # Second pass: batch LLM suggestions (ONE call for all columns!)
        if use_llm:
            llm_rules_dict = self.llm_suggest_rules_batch(profile["columns"])
            
            # Merge LLM rules into existing rules
            for col_name, llm_rules in llm_rules_dict.items():
                if col_name in all_rules:
                    all_rules[col_name].extend(llm_rules)
                else:
                    all_rules[col_name] = llm_rules
        
        return all_rules

