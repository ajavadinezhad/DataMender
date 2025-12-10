"""Rule Discovery - LLM-based and heuristic rule generation"""
from typing import Dict, Any, List
import json
from src.llm_client import LLMClient


class RuleDiscovery:
    """Discover data quality rules using LLM and heuristics"""
    
    def __init__(self, llm_provider: str = "ollama"):
        """Initialize rule discovery"""

        self._ALLOWED_PROVIDERS = {"groq", "ollama", "openai", "mock"}

        if llm_provider not in self._ALLOWED_PROVIDERS:
            raise ValueError(f"Invalid LLM provider: {llm_provider}")
            
        self._llm_provider = llm_provider
        self.llm = None
        self._llm_error = None


    def _ensure_llm(self):
        if self.llm is not None:
            return

        try:
            self.llm = LLMClient(provider=self._llm_provider)
            self._llm_error = None

        except Exception as e:
            self.llm = None
            self._llm_error = f"{self._llm_provider} unavailable: {e}"

    

    def universal_checks(self, column_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply universal sanity checks (heuristics)"""
        rules = []
        col_name = column_profile["name"]
        dtype = column_profile["dtype"]
        
        if column_profile["null_percentage"] > 0:
            rules.append({
                "column": col_name,
                "type": "null_check",
                "description": f"Column has {column_profile['null_percentage']}% null values",
                "action": "fill_null" if column_profile["null_percentage"] < 50 else "drop_rows",
                "severity": "high" if column_profile["null_percentage"] > 20 else "medium",
                "source": "heuristic"
            })
        
        if "min" in column_profile:
            if column_profile.get("negative_count", 0) > 0:
                if any(word in col_name.lower() for word in ['age', 'price', 'cost', 'amount', 'distance', 'duration', 'count', 'quantity']):
                    rules.append({
                        "column": col_name,
                        "type": "negative_check",
                        "description": f"Column '{col_name}' has {column_profile['negative_count']} negative values (likely invalid)",
                        "action": "abs_value",
                        "severity": "high",
                        "source": "heuristic"
                    })
            
            min_val = column_profile["min"]
            max_val = column_profile["max"]
            
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
        
        if dtype in ("Utf8", "String"):
            rules.append({
                "column": col_name,
                "type": "empty_string_check",
                "description": "Check for empty strings",
                "action": "treat_as_null",
                "severity": "low",
                "source": "heuristic"
            })
        
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
        """Use LLM to suggest rules based on column profile"""
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
            
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join([l for l in lines if not l.startswith("```")])
            
            rules = json.loads(response)
            
            if isinstance(rules, list):
                for rule in rules:
                    rule["source"] = "llm"
                return rules
            return []
        except Exception as e:
            print(f"LLM suggestion failed: {e}")
            return []
    
    def llm_suggest_rules_batch(self, columns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Suggest rules for multiple columns in one LLM call"""
        if not self.llm:
            return {}
        
        columns_summary = []
        for col in columns:
            summary = {
                "name": col["name"],
                "type": col["dtype"],
                "null_pct": col["null_percentage"],
                "unique": col["unique_count"]
            }
            
            if "min" in col:
                summary["range"] = f"{col['min']:.2f} to {col['max']:.2f}"
            if "unique_values" in col:
                summary["values"] = col["unique_values"][:5]
            
            columns_summary.append(summary)
        
        system_prompt = """You are a data quality expert. Given multiple column profiles, suggest validation rules for columns that need them.
Focus on columns with potential issues. Return ONLY a JSON object mapping column names to arrays of rules.

IMPORTANT: Use these EXACT action names (case-sensitive):
- "clip_range" - for clipping values to min/max bounds (include "min" and "max" in rule)
- "drop_rows" - for removing rows that violate conditions
- "fill_null" - for filling null values (include "strategy": "mean"/"median"/"mode"/"value")
- "abs_value" - for taking absolute value of numeric columns
- "treat_as_null" - for converting empty strings to nulls
- "mark_as_id" - for marking unique columns as IDs (no transformation)

Example format:
{
  "column1": [
    {"type": "range_check", "description": "Values outside 0-100 range", "action": "clip_range", "min": 0, "max": 100, "severity": "high"},
    {"type": "negative_check", "description": "Negative values found", "action": "drop_rows", "condition": "negative", "severity": "medium"}
  ],
  "column2": [...]
}

Skip columns that look fine. Return {} if no issues found."""

        user_prompt = f"""Analyze these columns and suggest validation rules:

{json.dumps(columns_summary, indent=2)}

Return only JSON object mapping column names to rule arrays."""

        try:
            response = self.llm.generate(user_prompt, system_prompt)

            
            response = response.strip()
            
            if "```" in response:
                lines = response.split("\n")
                response = "\n".join([l for l in lines if not l.startswith("```")])
                response = response.strip()
            
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start >= 0 and end > start:
                response = response[start:end]
            else:
                return {}
            

            rules_dict = json.loads(response)

            
            if isinstance(rules_dict, dict):
                for col_name, rules in rules_dict.items():
                    if isinstance(rules, list):
                        for rule in rules:
                            rule["column"] = col_name
                            rule["source"] = "llm"
                return rules_dict
            return {}
        except Exception as e:

            return {}
    
    def discover_rules(self, profile: Dict[str, Any], use_llm: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """Discover all rules for dataset profile"""
        all_rules = {}
        
        for column_profile in profile["columns"]:
            col_name = column_profile["name"]
            rules = self.universal_checks(column_profile)
            all_rules[col_name] = rules
        
        if use_llm:
            self._ensure_llm()
            llm_rules_dict = self.llm_suggest_rules_batch(profile["columns"]) if self.llm else {}
            
            for col_name, llm_rules in llm_rules_dict.items():
                if col_name in all_rules:
                    all_rules[col_name].extend(llm_rules)
                else:
                    all_rules[col_name] = llm_rules
        
        return all_rules

