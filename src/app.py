"""Week 4: Streamlit UI for Human-in-the-Loop Validation"""
import streamlit as st
import json
import yaml
from pathlib import Path
from src.profiler import DataProfiler
from src.rule_discovery import RuleDiscovery

st.set_page_config(
    page_title="DataMender", 
    page_icon="🔧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'profiler' not in st.session_state:
    st.session_state.profiler = None
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'rules' not in st.session_state:
    st.session_state.rules = None
if 'accepted_rules' not in st.session_state:
    st.session_state.accepted_rules = {}
if 'last_sample_size' not in st.session_state:
    st.session_state.last_sample_size = None
if 'last_file_name' not in st.session_state:
    st.session_state.last_file_name = None


def main():
    st.title("DataMender")
    st.markdown("*Smart data cleaning for large CSV/Parquet files*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        llm_provider = st.selectbox(
            "LLM Provider",
            ["groq", "ollama"],
            help="Groq: Cloud-based, faster\nOllama: Local, runs in Docker"
        )
        
        use_llm = st.checkbox("Use LLM for rule discovery", value=False)
        
        sample_size = st.number_input(
            "Sample size (0 = all rows)",
            min_value=0,
            value=10000,
            step=1000,
            help="Number of rows to profile (0 for all)",
            key="sample_size_input"
        )
        
        # Clear button
        if st.button("🗑️ Clear All", help="Reset and start fresh"):
            st.session_state.profiler = None
            st.session_state.profile = None
            st.session_state.rules = None
            st.session_state.accepted_rules = {}
            st.session_state.last_sample_size = None
            st.session_state.last_file_name = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Progress")
        if st.session_state.profile:
            st.success(f"Data profiled ({st.session_state.last_sample_size or 'all'} rows)")
        if st.session_state.rules:
            st.success("Rules discovered")
        if st.session_state.accepted_rules:
            st.success(f"{sum(len(r) for r in st.session_state.accepted_rules.values())} rules accepted")
    
    # Main workflow tabs
    tab1, tab2, tab3 = st.tabs(["📁 Load Data", "🔍 Review Rules", "💾 Export Rules"])
    
    with tab1:
        st.header("Step 1: Load and Profile Data")
        
        uploaded_file = st.file_uploader(
            "Upload CSV or Parquet file",
            type=["csv", "parquet"],
            help="Maximum file size: 5GB"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = Path(f"/tmp/{uploaded_file.name}")
            temp_path.write_bytes(uploaded_file.read())
            
            # Show info if file or settings changed
            if st.session_state.last_file_name and st.session_state.last_file_name != uploaded_file.name:
                st.info(f"New file: {uploaded_file.name}")
            
            # Show current configuration - use direct widget value
            display_sample = sample_size if sample_size > 0 else "all"
            
            # Show warning only if settings changed
            if st.session_state.last_sample_size and st.session_state.last_sample_size != sample_size:
                st.warning(f"Sample changed to {display_sample} rows - click 'Profile Dataset' to apply")
            else:
                st.caption(f"Ready to profile {display_sample} rows")
            
            if st.button("🚀 Profile Dataset", type="primary"):
                with st.spinner("Profiling dataset..."):
                    try:
                        # Use the direct widget value (sample_size)
                        profiler = DataProfiler(str(temp_path))
                        profiler.load_data(sample_size=sample_size if sample_size > 0 else None)
                        profile = profiler.profile_all()
                        
                        st.session_state.profiler = profiler
                        st.session_state.profile = profile
                        st.session_state.last_sample_size = sample_size if sample_size > 0 else profile['row_count']
                        st.session_state.last_file_name = uploaded_file.name
                        
                        st.success(f"Profiled {profile['row_count']} rows, {profile['column_count']} columns")
                        
                        # Discover rules
                        with st.spinner("Discovering rules..."):
                            try:
                                rule_discovery = RuleDiscovery(llm_provider=llm_provider)
                                rules = rule_discovery.discover_rules(profile, use_llm=use_llm)
                                st.session_state.rules = rules
                                
                                total_rules = sum(len(r) for r in rules.values())
                                st.success(f"Discovered {total_rules} rules")
                            except Exception as e:
                                st.error(f"LLM Error: {e}")
                                st.info("Tip: Uncheck 'Use LLM' to use heuristics only, or configure your API key in .env file")
                                # Fall back to heuristics only
                                rule_discovery = RuleDiscovery(llm_provider=llm_provider)
                                rules = rule_discovery.discover_rules(profile, use_llm=False)
                                st.session_state.rules = rules
                                total_rules = sum(len(r) for r in rules.values())
                                st.warning(f"Using heuristics only: {total_rules} rules discovered")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # Display profile summary
        if st.session_state.profile:
            st.markdown("---")
            st.subheader("Profile Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{st.session_state.profile['row_count']:,}")
            with col2:
                st.metric("Total Columns", st.session_state.profile['column_count'])
            with col3:
                null_cols = sum(1 for c in st.session_state.profile['columns'] if c['null_percentage'] > 0)
                st.metric("Columns with Nulls", null_cols)
            
            # Show column details
            with st.expander("📋 Column Details"):
                for col_profile in st.session_state.profile['columns']:
                    st.markdown(f"**{col_profile['name']}** ({col_profile['dtype']})")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.text(f"Unique: {col_profile['unique_count']:,}")
                    with col_b:
                        st.text(f"Nulls: {col_profile['null_percentage']}%")
                    with col_c:
                        if 'min' in col_profile:
                            st.text(f"Range: [{col_profile['min']:.2f}, {col_profile['max']:.2f}]")
                    st.markdown("---")
    
    with tab2:
        st.header("Step 2: Review and Accept Rules")
        
        if not st.session_state.rules:
            st.info("👈 Please load and profile data first")
            return
        
        st.markdown("Review proposed data quality rules. Check the boxes to accept rules.")
        
        # Calculate rule sources
        total_rules = sum(len(r) for r in st.session_state.rules.values())
        heuristic_count = sum(1 for rules in st.session_state.rules.values() for r in rules if r.get("source") == "heuristic")
        llm_count = sum(1 for rules in st.session_state.rules.values() for r in rules if r.get("source") == "llm")
        
        # Show breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rules", total_rules)
        with col2:
            st.metric("Heuristic", heuristic_count, help="Fast, rule-based detection")
        with col3:
            st.metric("AI Generated", llm_count, help="LLM-powered suggestions")
        
        st.markdown("---")
        
        # Initialize accepted rules for each column if not exists
        for col_name, rules in st.session_state.rules.items():
            if col_name not in st.session_state.accepted_rules:
                st.session_state.accepted_rules[col_name] = []
        
        # Display rules by column
        for col_name, rules in st.session_state.rules.items():
            if not rules:
                continue
            
            st.markdown(f"### Column: `{col_name}`")
            
            for idx, rule in enumerate(rules):
                rule_key = f"{col_name}_{idx}"
                
                # Severity color
                severity_color = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🔵",
                    "info": "ℹ️"
                }
                severity_icon = severity_color.get(rule.get("severity", "info"), "•")
                
                # Source indicator
                source = rule.get("source", "unknown")
                source_badge = {
                    "heuristic": "⚡ Heuristic",
                    "llm": "🤖 AI",
                    "unknown": "❓"
                }
                source_label = source_badge.get(source, source)
                
                col_a, col_b, col_c = st.columns([1, 4, 1])
                
                with col_a:
                    accepted = st.checkbox(
                        "Accept",
                        key=rule_key,
                        value=rule in st.session_state.accepted_rules.get(col_name, [])
                    )
                
                with col_b:
                    st.markdown(f"{severity_icon} **{rule['type']}**: {rule['description']}")
                    st.caption(f"Action: `{rule['action']}`")
                
                with col_c:
                    if source == "heuristic":
                        st.caption("Heuristic")
                    else:
                        st.caption("AI")
                
                # Update accepted rules
                if accepted and rule not in st.session_state.accepted_rules[col_name]:
                    st.session_state.accepted_rules[col_name].append(rule)
                elif not accepted and rule in st.session_state.accepted_rules[col_name]:
                    st.session_state.accepted_rules[col_name].remove(rule)
            
            st.markdown("---")
        
        # Summary
        total_accepted = sum(len(r) for r in st.session_state.accepted_rules.values())
        st.info(f"{total_accepted} rules accepted")
    
    with tab3:
        st.header("Step 3: Export Rules")
        
        if not st.session_state.accepted_rules:
            st.info("👈 Please review and accept rules first")
            return
        
        # Filter out empty rule lists
        export_rules = {k: v for k, v in st.session_state.accepted_rules.items() if v}
        
        if not export_rules:
            st.warning("No rules accepted yet")
            return
        
        st.markdown("Export accepted rules as YAML for reuse")
        
        # Convert to YAML
        yaml_str = yaml.dump(export_rules, default_flow_style=False, sort_keys=False)
        
        st.code(yaml_str, language="yaml")
        
        # Download button
        st.download_button(
            label="📥 Download Rules (YAML)",
            data=yaml_str,
            file_name="datamender_rules.yaml",
            mime="text/yaml"
        )
        
        # Also provide JSON option
        json_str = json.dumps(export_rules, indent=2)
        st.download_button(
            label="📥 Download Rules (JSON)",
            data=json_str,
            file_name="datamender_rules.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()

