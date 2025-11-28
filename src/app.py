"""Week 4-7: Streamlit UI for Human-in-the-Loop Validation and Data Cleaning"""
import streamlit as st
import json
import yaml
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import polars as pl
from src.profiler import DataProfiler
from src.rule_discovery import RuleDiscovery
from src.data_cleaner import DataCleaner
from src.metrics import CleaningMetrics
import hashlib


def generate_unique_rule_id(col_name: str, rule: dict) -> str:
    """Generate a unique rule ID for a rule"""
    rule_signature = (
        f"{col_name}_"
        f"{rule.get('type', 'unknown')}_"
        f"{rule.get('action', 'unknown')}_"
        f"{rule.get('description', '')}_"
        f"{rule.get('min', '')}_"
        f"{rule.get('max', '')}_"
        f"{rule.get('condition', '')}_"
        f"{rule.get('strategy', '')}"
    )
    return hashlib.md5(rule_signature.encode()).hexdigest()[:12]


st.set_page_config(
    page_title="DataMender", 
    page_icon="🔧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'profiler' not in st.session_state:
    st.session_state.profiler = None
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'rules' not in st.session_state:
    st.session_state.rules = None
if 'accepted_rules' not in st.session_state:
    st.session_state.accepted_rules = {}
if 'accepted_rule_ids' not in st.session_state:
    st.session_state.accepted_rule_ids = set()  # Simple set for counting
if 'last_sample_size' not in st.session_state:
    st.session_state.last_sample_size = None
if 'last_file_name' not in st.session_state:
    st.session_state.last_file_name = None
if 'cleaner' not in st.session_state:
    st.session_state.cleaner = None
if 'cleaned_profile' not in st.session_state:
    st.session_state.cleaned_profile = None
if 'cleaning_metrics' not in st.session_state:
    st.session_state.cleaning_metrics = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'full_data_path' not in st.session_state:
    st.session_state.full_data_path = None
if 'full_original_profile' not in st.session_state:
    st.session_state.full_original_profile = None
if 'metrics_calc' not in st.session_state:
    st.session_state.metrics_calc = None
if 'use_chunking' not in st.session_state:
    st.session_state.use_chunking = False
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = None
    if 'chunking_notified' not in st.session_state:
        st.session_state.chunking_notified = False
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'applied_rules_tracking' not in st.session_state:
        st.session_state.applied_rules_tracking = set()  # Track which rules have been applied
    if 'rows_clipped_history' not in st.session_state:
        st.session_state.rows_clipped_history = {}  # Store rows_clipped counts by column for clip_range rules
    if 'all_cleaning_logs' not in st.session_state:
        st.session_state.all_cleaning_logs = []  # Accumulate all cleaning logs across multiple cleaning passes
    if 'all_anomaly_details' not in st.session_state:
        st.session_state.all_anomaly_details = []  # Accumulate all anomaly details across multiple cleaning passes
    if 'cumulative_rows_removed' not in st.session_state:
        st.session_state.cumulative_rows_removed = 0  # Total rows removed across all cleaning passes
    if 'cumulative_nulls_filled' not in st.session_state:
        st.session_state.cumulative_nulls_filled = 0  # Total nulls filled across all cleaning passes
    if 'cumulative_rows_modified' not in st.session_state:
        st.session_state.cumulative_rows_modified = 0  # Total rows modified (clipped, converted, etc.) across all cleaning passes
    if 'original_row_count' not in st.session_state:
        st.session_state.original_row_count = None  # Store original row count for percentage calculations


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
            max_value=100000,  # Maximum sample size limit
            value=10000,
            step=1000,
            help="Number of rows to profile (0 for all, max 100,000 for performance)",
            key="sample_size_input"
        )
        
        # Auto-detect chunking for large files (fully automatic, no manual option)
        # Auto-detect if chunking should be enabled based on file size
        auto_chunking = False
        auto_chunk_size = 100000  # Default chunk size
        
        if st.session_state.full_data_path and Path(st.session_state.full_data_path).exists():
            file_size_mb = Path(st.session_state.full_data_path).stat().st_size / (1024 * 1024)
            file_size_gb = file_size_mb / 1024
            # Auto-enable for files >500MB
            if file_size_mb > 500:
                auto_chunking = True
                # Adjust chunk size based on file size
                if file_size_gb > 2:  # >2GB
                    auto_chunk_size = 50000  # Smaller chunks for very large files
                elif file_size_gb > 1:  # >1GB
                    auto_chunk_size = 100000  # Standard chunks
                else:
                    auto_chunk_size = 200000  # Larger chunks for medium files
                
                # Auto-enable automatically (no user choice needed)
                st.session_state.use_chunking = True
                st.session_state.chunk_size = auto_chunk_size
                
                # Show info message
                st.info(f"🔄 **Auto-chunking enabled** for {file_size_gb:.2f} GB file\nChunk size: {auto_chunk_size:,} rows")
            else:
                # Small files don't need chunking
                st.session_state.use_chunking = False
                st.session_state.chunk_size = None
        
        # Set chunk_size for use in cleaning/export
        chunk_size = st.session_state.chunk_size if st.session_state.use_chunking else None
        
        st.markdown("### Progress")
        
        # Beautiful progress indicators with icons and status
        progress_items = []
        
        # Step 1: Data Profiled
        if st.session_state.profile:
            row_count = st.session_state.last_sample_size or st.session_state.profile.get('row_count', 'all')
            progress_items.append({
                "icon": "✅",
                "text": f"Data profiled",
                "detail": f"{row_count:,} rows" if isinstance(row_count, int) else "all rows",
                "status": "complete"
            })
        else:
            progress_items.append({
                "icon": "⚪",
                "text": "Data profiled",
                "detail": "Pending",
                "status": "pending"
            })
        
        # Step 2: Rules Discovered
        if st.session_state.rules:
            total_rules = sum(len(r) for r in st.session_state.rules.values())
            progress_items.append({
                "icon": "✅",
                "text": "Rules discovered",
                "detail": f"{total_rules} rules",
                "status": "complete"
            })
        else:
            progress_items.append({
                "icon": "⚪",
                "text": "Rules discovered",
                "detail": "Pending",
                "status": "pending"
            })
        
        # Step 3: Rules Applied
        # Only show count after cleaning, not before
        if st.session_state.cleaner and st.session_state.applied_rules_tracking:
            applied_count = len(st.session_state.applied_rules_tracking)
            progress_items.append({
                "icon": "✅",
                "text": "Rules applied",
                "detail": f"{applied_count} rules",
                "status": "complete"
            })
        else:
            progress_items.append({
                "icon": "⚪",
                "text": "Rules applied",
                "detail": "Pending",
                "status": "pending"
            })
        
        # Step 4: Data Cleaned
        if st.session_state.cleaner:
            rows_cleaned = len(st.session_state.cleaner.df)
            progress_items.append({
                "icon": "✅",
                "text": "Data cleaned",
                "detail": f"{rows_cleaned:,} rows",
                "status": "complete"
            })
        else:
            progress_items.append({
                "icon": "⚪",
                "text": "Data cleaned",
                "detail": "Pending",
                "status": "pending"
            })
        
        # Display progress items with beautiful styling
        for item in progress_items:
            if item["status"] == "complete":
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    padding: 10px 14px;
                    border-radius: 10px;
                    margin: 6px 0;
                    color: white;
                    font-size: 14px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    border-left: 4px solid #4CAF50;
                ">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 18px;">{item['icon']}</span>
                        <div>
                            <strong>{item['text']}</strong><br>
                            <span style="font-size: 12px; opacity: 0.9;">{item['detail']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background: rgba(255, 255, 255, 0.03);
                    padding: 10px 14px;
                    border-radius: 10px;
                    margin: 6px 0;
                    color: rgba(255, 255, 255, 0.5);
                    font-size: 14px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-left: 4px solid rgba(255, 255, 255, 0.2);
                ">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 18px;">{item['icon']}</span>
                        <div>
                            {item['text']}<br>
                            <span style="font-size: 12px;">{item['detail']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        # Clear button - moved below progress section
        if st.button("🗑️ Clear All", help="Reset and start fresh"):
            st.session_state.profiler = None
            st.session_state.profile = None
            st.session_state.rules = None
            st.session_state.accepted_rules = {}
            st.session_state.accepted_rule_ids = set()
            st.session_state.last_sample_size = None
            st.session_state.last_file_name = None
            st.session_state.cleaner = None
            st.session_state.cleaned_profile = None
            st.session_state.cleaning_metrics = None
            st.session_state.original_df = None
            st.session_state.all_cleaning_logs = []  # Clear accumulated logs
            st.session_state.all_anomaly_details = []  # Clear accumulated anomaly details
            st.session_state.rows_clipped_history = {}  # Clear rows_clipped history
            st.session_state.applied_rules_tracking = set()  # Clear applied rules tracking
            st.session_state.cumulative_rows_removed = 0  # Reset cumulative statistics
            st.session_state.cumulative_nulls_filled = 0  # Reset cumulative statistics
            st.session_state.cumulative_rows_modified = 0  # Reset cumulative rows modified
            st.session_state.original_row_count = None  # Reset original row count
            st.session_state.full_data_path = None
            st.session_state.full_original_profile = None
            st.session_state.metrics_calc = None
            st.session_state.use_chunking = False
            st.session_state.chunk_size = None
            st.session_state.chunking_notified = False
            # Increment uploader key to force file uploader to reset
            st.session_state.uploader_key = (st.session_state.uploader_key or 0) + 1
            st.session_state.applied_rules_tracking = set()  # Reset applied rules tracking
            st.session_state.all_cleaning_logs = []  # Clear accumulated cleaning logs
            st.session_state.rows_clipped_history = {}  # Clear rows_clipped history
            st.session_state.accepted_rule_ids = set()  # Clear accepted rule IDs
            st.session_state.accepted_rule_ids = set()  # Reset accepted rule IDs
            st.rerun()
    
    # Main workflow tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Load Data", "🔍 Review Rules", "🧹 Clean Data", "💾 Export"])
    
    with tab1:
        st.header("Step 1: Load and Profile Data")
        
        uploaded_file = st.file_uploader(
            "Upload CSV or Parquet file",
            type=["csv", "parquet"],
            help="Maximum file size: 5GB",
            key=f"file_uploader_{st.session_state.uploader_key}"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = Path(f"/tmp/{uploaded_file.name}")
            temp_path.write_bytes(uploaded_file.read())
            
            # Check file size and auto-enable chunking if needed
            file_size_mb = temp_path.stat().st_size / (1024 * 1024)
            file_size_gb = file_size_mb / 1024
            
            # Auto-enable chunking for large files (>500MB) - fully automatic
            if file_size_mb > 500:
                st.session_state.use_chunking = True
                # Set appropriate chunk size based on file size
                if file_size_gb > 2:
                    st.session_state.chunk_size = 50000  # Smaller chunks for very large files
                elif file_size_gb > 1:
                    st.session_state.chunk_size = 100000  # Standard chunks
                else:
                    st.session_state.chunk_size = 200000  # Larger chunks for medium files
                st.session_state.chunking_notified = False  # Reset notification
                st.success(f"📊 Large file detected ({file_size_gb:.2f} GB). Chunked processing will be used automatically.")
            else:
                # Small files don't need chunking
                st.session_state.use_chunking = False
                st.session_state.chunk_size = None
            
            # Show info if file or settings changed
            if st.session_state.last_file_name and st.session_state.last_file_name != uploaded_file.name:
                st.info(f"New file: {uploaded_file.name}")
            
            # Show file size info
            if file_size_mb > 100:
                st.caption(f"📁 File size: {file_size_gb:.2f} GB ({file_size_mb:.0f} MB)")
            
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
                        # Store full data path for later cleaning
                        st.session_state.full_data_path = str(temp_path)
                        
                        # Check file size and auto-enable chunking if needed
                        file_size_mb = temp_path.stat().st_size / (1024 * 1024)
                        file_size_gb = file_size_mb / 1024
                        
                        # Auto-enable chunking for large files
                        if file_size_mb > 500 and not st.session_state.get('use_chunking', False):
                            st.session_state.use_chunking = True
                            if file_size_gb > 2:
                                st.session_state.chunk_size = 50000
                            elif file_size_gb > 1:
                                st.session_state.chunk_size = 100000
                            else:
                                st.session_state.chunk_size = 200000
                            st.info(f"📊 Large file detected ({file_size_gb:.2f} GB). Auto-enabled chunked processing with {st.session_state.chunk_size:,} row chunks.")
                        
                        # Use the direct widget value (sample_size) for profiling
                        profiler = DataProfiler(str(temp_path))
                        # Apply max sample size limit (100K)
                        effective_sample = sample_size if sample_size > 0 else None
                        if effective_sample and effective_sample > 100000:
                            st.warning(f"Sample size capped at 100,000 rows for performance")
                            effective_sample = 100000
                        profiler.load_data(sample_size=effective_sample, max_sample_size=100000)
                        profile = profiler.profile_all()
                        
                        # Store sampled DataFrame (for display/preview only)
                        # The full dataset will be loaded when cleaning
                        st.session_state.original_df = profiler.df  # This is the sampled data for preview
                        
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
                                
                                # Rerun to update progress sidebar immediately
                                st.rerun()
                            except Exception as e:
                                st.error(f"LLM Error: {e}")
                                st.info("Tip: Uncheck 'Use LLM' to use heuristics only, or configure your API key in .env file")
                                # Fall back to heuristics only
                                rule_discovery = RuleDiscovery(llm_provider=llm_provider)
                                rules = rule_discovery.discover_rules(profile, use_llm=False)
                                st.session_state.rules = rules
                                total_rules = sum(len(r) for r in rules.values())
                                st.warning(f"Using heuristics only: {total_rules} rules discovered")
                                
                                # Rerun to update progress sidebar immediately
                                st.rerun()
                    
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
        
        # Track if any changes were made during this render
        rules_changed = False
        
        # Display rules by column
        for col_name, rules in st.session_state.rules.items():
            if not rules:
                continue
            
            st.markdown(f"### Column: `{col_name}`")
            
            for idx, rule in enumerate(rules):
                rule_key = f"{col_name}_{idx}"
                
                # Check if this rule has been applied
                # Create a unique rule_id to distinguish between similar rules
                rule_id = generate_unique_rule_id(col_name, rule)
                is_applied = rule_id in st.session_state.applied_rules_tracking
                
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
                
                # Ensure the list exists first
                if col_name not in st.session_state.accepted_rules:
                    st.session_state.accepted_rules[col_name] = []
                
                # Simple check: is rule_id in the set? (use same rule_id generation)
                is_already_accepted = rule_id in st.session_state.accepted_rule_ids
                
                with col_a:
                    # Disable checkbox if rule has been applied, but show it as checked
                    if is_applied:
                        st.markdown("✅")
                        st.caption("Applied", help="This rule has already been applied during cleaning")
                        accepted = True  # Keep it in accepted list
                    else:
                        accepted = st.checkbox(
                        "Accept",
                        key=rule_key,
                            value=is_already_accepted
                        )
                        
                        # Simple: update state based on checkbox value
                        if accepted:
                            # Add if not already there
                            if not is_already_accepted:
                                st.session_state.accepted_rules[col_name].append(rule)
                                st.session_state.accepted_rule_ids.add(rule_id)  # Simple counting
                        else:
                            # Remove if it was there
                            if is_already_accepted and not is_applied:
                                # Remove by matching the same unique rule_id
                                st.session_state.accepted_rules[col_name] = [
                                    r for r in st.session_state.accepted_rules[col_name]
                                    if generate_unique_rule_id(col_name, r) != rule_id
                                ]
                                st.session_state.accepted_rule_ids.discard(rule_id)  # Simple counting
                
                with col_b:
                    # Style applied rules differently
                    if is_applied:
                        st.markdown(f"""
                        <div style="opacity: 0.7; border-left: 3px solid #4CAF50; padding-left: 8px;">
                            {severity_icon} **{rule['type']}**: {rule['description']}<br>
                            <span style="font-size: 12px; color: #4CAF50;">✅ Applied during cleaning</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(f"Action: `{rule['action']}`")
                    else:
                        st.markdown(f"{severity_icon} **{rule['type']}**: {rule['description']}")
                        st.caption(f"Action: `{rule['action']}`")
                
                with col_c:
                    if is_applied:
                        st.markdown("**✅ Applied**")
                    elif source == "heuristic":
                        st.caption("Heuristic")
                    else:
                        st.caption("AI")
            
            st.markdown("---")
        
        # Summary - simple count from set
        total_accepted = len(st.session_state.accepted_rule_ids)
        st.info(f"{total_accepted} rules accepted")
    
    with tab3:
        st.header("Step 3: Clean Data")
        
        if not st.session_state.accepted_rules:
            st.info("👈 Please review and accept rules first")
            return
        
        if st.session_state.full_data_path is None:
            st.info("👈 Please load and profile data first")
            return
        
        # Filter out empty rule lists
        export_rules = {k: v for k, v in st.session_state.accepted_rules.items() if v}
        
        if not export_rules:
            st.warning("No rules accepted yet. Please accept some rules in the 'Review Rules' tab.")
            return
        
        # Calculate new rules (rules that haven't been applied yet)
        applied_rules_tracking = st.session_state.applied_rules_tracking if st.session_state.applied_rules_tracking else set()
        new_rules = {}
        new_rules_count = 0
        already_applied_count = 0
        
        # Use same unique rule_id generation as everywhere else
        # Count both new and already-applied rules for accurate display
        for col_name, rules in export_rules.items():
            new_rules_for_col = []
            for rule in rules:
                rule_id = generate_unique_rule_id(col_name, rule)
                if rule_id not in applied_rules_tracking:
                    new_rules_for_col.append(rule)
                    new_rules_count += 1
                else:
                    already_applied_count += 1
            if new_rules_for_col:
                new_rules[col_name] = new_rules_for_col
        
        st.markdown("Apply accepted rules to clean your dataset")
        
        # Show both profiled and full dataset info
        # Only show "Rules to Apply" metric if there are new rules to apply
        total_rules = sum(len(r) for r in export_rules.values())
        
        if new_rules_count > 0:
            # Show metrics in 3 columns when there are new rules
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rules to Apply", f"{new_rules_count} new",
                         help=f"{new_rules_count} new rules to apply out of {total_rules} total accepted ({already_applied_count} already applied)" if already_applied_count > 0 else f"{new_rules_count} new rules to apply out of {total_rules} total accepted")
            with col2:
                st.metric("Profiled Rows", f"{st.session_state.profile['row_count']:,}", 
                         help="Number of rows used for profiling and rule discovery")
            with col3:
                if st.session_state.full_data_path:
                    st.metric("Full Dataset Rows", "Will load on clean", 
                             help="Full dataset will be loaded when you click 'Clean Data'")
                else:
                    st.metric("Full Dataset Rows", "N/A")
        else:
            # When all rules are applied, don't show redundant "Rules to Apply" metric
            # Sidebar and Cleaning Results already show the count
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Profiled Rows", f"{st.session_state.profile['row_count']:,}", 
                         help="Number of rows used for profiling and rule discovery")
            with col2:
                if st.session_state.full_data_path:
                    st.metric("Full Dataset Rows", "Will load on clean", 
                             help="Full dataset will be loaded when you click 'Clean Data'")
                else:
                    st.metric("Full Dataset Rows", "N/A")
        
        # Don't show redundant info - metrics already show profiled rows and cleaned rows
        # Disable button if no new rules to apply
        button_disabled = new_rules_count == 0 and st.session_state.cleaner is not None
        
        if button_disabled:
            st.info("ℹ️ All accepted rules have already been applied. Please accept new rules in the 'Review Rules' tab to clean again.")
        
        if st.button("🧹 Clean Data", type="primary", disabled=button_disabled):
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Load dataset (use cleaned data if available, otherwise original)
                status_text.text("📥 Step 1/5: Loading dataset...")
                progress_bar.progress(10)
                
                # If we've cleaned before, start from the cleaned data
                if st.session_state.cleaner and st.session_state.cleaner.df is not None:
                    full_df = st.session_state.cleaner.df.clone()  # Start from previously cleaned data
                    status_text.text(f"✅ Loaded {len(full_df):,} rows (from previous cleaning)")
                else:
                    # First time cleaning - load from original file
                    full_profiler = DataProfiler(st.session_state.full_data_path)
                    full_profiler.load_data(sample_size=None)  # Load ALL rows
                    full_df = full_profiler.df
                    status_text.text(f"✅ Loaded {len(full_df):,} rows (from original file)")
                
                progress_bar.progress(20)
                
                # Step 2: Profile baseline dataset for comparison
                # Always compare against the very first original dataset
                status_text.text("📊 Step 2/5: Loading baseline dataset for comparison...")
                progress_bar.progress(30)
                
                if st.session_state.full_original_profile is None:
                    # First time - profile the original dataset
                    if st.session_state.cleaner is None:
                        # Use current data as baseline
                        full_original_profiler = DataProfiler("")
                        full_original_profiler.df = full_df.clone()
                        full_original_profile = full_original_profiler.profile_all()
                        st.session_state.full_original_profile = full_original_profile
                        # Store original row count for cumulative calculations
                        st.session_state.original_row_count = full_original_profile.get("row_count", 0)
                    else:
                        # Shouldn't happen, but fallback
                        full_profiler = DataProfiler(st.session_state.full_data_path)
                        full_profiler.load_data(sample_size=None)
                        full_original_profiler = DataProfiler("")
                        full_original_profiler.df = full_profiler.df.clone()
                        full_original_profile = full_original_profiler.profile_all()
                        st.session_state.full_original_profile = full_original_profile
                        # Store original row count for cumulative calculations
                        st.session_state.original_row_count = full_original_profile.get("row_count", 0)
                else:
                    # Use stored original profile
                    full_original_profile = st.session_state.full_original_profile
                    # Ensure original_row_count is set if it wasn't before
                    if st.session_state.original_row_count is None:
                        st.session_state.original_row_count = full_original_profile.get("row_count", 0)
                
                status_text.text("✅ Baseline dataset ready for comparison")
                progress_bar.progress(40)
                
                # Step 3: Apply cleaning rules
                status_text.text("🧹 Step 3/5: Applying cleaning rules...")
                progress_bar.progress(50)
                cleaner = DataCleaner(full_df)
                
                # Use new_rules if available, otherwise use all export_rules
                rules_to_apply = new_rules if new_rules else export_rules
                
                # Count total rules for progress
                all_rules = []
                for col_name, col_rules in rules_to_apply.items():
                    for rule in col_rules:
                        all_rules.append((col_name, rule))
                
                total_rules = len(all_rules)
                effective_chunk_size = st.session_state.chunk_size if st.session_state.use_chunking else None
                
                if effective_chunk_size and len(full_df) > effective_chunk_size:
                    num_chunks = (len(full_df) + effective_chunk_size - 1) // effective_chunk_size
                    status_text.text(f"🔄 Processing {num_chunks:,} chunks ({effective_chunk_size:,} rows each)...")
                    progress_bar.progress(55)
                    cleaner.apply_rules(rules_to_apply, chunk_size=effective_chunk_size)
                    status_text.text(f"✅ Processed {num_chunks:,} chunks successfully")
                    progress_bar.progress(80)
                else:
                    # Apply rules one by one with progress updates
                    for idx, (col_name, rule) in enumerate(all_rules):
                        rule_type = rule.get("type", "unknown")
                        action = rule.get("action", "unknown")
                        status_text.text(f"🔧 Applying rule {idx+1}/{total_rules}: {col_name} - {rule_type} ({action})")
                        progress = 50 + int((idx + 1) / total_rules * 30)
                        progress_bar.progress(progress)
                        cleaner.apply_rule(rule)
                    
                    status_text.text("✅ All rules applied")
                    progress_bar.progress(80)
                
                # Step 4: Re-profile cleaned data (always re-profile to ensure fresh data)
                status_text.text("📊 Step 4/5: Re-profiling cleaned data...")
                progress_bar.progress(85)
                cleaning_stats = cleaner.get_cleaning_stats()
                cleaned_profiler = DataProfiler("")
                cleaned_profiler.df = cleaner.df.clone()  # Ensure fresh clone
                cleaned_profile = cleaned_profiler.profile_all()
                status_text.text("✅ Cleaned data profiled")
                progress_bar.progress(90)
                
                # Step 5: Calculate metrics (always recalculate with fresh profiles)
                status_text.text("📈 Step 5/5: Calculating metrics...")
                progress_bar.progress(95)
                
                # Get all applied rules from cleaner for this cleaning pass
                all_applied_rules = cleaner.get_applied_rules_log()
                
                # Store rows_clipped counts in session state for future reference
                # Store by column name (simplest key for lookup)
                for rule_log in all_applied_rules:
                    rule = rule_log.get("rule", {})
                    if rule.get("action") == "clip_range":
                        col_name = rule.get("column")
                        rows_clipped = rule_log.get("rows_clipped")
                        
                        # Always store rows_clipped if available (even if 0, to track that rule was applied)
                        if rows_clipped is not None:
                            # If this column doesn't have a stored value yet, store it
                            if col_name not in st.session_state.rows_clipped_history:
                                st.session_state.rows_clipped_history[col_name] = rows_clipped
                            # If stored value is 0 and we have a non-zero, update it
                            elif st.session_state.rows_clipped_history.get(col_name, 0) == 0 and rows_clipped > 0:
                                st.session_state.rows_clipped_history[col_name] = rows_clipped
                            # If both are non-zero, keep the larger one (represents more complete data)
                            elif rows_clipped > st.session_state.rows_clipped_history.get(col_name, 0):
                                st.session_state.rows_clipped_history[col_name] = rows_clipped
                
                # Accumulate all cleaning logs in session state (don't overwrite, append)
                st.session_state.all_cleaning_logs.extend(all_applied_rules)
                
                metrics_calc = CleaningMetrics(
                    full_original_profile,
                    cleaned_profile,
                    cleaning_stats,
                    all_applied_rules  # Pass applied rules to extract rows_clipped
                )
                metrics = metrics_calc.calculate_metrics()
                
                # Store original row count on first cleaning pass (if not already stored)
                if st.session_state.original_row_count is None:
                    st.session_state.original_row_count = full_original_profile.get("row_count", 0)
                
                # Calculate cumulative statistics
                # The metrics compare original to current cleaned, so they show cumulative totals
                # But we need to calculate them from the original count vs current cleaned count
                original_rows = st.session_state.original_row_count
                current_cleaned_rows = cleaned_profile.get("row_count", 0)
                
                # Cumulative rows removed = original - current cleaned
                st.session_state.cumulative_rows_removed = original_rows - current_cleaned_rows
                
                # Calculate cumulative nulls filled from original vs current profiles
                # Count total nulls in original profile
                original_total_nulls = 0
                for col in full_original_profile.get("columns", []):
                    null_pct = col.get("null_percentage", 0)
                    if null_pct > 0:
                        original_total_nulls += int(original_rows * null_pct / 100)
                
                # Count total nulls in current cleaned profile
                current_total_nulls = 0
                for col in cleaned_profile.get("columns", []):
                    null_pct = col.get("null_percentage", 0)
                    if null_pct > 0:
                        current_total_nulls += int(current_cleaned_rows * null_pct / 100)
                
                # Cumulative nulls filled = original nulls - current nulls
                st.session_state.cumulative_nulls_filled = original_total_nulls - current_total_nulls
                
                # Enhance anomaly details with stored rows_clipped history
                # Also calculate cumulative rows modified while we're at it
                # This helps when rules were applied in previous cleaning passes
                if "anomaly_metrics" in metrics and "anomaly_details" in metrics["anomaly_metrics"]:
                    # Get original profile columns for estimation
                    original_cols = {c["name"]: c for c in full_original_profile.get("columns", [])}
                    
                    # Build a map of negative_values anomalies by column name for quick lookup
                    # Include all columns with negatives (even if fixed_count is 0, they might have been dropped)
                    negative_values_map = {}
                    for anomaly in metrics["anomaly_metrics"]["anomaly_details"]:
                        if anomaly.get("type") == "negative_values":
                            col_name = anomaly.get("column")
                            fixed_count = anomaly.get("fixed", 0)
                            before_count = anomaly.get("before", 0)
                            after_count = anomaly.get("after", 0)
                            # Use fixed_count if > 0, otherwise use the difference (rows that disappeared)
                            # This handles cases where negatives were dropped (fixed_count=0 but before > after)
                            count_to_use = fixed_count if fixed_count > 0 else (before_count - after_count)
                            if count_to_use > 0:
                                negative_values_map[col_name] = count_to_use
                    
                    for anomaly in metrics["anomaly_metrics"]["anomaly_details"]:
                        if anomaly.get("type") == "out_of_range" and anomaly.get("rows_affected") is None:
                            col_name = anomaly.get("column")
                            
                            # First, try to look up stored rows_clipped for this column
                            stored_value = st.session_state.rows_clipped_history.get(col_name)
                            
                            # If stored value exists and is > 0, use it (actual clipping occurred)
                            if stored_value is not None and stored_value > 0:
                                anomaly["rows_affected"] = stored_value
                                continue
                            
                            # Second, calculate rows affected based on the range change itself
                            # This handles cases where range changed but stored value is 0 or missing
                            # This counts rows affected by range clipping even when abs_value converted negatives first
                            original_range = anomaly.get("original_range", [])
                            cleaned_range = anomaly.get("cleaned_range", [])
                            if len(original_range) == 2 and len(cleaned_range) == 2:
                                orig_min = original_range[0]
                                orig_max = original_range[1]
                                cleaned_min = cleaned_range[0]
                                cleaned_max = cleaned_range[1]
                                
                                # Calculate rows affected by the range change
                                rows_affected_by_range = 0
                                
                                # If min increased, count values that were affected by the range clipping
                                if cleaned_min > orig_min:
                                    # Check if negatives were dropped for this column (drop_rows action)
                                    # If so, those negatives were affected by the range change
                                    negatives_dropped = False
                                    if all_applied_rules:
                                        for rule_log in all_applied_rules:
                                            rule = rule_log.get("rule", {})
                                            if (rule.get("column") == col_name and 
                                                rule.get("action") == "drop_rows" and
                                                rule.get("condition") in ["negative", "non_positive"] and
                                                rule_log.get("success", False)):
                                                negatives_dropped = True
                                                break
                                    
                                    # If negatives were dropped, use that count (they were affected by range change)
                                    if negatives_dropped and col_name in negative_values_map:
                                        rows_affected_by_range = negative_values_map[col_name]
                                    # If negatives were converted (abs_value), use that count
                                    elif col_name in negative_values_map:
                                        rows_affected_by_range = negative_values_map[col_name]
                                    else:
                                        # Estimate based on the range change (min increased)
                                        # Calculate percentage of range that was clipped
                                        min_increase = cleaned_min - orig_min
                                        orig_range_size = orig_max - orig_min if orig_max > orig_min else 1
                                        if orig_range_size > 0:
                                            # Estimate: percentage of rows in the clipped range
                                            range_change_ratio = min_increase / orig_range_size
                                            # More conservative estimate for positive ranges
                                            estimated_rows = int(full_original_profile.get("row_count", 0) * range_change_ratio * 0.15)
                                            rows_affected_by_range = max(estimated_rows, 0)
                                
                                # If max decreased, add rows affected by upper bound clipping
                                if cleaned_max < orig_max:
                                    max_decrease = orig_max - cleaned_max
                                    orig_range_size = orig_max - orig_min if orig_max > orig_min else 1
                                    if orig_range_size > 0:
                                        range_change_ratio = max_decrease / orig_range_size
                                        estimated_rows = int(full_original_profile.get("row_count", 0) * range_change_ratio * 0.15)
                                        rows_affected_by_range += estimated_rows
                                
                                # Use the calculated count if we got a value
                                if rows_affected_by_range > 0:
                                    anomaly["rows_affected"] = rows_affected_by_range
                                    # Store it for future reference
                                    st.session_state.rows_clipped_history[col_name] = rows_affected_by_range
                                    continue
                            
                            # Third, if not found in history or negative_values, estimate from profile difference
                            # This is a fallback for cases where the rule was applied but count wasn't stored
                            if col_name in original_cols:
                                orig_col = original_cols[col_name]
                                if "min" in orig_col and "min" in anomaly.get("cleaned_range", []):
                                    # Estimate based on how much the range changed
                                    orig_min = orig_col["min"]
                                    cleaned_min = anomaly.get("cleaned_range", [])[0]
                                    orig_max = orig_col["max"]
                                    cleaned_max = anomaly.get("cleaned_range", [])[1]
                                    
                                    # Simple heuristic: if range was significantly reduced, estimate affected rows
                                    orig_range = orig_max - orig_min if orig_max > orig_min else 1
                                    min_change = cleaned_min - orig_min if cleaned_min > orig_min else 0
                                    max_change = orig_max - cleaned_max if cleaned_max < orig_max else 0
                                    
                                    if (min_change > 0 or max_change > 0) and orig_range > 0:
                                        # Estimate: assume values outside new range were clipped
                                        # Conservative estimate: 1-5% of rows per significant range change
                                        change_pct = max(min_change / orig_range, max_change / orig_range)
                                        estimated = int(full_original_profile.get("row_count", 0) * change_pct * 0.05)
                                        if estimated > 0:
                                            anomaly["rows_affected"] = estimated
                                            # Store the estimate for future reference
                                            st.session_state.rows_clipped_history[col_name] = estimated
                
                # Calculate cumulative rows modified from all anomaly details
                # Sum up all rows_affected from out_of_range anomalies and fixed from negative_values
                total_rows_modified = 0
                if "anomaly_metrics" in metrics and "anomaly_details" in metrics["anomaly_metrics"]:
                    # Use a set to track columns we've counted to avoid double-counting
                    columns_counted = set()
                    
                    for anomaly in metrics["anomaly_metrics"]["anomaly_details"]:
                        col_name = anomaly.get("column")
                        anomaly_type = anomaly.get("type")
                        
                        if anomaly_type == "out_of_range":
                            rows_affected = anomaly.get("rows_affected")
                            if rows_affected is not None and rows_affected > 0:
                                # Track this column to avoid double counting with negative_values
                                columns_counted.add(col_name)
                                total_rows_modified += rows_affected
                        elif anomaly_type == "negative_values":
                            # Only count if we haven't already counted this column via out_of_range
                            if col_name not in columns_counted:
                                fixed_count = anomaly.get("fixed", 0)
                                if fixed_count > 0:
                                    total_rows_modified += fixed_count
                
                # Update cumulative rows modified (always use the total from current metrics comparison)
                # This gives us the cumulative total since it compares original to current cleaned
                st.session_state.cumulative_rows_modified = total_rows_modified
                
                # Accumulate anomaly details from this cleaning pass
                if "anomaly_metrics" in metrics and "anomaly_details" in metrics["anomaly_metrics"]:
                    from datetime import datetime
                    current_pass_anomalies = metrics["anomaly_metrics"]["anomaly_details"]
                    # Add timestamp to each anomaly for tracking
                    for anomaly in current_pass_anomalies:
                        anomaly["cleaning_pass_timestamp"] = datetime.now().isoformat()
                    # Append to accumulated list
                    st.session_state.all_anomaly_details.extend(current_pass_anomalies)
                
                # Store in session state (overwrite to ensure fresh metrics)
                st.session_state.cleaner = cleaner
                st.session_state.cleaned_profile = cleaned_profile
                st.session_state.cleaning_metrics = metrics  # Always update with fresh metrics
                st.session_state.full_original_profile = full_original_profile
                st.session_state.metrics_calc = metrics_calc
                
                # Track which rules have been applied (only new ones or all if first time)
                rules_to_track = new_rules if new_rules else export_rules
                for col_name, col_rules in rules_to_track.items():
                    for rule in col_rules:
                        rule_id = generate_unique_rule_id(col_name, rule)
                        st.session_state.applied_rules_tracking.add(rule_id)
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Cleaning complete!")
                st.success(f"✅ Data cleaned! {cleaning_stats['rows_removed']:,} rows removed in {cleaning_stats['processing_time_seconds']:.2f}s")
                
                # Rerun to update progress sidebar immediately
                st.rerun()
                
                # Clear progress bar after a moment
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                    
            except Exception as e:
                st.error(f"Error cleaning data: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Show cleaning results if available
        if st.session_state.cleaner and st.session_state.cleaning_metrics:
            st.markdown("---")
            st.subheader("Cleaning Results")
            
            metrics = st.session_state.cleaning_metrics
            summary = metrics["summary"]
            
            # Get cumulative statistics (across all cleaning passes)
            original_row_count = st.session_state.original_row_count or summary['original_rows']
            cumulative_rows_removed = st.session_state.cumulative_rows_removed
            cumulative_nulls_filled = st.session_state.cumulative_nulls_filled
            cumulative_rows_modified = st.session_state.cumulative_rows_modified
            current_cleaned_rows = summary['cleaned_rows']
            
            # Calculate percentage based on original row count
            rows_removed_pct = (cumulative_rows_removed / original_row_count * 100) if original_row_count > 0 else 0
            rows_modified_pct = (cumulative_rows_modified / original_row_count * 100) if original_row_count > 0 else 0
            
            # First row: Main statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows Removed", f"{cumulative_rows_removed:,}", 
                         f"{rows_removed_pct:.1f}%",
                         help=f"Cumulative rows deleted across all cleaning passes (from original {original_row_count:,} rows)")
            with col2:
                st.metric("Rows Modified", f"{cumulative_rows_modified:,}",
                         f"{rows_modified_pct:.1f}%",
                         help=f"Cumulative rows transformed (clipped, converted, etc.) across all cleaning passes")
            with col3:
                st.metric("Cleaned Rows", f"{current_cleaned_rows:,}",
                         help=f"Current row count (from original {original_row_count:,} rows)")
            with col4:
                # Simple count from applied_rules_tracking
                applied_count = len(st.session_state.applied_rules_tracking) if st.session_state.applied_rules_tracking else 0
                st.metric("Rules Applied", applied_count)
            
            # Second row: Performance
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                perf = metrics["performance_metrics"]
                st.metric("Processing Time", f"{perf['processing_time_seconds']:.2f}s")
            
            # Null reduction - use cumulative values
            null_metrics = metrics["null_metrics"]
            null_reduction_pct = (cumulative_nulls_filled / original_row_count * 100) if original_row_count > 0 else 0
            
            st.markdown("### Null Value Reduction")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nulls Filled", f"{cumulative_nulls_filled:,}",
                         help="Cumulative null values filled in place across all cleaning passes (rows not deleted)")
            with col2:
                st.metric("Null Reduction", f"{null_reduction_pct:.2f}%")
            with col3:
                st.metric("Columns with Nulls", 
                         f"{null_metrics['columns_with_nulls_before']} → {null_metrics['columns_with_nulls_after']}")
            
            # Anomaly fixes - show all accumulated anomalies
            all_anomalies = st.session_state.get("all_anomaly_details", [])
            
            # If no accumulated anomalies yet, use current metrics (for backward compatibility)
            if not all_anomalies:
                anomaly_metrics = metrics["anomaly_metrics"]
                if anomaly_metrics.get("anomalies_fixed", 0) > 0:
                    all_anomalies = anomaly_metrics.get("anomaly_details", [])
            
            if all_anomalies:
                st.markdown("### Anomalies Fixed")
                st.metric("Anomalies Fixed", len(all_anomalies),
                         help=f"Total anomalies fixed across all cleaning passes")
                
                with st.expander("View Anomaly Details"):
                    st.caption(f"📋 Showing {len(all_anomalies)} total anomalies fixed across all cleaning passes")
                    st.markdown("---")
                    
                    # Group anomalies by column for better organization
                    anomalies_by_column = {}
                    for anomaly in all_anomalies:
                        col_name = anomaly.get("column", "unknown")
                        if col_name not in anomalies_by_column:
                            anomalies_by_column[col_name] = []
                        anomalies_by_column[col_name].append(anomaly)
                    
                    # Display grouped by column
                    for col_name, col_anomalies in anomalies_by_column.items():
                        st.markdown(f"#### Column: `{col_name}`")
                        
                        for anomaly in col_anomalies:
                            anomaly_type = anomaly.get("type", "unknown")
                            
                            # Format based on anomaly type
                            if anomaly_type == "negative_values":
                                st.markdown(f"**Issue:** Negative Values")
                                before_count = anomaly.get('before', 0)
                                after_count = anomaly.get('after', 0)
                                fixed_count = anomaly.get('fixed', 0)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Found", f"{before_count:,}", 
                                             help="Number of negative values found in original data")
                                with col2:
                                    st.metric("Fixed", f"{fixed_count:,}", 
                                             help="Number of negative values that were corrected")
                                with col3:
                                    st.metric("Remaining", f"{after_count:,}", 
                                             help="Number of negative values still in the data")
                            elif anomaly_type == "out_of_range":
                                st.markdown(f"**Type:** Out of Range")
                                original_range = anomaly.get("original_range", [])
                                cleaned_range = anomaly.get("cleaned_range", [])
                                rows_affected = anomaly.get("rows_affected")
                                
                                # Try multiple sources for rows_affected count:
                                # 1. From anomaly itself (set during metrics calculation)
                                # 2. From stored history (for previously applied rules)
                                # 3. From negative_values count (if range change was due to drop_rows or abs_value)
                                if rows_affected is None or rows_affected == 0:
                                    # First try stored history
                                    stored_rows_clipped = st.session_state.get("rows_clipped_history", {}).get(col_name)
                                    if stored_rows_clipped is not None and stored_rows_clipped > 0:
                                        rows_affected = stored_rows_clipped
                                    else:
                                        # Check if there's a negative_values anomaly for the same column in all_anomalies
                                        # and the range changed from negative to positive (indicating negatives were dropped/converted)
                                        for other_anomaly in all_anomalies:
                                            if (other_anomaly.get("type") == "negative_values" and 
                                                other_anomaly.get("column") == col_name):
                                                fixed_count = other_anomaly.get("fixed", 0)
                                                if fixed_count > 0 and len(original_range) == 2 and len(cleaned_range) == 2:
                                                    orig_min = original_range[0]
                                                    cleaned_min = cleaned_range[0]
                                                    # If original was negative and cleaned is positive/zero, use negative fix count
                                                    if orig_min < 0 and cleaned_min >= 0:
                                                        rows_affected = fixed_count
                                                        break
                                
                                # Always show 3 columns - rows affected might not be available for incremental cleaning
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if len(original_range) == 2:
                                        st.markdown(f"**Original Range:** {original_range[0]} to {original_range[1]}")
                                with col2:
                                    if len(cleaned_range) == 2:
                                        st.markdown(f"**Cleaned Range:** {cleaned_range[0]} to {cleaned_range[1]}")
                                with col3:
                                    if rows_affected is not None and rows_affected > 0:
                                        st.metric("Rows Affected", f"{rows_affected:,}",
                                                 help="Number of rows that had values adjusted (clipped or converted)")
                                    elif rows_affected == 0:
                                        st.metric("Rows Affected", "0",
                                                 help="No rows needed adjustment (values were already in range)")
                                    else:
                                        st.metric("Rows Affected", "—",
                                                 help="Row count not available (may have been applied in a previous cleaning pass)")
                            else:
                                # Fallback for other types
                                st.json(anomaly)
                            
                            st.markdown("---")  # Separator between anomalies
                        
                        st.markdown("---")  # Separator between columns
            
            # Note: Before/After comparison is done in background via metrics calculation
            # (anomaly_metrics already compares before/after to verify rules were applied correctly)
            
            # Preview cleaned data
            st.markdown("### Preview Cleaned Data")
            preview_rows = 100  # Fixed preview size, no slider needed
            cleaned_df = st.session_state.cleaner.df
            st.dataframe(cleaned_df.head(preview_rows), use_container_width=True)
            
            # Cleaning logs
            st.markdown("### Cleaning Log")
            # Display all accumulated cleaning logs (from all cleaning passes)
            all_logs = st.session_state.get("all_cleaning_logs", [])
            
            if not all_logs and st.session_state.cleaner:
                # Fallback: if no accumulated logs yet, use current cleaner's logs
                all_logs = st.session_state.cleaner.get_applied_rules_log()
            
            with st.expander("View Applied Rules Log"):
                if not all_logs:
                    st.info("No cleaning rules have been applied yet.")
                else:
                    # Filter out chunked processing entries for display
                    display_logs = [
                        log for log in all_logs 
                        if log.get("rule", {}).get("type") != "chunked_processing" 
                        and log.get("rule", {}).get("column") != "all"
                    ]
                    
                    if not display_logs:
                        st.info("No cleaning rules have been applied yet.")
                    else:
                        st.caption(f"📋 Showing {len(display_logs)} total rule applications across all cleaning passes")
                        st.markdown("---")
                        
                        for idx, log in enumerate(display_logs):
                            rule = log.get("rule", {})
                            status = "✅" if log.get("success", False) else "❌"
                            
                            # Handle different rule structures
                            column = rule.get("column", "unknown")
                            rule_type = rule.get("type", "unknown")
                            action = rule.get("action", rule.get("description", "N/A"))
                            
                            # Format timestamp
                            timestamp = log.get("timestamp", "")
                            if isinstance(timestamp, float):
                                from datetime import datetime
                                timestamp = datetime.fromtimestamp(timestamp).isoformat()
                            
                            st.markdown(f"{status} **{column}** - {rule_type} ({action})")
                            if log.get("success", False):
                                rows_before = log.get("rows_before", "?")
                                rows_after = log.get("rows_after", "?")
                                st.caption(f"Rows: {rows_before:,} → {rows_after:,} | {timestamp}")
                            else:
                                error = log.get("error", "Unknown error")
                                st.caption(f"Error: {error} | {timestamp}")
    
    with tab4:
        st.header("Step 4: Export")
        
        # Export Cleaned Data Section (main export functionality)
        st.subheader("Export Cleaned Data")
        
        if not st.session_state.cleaner:
            st.info("👈 Please clean data first in the 'Clean Data' tab")
        else:
            # Default to Parquet (better for large files), CSV as alternative
            export_format = st.radio(
                "Export Format", 
                ["Parquet", "CSV"], 
                horizontal=True,
                index=0,  # Default to Parquet
                help="Parquet: Smaller, faster, better for large files (recommended)\nCSV: Universal, human-readable"
            )
            
            if st.button("💾 Export Cleaned Data", type="primary"):
                try:
                    # Generate filename
                    original_name = st.session_state.last_file_name or "data"
                    base_name = Path(original_name).stem
                    extension = "parquet" if export_format == "Parquet" else "csv"
                    output_name = f"{base_name}_cleaned.{extension}"
                    
                    # Save to temp location
                    temp_output = Path(f"/tmp/{output_name}")
                    
                    # Use chunked export if enabled and file is large
                    effective_chunk_size = st.session_state.chunk_size if st.session_state.use_chunking else None
                    if effective_chunk_size and len(st.session_state.cleaner.df) > effective_chunk_size:
                        st.info(f"💾 Exporting in chunks of {effective_chunk_size:,} rows...")
                        st.session_state.cleaner.export_cleaned_data(
                            str(temp_output), 
                            extension, 
                            chunk_size=effective_chunk_size
                        )
                    else:
                        st.session_state.cleaner.export_cleaned_data(str(temp_output), extension)
                    
                    # Check file size for download strategy
                    file_size_mb = temp_output.stat().st_size / (1024 * 1024)
                    file_size_gb = file_size_mb / 1024
                    
                    # For very large files (>500MB), show path instead of loading into memory
                    if file_size_mb > 500:
                        st.success(f"✅ Cleaned data exported successfully!")
                        st.info(f"📁 **File saved to:** `{temp_output}`\n\n**Size:** {file_size_gb:.2f} GB ({file_size_mb:.0f} MB)\n\n⚠️ File is too large for browser download. Please access it directly from the server.")
                        st.code(str(temp_output), language=None)
                    else:
                        # For smaller files, provide download button
                        file_bytes = temp_output.read_bytes()
                        st.download_button(
                            label=f"📥 Download {output_name}",
                            data=file_bytes,
                            file_name=output_name,
                            mime="application/octet-stream" if extension == "parquet" else "text/csv"
                        )
                        st.success(f"✅ Cleaned data ready for download: {output_name} ({file_size_mb:.1f} MB)")
                    
                except Exception as e:
                    st.error(f"Error exporting data: {e}")


if __name__ == "__main__":
    main()

