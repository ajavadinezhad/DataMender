"""Streamlit UI for Human-in-the-Loop Validation and Data Cleaning"""
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
    st.session_state.accepted_rule_ids = set()
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
        st.session_state.applied_rules_tracking = set()
    if 'rows_clipped_history' not in st.session_state:
        st.session_state.rows_clipped_history = {}
    if 'all_cleaning_logs' not in st.session_state:
        st.session_state.all_cleaning_logs = []
    if 'all_anomaly_details' not in st.session_state:
        st.session_state.all_anomaly_details = []
    if 'cumulative_rows_removed' not in st.session_state:
        st.session_state.cumulative_rows_removed = 0
    if 'cumulative_nulls_filled' not in st.session_state:
        st.session_state.cumulative_nulls_filled = 0
    if 'cumulative_rows_modified' not in st.session_state:
        st.session_state.cumulative_rows_modified = 0
    if 'original_row_count' not in st.session_state:
        st.session_state.original_row_count = None


def main():
    st.title("DataMender")
    st.markdown("*Smart data cleaning for large CSV/Parquet files*")
    
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
            max_value=100000,
            value=10000,
            step=1000,
            help="Number of rows to profile (0 for all, max 100,000 for performance)",
            key="sample_size_input"
        )
        

        auto_chunking = False
        auto_chunk_size = 100000
        
        if st.session_state.full_data_path and Path(st.session_state.full_data_path).exists():
            file_size_mb = Path(st.session_state.full_data_path).stat().st_size / (1024 * 1024)
            file_size_gb = file_size_mb / 1024
            if file_size_mb > 500:
                auto_chunking = True
                if file_size_gb > 2:
                    auto_chunk_size = 50000
                elif file_size_gb > 1:
                    auto_chunk_size = 100000
                else:
                    auto_chunk_size = 200000
                

                st.session_state.use_chunking = True
                st.session_state.chunk_size = auto_chunk_size
                

                st.info(f"🔄 **Auto-chunking enabled** for {file_size_gb:.2f} GB file\nChunk size: {auto_chunk_size:,} rows")
            else:
                st.session_state.use_chunking = False
                st.session_state.chunk_size = None
        

        chunk_size = st.session_state.chunk_size if st.session_state.use_chunking else None
        
        st.markdown("### Progress")
        
        progress_items = []
        
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
            st.session_state.all_cleaning_logs = []
            st.session_state.all_anomaly_details = []
            st.session_state.rows_clipped_history = {}
            st.session_state.applied_rules_tracking = set()
            st.session_state.cumulative_rows_removed = 0
            st.session_state.cumulative_nulls_filled = 0
            st.session_state.cumulative_rows_modified = 0
            st.session_state.original_row_count = None
            st.session_state.full_data_path = None
            st.session_state.full_original_profile = None
            st.session_state.metrics_calc = None
            st.session_state.use_chunking = False
            st.session_state.chunk_size = None
            st.session_state.chunking_notified = False
            st.session_state.uploader_key = (st.session_state.uploader_key or 0) + 1
            st.session_state.applied_rules_tracking = set()
            st.session_state.all_cleaning_logs = []
            st.session_state.rows_clipped_history = {}
            st.session_state.accepted_rule_ids = set()
            st.session_state.accepted_rule_ids = set()
            st.rerun()
    
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
            temp_path = Path(f"/tmp/{uploaded_file.name}")
            temp_path.write_bytes(uploaded_file.read())
            
            file_size_mb = temp_path.stat().st_size / (1024 * 1024)
            file_size_gb = file_size_mb / 1024
            
            if file_size_mb > 500:
                st.session_state.use_chunking = True
                if file_size_gb > 2:
                    st.session_state.chunk_size = 50000
                elif file_size_gb > 1:
                    st.session_state.chunk_size = 100000
                else:
                    st.session_state.chunk_size = 200000
                st.session_state.chunking_notified = False
                st.success(f"📊 Large file detected ({file_size_gb:.2f} GB). Chunked processing will be used automatically.")
            else:
                st.session_state.use_chunking = False
                st.session_state.chunk_size = None
            
            if st.session_state.last_file_name and st.session_state.last_file_name != uploaded_file.name:
                st.info(f"New file: {uploaded_file.name}")
            
            if file_size_mb > 100:
                st.caption(f"📁 File size: {file_size_gb:.2f} GB ({file_size_mb:.0f} MB)")
            
            display_sample = sample_size if sample_size > 0 else "all"
            
            if st.session_state.last_sample_size and st.session_state.last_sample_size != sample_size:
                st.warning(f"Sample changed to {display_sample} rows - click 'Profile Dataset' to apply")
            else:
                st.caption(f"Ready to profile {display_sample} rows")
            
            if st.button("🚀 Profile Dataset", type="primary"):
                with st.spinner("Profiling dataset..."):
                    try:
                        st.session_state.full_data_path = str(temp_path)
                        
                        file_size_mb = temp_path.stat().st_size / (1024 * 1024)
                        file_size_gb = file_size_mb / 1024
                        
                        if file_size_mb > 500 and not st.session_state.get('use_chunking', False):
                            st.session_state.use_chunking = True
                            if file_size_gb > 2:
                                st.session_state.chunk_size = 50000
                            elif file_size_gb > 1:
                                st.session_state.chunk_size = 100000
                            else:
                                st.session_state.chunk_size = 200000
                            st.info(f"📊 Large file detected ({file_size_gb:.2f} GB). Auto-enabled chunked processing with {st.session_state.chunk_size:,} row chunks.")
                        
                        profiler = DataProfiler(str(temp_path))
                        effective_sample = sample_size if sample_size > 0 else None
                        if effective_sample and effective_sample > 100000:
                            st.warning(f"Sample size capped at 100,000 rows for performance")
                            effective_sample = 100000
                        profiler.load_data(sample_size=effective_sample, max_sample_size=100000)
                        profile = profiler.profile_all()
                        
                        st.session_state.original_df = profiler.df
                        
                        st.session_state.profiler = profiler
                        st.session_state.profile = profile
                        st.session_state.last_sample_size = sample_size if sample_size > 0 else profile['row_count']
                        st.session_state.last_file_name = uploaded_file.name
                        
                        st.success(f"Profiled {profile['row_count']} rows, {profile['column_count']} columns")
                        
                        with st.spinner("Discovering rules..."):
                            try:
                                rule_discovery = RuleDiscovery(llm_provider=llm_provider)
                                rules = rule_discovery.discover_rules(profile, use_llm=use_llm)
                                st.session_state.rules = rules
                                
                                total_rules = sum(len(r) for r in rules.values())
                                st.success(f"Discovered {total_rules} rules")
                                
                                st.rerun()
                            except Exception as e:
                                st.error(f"LLM Error: {e}")
                                st.info("Tip: Uncheck 'Use LLM' to use heuristics only, or configure your API key in .env file")
                                rule_discovery = RuleDiscovery(llm_provider=llm_provider)
                                rules = rule_discovery.discover_rules(profile, use_llm=False)
                                st.session_state.rules = rules
                                total_rules = sum(len(r) for r in rules.values())
                                st.warning(f"Using heuristics only: {total_rules} rules discovered")
                                
                                st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
        
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
        
        total_rules = sum(len(r) for r in st.session_state.rules.values())
        heuristic_count = sum(1 for rules in st.session_state.rules.values() for r in rules if r.get("source") == "heuristic")
        llm_count = sum(1 for rules in st.session_state.rules.values() for r in rules if r.get("source") == "llm")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rules", total_rules)
        with col2:
            st.metric("Heuristic", heuristic_count, help="Fast, rule-based detection")
        with col3:
            st.metric("AI Generated", llm_count, help="LLM-powered suggestions")
        
        st.markdown("---")
        
        for col_name, rules in st.session_state.rules.items():
            if col_name not in st.session_state.accepted_rules:
                st.session_state.accepted_rules[col_name] = []
        
        rules_changed = False
        
        for col_name, rules in st.session_state.rules.items():
            if not rules:
                continue
            
            st.markdown(f"### Column: `{col_name}`")
            
            for idx, rule in enumerate(rules):
                rule_key = f"{col_name}_{idx}"
                
                rule_id = generate_unique_rule_id(col_name, rule)
                is_applied = rule_id in st.session_state.applied_rules_tracking
                
                severity_color = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🔵",
                    "info": "ℹ️"
                }
                severity_icon = severity_color.get(rule.get("severity", "info"), "•")
                
                source = rule.get("source", "unknown")
                source_badge = {
                    "heuristic": "⚡ Heuristic",
                    "llm": "🤖 AI",
                    "unknown": "❓"
                }
                source_label = source_badge.get(source, source)
                
                col_a, col_b, col_c = st.columns([1, 4, 1])
                
                if col_name not in st.session_state.accepted_rules:
                    st.session_state.accepted_rules[col_name] = []
                
                is_already_accepted = rule_id in st.session_state.accepted_rule_ids
                
                with col_a:
                    if is_applied:
                        st.markdown("✅")
                        st.caption("Applied", help="This rule has already been applied during cleaning")
                        accepted = True
                    else:
                        accepted = st.checkbox(
                        "Accept",
                        key=rule_key,
                            value=is_already_accepted
                        )
                        
                        if accepted:
                            if not is_already_accepted:
                                st.session_state.accepted_rules[col_name].append(rule)
                                st.session_state.accepted_rule_ids.add(rule_id)
                        else:
                            if is_already_accepted and not is_applied:
                                st.session_state.accepted_rules[col_name] = [
                                    r for r in st.session_state.accepted_rules[col_name]
                                    if generate_unique_rule_id(col_name, r) != rule_id
                                ]
                                st.session_state.accepted_rule_ids.discard(rule_id)
                
                with col_b:
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
        
        export_rules = {k: v for k, v in st.session_state.accepted_rules.items() if v}
        
        if not export_rules:
            st.warning("No rules accepted yet. Please accept some rules in the 'Review Rules' tab.")
            return
        
        applied_rules_tracking = st.session_state.applied_rules_tracking if st.session_state.applied_rules_tracking else set()
        new_rules = {}
        new_rules_count = 0
        already_applied_count = 0
        
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
        
        total_rules = sum(len(r) for r in export_rules.values())
        
        if new_rules_count > 0:
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
        
        button_disabled = new_rules_count == 0 and st.session_state.cleaner is not None
        
        if button_disabled:
            st.info("ℹ️ All accepted rules have already been applied. Please accept new rules in the 'Review Rules' tab to clean again.")
        
        if st.button("🧹 Clean Data", type="primary", disabled=button_disabled):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📥 Step 1/5: Loading dataset...")
                progress_bar.progress(10)
                
                if st.session_state.cleaner and st.session_state.cleaner.df is not None:
                    full_df = st.session_state.cleaner.df.clone()
                    status_text.text(f"✅ Loaded {len(full_df):,} rows (from previous cleaning)")
                else:
                    full_profiler = DataProfiler(st.session_state.full_data_path)
                    full_profiler.load_data(sample_size=None)
                    full_df = full_profiler.df
                    status_text.text(f"✅ Loaded {len(full_df):,} rows (from original file)")
                
                progress_bar.progress(20)
                
                status_text.text("📊 Step 2/5: Loading baseline dataset for comparison...")
                progress_bar.progress(30)
                
                if st.session_state.full_original_profile is None:
                    if st.session_state.cleaner is None:
                        full_original_profiler = DataProfiler("")
                        full_original_profiler.df = full_df.clone()
                        full_original_profile = full_original_profiler.profile_all()
                        st.session_state.full_original_profile = full_original_profile
                        st.session_state.original_row_count = full_original_profile.get("row_count", 0)
                    else:
                        full_profiler = DataProfiler(st.session_state.full_data_path)
                        full_profiler.load_data(sample_size=None)
                        full_original_profiler = DataProfiler("")
                        full_original_profiler.df = full_profiler.df.clone()
                        full_original_profile = full_original_profiler.profile_all()
                        st.session_state.full_original_profile = full_original_profile
                        st.session_state.original_row_count = full_original_profile.get("row_count", 0)
                else:
                    full_original_profile = st.session_state.full_original_profile
                    if st.session_state.original_row_count is None:
                        st.session_state.original_row_count = full_original_profile.get("row_count", 0)
                
                status_text.text("✅ Baseline dataset ready for comparison")
                progress_bar.progress(40)
                
                status_text.text("🧹 Step 3/5: Applying cleaning rules...")
                progress_bar.progress(50)
                cleaner = DataCleaner(full_df)
                
                rules_to_apply = new_rules if new_rules else export_rules
                
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
                    for idx, (col_name, rule) in enumerate(all_rules):
                        rule_type = rule.get("type", "unknown")
                        action = rule.get("action", "unknown")
                        status_text.text(f"🔧 Applying rule {idx+1}/{total_rules}: {col_name} - {rule_type} ({action})")
                        progress = 50 + int((idx + 1) / total_rules * 30)
                        progress_bar.progress(progress)
                        cleaner.apply_rule(rule)
                    
                    status_text.text("✅ All rules applied")
                    progress_bar.progress(80)
                
                status_text.text("📊 Step 4/5: Re-profiling cleaned data...")
                progress_bar.progress(85)
                cleaning_stats = cleaner.get_cleaning_stats()
                cleaned_profiler = DataProfiler("")
                cleaned_profiler.df = cleaner.df.clone()
                cleaned_profile = cleaned_profiler.profile_all()
                status_text.text("✅ Cleaned data profiled")
                progress_bar.progress(90)
                
                status_text.text("📈 Step 5/5: Calculating metrics...")
                progress_bar.progress(95)
                
                all_applied_rules = cleaner.get_applied_rules_log()
                
                for rule_log in all_applied_rules:
                    rule = rule_log.get("rule", {})
                    if rule.get("action") == "clip_range":
                        col_name = rule.get("column")
                        rows_clipped = rule_log.get("rows_clipped")
                        
                        if rows_clipped is not None:
                            if col_name not in st.session_state.rows_clipped_history:
                                st.session_state.rows_clipped_history[col_name] = rows_clipped
                            elif st.session_state.rows_clipped_history.get(col_name, 0) == 0 and rows_clipped > 0:
                                st.session_state.rows_clipped_history[col_name] = rows_clipped
                            elif rows_clipped > st.session_state.rows_clipped_history.get(col_name, 0):
                                st.session_state.rows_clipped_history[col_name] = rows_clipped
                
                st.session_state.all_cleaning_logs.extend(all_applied_rules)
                
                metrics_calc = CleaningMetrics(
                    full_original_profile,
                    cleaned_profile,
                    cleaning_stats,
                    all_applied_rules
                )
                metrics = metrics_calc.calculate_metrics()
                
                if st.session_state.original_row_count is None:
                    st.session_state.original_row_count = full_original_profile.get("row_count", 0)
                
                original_rows = st.session_state.original_row_count
                current_cleaned_rows = cleaned_profile.get("row_count", 0)
                
                st.session_state.cumulative_rows_removed = original_rows - current_cleaned_rows
                
                original_total_nulls = 0
                for col in full_original_profile.get("columns", []):
                    null_pct = col.get("null_percentage", 0)
                    if null_pct > 0:
                        original_total_nulls += int(original_rows * null_pct / 100)
                
                current_total_nulls = 0
                for col in cleaned_profile.get("columns", []):
                    null_pct = col.get("null_percentage", 0)
                    if null_pct > 0:
                        current_total_nulls += int(current_cleaned_rows * null_pct / 100)
                
                st.session_state.cumulative_nulls_filled = original_total_nulls - current_total_nulls
                

                if "anomaly_metrics" in metrics and "anomaly_details" in metrics["anomaly_metrics"]:
                    original_cols = {c["name"]: c for c in full_original_profile.get("columns", [])}
                    
                    negative_values_map = {}
                    for anomaly in metrics["anomaly_metrics"]["anomaly_details"]:
                        if anomaly.get("type") == "negative_values":
                            col_name = anomaly.get("column")
                            fixed_count = anomaly.get("fixed", 0)
                            before_count = anomaly.get("before", 0)
                            after_count = anomaly.get("after", 0)

                            count_to_use = fixed_count if fixed_count > 0 else (before_count - after_count)
                            if count_to_use > 0:
                                negative_values_map[col_name] = count_to_use
                    
                    for anomaly in metrics["anomaly_metrics"]["anomaly_details"]:
                        if anomaly.get("type") == "out_of_range" and anomaly.get("rows_affected") is None:
                            col_name = anomaly.get("column")
                            
                            stored_value = st.session_state.rows_clipped_history.get(col_name)
                            
                            if stored_value is not None and stored_value > 0:
                                anomaly["rows_affected"] = stored_value
                                continue
                            

                            original_range = anomaly.get("original_range", [])
                            cleaned_range = anomaly.get("cleaned_range", [])
                            if len(original_range) == 2 and len(cleaned_range) == 2:
                                orig_min = original_range[0]
                                orig_max = original_range[1]
                                cleaned_min = cleaned_range[0]
                                cleaned_max = cleaned_range[1]
                                
                                rows_affected_by_range = 0
                                
                                if cleaned_min > orig_min:
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
                                    
                                    if negatives_dropped and col_name in negative_values_map:
                                        rows_affected_by_range = negative_values_map[col_name]
                                    elif col_name in negative_values_map:
                                        rows_affected_by_range = negative_values_map[col_name]
                                    else:
                                        min_increase = cleaned_min - orig_min
                                        orig_range_size = orig_max - orig_min if orig_max > orig_min else 1
                                        if orig_range_size > 0:
                                            range_change_ratio = min_increase / orig_range_size
                                            estimated_rows = int(full_original_profile.get("row_count", 0) * range_change_ratio * 0.15)
                                            rows_affected_by_range = max(estimated_rows, 0)
                                
                                if cleaned_max < orig_max:
                                    max_decrease = orig_max - cleaned_max
                                    orig_range_size = orig_max - orig_min if orig_max > orig_min else 1
                                    if orig_range_size > 0:
                                        range_change_ratio = max_decrease / orig_range_size
                                        estimated_rows = int(full_original_profile.get("row_count", 0) * range_change_ratio * 0.15)
                                        rows_affected_by_range += estimated_rows
                                
                                if rows_affected_by_range > 0:
                                    anomaly["rows_affected"] = rows_affected_by_range
                                    st.session_state.rows_clipped_history[col_name] = rows_affected_by_range
                                    continue
                            

                            if col_name in original_cols:
                                orig_col = original_cols[col_name]
                                if "min" in orig_col and "min" in anomaly.get("cleaned_range", []):
                                    orig_min = orig_col["min"]
                                    cleaned_min = anomaly.get("cleaned_range", [])[0]
                                    orig_max = orig_col["max"]
                                    cleaned_max = anomaly.get("cleaned_range", [])[1]
                                    

                                    orig_range = orig_max - orig_min if orig_max > orig_min else 1
                                    min_change = cleaned_min - orig_min if cleaned_min > orig_min else 0
                                    max_change = orig_max - cleaned_max if cleaned_max < orig_max else 0
                                    
                                    if (min_change > 0 or max_change > 0) and orig_range > 0:
                                        change_pct = max(min_change / orig_range, max_change / orig_range)
                                        estimated = int(full_original_profile.get("row_count", 0) * change_pct * 0.05)
                                        if estimated > 0:
                                            anomaly["rows_affected"] = estimated
                                            st.session_state.rows_clipped_history[col_name] = estimated
                
                total_rows_modified = 0
                if "anomaly_metrics" in metrics and "anomaly_details" in metrics["anomaly_metrics"]:
                    columns_counted = set()
                    
                    for anomaly in metrics["anomaly_metrics"]["anomaly_details"]:
                        col_name = anomaly.get("column")
                        anomaly_type = anomaly.get("type")
                        
                        if anomaly_type == "out_of_range":
                            rows_affected = anomaly.get("rows_affected")
                            if rows_affected is not None and rows_affected > 0:
                                columns_counted.add(col_name)
                                total_rows_modified += rows_affected
                        elif anomaly_type == "negative_values":
                            if col_name not in columns_counted:
                                fixed_count = anomaly.get("fixed", 0)
                                if fixed_count > 0:
                                    total_rows_modified += fixed_count
                
                st.session_state.cumulative_rows_modified = total_rows_modified
                
                if "anomaly_metrics" in metrics and "anomaly_details" in metrics["anomaly_metrics"]:
                    from datetime import datetime
                    current_pass_anomalies = metrics["anomaly_metrics"]["anomaly_details"]
                    for anomaly in current_pass_anomalies:
                        anomaly["cleaning_pass_timestamp"] = datetime.now().isoformat()
                    st.session_state.all_anomaly_details.extend(current_pass_anomalies)
                
                st.session_state.cleaner = cleaner
                st.session_state.cleaned_profile = cleaned_profile
                st.session_state.cleaning_metrics = metrics
                st.session_state.full_original_profile = full_original_profile
                st.session_state.metrics_calc = metrics_calc
                
                rules_to_track = new_rules if new_rules else export_rules
                for col_name, col_rules in rules_to_track.items():
                    for rule in col_rules:
                        rule_id = generate_unique_rule_id(col_name, rule)
                        st.session_state.applied_rules_tracking.add(rule_id)
                
                progress_bar.progress(100)
                status_text.text("✅ Cleaning complete!")
                st.success(f"✅ Data cleaned! {cleaning_stats['rows_removed']:,} rows removed in {cleaning_stats['processing_time_seconds']:.2f}s")
                
                st.rerun()
                
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                    
            except Exception as e:
                st.error(f"Error cleaning data: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        if st.session_state.cleaner and st.session_state.cleaning_metrics:
            st.markdown("---")
            st.subheader("Cleaning Results")
            
            metrics = st.session_state.cleaning_metrics
            summary = metrics["summary"]
            
            original_row_count = st.session_state.original_row_count or summary['original_rows']
            cumulative_rows_removed = st.session_state.cumulative_rows_removed
            cumulative_nulls_filled = st.session_state.cumulative_nulls_filled
            cumulative_rows_modified = st.session_state.cumulative_rows_modified
            current_cleaned_rows = summary['cleaned_rows']
            
            rows_removed_pct = (cumulative_rows_removed / original_row_count * 100) if original_row_count > 0 else 0
            rows_modified_pct = (cumulative_rows_modified / original_row_count * 100) if original_row_count > 0 else 0
            
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
                applied_count = len(st.session_state.applied_rules_tracking) if st.session_state.applied_rules_tracking else 0
                st.metric("Rules Applied", applied_count)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                perf = metrics["performance_metrics"]
                st.metric("Processing Time", f"{perf['processing_time_seconds']:.2f}s")
            
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
            
            all_anomalies = st.session_state.get("all_anomaly_details", [])
            
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
                    
                    anomalies_by_column = {}
                    for anomaly in all_anomalies:
                        col_name = anomaly.get("column", "unknown")
                        if col_name not in anomalies_by_column:
                            anomalies_by_column[col_name] = []
                        anomalies_by_column[col_name].append(anomaly)
                    
                    for col_name, col_anomalies in anomalies_by_column.items():
                        st.markdown(f"#### Column: `{col_name}`")
                        
                        for anomaly in col_anomalies:
                            anomaly_type = anomaly.get("type", "unknown")
                            
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
                                
                                if rows_affected is None or rows_affected == 0:
                                    stored_rows_clipped = st.session_state.get("rows_clipped_history", {}).get(col_name)
                                    if stored_rows_clipped is not None and stored_rows_clipped > 0:
                                        rows_affected = stored_rows_clipped
                                    else:
                                        for other_anomaly in all_anomalies:
                                            if (other_anomaly.get("type") == "negative_values" and 
                                                other_anomaly.get("column") == col_name):
                                                fixed_count = other_anomaly.get("fixed", 0)
                                                if fixed_count > 0 and len(original_range) == 2 and len(cleaned_range) == 2:
                                                    orig_min = original_range[0]
                                                    cleaned_min = cleaned_range[0]
                                                    if orig_min < 0 and cleaned_min >= 0:
                                                        rows_affected = fixed_count
                                                        break
                                
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
                                st.json(anomaly)
                            
                            st.markdown("---")
                        
                        st.markdown("---")
            
            
            st.markdown("### Preview Cleaned Data")
            preview_rows = 100
            cleaned_df = st.session_state.cleaner.df
            st.dataframe(cleaned_df.head(preview_rows), use_container_width=True)
            
            st.markdown("### Cleaning Log")
            all_logs = st.session_state.get("all_cleaning_logs", [])
            
            if not all_logs and st.session_state.cleaner:
                all_logs = st.session_state.cleaner.get_applied_rules_log()
            
            with st.expander("View Applied Rules Log"):
                if not all_logs:
                    st.info("No cleaning rules have been applied yet.")
                else:
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
                            
                            column = rule.get("column", "unknown")
                            rule_type = rule.get("type", "unknown")
                            action = rule.get("action", rule.get("description", "N/A"))
                            
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
        
        st.subheader("Export Cleaned Data")
        
        if not st.session_state.cleaner:
            st.info("👈 Please clean data first in the 'Clean Data' tab")
        else:
            export_format = st.radio(
                "Export Format", 
                ["Parquet", "CSV"], 
                horizontal=True,
                index=0,
                help="Parquet: Smaller, faster, better for large files (recommended)\nCSV: Universal, human-readable"
            )
            
            if st.button("💾 Export Cleaned Data", type="primary"):
                try:
                    original_name = st.session_state.last_file_name or "data"
                    base_name = Path(original_name).stem
                    extension = "parquet" if export_format == "Parquet" else "csv"
                    output_name = f"{base_name}_cleaned.{extension}"
                    
                    temp_output = Path(f"/tmp/{output_name}")
                    
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
                    
                    file_size_mb = temp_output.stat().st_size / (1024 * 1024)
                    file_size_gb = file_size_mb / 1024
                    
                    if file_size_mb > 500:
                        st.success(f"✅ Cleaned data exported successfully!")
                        st.info(f"📁 **File saved to:** `{temp_output}`\n\n**Size:** {file_size_gb:.2f} GB ({file_size_mb:.0f} MB)\n\n⚠️ File is too large for browser download. Please access it directly from the server.")
                        st.code(str(temp_output), language=None)
                    else:
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

