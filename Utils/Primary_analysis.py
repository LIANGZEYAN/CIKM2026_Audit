"""
Click Data Analysis Tool
Analyzes user interaction logs from SQL/CSV files and generates visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import os


def load_data(file_path, is_sql=True):
    """
    Load data from SQL file or CSV file
    
    Args:
        file_path: Path to the data file (.sql or .csv)
        is_sql: Boolean indicating if the file is SQL format
    
    Returns:
        DataFrame containing the loaded data
    """
    if is_sql:
        try:
            with open(file_path, 'r') as f:
                # Check if file contains INSERT statements
                content = f.read()
                if 'INSERT' in content:
                    # Data is in SQL file, need to parse
                    print("SQL file contains INSERT statements, attempting to parse...")
                    data = []
                    # Simple processing of INSERT statements
                    insert_statements = [s for s in content.split(';') if 'INSERT INTO' in s and 'logs' in s]
                    
                    for statement in insert_statements:
                        values_part = statement.split('VALUES')[1] if 'VALUES' in statement else ""
                        rows = values_part.strip(';').strip().split('),(')
                        
                        for row in rows:
                            row = row.strip('(').strip(')').strip()
                            values = row.split(',')
                            cleaned_values = []
                            for v in values:
                                cleaned_values.append(v.strip().strip("'").strip('"'))
                            
                            if len(cleaned_values) == 10:  # Assuming 10 columns
                                data.append(cleaned_values)
                    
                    if data:
                        df = pd.DataFrame(data, columns=['id', 'user_id', 'qid', 'docno', 'event_type', 
                                                       'start_idx', 'end_idx', 'duration', 'pass_flag', 'timestamp'])
                        # Convert data types
                        df['id'] = pd.to_numeric(df['id'], errors='coerce')
                        df['qid'] = pd.to_numeric(df['qid'], errors='coerce')
                        df['start_idx'] = pd.to_numeric(df['start_idx'], errors='coerce')
                        df['end_idx'] = pd.to_numeric(df['end_idx'], errors='coerce')
                        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
                        df['pass_flag'] = pd.to_numeric(df['pass_flag'], errors='coerce')
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        return df
                    else:
                        raise Exception("Unable to extract data from SQL file")
                else:
                    print("SQL file doesn't contain INSERT statements, attempting to connect to database...")
                    raise Exception("Database connection information required")
        except Exception as e:
            print(f"Error loading data from SQL: {e}")
            print("Please try exporting the data to CSV format")
            return None
    else:
        # Load from CSV
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            return None


def analyze_click_data(df, df_label=None):
    """
    Analyze click data, return user activity summary and source comparison results
    
    Args:
        df: DataFrame containing click event logs
        df_label: Optional DataFrame containing document labels and sources
    
    Returns:
        Tuple of (processed_df, user_summary, source_comparison)
    """
    if df is None:
        return None, None, None
    
    # Ensure event_type is string type
    df['event_type'] = df['event_type'].astype(str)
    
    # 1. User activity analysis
    print("Analyzing user activity data...")
    
    # Number of clicked documents per user
    user_clicks = df[df['event_type'] == 'PASSAGE_SELECTION'].groupby('user_id').size()
    
    # Average click duration per user (milliseconds to seconds)
    user_click_duration = df[df['event_type'] == 'PASSAGE_SELECTION'].groupby('user_id')['duration'].mean() / 1000
    
    # Number of documents opened per user
    user_docs_opened = df[df['event_type'] == 'OPEN_DOC'].groupby('user_id').size()
    
    # Number of documents canceled per user
    user_docs_canceled = df[df['event_type'] == 'CANCEL_DOC'].groupby('user_id').size()
    
    # Build user activity summary
    user_summary = pd.DataFrame({
        'clicks': user_clicks,
        'avg_duration': user_click_duration,
        'docs_opened': user_docs_opened,
        'docs_canceled': user_docs_canceled
    }).fillna(0)
    
    # 2. Source analysis (if label data is provided)
    source_comparison = None
    if df_label is not None:
        print("Analyzing source data...")
        
        # Ensure qid and docno types are consistent
        df['qid'] = df['qid'].astype(str)
        df['docno'] = df['docno'].astype(str)
        df_label['qid'] = df_label['qid'].astype(str)
        df_label['docno'] = df_label['docno'].astype(str)
        
        # Filter click events
        click_df = df[df['event_type'] == 'PASSAGE_SELECTION'].copy()
        
        # Group by qid and docno to count clicks per document
        doc_clicks = click_df.groupby(['qid', 'docno']).size().reset_index(name='clicks')
        
        # Merge label data to get source information
        doc_clicks_with_source = pd.merge(
            doc_clicks, 
            df_label[['qid', 'docno', 'source']], 
            on=['qid', 'docno'], 
            how='inner'
        )
        
        # Calculate total clicks and click rates by source
        total_clicks_by_source = doc_clicks_with_source.groupby('source')['clicks'].sum()
        total_clicks = total_clicks_by_source.sum()
        click_rate_by_source = [(clicks / total_clicks * 100) for clicks in total_clicks_by_source]
        click_rate_by_source = pd.Series(click_rate_by_source, index=total_clicks_by_source.index).round(2)
        
        # Analyze comparison between ColBERT and SBR documents
        colbert_docs = doc_clicks_with_source[doc_clicks_with_source['source'] == 'top from biased']
        sbr_docs = doc_clicks_with_source[doc_clicks_with_source['source'] == 'top from debiased']
        
        # Count number of queries with SBR documents
        queries_with_sbr = sbr_docs['qid'].nunique()
        
        # For each query, find the minimum clicks for ColBERT documents
        min_colbert_clicks_by_query = colbert_docs.groupby('qid')['clicks'].min().reset_index()
        min_colbert_clicks_by_query.rename(columns={'clicks': 'min_colbert_clicks'}, inplace=True)
        
        # For each query, find the maximum clicks for ColBERT documents
        max_colbert_clicks_by_query = colbert_docs.groupby('qid')['clicks'].max().reset_index()
        max_colbert_clicks_by_query.rename(columns={'clicks': 'max_colbert_clicks'}, inplace=True)
        
        # For each query, find the average clicks for ColBERT documents
        avg_colbert_clicks_by_query = colbert_docs.groupby('qid')['clicks'].mean().reset_index()
        avg_colbert_clicks_by_query.rename(columns={'clicks': 'avg_colbert_clicks'}, inplace=True)
        
        # Merge minimum, maximum, and average ColBERT clicks for each query
        colbert_stats_by_query = pd.merge(min_colbert_clicks_by_query, max_colbert_clicks_by_query, on='qid', how='outer')
        colbert_stats_by_query = pd.merge(colbert_stats_by_query, avg_colbert_clicks_by_query, on='qid', how='outer')
        
        # For each query, get SBR document clicks
        sbr_clicks_by_query = sbr_docs.groupby('qid')['clicks'].apply(list).reset_index()
        sbr_clicks_by_query.rename(columns={'clicks': 'sbr_clicks'}, inplace=True)
        
        # Merge ColBERT and SBR data
        comparison_df = pd.merge(colbert_stats_by_query, sbr_clicks_by_query, on='qid', how='outer')
        
        # For queries without SBR documents, set sbr_clicks to empty list
        comparison_df['sbr_clicks'] = comparison_df['sbr_clicks'].apply(lambda x: x if isinstance(x, list) else [])
        
        # Compare each SBR document click count with ColBERT documents
        def compare_sbr_with_colbert(row):
            if not row['sbr_clicks']:
                return {'above_max': 0, 'between': 0, 'below_min': 0}
            
            min_colbert = row['min_colbert_clicks'] if pd.notna(row['min_colbert_clicks']) else float('inf')
            max_colbert = row['max_colbert_clicks'] if pd.notna(row['max_colbert_clicks']) else float('-inf')
            
            above_max = sum(1 for clicks in row['sbr_clicks'] if clicks > max_colbert)
            below_min = sum(1 for clicks in row['sbr_clicks'] if clicks < min_colbert)
            between = len(row['sbr_clicks']) - above_max - below_min
            
            return {'above_max': above_max, 'between': between, 'below_min': below_min}
        
        # Apply comparison function
        comparison_results = comparison_df.apply(compare_sbr_with_colbert, axis=1, result_type='expand')
        
        # Calculate summary statistics
        total_sbr_docs = sum(len(x) for x in comparison_df['sbr_clicks'])
        total_above_max = comparison_results['above_max'].sum()
        total_between = comparison_results['between'].sum()
        total_below_min = comparison_results['below_min'].sum()
        
        # Build source comparison summary
        source_comparison = {
            'total_clicks_by_source': total_clicks_by_source,
            'click_rate_by_source': click_rate_by_source,
            'queries_with_sbr': queries_with_sbr,
            'total_sbr_docs': total_sbr_docs,
            'sbr_above_max_colbert': total_above_max,
            'sbr_between_colbert': total_between,
            'sbr_below_min_colbert': total_below_min,
            'comparison_details': comparison_df
        }
        
        # Print summary
        print("\n=== Source Analysis Summary ===")
        print(f"Total clicks by source:\n{total_clicks_by_source}")
        print(f"\nClick rate by source (%):\n{click_rate_by_source}")
        print(f"\nNumber of queries with SBR documents: {queries_with_sbr}")
        print(f"Total number of SBR documents: {total_sbr_docs}")
        print(f"  - SBR docs with clicks > max ColBERT: {total_above_max}")
        print(f"  - SBR docs with clicks between min and max ColBERT: {total_between}")
        print(f"  - SBR docs with clicks < min ColBERT: {total_below_min}")
    
    return df, user_summary, source_comparison


def analyze_cancel_behaviors(df):
    """
    Analyze document cancel behaviors
    
    Args:
        df: DataFrame containing event logs
    
    Returns:
        Dictionary containing cancel behavior statistics
    """
    if df is None:
        return None
    
    print("Analyzing document cancel behaviors...")
    
    # Get all cancel events
    cancel_events = df[df['event_type'] == 'CANCEL_DOC'].copy()
    total_cancels = len(cancel_events)
    
    if total_cancels == 0:
        print("No cancel events found in the data.")
        return None
    
    # For each cancel event, check if there was a previous PASSAGE_SELECTION for the same document
    cancels_with_selection = 0
    cancels_without_selection = 0
    
    # Sort by timestamp to ensure chronological order
    df_sorted = df.sort_values('timestamp')
    
    for idx, cancel_row in cancel_events.iterrows():
        user_id = cancel_row['user_id']
        qid = cancel_row['qid']
        docno = cancel_row['docno']
        cancel_time = cancel_row['timestamp']
        
        # Find OPEN_DOC event for this document
        open_event = df_sorted[
            (df_sorted['user_id'] == user_id) &
            (df_sorted['qid'] == qid) &
            (df_sorted['docno'] == docno) &
            (df_sorted['event_type'] == 'OPEN_DOC') &
            (df_sorted['timestamp'] < cancel_time)
        ]
        
        if not open_event.empty:
            open_time = open_event.iloc[-1]['timestamp']
            
            # Check for PASSAGE_SELECTION between OPEN_DOC and CANCEL_DOC
            had_selection = df_sorted[
                (df_sorted['user_id'] == user_id) &
                (df_sorted['qid'] == qid) &
                (df_sorted['docno'] == docno) &
                (df_sorted['event_type'] == 'PASSAGE_SELECTION') &
                (df_sorted['timestamp'] > open_time) &
                (df_sorted['timestamp'] < cancel_time)
            ]
            
            if not had_selection.empty:
                cancels_with_selection += 1
            else:
                cancels_without_selection += 1
        else:
            # If no corresponding OPEN_DOC found, count as without selection
            cancels_without_selection += 1
    
    cancel_summary = {
        'total_cancels': total_cancels,
        'cancels_with_selection': cancels_with_selection,
        'cancels_without_selection': cancels_without_selection,
        'pct_with_selection': (cancels_with_selection / total_cancels * 100) if total_cancels > 0 else 0,
        'pct_without_selection': (cancels_without_selection / total_cancels * 100) if total_cancels > 0 else 0
    }
    
    print(f"Cancel behavior analysis complete. Found {total_cancels} total cancels.")
    print(f"  - Cancels after content selection: {cancels_with_selection} ({cancel_summary['pct_with_selection']:.2f}%)")
    print(f"  - Cancels after viewing without selection: {cancels_without_selection} ({cancel_summary['pct_without_selection']:.2f}%)")
    
    return cancel_summary


def create_focused_visualizations(df, user_summary, source_comparison):
    """
    Create visualizations for user activity and source comparison
    
    Args:
        df: DataFrame containing event logs
        user_summary: DataFrame with user activity statistics
        source_comparison: Dictionary with source comparison data
    """
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Set global style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 1. User Activity Visualization
    if user_summary is not None:
        print("Creating user activity visualization...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('User Activity Analysis', fontsize=20, fontweight='bold', y=0.995)
        
        # Clicks distribution
        axes[0, 0].hist(user_summary['clicks'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution of Clicks per User', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Number of Clicks', fontsize=12)
        axes[0, 0].set_ylabel('Number of Users', fontsize=12)
        axes[0, 0].axvline(user_summary['clicks'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {user_summary["clicks"].mean():.2f}')
        axes[0, 0].legend()
        
        # Average duration distribution
        axes[0, 1].hist(user_summary['avg_duration'], bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Distribution of Average Click Duration per User', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Average Duration (seconds)', fontsize=12)
        axes[0, 1].set_ylabel('Number of Users', fontsize=12)
        axes[0, 1].axvline(user_summary['avg_duration'].mean(), color='darkred', linestyle='dashed', linewidth=2, label=f'Mean: {user_summary["avg_duration"].mean():.2f}s')
        axes[0, 1].legend()
        
        # Documents opened distribution
        axes[1, 0].hist(user_summary['docs_opened'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Distribution of Documents Opened per User', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Number of Documents Opened', fontsize=12)
        axes[1, 0].set_ylabel('Number of Users', fontsize=12)
        axes[1, 0].axvline(user_summary['docs_opened'].mean(), color='darkgreen', linestyle='dashed', linewidth=2, label=f'Mean: {user_summary["docs_opened"].mean():.2f}')
        axes[1, 0].legend()
        
        # Documents canceled distribution
        axes[1, 1].hist(user_summary['docs_canceled'], bins=20, color='plum', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Distribution of Documents Canceled per User', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Number of Documents Canceled', fontsize=12)
        axes[1, 1].set_ylabel('Number of Users', fontsize=12)
        axes[1, 1].axvline(user_summary['docs_canceled'].mean(), color='purple', linestyle='dashed', linewidth=2, label=f'Mean: {user_summary["docs_canceled"].mean():.2f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/user_activity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Source Comparison Visualization
    if source_comparison is not None:
        print("Creating source comparison visualization...")
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Document Source Comparison: ColBERT vs SBR', fontsize=20, fontweight='bold')
        
        # Click distribution by source (pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        wedges, texts, autotexts = ax1.pie(
            source_comparison['total_clicks_by_source'],
            labels=source_comparison['total_clicks_by_source'].index,
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=90,
            textprops={'fontsize': 11}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax1.set_title('Click Distribution by Source', fontsize=14, fontweight='bold', pad=20)
        
        # Click count by source (bar chart)
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(
            range(len(source_comparison['total_clicks_by_source'])),
            source_comparison['total_clicks_by_source'].values,
            color=colors_pie,
            edgecolor='black',
            linewidth=1.5
        )
        ax2.set_xticks(range(len(source_comparison['total_clicks_by_source'])))
        ax2.set_xticklabels(source_comparison['total_clicks_by_source'].index, rotation=15, ha='right')
        ax2.set_ylabel('Number of Clicks', fontsize=12)
        ax2.set_title('Total Clicks by Source', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # SBR document performance comparison
        ax3 = fig.add_subplot(gs[1, :])
        categories = ['Above Max\nColBERT', 'Between Min-Max\nColBERT', 'Below Min\nColBERT']
        values = [
            source_comparison['sbr_above_max_colbert'],
            source_comparison['sbr_between_colbert'],
            source_comparison['sbr_below_min_colbert']
        ]
        colors_bar = ['#2ecc71', '#f39c12', '#e74c3c']
        
        bars = ax3.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1.5, width=0.6)
        ax3.set_ylabel('Number of SBR Documents', fontsize=12)
        ax3.set_title('SBR Document Click Performance Compared to ColBERT', fontsize=14, fontweight='bold', pad=20)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels and percentages
        total_sbr = sum(values)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            percentage = (val / total_sbr * 100) if total_sbr > 0 else 0
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.savefig('visualizations/source_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Visualizations created successfully, saved in the 'visualizations' directory.")


def visualize_cancel_behaviors(cancel_summary):
    """
    Create visualization for cancel behavior analysis
    
    Args:
        cancel_summary: Dictionary containing cancel behavior statistics
    """
    print("Creating cancel behavior visualization...")
    
    # Extract data
    total_cancels = cancel_summary['total_cancels']
    cancels_with_selection = cancel_summary['cancels_with_selection']
    cancels_without_selection = cancel_summary['cancels_without_selection']
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.5, 1.2, 2], hspace=0.35, wspace=0.3)
    
    # Main title
    fig.suptitle('Document Cancel Behavior Analysis', fontsize=22, fontweight='bold', y=0.96)
    
    # Summary statistics table
    ax_table = plt.subplot(gs[0, :])
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table_data = [
        ['Total Cancel Events', f'{total_cancels}'],
        ['Cancels After Selection', f'{cancels_with_selection} ({cancels_with_selection/total_cancels*100:.1f}%)'],
        ['Cancels Without Selection', f'{cancels_without_selection} ({cancels_without_selection/total_cancels*100:.1f}%)']
    ]
    
    table = ax_table.table(cellText=table_data, cellLoc='left',
                          colWidths=[0.4, 0.3], loc='center',
                          bbox=[0.15, 0, 0.7, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    # Style table cells
    for i in range(len(table_data)):
        cell = table[(i, 0)]
        cell.set_facecolor('#e8f4f8')
        cell.set_text_props(weight='bold')
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
        
        cell = table[(i, 1)]
        cell.set_facecolor('#ffffff')
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
    
    ax_table.set_title('Summary Statistics', fontsize=16, pad=15, fontweight='bold')
    
    # Explanation text
    ax_explanation = plt.subplot(gs[1, :])
    ax_explanation.axis('off')
    
    explanation_text = (
        f"Analysis of {total_cancels} document cancel events:\n\n"
        f"• After Selection ({cancels_with_selection} events, {cancels_with_selection/total_cancels*100:.1f}%): "
        f"Users canceled the document after selecting content, suggesting they found relevant information.\n\n"
        f"• Without Selection ({cancels_without_selection} events, {cancels_without_selection/total_cancels*100:.1f}%): "
        f"Users canceled without selecting any content, suggesting they quickly "
        f"determined document irrelevance without deeper interaction."
    )
    
    ax_explanation.text(0.5, 0.5, explanation_text, 
                      ha='center', va='center', fontsize=14,
                      bbox=dict(facecolor='#f0f8ff', alpha=0.5, boxstyle='round,pad=0.8'))
    
    # Pie chart
    ax_pie = plt.subplot(gs[2, 0])
    labels = ['After Selection', 'Without Selection']
    sizes = [cancels_with_selection, cancels_without_selection]
    colors = ['#5ec962', '#f85a3e']
    explode = (0.1, 0)
    
    wedges, texts, autotexts = ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax_pie.axis('equal')
    
    for text in texts:
        text.set_fontsize(14)
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_color('white')
    
    ax_pie.set_title('Distribution of Cancel Behaviors', fontsize=16, pad=10)
    
    # Bar chart
    ax_bar = plt.subplot(gs[2, 1])
    categories = ['After Selection', 'Without Selection']
    values = [cancels_with_selection, cancels_without_selection]
    
    bars = ax_bar.bar(categories, values, color=['#5ec962', '#f85a3e'])
    
    # Calculate Y-axis upper limit
    y_max = max(values) * 1.15
    ax_bar.set_ylim(0, y_max)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height == max(values):
            # Place label inside the tallest bar
            ax_bar.annotate(f'{height}\n({height/total_cancels*100:.1f}%)',
                        xy=(bar.get_x() + bar.get_width() / 2, height * 0.9),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='center',
                        fontsize=14,
                        color='white')
        else:
            # Place label on top of shorter bar
            ax_bar.annotate(f'{height}\n({height/total_cancels*100:.1f}%)',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=14)
    
    ax_bar.set_title('Number of Cancels by Behavior Type', fontsize=16, pad=25)
    ax_bar.set_ylabel('Number of Cancel Events', fontsize=14)
    ax_bar.tick_params(axis='both', which='major', labelsize=12)
    
    plt.subplots_adjust(hspace=0.15, wspace=0.25, top=0.92, bottom=0.08, left=0.12, right=0.88)
    plt.savefig('visualizations/cancel_behavior_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Cancel behavior visualization created successfully.")
    return


def main(click_data_path, label_data_path=None):
    """
    Main analysis function
    
    Args:
        click_data_path: Path to click event data file (.sql or .csv)
        label_data_path: Optional path to document label data (.csv)
    """
    # Load click data
    print(f"Loading click data from {click_data_path}...")
    is_sql = click_data_path.lower().endswith('.sql')
    df = load_data(click_data_path, is_sql)
    
    if df is None:
        print("Unable to load click data, analysis terminated.")
        return
    
    # Load label data (if provided)
    df_label = None
    if label_data_path and os.path.exists(label_data_path):
        print(f"Loading label data from {label_data_path}...")
        df_label = pd.read_csv(label_data_path)
    
    # Analyze data
    df, user_summary, source_comparison = analyze_click_data(df, df_label)
    
    # Analyze cancel behaviors
    cancel_behavior_summary = analyze_cancel_behaviors(df)
    
    # Save analysis results
    if user_summary is not None:
        user_summary.to_csv("user_summary.csv")
        print("User activity summary has been saved to 'user_summary.csv'")
    
    # Create visualizations
    create_focused_visualizations(df, user_summary, source_comparison)
    
    # Create cancel behavior visualization
    if cancel_behavior_summary is not None:
        visualize_cancel_behaviors(cancel_behavior_summary)
        
    return


if __name__ == "__main__":
    # Example usage
    click_data_path = "click_logs.sql"  # or "click_logs.csv"
    label_data_path = "document_labels.csv"  # CSV file containing document labels
    
    main(click_data_path, label_data_path)
