"""
Dwell Time Analysis for User Study
Analyzes user engagement duration for PASSAGE_SELECTION and OPEN_DOC events.

This script answers RQ 5.3: How long do users engage with documents?
"""

import pandas as pd
import numpy as np
import re
import argparse


def load_data_from_sql(file_path):
    """
    Parse INSERT statements from SQL file and load data.
    
    Args:
        file_path: Path to SQL dump file
    
    Returns:
        DataFrame with parsed event log data
    """
    print(f"Loading data from SQL file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for INSERT statements
        if 'INSERT' not in content.upper():
            raise Exception("No INSERT statements found in SQL file")
        
        print("Found INSERT statements, parsing...")
        data = []
        
        # Find all INSERT statements for logs table
        # Pattern: INSERT INTO `logs` VALUES (...)
        insert_pattern = r"INSERT INTO.*?logs.*?VALUES\s*(.*?);"
        matches = re.findall(insert_pattern, content, re.IGNORECASE | re.DOTALL)
        
        print(f"Found {len(matches)} INSERT statements")
        
        for match in matches:
            # Extract all row data: (value1, value2, ...), (value1, value2, ...)
            rows_pattern = r'\((.*?)\)(?=,\s*\(|\s*$)'
            rows = re.findall(rows_pattern, match, re.DOTALL)
            
            for row in rows:
                # Split values in each row
                # Handle content with commas inside quotes
                values = []
                current_value = ""
                in_quotes = False
                quote_char = None
                
                for char in row + ',':
                    if char in ('"', "'") and (not in_quotes or char == quote_char):
                        if not in_quotes:
                            in_quotes = True
                            quote_char = char
                        else:
                            in_quotes = False
                            quote_char = None
                    elif char == ',' and not in_quotes:
                        values.append(current_value.strip().strip('"').strip("'"))
                        current_value = ""
                    else:
                        current_value += char
                
                # Clean last value
                if current_value.strip():
                    values.append(current_value.strip().strip('"').strip("'"))
                
                # Keep only rows with 10 fields (valid data rows)
                if len(values) == 10:
                    data.append(values)
        
        if not data:
            raise Exception("Unable to extract data from SQL file")
        
        print(f"Successfully extracted {len(data)} rows")
        
        # Create DataFrame
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
        
        print(f"Data loaded successfully! DataFrame shape: {df.shape}")
        print(f"\nData preview:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error loading SQL file: {e}")
        return None


def load_data(file_path):
    """
    Smart data loader: automatically detects SQL or CSV format.
    
    Args:
        file_path: Path to data file (.sql or .csv)
    
    Returns:
        DataFrame with event log data
    """
    if file_path.endswith('.sql'):
        return load_data_from_sql(file_path)
    elif file_path.endswith('.csv'):
        print(f"Loading data from CSV file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully! DataFrame shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
    else:
        print("Unsupported file format. Please use .sql or .csv file")
        return None


def analyze_event_duration(df, output_csv=None):
    """
    Analyze dwell time statistics for PASSAGE_SELECTION and OPEN_DOC events.
    
    Args:
        df: DataFrame with event log data
        output_csv: Optional path to save results as CSV
    
    Returns:
        Tuple of (results_dict, passage_selection_df, open_doc_df)
    """
    if df is None:
        print("Error: DataFrame is empty")
        return None, None, None
    
    # Ensure event_type is string
    df['event_type'] = df['event_type'].astype(str)
    
    # Ensure duration is numeric
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    
    # Filter PASSAGE_SELECTION events
    passage_selection = df[df['event_type'] == 'PASSAGE_SELECTION'].copy()
    
    # Filter OPEN_DOC events
    open_doc = df[df['event_type'] == 'OPEN_DOC'].copy()
    
    # Calculate statistics
    results = {}
    
    print("\n" + "="*80)
    print("User Study - Event Dwell Time Analysis")
    print("="*80)
    
    # PASSAGE_SELECTION statistics
    if len(passage_selection) > 0:
        ps_duration_ms = passage_selection['duration']
        ps_duration_sec = ps_duration_ms / 1000
        
        results['passage_selection'] = {
            'event_count': len(passage_selection),
            'mean_duration_sec': ps_duration_sec.mean(),
            'median_duration_sec': ps_duration_sec.median(),
            'mean_duration_ms': ps_duration_ms.mean(),
            'median_duration_ms': ps_duration_ms.median(),
            'std_duration_sec': ps_duration_sec.std(),
            'min_duration_sec': ps_duration_sec.min(),
            'max_duration_sec': ps_duration_sec.max(),
            'q25_duration_sec': ps_duration_sec.quantile(0.25),
            'q75_duration_sec': ps_duration_sec.quantile(0.75)
        }
        
        print("\n📊 PASSAGE_SELECTION Event Analysis:")
        print(f"   Total events:     {len(passage_selection):,}")
        print(f"   Mean dwell time:  {ps_duration_sec.mean():.2f} sec ({ps_duration_ms.mean():.2f} ms)")
        print(f"   Median dwell time: {ps_duration_sec.median():.2f} sec ({ps_duration_ms.median():.2f} ms)")
        print(f"   Std deviation:    {ps_duration_sec.std():.2f} sec")
        print(f"   Min:              {ps_duration_sec.min():.2f} sec")
        print(f"   Max:              {ps_duration_sec.max():.2f} sec")
        print(f"   25th percentile:  {ps_duration_sec.quantile(0.25):.2f} sec")
        print(f"   75th percentile:  {ps_duration_sec.quantile(0.75):.2f} sec")
    else:
        print("\n⚠️  Warning: No PASSAGE_SELECTION events found")
        results['passage_selection'] = None
    
    # OPEN_DOC statistics
    if len(open_doc) > 0:
        od_duration_ms = open_doc['duration']
        od_duration_sec = od_duration_ms / 1000
        
        results['open_doc'] = {
            'event_count': len(open_doc),
            'mean_duration_sec': od_duration_sec.mean(),
            'median_duration_sec': od_duration_sec.median(),
            'mean_duration_ms': od_duration_ms.mean(),
            'median_duration_ms': od_duration_ms.median(),
            'std_duration_sec': od_duration_sec.std(),
            'min_duration_sec': od_duration_sec.min(),
            'max_duration_sec': od_duration_sec.max(),
            'q25_duration_sec': od_duration_sec.quantile(0.25),
            'q75_duration_sec': od_duration_sec.quantile(0.75)
        }
        
        print("\n📄 OPEN_DOC Event Analysis:")
        print(f"   Total events:     {len(open_doc):,}")
        print(f"   Mean dwell time:  {od_duration_sec.mean():.2f} sec ({od_duration_ms.mean():.2f} ms)")
        print(f"   Median dwell time: {od_duration_sec.median():.2f} sec ({od_duration_ms.median():.2f} ms)")
        print(f"   Std deviation:    {od_duration_sec.std():.2f} sec")
        print(f"   Min:              {od_duration_sec.min():.2f} sec")
        print(f"   Max:              {od_duration_sec.max():.2f} sec")
        print(f"   25th percentile:  {od_duration_sec.quantile(0.25):.2f} sec")
        print(f"   75th percentile:  {od_duration_sec.quantile(0.75):.2f} sec")
    else:
        print("\n⚠️  Warning: No OPEN_DOC events found")
        results['open_doc'] = None
    
    print("\n" + "="*80 + "\n")
    
    # Export to CSV if requested
    if output_csv:
        export_results_to_csv(results, output_csv)
    
    return results, passage_selection, open_doc


def export_results_to_csv(results, filename='duration_summary.csv'):
    """
    Export analysis results to CSV file.
    
    Args:
        results: Dictionary with analysis results
        filename: Output CSV file path
    
    Returns:
        DataFrame with exported results
    """
    data = []
    for event_type, stats in results.items():
        if stats is not None:
            row = {'event_type': event_type}
            row.update(stats)
            data.append(row)
    
    if data:
        results_df = pd.DataFrame(data)
        results_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ Analysis results saved to '{filename}'")
        return results_df
    return None


def main(data_file, output_csv=None):
    """
    Main workflow for dwell time analysis.
    
    Args:
        data_file: Path to event log file (SQL or CSV)
        output_csv: Optional path to save summary results
    """
    print("="*80)
    print("Dwell Time Analysis - User Engagement Duration")
    print("="*80)
    
    # Load data
    df = load_data(data_file)
    
    if df is not None:
        # Run analysis
        results, ps_df, od_df = analyze_event_duration(df, output_csv=output_csv)
        
        print("\n" + "="*80)
        print("Analysis Complete!")
        print("="*80)
        
        # Summary
        if results.get('passage_selection'):
            print(f"\n📊 PASSAGE_SELECTION Summary:")
            print(f"   Events: {results['passage_selection']['event_count']:,}")
            print(f"   Mean: {results['passage_selection']['mean_duration_sec']:.2f} sec")
            print(f"   Median: {results['passage_selection']['median_duration_sec']:.2f} sec")
        
        if results.get('open_doc'):
            print(f"\n📄 OPEN_DOC Summary:")
            print(f"   Events: {results['open_doc']['event_count']:,}")
            print(f"   Mean: {results['open_doc']['mean_duration_sec']:.2f} sec")
            print(f"   Median: {results['open_doc']['median_duration_sec']:.2f} sec")
        
        return results
    else:
        print("\nData loading failed. Please check file path and format.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze dwell time for user study events (RQ 5.3)"
    )
    
    parser.add_argument(
        "data_file",
        type=str,
        nargs="?",
        default="click_logs.sql",
        help="Path to event log file (.sql or .csv)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV file to save summary statistics"
    )
    
    args = parser.parse_args()
    
    main(args.data_file, output_csv=args.output)
