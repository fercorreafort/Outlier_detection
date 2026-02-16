import kagglehub
import pandas as pd
import os
from typing import Dict, Set

def load_streaming_datasets() -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load titles and credits datasets from HBO, Amazon Prime, and Netflix.
    
    Returns:
        tuple: (titles_dict, credits_dict) where each dict maps platform name to DataFrame
    """
    credits = {}
    titles = {}
    
    datasets = {
        'hbo': "victorsoeiro/hbo-max-tv-shows-and-movies",
        'amazon': "victorsoeiro/amazon-prime-tv-shows-and-movies",
        'netflix': 'victorsoeiro/netflix-tv-shows-and-movies'
    }
    
    for platform, handle in datasets.items():
        try:
            # Download dataset
            path = kagglehub.dataset_download(handle)
            
            # Load CSV files
            for file in os.listdir(path):
                if not file.endswith('.csv'):
                    continue
                    
                file_path = os.path.join(path, file)
                
                if 'titles' in file.lower():
                    titles[platform] = pd.read_csv(file_path)
                elif 'credits' in file.lower():
                    credits[platform] = pd.read_csv(file_path)
                    
        except Exception as e:
            # Log error but continue with other platforms
            print(f"Warning: Failed to load {platform} dataset - {str(e)}")
            continue
    
    return titles, credits


def get_title_sets(titles: Dict[str, pd.DataFrame]) -> Dict[str, Set[str]]:
    """
    Extract sets of unique titles from each platform.
    
    Args:
        titles: Dictionary mapping platform names to title DataFrames
        
    Returns:
        Dictionary mapping platform names to sets of title strings
    """
    title_sets = {}
    
    for platform, df in titles.items():
        # Find title column
        title_col = next((col for col in ['title', 'name', 'Title', 'Name'] if col in df.columns), None)
        
        if title_col:
            # Normalize titles: strip whitespace, convert to lowercase for comparison
            title_sets[platform] = set(df[title_col].dropna().str.strip().str.lower())
    
    return title_sets


def print_basic_summary(titles: Dict[str, pd.DataFrame], credits: Dict[str, pd.DataFrame]) -> None:
    """
    Print basic dataset statistics.
    
    Args:
        titles: Dictionary mapping platform names to title DataFrames
        credits: Dictionary mapping platform names to credit DataFrames
    """
    print("\n" + "="*70)
    print("STREAMING PLATFORMS DATA OVERVIEW".center(70))
    print("="*70)
    
    # Overall summary
    total_titles = sum(len(df) for df in titles.values())
    total_credits = sum(len(df) for df in credits.values())
    
    print(f"\n📊 Total Records Loaded:")
    print(f"   • Title records: {total_titles:,}")
    print(f"   • Credit records: {total_credits:,}")
    print(f"   • Platforms: {len(titles)}")
    
    # Titles summary
    print(f"\n{'─'*70}")
    print("CONTENT LIBRARY SIZE BY PLATFORM")
    print(f"{'─'*70}")
    
    for platform in sorted(titles.keys()):
        df = titles[platform]
        title_col = next((col for col in ['title', 'name', 'Title', 'Name'] if col in df.columns), None)
        
        if title_col:
            unique_count = df[title_col].nunique()
            total_count = len(df)
            print(f"  📺 {platform.upper():10s}: {unique_count:,} unique titles ({total_count:,} total records)")
        else:
            print(f"  📺 {platform.upper():10s}: {len(df):,} total records")


def analyze_content_types(titles: Dict[str, pd.DataFrame]) -> None:
    """
    Analyze content type distribution (Movies vs Shows).
    
    Args:
        titles: Dictionary mapping platform names to title DataFrames
    """
    print("\n" + "="*70)
    print("CONTENT TYPE DISTRIBUTION".center(70))
    print("="*70)
    
    for platform in sorted(titles.keys()):
        df = titles[platform]
        type_col = next((col for col in ['type', 'Type', 'content_type'] if col in df.columns), None)
        
        if type_col:
            print(f"\n{platform.upper()}:")
            type_counts = df[type_col].value_counts()
            total = len(df)
            
            for content_type, count in type_counts.items():
                pct = (count / total * 100)
                bar_length = int(pct / 2)
                bar = '█' * bar_length
                print(f"  {content_type:10s} {bar} {count:,} ({pct:.1f}%)")


def analyze_title_overlap(title_sets: Dict[str, Set[str]]) -> None:
    """
    Analyze and print overlap statistics between platforms.
    
    Args:
        title_sets: Dictionary mapping platform names to sets of titles
    """
    platforms = sorted(title_sets.keys())
    
    if len(platforms) < 2:
        print("\nNot enough platforms to analyze overlap")
        return
    
    print("\n" + "="*70)
    print("CROSS-PLATFORM TITLE OVERLAP ANALYSIS".center(70))
    print("="*70)
    
    # Platform exclusives first (most interesting insight)
    print("\n🎯 PLATFORM EXCLUSIVITY:")
    print(f"{'─'*70}")
    
    for platform in platforms:
        # Find titles only on this platform
        other_platforms = [p for p in platforms if p != platform]
        exclusive = title_sets[platform].copy()
        
        for other in other_platforms:
            exclusive -= title_sets[other]
        
        total = len(title_sets[platform])
        pct_exclusive = (len(exclusive) / total * 100) if total else 0
        
        # Visual bar
        bar_length = int(pct_exclusive / 2)
        bar = '█' * bar_length
        
        print(f"\n  {platform.upper():10s} {bar}")
        print(f"    Exclusive: {len(exclusive):,} titles ({pct_exclusive:.1f}% of library)")
        print(f"    Shared:    {total - len(exclusive):,} titles ({100 - pct_exclusive:.1f}% of library)")
    
    # Pairwise comparisons
    print(f"\n{'─'*70}")
    print("🔗 PAIRWISE OVERLAP:")
    print(f"{'─'*70}")
    
    for i, platform1 in enumerate(platforms):
        for platform2 in platforms[i+1:]:
            set1 = title_sets[platform1]
            set2 = title_sets[platform2]
            
            overlap = set1 & set2
            overlap_count = len(overlap)
            
            pct_of_platform1 = (overlap_count / len(set1) * 100) if set1 else 0
            pct_of_platform2 = (overlap_count / len(set2) * 100) if set2 else 0
            
            print(f"\n  {platform1.upper()} ↔ {platform2.upper()}:")
            print(f"    Shared titles: {overlap_count:,}")
            print(f"    • {pct_of_platform1:.1f}% of {platform1.upper()}'s library")
            print(f"    • {pct_of_platform2:.1f}% of {platform2.upper()}'s library")
    
    # Three-way comparison (if applicable)
    if len(platforms) == 3:
        print(f"\n{'─'*70}")
        print("🌐 UNIVERSAL TITLES (Available on ALL platforms):")
        print(f"{'─'*70}")
        
        all_three = title_sets[platforms[0]] & title_sets[platforms[1]] & title_sets[platforms[2]]
        total_unique = len(set.union(*title_sets.values()))
        pct = (len(all_three) / total_unique * 100) if total_unique else 0
        
        print(f"\n  Total: {len(all_three):,} titles ({pct:.1f}% of all unique content)")
        
        if len(all_three) > 0:
            sample_size = min(10, len(all_three))
            print(f"\n  Sample ({sample_size} of {len(all_three)}):")
            for title in sorted(list(all_three)[:sample_size]):
                print(f"    • {title.title()}")


def analyze_market_reach(title_sets: Dict[str, Set[str]]) -> None:
    """
    Analyze total market coverage and content availability.
    
    Args:
        title_sets: Dictionary mapping platform names to sets of titles
    """
    print("\n" + "="*70)
    print("MARKET COVERAGE ANALYSIS".center(70))
    print("="*70)
    
    # Total unique titles across all platforms
    all_titles = set.union(*title_sets.values()) if title_sets else set()
    
    print(f"\n📈 Content Availability:")
    print(f"   • Total unique titles across all platforms: {len(all_titles):,}")
    
    # Calculate how many platforms each title appears on
    title_platform_count = {}
    for title in all_titles:
        count = sum(1 for titles in title_sets.values() if title in titles)
        title_platform_count[title] = count
    
    print(f"\n🎬 Title Distribution by Platform Availability:")
    print(f"{'─'*70}")
    
    for num_platforms in range(len(title_sets), 0, -1):
        count = sum(1 for c in title_platform_count.values() if c == num_platforms)
        pct = (count / len(all_titles) * 100) if all_titles else 0
        
        bar_length = int(pct / 2)
        bar = '█' * bar_length
        
        platform_text = "platform" if num_platforms == 1 else "platforms"
        print(f"  {num_platforms} {platform_text:9s} {bar} {count:,} titles ({pct:.1f}%)")


def analyze_credits(credits: Dict[str, pd.DataFrame]) -> None:
    """
    Analyze credits data - top talent and unique contributors.
    
    Args:
        credits: Dictionary mapping platform names to credit DataFrames
    """
    print("\n" + "="*70)
    print("TALENT & CREDITS ANALYSIS".center(70))
    print("="*70)
    
    for platform in sorted(credits.keys()):
        df = credits[platform]
        name_col = next((col for col in ['name', 'person_name', 'Name', 'actor', 'Person'] if col in df.columns), None)
        
        if name_col:
            unique_count = df[name_col].nunique()
            print(f"\n⭐ {platform.upper()}: {unique_count:,} unique contributors")
            
            # Get top 3
            top_3 = df[name_col].value_counts().head(3)
            print(f"   Top 3 most credited:")
            for rank, (name, count) in enumerate(top_3.items(), 1):
                print(f"     {rank}. {name}: {count:,} credits")
        else:
            print(f"\n⭐ {platform.upper()}: Name column not found")


if __name__ == "__main__":
    # Load datasets
    titles, credits = load_streaming_datasets()
    
    # 1. Basic overview
    print_basic_summary(titles, credits)
    
    # 2. Content type distribution
    analyze_content_types(titles)
    
    # 3. Overlap analysis
    title_sets = get_title_sets(titles)
    analyze_title_overlap(title_sets)
    
    # 4. Market coverage
    analyze_market_reach(title_sets)
    
    # 5. Credits analysis
    analyze_credits(credits)
    
    print("\n" + "="*70)
    print("Analysis Complete!".center(70))
    print("="*70 + "\n")