import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional

class StatisticsTable:
    def __init__(self, 
                 stats_list: List[Dict],
                 stat_group: str,
                 metrics: List[str],
                 directions: Optional[List[str]] = None,
                 experiment_names: Optional[List[str]] = None,
                 precision: int = 4):
        """
        Create a formatted table from statistics dictionaries.
        
        Parameters:
        -----------
        stats_list : List[Dict]
            List of statistics dictionaries from calculate_stretch_statistics
        stat_group : str
            Statistics group to extract (e.g., 'abs_diff', 'rel_diff')
        metrics : List[str]
            List of metrics to include (e.g., ['mean', 'median', 'max'])
        directions : List[str], optional
            List of directions to include (e.g., ['u_direction', 'v_direction'])
            If None, uses global_metrics
        experiment_names : List[str], optional
            Names for each experiment/dictionary in stats_list
        precision : int
            Number of decimal places for floating point numbers
        """
        self.stats_list = stats_list
        self.stat_group = stat_group
        self.metrics = metrics
        self.directions = directions if directions else ['global_metrics']
        self.precision = precision
        
        # Generate default experiment names if not provided
        if experiment_names is None:
            self.experiment_names = [f"Experiment {i+1}" 
                                   for i in range(len(stats_list))]
        else:
            if len(experiment_names) != len(stats_list):
                raise ValueError("Number of experiment names must match number of dictionaries")
            self.experiment_names = experiment_names
            
        self._create_dataframe()
    
    def _extract_values(self, stats_dict: Dict) -> List[float]:
        """Extract values from a single statistics dictionary."""
        values = []
        for direction in self.directions:
            if direction == 'global_metrics':
                # Handle global metrics differently as they might have a different structure
                for metric in self.metrics:
                    try:
                        if self.stat_group in stats_dict[direction]:
                            val = stats_dict[direction][self.stat_group].get(metric)
                        else:
                            val = stats_dict[direction].get(f"{self.stat_group}_{metric}")
                        values.append(val if val is not None else np.nan)
                    except (KeyError, AttributeError):
                        values.append(np.nan)
            else:
                # Handle directional metrics
                for metric in self.metrics:
                    try:
                        val = stats_dict[direction][self.stat_group][metric]
                        values.append(val if val is not None else np.nan)
                    except (KeyError, AttributeError):
                        values.append(np.nan)
        return values
    
    def _create_dataframe(self):
        """Create a pandas DataFrame from the statistics."""
        # Create column names
        if len(self.directions) > 1:
            columns = [f"{dir}_{metric}" 
                      for dir in self.directions 
                      for metric in self.metrics]
        else:
            columns = self.metrics
            
        # Extract values from all dictionaries
        data = [self._extract_values(stats) for stats in self.stats_list]
        
        # Create DataFrame
        self.df = pd.DataFrame(data, 
                             index=self.experiment_names,
                             columns=columns)
    
    def to_latex(self, caption: str = "", label: str = "") -> str:
        """Convert to LaTeX format."""
        latex_str = self.df.round(self.precision).to_latex(
            escape=False,
            caption=caption,
            label=label
        )
        return latex_str
    
    def to_markdown(self, filepath: Optional[str] = None) -> Optional[str]:
        """
        Convert to Markdown format and optionally save to file.
        
        Parameters:
        -----------
        filepath : str, optional
            If provided, saves the markdown to this file
            
        Returns:
        --------
        str or None
            Returns the markdown string if filepath is None, otherwise None
        """
        markdown_str = self.df.round(self.precision).to_markdown()
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_str)
        else:
            return markdown_str
    
    def to_csv(self, filepath: str):
        """Save as CSV file."""
        self.df.round(self.precision).to_csv(filepath)
    
    def to_excel(self, filepath: str):
        """Save as Excel file."""
        self.df.round(self.precision).to_excel(filepath)
    
    def to_html(self, 
                classes: List[str] = ["table", "table-striped"],
                filepath: Optional[str] = None) -> Optional[str]:
        """
        Convert to HTML format with optional CSS classes and file output.
        
        Parameters:
        -----------
        classes : List[str]
            CSS classes to apply to the table
        filepath : str, optional
            If provided, saves the HTML to this file
            
        Returns:
        --------
        str or None
            Returns the HTML string if filepath is None, otherwise None
        """
        html_str = self.df.round(self.precision).to_html(classes=classes)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_str)
        else:
            return html_str

# Example usage:
if __name__ == "__main__":
    # Example statistics dictionaries
    example_stats_list = [
        {
            'u_direction': {
                'abs_diff': {
                    'mean': 0.1234,
                    'median': 0.1111,
                    'max': 0.2222
                }
            },
            'v_direction': {
                'abs_diff': {
                    'mean': 0.2345,
                    'median': 0.2222,
                    'max': 0.3333
                }
            },
            'global_metrics': {
                'abs_diff_mean': 0.1789,
                'abs_diff_median': 0.1667,
                'abs_diff_max': 0.2778
            }
        },
        # Add more dictionaries for more experiments...
    ]
    
    # Create table for absolute differences
    table = StatisticsTable(
        stats_list=example_stats_list,
        stat_group='abs_diff',
        metrics=['mean', 'median', 'max'],
        directions=['u_direction', 'v_direction'],
        experiment_names=['Test 1'],
        precision=4
    )
    
    # Get output in different formats
    latex_output = table.to_latex(
        caption="Statistics for absolute differences",
        label="tab:abs_diff_stats"
    )
    markdown_output = table.to_markdown()
    
    # Print examples
    print("LaTeX format:")
    print(latex_output)
    print("\nMarkdown format:")
    print(markdown_output)
