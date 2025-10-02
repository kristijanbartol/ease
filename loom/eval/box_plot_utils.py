import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Tuple

from loom.eval.table_utils import StatisticsTable


def compute_box_plot_stats(values):
        """Compute statistics needed for box plot visualization"""
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        # Find outliers
        outliers = values[(values < lower_fence) | (values > upper_fence)]
        
        # Find whisker values (min/max excluding outliers)
        non_outliers = values[(values >= lower_fence) & (values <= upper_fence)]
        min_val = np.min(non_outliers) if len(non_outliers) > 0 else np.min(values)
        max_val = np.max(non_outliers) if len(non_outliers) > 0 else np.max(values)
        
        return {
            'q1': q1,
            'median': median,
            'q3': q3,
            'min_val': min_val,
            'max_val': max_val,
            'outliers': outliers.tolist() if len(outliers) > 0 else None
        }


@dataclass
class BoxPlotStats:
    """Class to hold statistics needed for a box plot."""
    q1: float
    median: float
    q3: float
    min_val: float
    max_val: float
    outliers: Optional[List[float]] = None

class BoxPlotVisualizer:
    def __init__(self, stats_table):
        """
        Initialize visualizer with StatisticsTable object.
        
        Parameters:
        -----------
        stats_table : StatisticsTable
            StatisticsTable object containing the statistics
        """
        self.stats_table = stats_table
        self.df = stats_table.df
        
    def _extract_box_plot_stats(self, row: str, metrics: List[str]) -> BoxPlotStats:
        """Extract box plot statistics from a row of the DataFrame."""
        stats = {}
        for metric in metrics:
            if f"{metric}" in self.df.columns:
                stats[metric] = self.df.loc[row, f"{metric}"]
            elif f"box_plot_stats_{metric}" in self.df.columns:
                stats[metric] = self.df.loc[row, f"box_plot_stats_{metric}"]
        
        return BoxPlotStats(
            q1=stats.get('q1', stats.get('box_plot_stats_q1')),
            median=stats.get('median', stats.get('box_plot_stats_median')),
            q3=stats.get('q3', stats.get('box_plot_stats_q3')),
            min_val=stats.get('min_val', stats.get('box_plot_stats_min_val')),
            max_val=stats.get('max_val', stats.get('box_plot_stats_max_val')),
            outliers=stats.get('outliers', stats.get('box_plot_stats_outliers', []))
        )

    def _create_custom_boxplot(self, 
                             ax: plt.Axes,
                             stats: List[BoxPlotStats],
                             positions: np.ndarray,
                             width: float = 0.8,
                             color: str = 'C0',
                             label: Optional[str] = None):
        """Create a box plot using pre-computed statistics."""
        for pos, stat in zip(positions, stats):
            # Draw the box
            box_left = pos - width/2
            box_height = stat.q3 - stat.q1
            
            # Box
            box = plt.Rectangle((box_left, stat.q1), width, box_height,
                              facecolor=color, alpha=0.5, label=label if pos == positions[0] else "")
            ax.add_patch(box)
            
            # Median line
            ax.hlines(stat.median, box_left, box_left + width, color='black', lw=2)
            
            # Whiskers
            ax.vlines(pos, stat.q1, stat.min_val, color='black')
            ax.vlines(pos, stat.q3, stat.max_val, color='black')
            ax.hlines(stat.min_val, pos - width/4, pos + width/4, color='black')
            ax.hlines(stat.max_val, pos - width/4, pos + width/4, color='black')
            
            # Outliers
            if stat.outliers:
                ax.plot([pos] * len(stat.outliers), stat.outliers, 
                       'o', color='black', ms=4)

    def create_dual_boxplot(self,
                          ratio_metrics: List[str],
                          diff_metrics: List[str],
                          figsize: Tuple[int, int] = (12, 8),
                          title: Optional[str] = None,
                          ratio_ylabel: str = "Ratio",
                          diff_ylabel: str = "Difference",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a dual box plot showing distributions of ratios and differences.
        
        Parameters:
        -----------
        ratio_metrics : List[str]
            List of column names for ratio statistics
        diff_metrics : List[str]
            List of column names for difference statistics
        figsize : Tuple[int, int]
            Figure size (width, height)
        title : str, optional
            Title for the figure
        ratio_ylabel : str
            Y-axis label for ratio plot
        diff_ylabel : str
            Y-axis label for difference plot
        save_path : str, optional
            If provided, saves the figure to this path
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Get positions for box plots
        positions = np.arange(len(self.df.index))
        
        # Extract statistics for both plots
        ratio_stats = [self._extract_box_plot_stats(idx, ratio_metrics) 
                      for idx in self.df.index]
        diff_stats = [self._extract_box_plot_stats(idx, diff_metrics) 
                     for idx in self.df.index]
        
        # Create box plots for ratios
        self._create_custom_boxplot(ax1, ratio_stats, positions, color='C0', label='Ratio')
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Distribution of Target/Estimated Ratios')
        ax1.set_ylabel(ratio_ylabel)
        ax1.legend()
        
        # Create box plots for differences
        self._create_custom_boxplot(ax2, diff_stats, positions, color='C1', label='Difference')
        ax2.axhline(y=0.0, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Distribution of Differences')
        ax2.set_ylabel(diff_ylabel)
        ax2.legend()
        
        # Set x-axis labels
        ax2.set_xticks(positions)
        ax2.set_xticklabels(self.df.index, rotation=45)
        ax2.set_xlabel('Experiment')
        
        if title:
            fig.suptitle(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig

# Example usage:
if __name__ == "__main__":
    # Create example StatisticsTable object
    example_stats_list = [
        {
            'global_metrics': {
                'rel_diff': {
                    'q1': 0.9,
                    'median': 1.0,
                    'q3': 1.1,
                    'min_val': 0.8,
                    'max_val': 1.2,
                    'outliers': [0.7, 1.3]
                },
                'abs_diff': {
                    'q1': -0.1,
                    'median': 0.0,
                    'q3': 0.1,
                    'min_val': -0.2,
                    'max_val': 0.2,
                    'outliers': [-0.3, 0.3]
                }
            }
        }
    ]
    
    stats_table = StatisticsTable(
        stats_list=example_stats_list,
        stat_group='box_plot_stats',
        metrics=['q1', 'median', 'q3', 'min_val', 'max_val', 'outliers'],
        experiment_names=['Test 1']
    )
    
    # Create visualization
    visualizer = BoxPlotVisualizer(stats_table)
    fig = visualizer.create_dual_boxplot(
        ratio_metrics=['q1', 'median', 'q3', 'min_val', 'max_val', 'outliers'],
        diff_metrics=['q1', 'median', 'q3', 'min_val', 'max_val', 'outliers'],
        title='Distribution Analysis',
        save_path='boxplot.png'
    )
    plt.show()
