import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging 
from utils import setup_logger, ensure_directory

class Visualizations:
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()
        self.figsize_univariate = self.config['visualization']['figure_size_univariate']
        self.figsize_multivariate = self.config['visualization']['figure_size_multivariate']
        self.save_plots = self.config['visualization']['save_plots']
        self.output_dir = self.config['output']['plots_dir']

    def _setup_logging(self):
        '''setup logging system'''
        try:
            log_config = self.config.get('logging',{})
            log_dir = Path(log_config.get('log_dir','logs/'))

            ensure_directory(log_dir)

            logger = setup_logger(
                name='data_overview',
                log_dir=log_dir,
                log_level=log_config.get('log_level', 'INFO'),
                max_bytes=log_config.get('max_bytes', 10485760),
                backup_count=log_config.get('backup_count', 7)
            )
            return logger 
        except Exception as e:
            print(f'Error setting up logging system: {e}')

    def setup_plot_style(self):
        STYLE = self.config['visualization']['style']
        CONTEXT = self.config['visualization']['context']
        try:
            plt.style.use(STYLE)
        except:
            plt.style.use('default')
        sns.set_context(CONTEXT)

    def plot_numeric_distributions(self,df):
        '''Plot distributions for numeric columns'''
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3

        self.logger.info(f'Plotting distributions for {len(numeric_cols)} numeric columns')

        fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=self.figsize_multivariate)
        axes = axes.flatten() if n_cols > 1 else [axes]

        kde = self.config['visualization']['histogram'].get('kde',True)
        alpha = self.config['visualization']['histogram'].get('alpha',0.7)

        for idx, col in enumerate(numeric_cols):
            try:
                sns.histplot(
                    data=df, x=col, kde=kde,
                    ax= axes[idx], color='purple',
                    alpha=alpha
                )
                axes[idx].set_title(f'Distibution of {col}', fontweight='bold')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
            except Exception as e:
                self.logger.error(f'Error plotting {col}:{e}')

        # hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        DPI = self.config['output'].get('plot_dpi', 300)
        output_file = Path(f'{self.output_dir}/numeric_distributions.png')
        if self.save_plots:
            plt.savefig(output_file, dpi=DPI,bbox_inches='tight')
            plt.close(fig)

        self.logger.info(f'Saved: {output_file}')


    def plot_categorical_distributions(self, df):
        '''Plot count plots for categorical columns'''

        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        n_cols = len(categorical_cols)
        n_rows = (n_cols + 2) // 3

        self.logger.info(f'Plotting frequencies for {len(categorical_cols)} categorical columns')
        fig, axes = plt.subplots(n_rows, 3, figsize=self.figsize_multivariate)
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(categorical_cols):
            try:
                ax = sns.countplot(data=df, x=col, ax=axes[idx], color='green', width=0.4)

                # add label values
                for container in ax.containers:
                    ax.bar_label(container, label_type='edge')

                ax.set_title(f'{col}', fontweight='bold')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x', rotation=45)

            except Exception as e:
                self.logger.error(f'Error plotting {col}:{e}')

        # hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        output_file = Path(f'{self.output_dir}/categorical_distributions.png')
        DPI = self.config['output'].get('plot_dpi', 300)

        if self.save_plots:
            plt.savefig(output_file, dpi=DPI,bbox_inches='tight')
            plt.close(fig)

        self.logger.info(f'Saved: {output_file}')

    def plot_boxplots(self, df):
        '''Plot boxplots for outlier detection'''

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3

        self.logger.info(f'Plotting boxplots for {len(numeric_cols)} numerical columns')
        fig, axes = plt.subplots(n_rows, 3, figsize=self.figsize_multivariate)
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            try:
                sns.boxplot(
                    data=df, y=col, ax=axes[idx], color='gold',linewidth=2, linecolor='blue'
                )

                axes[idx].set_title(f'Boxplot - {col}', fontweight='bold')
                axes[idx].grid(True, alpha=0.3)

            except Exception as e:
                self.logger.error(f'Error plotting {col}:{e}')

        # hide empty subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        output_file = Path(f'{self.output_dir}/boxplots_outliers.png')
        DPI = self.config['output'].get('plot_dpi', 300)

        if self.save_plots:
            plt.savefig(output_file, dpi=DPI,bbox_inches='tight')
            plt.close(fig)

    def correlation_heatmap(self, df):
        '''Plot correlation heatmap'''

        method = self.config['visualization']['heatmap'].get('method','spearman')

        self.logger.info(f'Plotting correlation heatmap ({method} method)')

        try:
            corr = df.corr(numeric_only=True, method=method)

            fig, ax = plt.subplot(figize=self.figsize_univariate)
            sns.heatmap(
                data=corr, annot=True, fmt='.2f', cmap='Blues',ax=ax
            )
            ax.set_tile(f'Correlation Heatmap ({method.capitalize()})')


            plt.tight_layout()
            output_file = Path(f'{self.output_dir}/correlation_heatmap_{method}.png')
            DPI = self.config['output'].get('plot_dpi', 300)

            if self.save_plots:
                plt.savefig(output_file, dpi=DPI,bbox_inches='tight')
                plt.close(fig)

                self.logger.error(f'Saved : {output_file}')

        except Exception as e:
            self.logger.error(f'Error creating correlation heatmap: {e}')

    def plot_target_distribution(self, df):
        '''Plot target variable distribution'''

        target_column = self.config['data']['target_column'] 

        self.logger.info(f'Plotting target distribution for {target_column}')

        try:
            fig, ax = plt.subplots(figsize=self.figsize_univariate)

            ax = sns.countplot(
                data=df, 
                x=target_column,
                color='blue',
                width=0.5
            )

            # add labels
            for container in ax.containers:
                ax.bar_label(container, label_type='edge')

            ax.set_title(f'Target variable distribution: {target_column}',
                        fontweight='bold', fontsize=13)
            ax.set_ylabel('Count', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            output_file = Path(f'{self.output_dir}/target_distribution.png')
            DPI = self.config['output'].get('plot_dpi', 300)

            if self.save_plots:
                plt.savefig(output_file, dpi=DPI,bbox_inches='tight')
                plt.close(fig)

                self.logger.error(f'Saved : {output_file}')

        except Exception as e:
            self.logger.error(f'Error plotting target distribution: {e}')


    def run_visualizations(self, df):
        '''Run all visualizations'''
        visuals = {}
        self.logger.info(f'Running visualizations....')
        visuals['numeric_distributions'] = self.plot_numeric_distributions(df)
        visuals['categorical_distributions'] = self.plot_categorical_distributions(df)
        visuals['heatmap'] = self.correlation_heatmap(df)
        visuals['boxplots'] = self.plot_boxplots(df)
        visuals['target_distributions'] = self.plot_target_distribution(df)

        return visuals
    
