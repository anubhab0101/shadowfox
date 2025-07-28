import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class VisualizationManager:
    """Handles all visualization tasks for the NLP analysis platform"""
    
    def __init__(self):
        # Define color schemes
        self.color_schemes = {
            'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'performance': ['#2E86C1', '#28B463', '#F39C12', '#E74C3C', '#8E44AD'],
            'analysis': ['#3498DB', '#E67E22', '#27AE60', '#E74C3C', '#9B59B6'],
            'research': ['#5DADE2', '#F8C471', '#82E0AA', '#F1948A', '#D2B4DE']
        }
        
        # Chart configuration
        self.chart_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
    
    def create_model_comparison_chart(self, comparison_df: pd.DataFrame) -> go.Figure:
        """Create comprehensive model comparison visualization"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Response Time', 'Token Efficiency', 'Citation Quality', 'Quality Score'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Response Time
        fig.add_trace(
            go.Bar(
                x=comparison_df['model'],
                y=comparison_df['response_time'],
                name='Response Time (s)',
                marker_color=self.color_schemes['performance'][0]
            ),
            row=1, col=1
        )
        
        # Token Efficiency (inverse of token count for efficiency)
        token_efficiency = 1000 / comparison_df['response_length'].replace(0, 1)
        fig.add_trace(
            go.Bar(
                x=comparison_df['model'],
                y=token_efficiency,
                name='Token Efficiency',
                marker_color=self.color_schemes['performance'][1]
            ),
            row=1, col=2
        )
        
        # Citation Quality
        fig.add_trace(
            go.Bar(
                x=comparison_df['model'],
                y=comparison_df['citation_count'],
                name='Citation Count',
                marker_color=self.color_schemes['performance'][2]
            ),
            row=2, col=1
        )
        
        # Quality Score
        fig.add_trace(
            go.Bar(
                x=comparison_df['model'],
                y=comparison_df['quality_score'],
                name='Quality Score',
                marker_color=self.color_schemes['performance'][3]
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Model Performance Comparison",
            showlegend=False,
            height=600,
            title_x=0.5
        )
        
        return fig
    
    def create_response_time_chart(self, performance_data: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create response time comparison chart"""
        
        models = list(performance_data.keys())
        avg_times = [performance_data[model].get('avg_response_time', 0) for model in models]
        min_times = [performance_data[model].get('min_response_time', 0) for model in models]
        max_times = [performance_data[model].get('max_response_time', 0) for model in models]
        
        fig = go.Figure()
        
        # Add bar chart for average response time
        fig.add_trace(go.Bar(
            x=models,
            y=avg_times,
            name='Average Response Time',
            marker_color=self.color_schemes['performance'][0],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[max_times[i] - avg_times[i] for i in range(len(models))],
                arrayminus=[avg_times[i] - min_times[i] for i in range(len(models))],
                visible=True
            )
        ))
        
        fig.update_layout(
            title='Model Response Time Comparison',
            xaxis_title='Model',
            yaxis_title='Response Time (seconds)',
            title_x=0.5,
            showlegend=False
        )
        
        return fig
    
    def create_performance_radar(self, performance_data: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Create radar chart for overall performance comparison"""
        
        fig = go.Figure()
        
        # Define metrics for radar chart
        metrics = ['Token Efficiency', 'Citation Score', 'Quality Score', 'Consistency Score', 'Reliability']
        
        for i, (model, data) in enumerate(performance_data.items()):
            # Normalize values to 0-1 scale for radar chart
            values = [
                data.get('token_efficiency', 0),
                data.get('citation_score', 0),
                data.get('quality_score', 0),
                data.get('consistency_score', 0),
                max(0, 1 - data.get('error_rate', 0) / 100)  # Reliability
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model,
                line_color=self.color_schemes['performance'][i % len(self.color_schemes['performance'])]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart",
            title_x=0.5
        )
        
        return fig
    
    def create_research_findings_chart(self, findings_data: Dict[str, Any]) -> go.Figure:
        """Create visualization for research findings"""
        
        # Create different visualizations based on findings data structure
        if 'hypothesis_tests' in findings_data:
            return self._create_hypothesis_test_chart(findings_data['hypothesis_tests'])
        elif 'statistical_results' in findings_data:
            return self._create_statistical_results_chart(findings_data['statistical_results'])
        else:
            return self._create_generic_findings_chart(findings_data)
    
    def _create_hypothesis_test_chart(self, hypothesis_data: Dict[str, Any]) -> go.Figure:
        """Create chart for hypothesis test results"""
        
        fig = go.Figure()
        
        # Extract test results
        tests = list(hypothesis_data.keys())
        p_values = [hypothesis_data[test].get('p_value', 0) for test in tests]
        significance = [hypothesis_data[test].get('significant', False) for test in tests]
        
        # Color based on significance
        colors = ['green' if sig else 'red' for sig in significance]
        
        fig.add_trace(go.Bar(
            x=tests,
            y=p_values,
            marker_color=colors,
            name='P-Values'
        ))
        
        # Add significance line
        fig.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                     annotation_text="Significance Threshold (0.05)")
        
        fig.update_layout(
            title='Hypothesis Test Results',
            xaxis_title='Statistical Tests',
            yaxis_title='P-Value',
            title_x=0.5
        )
        
        return fig
    
    def _create_statistical_results_chart(self, statistical_data: Dict[str, Any]) -> go.Figure:
        """Create chart for statistical analysis results"""
        
        # Create correlation matrix if available
        if 'correlations' in statistical_data:
            corr_data = statistical_data['correlations']
            
            fig = go.Figure(data=go.Heatmap(
                z=list(corr_data.values()),
                x=list(corr_data.keys()),
                y=list(corr_data.keys()),
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title='Correlation Matrix',
                title_x=0.5
            )
            
            return fig
        
        # Default bar chart for statistical measures
        metrics = list(statistical_data.keys())
        values = list(statistical_data.values())
        
        fig = go.Figure(data=[go.Bar(
            x=metrics,
            y=values,
            marker_color=self.color_schemes['research'][0]
        )])
        
        fig.update_layout(
            title='Statistical Analysis Results',
            xaxis_title='Metrics',
            yaxis_title='Values',
            title_x=0.5
        )
        
        return fig
    
    def _create_generic_findings_chart(self, findings_data: Dict[str, Any]) -> go.Figure:
        """Create generic chart for research findings"""
        
        # Extract numeric data for visualization
        numeric_data = {}
        for key, value in findings_data.items():
            if isinstance(value, (int, float)):
                numeric_data[key] = value
            elif isinstance(value, dict) and 'score' in value:
                numeric_data[key] = value['score']
        
        if not numeric_data:
            # Create a simple text-based figure
            fig = go.Figure()
            fig.add_annotation(
                text="Research findings available in detailed results section",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Research Findings")
            return fig
        
        # Create bar chart
        fig = go.Figure(data=[go.Bar(
            x=list(numeric_data.keys()),
            y=list(numeric_data.values()),
            marker_color=self.color_schemes['research'][:len(numeric_data)]
        )])
        
        fig.update_layout(
            title='Research Findings Summary',
            xaxis_title='Findings',
            yaxis_title='Scores',
            title_x=0.5
        )
        
        return fig
    
    def create_sentiment_analysis_chart(self, sentiment_data: List[Dict[str, Any]]) -> go.Figure:
        """Create sentiment analysis visualization"""
        
        df = pd.DataFrame(sentiment_data)
        
        fig = px.scatter(df, 
                        x='timestamp', 
                        y='sentiment_score',
                        color='model',
                        size='response_length',
                        hover_data=['word_count'],
                        title='Sentiment Analysis Over Time')
        
        fig.update_layout(title_x=0.5)
        
        return fig
    
    def create_complexity_distribution(self, complexity_data: List[Dict[str, Any]]) -> go.Figure:
        """Create text complexity distribution chart"""
        
        df = pd.DataFrame(complexity_data)
        
        fig = px.histogram(df, 
                          x='complexity_score', 
                          color='model',
                          marginal='box',
                          title='Text Complexity Distribution')
        
        fig.update_layout(title_x=0.5)
        
        return fig
    
    def create_token_efficiency_chart(self, efficiency_data: List[Dict[str, Any]]) -> go.Figure:
        """Create token efficiency analysis chart"""
        
        df = pd.DataFrame(efficiency_data)
        
        fig = px.scatter(df,
                        x='word_count',
                        y='token_count',
                        color='model',
                        trendline='ols',
                        title='Token Usage Efficiency')
        
        fig.update_layout(
            title_x=0.5,
            xaxis_title='Word Count',
            yaxis_title='Token Count'
        )
        
        return fig
    
    def create_citation_network(self, citation_data: List[Dict[str, Any]]) -> go.Figure:
        """Create citation network visualization"""
        
        # Count citation frequencies
        all_citations = []
        for data in citation_data:
            all_citations.extend(data.get('citations', []))
        
        citation_counts = pd.Series(all_citations).value_counts().head(20)
        
        fig = px.bar(
            x=citation_counts.values,
            y=citation_counts.index,
            orientation='h',
            title='Most Frequently Cited Sources'
        )
        
        fig.update_layout(
            title_x=0.5,
            xaxis_title='Citation Frequency',
            yaxis_title='Source'
        )
        
        return fig
    
    def create_comparative_analysis_dashboard(self, 
                                            analysis_results: List[Dict[str, Any]]) -> List[go.Figure]:
        """Create comprehensive dashboard with multiple charts"""
        
        figures = []
        
        # Prepare data
        if analysis_results:
            # Sentiment comparison
            sentiment_data = []
            complexity_data = []
            efficiency_data = []
            
            for result in analysis_results:
                if result.get('analysis'):
                    analysis = result['analysis']
                    
                    sentiment_data.append({
                        'model': result.get('model', 'Unknown'),
                        'sentiment_score': analysis.get('sentiment', {}).get('score', 0),
                        'response_length': analysis.get('token_count', 0),
                        'word_count': analysis.get('word_count', 0),
                        'timestamp': result.get('timestamp', '')
                    })
                    
                    complexity_data.append({
                        'model': result.get('model', 'Unknown'),
                        'complexity_score': analysis.get('complexity', {}).get('score', 0)
                    })
                    
                    efficiency_data.append({
                        'model': result.get('model', 'Unknown'),
                        'word_count': analysis.get('word_count', 0),
                        'token_count': analysis.get('token_count', 0)
                    })
            
            # Create individual charts
            if sentiment_data:
                figures.append(self.create_sentiment_analysis_chart(sentiment_data))
            
            if complexity_data:
                figures.append(self.create_complexity_distribution(complexity_data))
            
            if efficiency_data:
                figures.append(self.create_token_efficiency_chart(efficiency_data))
            
            if any(result.get('citations') for result in analysis_results):
                figures.append(self.create_citation_network(analysis_results))
        
        return figures
