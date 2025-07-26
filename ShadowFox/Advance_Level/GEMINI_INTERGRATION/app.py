import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
import os

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

from modules.gemini_client import GeminiClient
from modules.analysis_engine import AnalysisEngine
from modules.visualization import VisualizationManager
from modules.research_framework import ResearchFramework
from data.sample_prompts import SAMPLE_PROMPTS, RESEARCH_QUESTIONS
from utils.rate_limiter import RateLimiter
from utils.text_processor import TextProcessor

st.set_page_config(
    page_title="NLP/ML Analysis Platform - Gemini API",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'research_data' not in st.session_state:
    st.session_state.research_data = {}

@st.cache_resource
def initialize_components():
    """Initialize all analysis components"""
    try:
        gemini_client = GeminiClient()
        analysis_engine = AnalysisEngine(gemini_client)
        viz_manager = VisualizationManager()
        research_framework = ResearchFramework()
        text_processor = TextProcessor()
        
        return gemini_client, analysis_engine, viz_manager, research_framework, text_processor
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return None, None, None, None, None

# Header
st.title("🧠 NLP/ML Analysis Platform")
st.markdown("### Comprehensive Language Model Analysis using Google Gemini API & Advanced NLP Techniques")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Analysis Module",
    ["Overview", "Model Testing", "Performance Analysis", "Research Framework", "Visualization Lab", "Results Dashboard"]
)

# Initialize components
gemini_client, analysis_engine, viz_manager, research_framework, text_processor = initialize_components()

if any(component is None for component in [gemini_client, analysis_engine, viz_manager, research_framework, text_processor]):
    st.error("Failed to initialize application components. Please check your API configuration.")
    st.stop()

# Main content based on selected page
if page == "Overview":
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Project Objectives")
        st.markdown("""
        **Primary Goals:**
        - Implement and analyze Language Models using Google Gemini API
        - Conduct comprehensive performance evaluation
        - Explore model capabilities across diverse NLP tasks
        - Generate insights through systematic research framework
        - Create interactive visualizations of model behavior
        """)
        
        st.subheader("Available Models")
        models = gemini_client.get_available_models()
        for model in models:
            st.markdown(f"• **{model['name']}**: {model['description']}")
    
    with col2:
        st.subheader("Research Framework")
        st.markdown("""
        **Analysis Dimensions:**
        1. **Contextual Understanding**: Evaluate context retention and reasoning
        2. **Text Generation Quality**: Assess creativity and coherence
        3. **Domain Adaptability**: Test performance across different fields
        4. **Response Consistency**: Measure reliability and stability
        5. **Citation Quality**: Analyze source attribution accuracy
        """)
        
        # API Status Check
        st.subheader("System Status")
        with st.spinner("Checking API connectivity..."):
            status = gemini_client.check_api_status()
            if status['connected']:
                st.success("✅ Google Gemini API Connected")
                st.info(f"Available Credits: {status.get('credits', 'Available')}")
            else:
                st.error("❌ API Connection Failed")

elif page == "Model Testing":
    st.header("Interactive Model Testing")
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select Language Model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-preview-image-generation"],
            index=0
        )
    
    with col2:
        max_tokens = st.number_input("Max Tokens", min_value=50, max_value=2000, value=500)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Input methods
    st.subheader("Input Configuration")
    input_method = st.radio("Choose input method:", ["Custom Input", "Sample Prompts", "Research Questions"])
    
    if input_method == "Custom Input":
        user_prompt = st.text_area("Enter your prompt:", height=100, placeholder="Type your question or prompt here...")
    elif input_method == "Sample Prompts":
        category = st.selectbox("Select category:", list(SAMPLE_PROMPTS.keys()))
        prompt_option = st.selectbox("Select prompt:", SAMPLE_PROMPTS[category])
        user_prompt = prompt_option
        st.text_area("Selected prompt:", value=user_prompt, height=100, disabled=True)
    else:
        research_question = st.selectbox("Select research question:", RESEARCH_QUESTIONS)
        user_prompt = research_question
        st.text_area("Selected research question:", value=user_prompt, height=100, disabled=True)
    
    # Analysis options
    st.subheader("Analysis Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analyze_sentiment = st.checkbox("Sentiment Analysis", value=True)
        analyze_complexity = st.checkbox("Text Complexity", value=True)
    
    with col2:
        analyze_tokens = st.checkbox("Token Analysis", value=True)
        analyze_citations = st.checkbox("Citation Quality", value=True)
    
    with col3:
        compare_models = st.checkbox("Compare Models", value=False)
        save_results = st.checkbox("Save Results", value=True)
    
    # Execute analysis
    if st.button("🚀 Run Analysis", type="primary"):
        if user_prompt:
            with st.spinner("Analyzing with Perplexity API..."):
                try:
                    # Single model analysis
                    result = analysis_engine.analyze_prompt(
                        prompt=user_prompt,
                        model=selected_model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        analyze_sentiment=analyze_sentiment,
                        analyze_complexity=analyze_complexity,
                        analyze_tokens=analyze_tokens,
                        analyze_citations=analyze_citations
                    )
                    
                    if compare_models:
                        # Compare with other models
                        comparison_results = analysis_engine.compare_models(
                            prompt=user_prompt,
                            models=["gemini-2.5-flash", "gemini-2.5-pro"],
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        result['comparison'] = comparison_results
                    
                    # Display results
                    st.success("Analysis completed!")
                    
                    # Response
                    st.subheader("Model Response")
                    st.markdown(result['response'])
                    
                    # Analysis metrics
                    if result['analysis']:
                        st.subheader("Analysis Metrics")
                        
                        cols = st.columns(4)
                        metrics = result['analysis']
                        
                        with cols[0]:
                            st.metric("Response Length", f"{metrics.get('token_count', 0)} tokens")
                            st.metric("Word Count", metrics.get('word_count', 0))
                        
                        with cols[1]:
                            if 'sentiment' in metrics:
                                st.metric("Sentiment Score", f"{metrics['sentiment']['score']:.2f}")
                                st.metric("Sentiment", metrics['sentiment']['label'])
                        
                        with cols[2]:
                            if 'complexity' in metrics:
                                st.metric("Complexity Score", f"{metrics['complexity']['score']:.2f}")
                                st.metric("Reading Level", metrics['complexity']['level'])
                        
                        with cols[3]:
                            if 'citations' in metrics:
                                st.metric("Citations Found", len(result.get('citations', [])))
                                st.metric("Unique Sources", metrics['citations']['unique_sources'])
                    
                    # Citations
                    if result.get('citations'):
                        st.subheader("Source Citations")
                        for i, citation in enumerate(result['citations'], 1):
                            st.markdown(f"{i}. [{citation}]({citation})")
                    
                    # Model comparison
                    if compare_models and 'comparison' in result:
                        st.subheader("Model Comparison")
                        comparison_df = pd.DataFrame(result['comparison'])
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Comparison visualization
                        if len(comparison_df) > 1:
                            fig = viz_manager.create_model_comparison_chart(comparison_df)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Save results
                    if save_results:
                        result['timestamp'] = datetime.now().isoformat()
                        result['prompt'] = user_prompt
                        result['model'] = selected_model
                        st.session_state.analysis_results.append(result)
                        st.success("Results saved to session!")
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        else:
            st.warning("Please enter a prompt to analyze.")

elif page == "Performance Analysis":
    st.header("Performance Analysis & Benchmarking")
    
    # Performance testing configuration
    st.subheader("Benchmark Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_suite = st.selectbox(
            "Select Test Suite",
            ["Quick Evaluation", "Comprehensive Analysis", "Domain-Specific Testing", "Custom Benchmark"]
        )
        
        models_to_test = st.multiselect(
            "Models to Test",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-preview-image-generation"],
            default=["gemini-2.5-flash"]
        )
    
    with col2:
        test_iterations = st.number_input("Test Iterations", min_value=1, max_value=10, value=3)
        include_metrics = st.multiselect(
            "Include Metrics",
            ["Response Time", "Token Efficiency", "Citation Quality", "Consistency Score", "Sentiment Analysis"],
            default=["Response Time", "Token Efficiency", "Citation Quality"]
        )
    
    # Test execution
    if st.button("🧪 Run Performance Tests", type="primary"):
        if models_to_test:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                results = analysis_engine.run_performance_benchmark(
                    models=models_to_test,
                    test_suite=test_suite,
                    iterations=test_iterations,
                    metrics=include_metrics,
                    progress_callback=lambda p, s: (progress_bar.progress(p), status_text.text(s))
                )
                
                st.success("Performance analysis completed!")
                
                # Results summary
                st.subheader("Performance Summary")
                
                # Create summary metrics
                summary_data = []
                for model, metrics in results.items():
                    summary_data.append({
                        'Model': model,
                        'Avg Response Time': f"{metrics.get('avg_response_time', 0):.2f}s",
                        'Token Efficiency': f"{metrics.get('token_efficiency', 0):.2f}",
                        'Citation Score': f"{metrics.get('citation_score', 0):.2f}",
                        'Overall Score': f"{metrics.get('overall_score', 0):.2f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Visualizations
                st.subheader("Performance Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Response time comparison
                    fig_time = viz_manager.create_response_time_chart(results)
                    st.plotly_chart(fig_time, use_container_width=True)
                
                with col2:
                    # Overall performance radar
                    fig_radar = viz_manager.create_performance_radar(results)
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # Detailed metrics
                st.subheader("Detailed Metrics")
                
                for model in models_to_test:
                    with st.expander(f"📊 {model} - Detailed Results"):
                        model_results = results[model]
                        
                        metrics_cols = st.columns(3)
                        
                        with metrics_cols[0]:
                            st.metric("Response Time", f"{model_results.get('avg_response_time', 0):.2f}s")
                            st.metric("Min Response Time", f"{model_results.get('min_response_time', 0):.2f}s")
                            st.metric("Max Response Time", f"{model_results.get('max_response_time', 0):.2f}s")
                        
                        with metrics_cols[1]:
                            st.metric("Token Efficiency", f"{model_results.get('token_efficiency', 0):.2f}")
                            st.metric("Avg Tokens/Response", model_results.get('avg_tokens', 0))
                            st.metric("Token Variance", f"{model_results.get('token_variance', 0):.2f}")
                        
                        with metrics_cols[2]:
                            st.metric("Citation Quality", f"{model_results.get('citation_score', 0):.2f}")
                            st.metric("Consistency Score", f"{model_results.get('consistency_score', 0):.2f}")
                            st.metric("Error Rate", f"{model_results.get('error_rate', 0):.1f}%")
                
                # Save performance data
                st.session_state.research_data['performance_results'] = results
                
            except Exception as e:
                st.error(f"Performance analysis failed: {str(e)}")
        else:
            st.warning("Please select at least one model to test.")

elif page == "Research Framework":
    st.header("Research Questions & Hypothesis Testing")
    
    # Research question formulation
    st.subheader("Research Question Development")
    
    research_type = st.selectbox(
        "Research Focus Area",
        ["Contextual Understanding", "Creative Text Generation", "Domain Adaptation", "Factual Accuracy", "Reasoning Capabilities"]
    )
    
    # Dynamic research questions based on type
    questions = research_framework.get_research_questions(research_type)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_question = st.selectbox("Pre-defined Research Questions", questions)
        
        custom_question = st.text_area(
            "Or Define Custom Research Question:",
            placeholder="Example: How does model performance vary with prompt complexity across different domains?",
            height=100
        )
        
        final_question = custom_question if custom_question else selected_question
    
    with col2:
        st.subheader("Hypothesis Formation")
        hypothesis = st.text_area(
            "Research Hypothesis:",
            placeholder="State your hypothesis about the expected model behavior...",
            height=100
        )
        
        methodology = st.multiselect(
            "Research Methodology",
            ["Controlled Prompt Testing", "A/B Model Comparison", "Domain Cross-Validation", "Temporal Consistency Testing", "Statistical Analysis"],
            default=["Controlled Prompt Testing"]
        )
    
    # Experimental design
    st.subheader("Experimental Design")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sample_size = st.number_input("Sample Size", min_value=5, max_value=50, value=10)
        control_variables = st.multiselect(
            "Control Variables",
            ["Temperature", "Max Tokens", "Model Type", "Prompt Structure", "Domain Context"],
            default=["Temperature", "Max Tokens"]
        )
    
    with col2:
        test_variables = st.multiselect(
            "Test Variables",
            ["Prompt Complexity", "Domain Type", "Response Length", "Citation Requirements", "Creative vs Factual"],
            default=["Prompt Complexity"]
        )
        
        success_metrics = st.multiselect(
            "Success Metrics",
            ["Response Quality", "Factual Accuracy", "Citation Quality", "Response Time", "Consistency"],
            default=["Response Quality", "Factual Accuracy"]
        )
    
    with col3:
        statistical_tests = st.multiselect(
            "Statistical Analysis",
            ["Descriptive Statistics", "T-Test", "ANOVA", "Correlation Analysis", "Regression Analysis"],
            default=["Descriptive Statistics"]
        )
    
    # Execute research
    if st.button("🔬 Execute Research Study", type="primary"):
        if final_question and hypothesis:
            with st.spinner("Conducting research study..."):
                try:
                    research_results = research_framework.conduct_study(
                        research_question=final_question,
                        hypothesis=hypothesis,
                        methodology=methodology,
                        sample_size=sample_size,
                        control_variables=control_variables,
                        test_variables=test_variables,
                        success_metrics=success_metrics,
                        statistical_tests=statistical_tests,
                        gemini_client=gemini_client
                    )
                    
                    st.success("Research study completed!")
                    
                    # Results presentation
                    st.subheader("Research Results")
                    
                    # Executive summary
                    st.markdown("#### Executive Summary")
                    st.info(research_results.get('summary', 'No summary available'))
                    
                    # Hypothesis evaluation
                    st.markdown("#### Hypothesis Evaluation")
                    hypothesis_result = research_results.get('hypothesis_result', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Hypothesis Supported", "Yes" if hypothesis_result.get('supported', False) else "No")
                        st.metric("Confidence Level", f"{hypothesis_result.get('confidence', 0):.1f}%")
                    
                    with col2:
                        st.metric("Statistical Significance", "Yes" if hypothesis_result.get('significant', False) else "No")
                        st.metric("P-Value", f"{hypothesis_result.get('p_value', 0):.4f}")
                    
                    # Detailed findings
                    st.markdown("#### Detailed Findings")
                    
                    findings_data = research_results.get('findings', {})
                    
                    if findings_data:
                        # Create findings visualization
                        fig_findings = viz_manager.create_research_findings_chart(findings_data)
                        st.plotly_chart(fig_findings, use_container_width=True)
                        
                        # Statistical analysis results
                        if 'statistical_analysis' in research_results:
                            st.markdown("#### Statistical Analysis")
                            stats_df = pd.DataFrame(research_results['statistical_analysis'])
                            st.dataframe(stats_df, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("#### Recommendations")
                    recommendations = research_results.get('recommendations', [])
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"{i}. {rec}")
                    
                    # Save research results
                    st.session_state.research_data[f'study_{len(st.session_state.research_data)}'] = research_results
                
                except Exception as e:
                    st.error(f"Research study failed: {str(e)}")
        else:
            st.warning("Please provide both a research question and hypothesis.")

elif page == "Visualization Lab":
    st.header("Advanced Visualization Laboratory")
    
    # Check if we have data to visualize
    if not st.session_state.analysis_results and not st.session_state.research_data:
        st.warning("No analysis data available. Please run some analyses first.")
        st.stop()
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Response Analysis", "Performance Comparison", "Research Insights", "Token Patterns", "Citation Networks", "Temporal Analysis"]
    )
    
    if viz_type == "Response Analysis":
        st.subheader("Response Quality Analysis")
        
        if st.session_state.analysis_results:
            # Prepare data for response analysis
            response_data = []
            for result in st.session_state.analysis_results:
                if result.get('analysis'):
                    analysis = result['analysis']
                    response_data.append({
                        'Model': result.get('model', 'Unknown'),
                        'Response Length': analysis.get('token_count', 0),
                        'Sentiment Score': analysis.get('sentiment', {}).get('score', 0),
                        'Complexity Score': analysis.get('complexity', {}).get('score', 0),
                        'Citations': len(result.get('citations', [])),
                        'Timestamp': result.get('timestamp', '')
                    })
            
            if response_data:
                df = pd.DataFrame(response_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Response length distribution
                    fig_length = px.histogram(df, x='Response Length', title='Response Length Distribution')
                    st.plotly_chart(fig_length, use_container_width=True)
                
                with col2:
                    # Sentiment vs Complexity scatter
                    fig_scatter = px.scatter(df, x='Sentiment Score', y='Complexity Score', 
                                           color='Model', size='Citations',
                                           title='Sentiment vs Complexity Analysis')
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Model performance heatmap
                if len(df['Model'].unique()) > 1:
                    model_metrics = df.groupby('Model').agg({
                        'Response Length': 'mean',
                        'Sentiment Score': 'mean',
                        'Complexity Score': 'mean',
                        'Citations': 'mean'
                    }).round(2)
                    
                    fig_heatmap = px.imshow(model_metrics.T, 
                                          title='Model Performance Heatmap',
                                          labels=dict(x="Model", y="Metric", color="Score"))
                    st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No response analysis data available.")
    
    elif viz_type == "Performance Comparison":
        st.subheader("Model Performance Comparison")
        
        if 'performance_results' in st.session_state.research_data:
            perf_data = st.session_state.research_data['performance_results']
            
            # Create comprehensive performance dashboard
            metrics = ['avg_response_time', 'token_efficiency', 'citation_score', 'consistency_score']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance radar chart
                fig_radar = viz_manager.create_performance_radar(perf_data)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                # Response time comparison
                fig_time = viz_manager.create_response_time_chart(perf_data)
                st.plotly_chart(fig_time, use_container_width=True)
            
            # Detailed performance matrix
            st.subheader("Performance Matrix")
            
            matrix_data = []
            for model, metrics in perf_data.items():
                matrix_data.append({
                    'Model': model,
                    'Response Time (s)': f"{metrics.get('avg_response_time', 0):.2f}",
                    'Token Efficiency': f"{metrics.get('token_efficiency', 0):.2f}",
                    'Citation Score': f"{metrics.get('citation_score', 0):.2f}",
                    'Consistency': f"{metrics.get('consistency_score', 0):.2f}",
                    'Overall Score': f"{metrics.get('overall_score', 0):.2f}"
                })
            
            matrix_df = pd.DataFrame(matrix_data)
            st.dataframe(matrix_df, use_container_width=True)
        else:
            st.info("No performance data available. Run performance analysis first.")
    
    elif viz_type == "Research Insights":
        st.subheader("Research Study Insights")
        
        research_studies = [k for k in st.session_state.research_data.keys() if k.startswith('study_')]
        
        if research_studies:
            selected_study = st.selectbox("Select Study", research_studies)
            study_data = st.session_state.research_data[selected_study]
            
            # Study overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Hypothesis Supported", "Yes" if study_data.get('hypothesis_result', {}).get('supported', False) else "No")
                st.metric("Confidence Level", f"{study_data.get('hypothesis_result', {}).get('confidence', 0):.1f}%")
            
            with col2:
                st.metric("Sample Size", study_data.get('sample_size', 0))
                st.metric("Statistical Significance", "Yes" if study_data.get('hypothesis_result', {}).get('significant', False) else "No")
            
            # Research findings visualization
            if 'findings' in study_data:
                fig_research = viz_manager.create_research_findings_chart(study_data['findings'])
                st.plotly_chart(fig_research, use_container_width=True)
        else:
            st.info("No research studies available. Conduct research studies first.")
    
    elif viz_type == "Token Patterns":
        st.subheader("Token Usage Patterns")
        
        if st.session_state.analysis_results:
            # Analyze token patterns across responses
            token_data = []
            for result in st.session_state.analysis_results:
                if result.get('analysis'):
                    analysis = result['analysis']
                    token_data.append({
                        'Model': result.get('model', 'Unknown'),
                        'Token Count': analysis.get('token_count', 0),
                        'Word Count': analysis.get('word_count', 0),
                        'Avg Token Length': analysis.get('token_count', 0) / max(analysis.get('word_count', 1), 1),
                        'Prompt': result.get('prompt', '')[:50] + '...' if len(result.get('prompt', '')) > 50 else result.get('prompt', '')
                    })
            
            if token_data:
                df = pd.DataFrame(token_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Token distribution
                    fig_tokens = px.box(df, x='Model', y='Token Count', title='Token Count Distribution by Model')
                    st.plotly_chart(fig_tokens, use_container_width=True)
                
                with col2:
                    # Token efficiency
                    fig_efficiency = px.scatter(df, x='Word Count', y='Token Count', 
                                               color='Model', title='Token Efficiency Analysis')
                    st.plotly_chart(fig_efficiency, use_container_width=True)
        else:
            st.info("No token analysis data available.")
    
    elif viz_type == "Citation Networks":
        st.subheader("Citation Network Analysis")
        
        if st.session_state.analysis_results:
            # Analyze citation patterns
            citation_data = []
            all_citations = []
            
            for result in st.session_state.analysis_results:
                citations = result.get('citations', [])
                all_citations.extend(citations)
                citation_data.append({
                    'Model': result.get('model', 'Unknown'),
                    'Citation Count': len(citations),
                    'Unique Citations': len(set(citations)),
                    'Prompt': result.get('prompt', '')[:50] + '...' if len(result.get('prompt', '')) > 50 else result.get('prompt', '')
                })
            
            if citation_data:
                df = pd.DataFrame(citation_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Citation frequency
                    fig_citations = px.bar(df, x='Model', y='Citation Count', 
                                         title='Average Citations per Model')
                    st.plotly_chart(fig_citations, use_container_width=True)
                
                with col2:
                    # Citation uniqueness
                    fig_unique = px.scatter(df, x='Citation Count', y='Unique Citations',
                                          color='Model', title='Citation Diversity Analysis')
                    st.plotly_chart(fig_unique, use_container_width=True)
                
                # Most cited sources
                if all_citations:
                    citation_counts = pd.Series(all_citations).value_counts().head(10)
                    fig_top = px.bar(x=citation_counts.values, y=citation_counts.index,
                                   orientation='h', title='Top 10 Most Cited Sources')
                    st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No citation data available.")
    
    elif viz_type == "Temporal Analysis":
        st.subheader("Temporal Pattern Analysis")
        
        if st.session_state.analysis_results:
            # Analyze patterns over time
            temporal_data = []
            for result in st.session_state.analysis_results:
                if result.get('timestamp') and result.get('analysis'):
                    analysis = result['analysis']
                    temporal_data.append({
                        'Timestamp': pd.to_datetime(result['timestamp']),
                        'Model': result.get('model', 'Unknown'),
                        'Response Length': analysis.get('token_count', 0),
                        'Sentiment Score': analysis.get('sentiment', {}).get('score', 0),
                        'Citations': len(result.get('citations', []))
                    })
            
            if temporal_data:
                df = pd.DataFrame(temporal_data)
                df = df.sort_values('Timestamp')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Response length over time
                    fig_time = px.line(df, x='Timestamp', y='Response Length', 
                                     color='Model', title='Response Length Over Time')
                    st.plotly_chart(fig_time, use_container_width=True)
                
                with col2:
                    # Sentiment trends
                    fig_sentiment = px.line(df, x='Timestamp', y='Sentiment Score',
                                          color='Model', title='Sentiment Trends Over Time')
                    st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.info("No temporal data available.")

elif page == "Results Dashboard":
    st.header("Comprehensive Results Dashboard")
    
    # Summary statistics
    st.subheader("Session Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", len(st.session_state.analysis_results))
    
    with col2:
        research_studies = len([k for k in st.session_state.research_data.keys() if k.startswith('study_')])
        st.metric("Research Studies", research_studies)
    
    with col3:
        total_tokens = sum(result.get('analysis', {}).get('token_count', 0) for result in st.session_state.analysis_results)
        st.metric("Total Tokens", f"{total_tokens:,}")
    
    with col4:
        unique_models = len(set(result.get('model', 'Unknown') for result in st.session_state.analysis_results))
        st.metric("Models Tested", unique_models)
    
    # Data export options
    st.subheader("Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Analysis Results"):
            if st.session_state.analysis_results:
                # Prepare data for export
                export_data = []
                for result in st.session_state.analysis_results:
                    export_data.append({
                        'timestamp': result.get('timestamp'),
                        'model': result.get('model'),
                        'prompt': result.get('prompt'),
                        'response': result.get('response'),
                        'token_count': result.get('analysis', {}).get('token_count', 0),
                        'sentiment_score': result.get('analysis', {}).get('sentiment', {}).get('score', 0),
                        'complexity_score': result.get('analysis', {}).get('complexity', {}).get('score', 0),
                        'citation_count': len(result.get('citations', []))
                    })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Analysis Results CSV",
                    data=csv,
                    file_name=f"nlp_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No analysis results to export.")
    
    with col2:
        if st.button("🔬 Export Research Data"):
            if st.session_state.research_data:
                research_json = json.dumps(st.session_state.research_data, indent=2, default=str)
                
                st.download_button(
                    label="Download Research Data JSON",
                    data=research_json,
                    file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No research data to export.")
    
    # Recent activity
    st.subheader("Recent Activity")
    
    if st.session_state.analysis_results:
        recent_results = sorted(st.session_state.analysis_results, 
                              key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
        
        for result in recent_results:
            with st.expander(f"📝 {result.get('model', 'Unknown')} - {result.get('timestamp', 'Unknown time')[:19]}"):
                st.markdown(f"**Prompt:** {result.get('prompt', 'N/A')[:200]}...")
                st.markdown(f"**Response:** {result.get('response', 'N/A')[:300]}...")
                
                if result.get('analysis'):
                    analysis = result['analysis']
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Tokens", analysis.get('token_count', 0))
                    with cols[1]:
                        st.metric("Sentiment", f"{analysis.get('sentiment', {}).get('score', 0):.2f}")
                    with cols[2]:
                        st.metric("Complexity", f"{analysis.get('complexity', {}).get('score', 0):.2f}")
                    with cols[3]:
                        st.metric("Citations", len(result.get('citations', [])))
    else:
        st.info("No recent activity to display.")
    
    # Clear data option
    st.subheader("Session Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Analysis Results", type="secondary"):
            st.session_state.analysis_results = []
            st.success("Analysis results cleared!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Research Data", type="secondary"):
            st.session_state.research_data = {}
            st.success("Research data cleared!")
            st.rerun()

# Footer
st.markdown("---")
st.markdown("### 🧠 NLP/ML Analysis Platform | Powered by Perplexity API")
st.markdown("*Advanced Language Model Analysis & Research Framework*")
