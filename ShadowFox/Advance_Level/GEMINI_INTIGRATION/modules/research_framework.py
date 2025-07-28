import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import statistics
import scipy.stats as stats
from datetime import datetime
import random

from modules.gemini_client import GeminiClient

class ResearchFramework:
    """Framework for conducting systematic research studies on language models"""
    
    def __init__(self):
        self.research_questions_db = {
            "Contextual Understanding": [
                "How well does the model maintain context across long conversations?",
                "Does the model's contextual understanding vary with prompt complexity?",
                "How does context length affect response quality and accuracy?",
                "Can the model distinguish between different contextual frameworks?",
                "How does the model handle ambiguous contextual references?"
            ],
            "Creative Text Generation": [
                "What is the relationship between temperature settings and creative output?",
                "How does the model balance creativity with factual accuracy?",
                "Does the model show consistent creative patterns across different domains?",
                "How does prompt structure influence creative response generation?",
                "What factors determine the originality of generated content?"
            ],
            "Domain Adaptation": [
                "How well does the model adapt to specialized technical domains?",
                "Does performance vary significantly across different academic fields?",
                "How does the model handle domain-specific terminology and concepts?",
                "What is the model's performance in cross-domain knowledge transfer?",
                "How does domain expertise affect response quality and accuracy?"
            ],
            "Factual Accuracy": [
                "What is the correlation between citation quantity and factual accuracy?",
                "How does the model's confidence relate to actual accuracy?",
                "Does factual accuracy vary with response length?",
                "How well does the model identify and correct factual errors?",
                "What factors influence the model's ability to distinguish fact from opinion?"
            ],
            "Reasoning Capabilities": [
                "How does the model perform on different types of logical reasoning tasks?",
                "What is the relationship between problem complexity and reasoning accuracy?",
                "How does the model handle multi-step reasoning processes?",
                "Does the model show consistent reasoning patterns across different contexts?",
                "How does reasoning performance correlate with other model capabilities?"
            ]
        }
        
        self.statistical_tests = {
            "Descriptive Statistics": self._descriptive_analysis,
            "T-Test": self._t_test_analysis,
            "ANOVA": self._anova_analysis,
            "Correlation Analysis": self._correlation_analysis,
            "Regression Analysis": self._regression_analysis
        }
    
    def get_research_questions(self, focus_area: str) -> List[str]:
        """Get predefined research questions for a focus area"""
        return self.research_questions_db.get(focus_area, [])
    
    def conduct_study(
        self,
        research_question: str,
        hypothesis: str,
        methodology: List[str],
        sample_size: int,
        control_variables: List[str],
        test_variables: List[str],
        success_metrics: List[str],
        statistical_tests: List[str],
        gemini_client: GeminiClient
    ) -> Dict[str, Any]:
        """
        Conduct a comprehensive research study
        """
        study_results = {
            'research_question': research_question,
            'hypothesis': hypothesis,
            'methodology': methodology,
            'sample_size': sample_size,
            'start_time': datetime.now().isoformat(),
            'data_collection': {},
            'statistical_analysis': {},
            'findings': {},
            'hypothesis_result': {},
            'recommendations': []
        }
        
        try:
            # Generate test scenarios
            test_scenarios = self._generate_test_scenarios(
                sample_size=sample_size,
                control_variables=control_variables,
                test_variables=test_variables
            )
            
            # Collect data
            collected_data = self._collect_research_data(
                test_scenarios=test_scenarios,
                success_metrics=success_metrics,
                gemini_client=gemini_client
            )
            
            study_results['data_collection'] = {
                'scenarios_tested': len(test_scenarios),
                'successful_collections': len(collected_data),
                'data_quality': self._assess_data_quality(collected_data)
            }
            
            # Perform statistical analysis
            statistical_results = {}
            for test_name in statistical_tests:
                if test_name in self.statistical_tests:
                    test_result = self.statistical_tests[test_name](collected_data)
                    statistical_results[test_name] = test_result
            
            study_results['statistical_analysis'] = statistical_results
            
            # Generate findings
            findings = self._generate_findings(collected_data, statistical_results)
            study_results['findings'] = findings
            
            # Evaluate hypothesis
            hypothesis_result = self._evaluate_hypothesis(
                hypothesis=hypothesis,
                findings=findings,
                statistical_results=statistical_results
            )
            study_results['hypothesis_result'] = hypothesis_result
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                findings=findings,
                hypothesis_result=hypothesis_result
            )
            study_results['recommendations'] = recommendations
            
            # Generate summary
            study_results['summary'] = self._generate_study_summary(study_results)
            
            study_results['end_time'] = datetime.now().isoformat()
            study_results['status'] = 'completed'
            
        except Exception as e:
            study_results['error'] = str(e)
            study_results['status'] = 'failed'
            study_results['end_time'] = datetime.now().isoformat()
        
        return study_results
    
    def _generate_test_scenarios(
        self,
        sample_size: int,
        control_variables: List[str],
        test_variables: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate test scenarios for the research study"""
        
        scenarios = []
        
        # Define variable ranges
        variable_ranges = {
            "Temperature": [0.1, 0.3, 0.5, 0.7, 0.9],
            "Max Tokens": [100, 200, 300, 500, 800],
            "Model Type": ["gemini-2.5-flash", "gemini-2.5-pro"],
            "Prompt Structure": ["direct", "detailed", "creative", "analytical"],
            "Domain Context": ["science", "technology", "literature", "history", "general"],
            "Prompt Complexity": ["simple", "moderate", "complex"],
            "Response Length": ["short", "medium", "long"],
            "Citation Requirements": ["none", "minimal", "comprehensive"],
            "Creative vs Factual": ["factual", "balanced", "creative"]
        }
        
        # Base prompts for different categories
        base_prompts = {
            "simple": [
                "What is artificial intelligence?",
                "Explain photosynthesis.",
                "What is democracy?"
            ],
            "moderate": [
                "Compare and contrast machine learning and deep learning approaches.",
                "Analyze the impact of renewable energy on global economics.",
                "Discuss the role of genetics in modern medicine."
            ],
            "complex": [
                "Evaluate the ethical implications of autonomous AI systems in healthcare decision-making.",
                "Analyze the intersection of quantum computing and cryptographic security in the digital age.",
                "Examine the socioeconomic factors influencing climate change adaptation strategies."
            ]
        }
        
        for i in range(sample_size):
            scenario = {
                'scenario_id': i + 1,
                'prompt': self._select_prompt(base_prompts, test_variables),
                'parameters': {}
            }
            
            # Set control variables to consistent values
            for var in control_variables:
                if var in variable_ranges:
                    scenario['parameters'][var] = variable_ranges[var][0]  # Use first value as control
            
            # Vary test variables
            for var in test_variables:
                if var in variable_ranges:
                    scenario['parameters'][var] = random.choice(variable_ranges[var])
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _select_prompt(self, base_prompts: Dict[str, List[str]], test_variables: List[str]) -> str:
        """Select appropriate prompt based on test variables"""
        
        if "Prompt Complexity" in test_variables:
            complexity = random.choice(["simple", "moderate", "complex"])
            return random.choice(base_prompts[complexity])
        else:
            # Default to moderate complexity
            return random.choice(base_prompts["moderate"])
    
    def _collect_research_data(
        self,
        test_scenarios: List[Dict[str, Any]],
        success_metrics: List[str],
        gemini_client: GeminiClient
    ) -> List[Dict[str, Any]]:
        """Collect data for research analysis"""
        
        collected_data = []
        
        for scenario in test_scenarios:
            try:
                # Extract parameters
                prompt = scenario['prompt']
                params = scenario['parameters']
                
                # Set up API call parameters
                model = params.get('Model Type', 'gemini-2.5-flash')
                max_tokens = params.get('Max Tokens', 300)
                temperature = params.get('Temperature', 0.7)
                
                # Make API call
                response = gemini_client.simple_query(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Extract response data
                response_content = gemini_client.extract_response_content(response)
                citations = gemini_client.extract_citations(response)
                usage_info = gemini_client.get_usage_info(response)
                response_time = response.get('_response_time', 0)
                
                # Calculate metrics based on success_metrics
                metrics = self._calculate_research_metrics(
                    response_content=response_content,
                    citations=citations,
                    usage_info=usage_info,
                    response_time=response_time,
                    success_metrics=success_metrics
                )
                
                # Compile data point
                data_point = {
                    'scenario_id': scenario['scenario_id'],
                    'prompt': prompt,
                    'parameters': params,
                    'response': response_content,
                    'citations': citations,
                    'usage': usage_info,
                    'response_time': response_time,
                    'metrics': metrics
                }
                
                collected_data.append(data_point)
                
            except Exception as e:
                # Record failed collection
                collected_data.append({
                    'scenario_id': scenario['scenario_id'],
                    'error': str(e),
                    'failed': True
                })
        
        return collected_data
    
    def _calculate_research_metrics(
        self,
        response_content: str,
        citations: List[str],
        usage_info: Dict[str, int],
        response_time: float,
        success_metrics: List[str]
    ) -> Dict[str, Any]:
        """Calculate research-specific metrics"""
        
        metrics = {}
        
        # Basic text metrics
        words = response_content.split()
        sentences = response_content.split('.')
        
        if "Response Quality" in success_metrics:
            # Simple quality assessment
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            quality_score = min(1.0, (
                (word_count / 200) * 0.4 +  # Optimal around 200 words
                (len(citations) / 3) * 0.3 +  # Good citation count
                (1 / (1 + abs(avg_sentence_length - 15) * 0.1)) * 0.3  # Optimal sentence length
            ))
            
            metrics['response_quality'] = quality_score
        
        if "Factual Accuracy" in success_metrics:
            # Proxy for factual accuracy based on citations and structure
            citation_score = min(1.0, len(citations) / 5)
            structure_score = 1.0 if len(sentences) >= 3 else 0.5
            
            accuracy_score = (citation_score * 0.7 + structure_score * 0.3)
            metrics['factual_accuracy'] = accuracy_score
        
        if "Citation Quality" in success_metrics:
            unique_citations = len(set(citations))
            citation_quality = min(1.0, unique_citations / max(len(citations), 1))
            metrics['citation_quality'] = citation_quality
        
        if "Response Time" in success_metrics:
            metrics['response_time'] = response_time
        
        if "Consistency" in success_metrics:
            # Consistency measured by text structure regularity
            word_lengths = [len(word) for word in words]
            consistency_score = 1.0 / (1.0 + np.var(word_lengths) * 0.1) if word_lengths else 0
            metrics['consistency'] = consistency_score
        
        # Token efficiency
        metrics['token_efficiency'] = usage_info.get('total_tokens', 0) / max(len(words), 1)
        
        return metrics
    
    def _assess_data_quality(self, collected_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of collected research data"""
        
        total_scenarios = len(collected_data)
        successful_collections = len([d for d in collected_data if not d.get('failed', False)])
        
        return {
            'completion_rate': successful_collections / max(total_scenarios, 1),
            'total_scenarios': total_scenarios,
            'successful_collections': successful_collections,
            'failed_collections': total_scenarios - successful_collections
        }
    
    def _descriptive_analysis(self, collected_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform descriptive statistical analysis"""
        
        # Filter successful data
        valid_data = [d for d in collected_data if not d.get('failed', False)]
        
        if not valid_data:
            return {'error': 'No valid data for analysis'}
        
        # Extract metrics
        response_times = [d.get('response_time', 0) for d in valid_data]
        quality_scores = [d.get('metrics', {}).get('response_quality', 0) for d in valid_data]
        token_counts = [d.get('usage', {}).get('total_tokens', 0) for d in valid_data]
        
        results = {
            'sample_size': len(valid_data),
            'response_time': {
                'mean': np.mean(response_times),
                'median': np.median(response_times),
                'std': np.std(response_times),
                'min': np.min(response_times),
                'max': np.max(response_times)
            },
            'quality_scores': {
                'mean': np.mean(quality_scores),
                'median': np.median(quality_scores),
                'std': np.std(quality_scores)
            },
            'token_usage': {
                'mean': np.mean(token_counts),
                'median': np.median(token_counts),
                'std': np.std(token_counts)
            }
        }
        
        return results
    
    def _t_test_analysis(self, collected_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform t-test analysis comparing two groups"""
        
        valid_data = [d for d in collected_data if not d.get('failed', False)]
        
        if len(valid_data) < 4:  # Need at least 2 data points per group
            return {'error': 'Insufficient data for t-test'}
        
        # Split data into two groups (e.g., first half vs second half)
        mid_point = len(valid_data) // 2
        group1 = valid_data[:mid_point]
        group2 = valid_data[mid_point:]
        
        # Compare response quality scores
        group1_scores = [d.get('metrics', {}).get('response_quality', 0) for d in group1]
        group2_scores = [d.get('metrics', {}).get('response_quality', 0) for d in group2]
        
        try:
            t_stat, p_value = stats.ttest_ind(group1_scores, group2_scores)
            
            return {
                'test_type': 'independent_t_test',
                'group1_mean': np.mean(group1_scores),
                'group2_mean': np.mean(group2_scores),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': abs(np.mean(group1_scores) - np.mean(group2_scores)) / 
                              np.sqrt((np.var(group1_scores) + np.var(group2_scores)) / 2)
            }
        except Exception as e:
            return {'error': f'T-test failed: {str(e)}'}
    
    def _anova_analysis(self, collected_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform ANOVA analysis for multiple groups"""
        
        valid_data = [d for d in collected_data if not d.get('failed', False)]
        
        if len(valid_data) < 6:  # Need at least 2 data points per group (3 groups minimum)
            return {'error': 'Insufficient data for ANOVA'}
        
        # Create groups based on model type if available
        groups = {}
        for d in valid_data:
            model = d.get('parameters', {}).get('Model Type', 'default')
            if model not in groups:
                groups[model] = []
            quality_score = d.get('metrics', {}).get('response_quality', 0)
            groups[model].append(quality_score)
        
        # Need at least 2 groups for ANOVA
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for ANOVA'}
        
        try:
            group_values = list(groups.values())
            f_stat, p_value = stats.f_oneway(*group_values)
            
            return {
                'test_type': 'one_way_anova',
                'groups': list(groups.keys()),
                'group_means': {group: np.mean(values) for group, values in groups.items()},
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except Exception as e:
            return {'error': f'ANOVA failed: {str(e)}'}
    
    def _correlation_analysis(self, collected_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform correlation analysis between variables"""
        
        valid_data = [d for d in collected_data if not d.get('failed', False)]
        
        if len(valid_data) < 3:
            return {'error': 'Insufficient data for correlation analysis'}
        
        # Extract variables for correlation
        response_times = [d.get('response_time', 0) for d in valid_data]
        quality_scores = [d.get('metrics', {}).get('response_quality', 0) for d in valid_data]
        token_counts = [d.get('usage', {}).get('total_tokens', 0) for d in valid_data]
        citation_counts = [len(d.get('citations', [])) for d in valid_data]
        
        variables = {
            'response_time': response_times,
            'quality_score': quality_scores,
            'token_count': token_counts,
            'citation_count': citation_counts
        }
        
        correlations = {}
        
        try:
            # Calculate pairwise correlations
            for var1 in variables:
                for var2 in variables:
                    if var1 != var2:
                        corr_coef, p_value = stats.pearsonr(variables[var1], variables[var2])
                        correlations[f"{var1}_vs_{var2}"] = {
                            'correlation': corr_coef,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
            
            return {
                'test_type': 'pearson_correlation',
                'correlations': correlations,
                'sample_size': len(valid_data)
            }
        except Exception as e:
            return {'error': f'Correlation analysis failed: {str(e)}'}
    
    def _regression_analysis(self, collected_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform regression analysis"""
        
        valid_data = [d for d in collected_data if not d.get('failed', False)]
        
        if len(valid_data) < 5:
            return {'error': 'Insufficient data for regression analysis'}
        
        try:
            # Prepare data for regression
            y = [d.get('metrics', {}).get('response_quality', 0) for d in valid_data]  # Dependent variable
            x1 = [d.get('response_time', 0) for d in valid_data]  # Independent variable 1
            x2 = [len(d.get('citations', [])) for d in valid_data]  # Independent variable 2
            
            # Simple linear regression (quality vs response time)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x1, y)
            
            return {
                'test_type': 'linear_regression',
                'dependent_variable': 'response_quality',
                'independent_variable': 'response_time',
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'standard_error': std_err,
                'significant': p_value < 0.05
            }
        except Exception as e:
            return {'error': f'Regression analysis failed: {str(e)}'}
    
    def _generate_findings(
        self,
        collected_data: List[Dict[str, Any]],
        statistical_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate research findings from collected data and statistical analysis"""
        
        valid_data = [d for d in collected_data if not d.get('failed', False)]
        
        findings = {
            'data_summary': {
                'total_samples': len(collected_data),
                'valid_samples': len(valid_data),
                'success_rate': len(valid_data) / max(len(collected_data), 1)
            }
        }
        
        if valid_data:
            # Performance findings
            response_times = [d.get('response_time', 0) for d in valid_data]
            quality_scores = [d.get('metrics', {}).get('response_quality', 0) for d in valid_data]
            
            findings['performance'] = {
                'avg_response_time': np.mean(response_times),
                'avg_quality_score': np.mean(quality_scores),
                'quality_consistency': 1.0 - np.std(quality_scores)  # Higher is more consistent
            }
            
            # Model comparison findings
            models = set(d.get('parameters', {}).get('Model Type', 'Unknown') for d in valid_data)
            if len(models) > 1:
                model_performance = {}
                for model in models:
                    model_data = [d for d in valid_data 
                                if d.get('parameters', {}).get('Model Type') == model]
                    if model_data:
                        model_quality = [d.get('metrics', {}).get('response_quality', 0) 
                                       for d in model_data]
                        model_performance[model] = {
                            'avg_quality': np.mean(model_quality),
                            'sample_count': len(model_data)
                        }
                
                findings['model_comparison'] = model_performance
        
        # Statistical significance findings
        significant_tests = []
        for test_name, results in statistical_results.items():
            if isinstance(results, dict) and results.get('significant', False):
                significant_tests.append(test_name)
        
        findings['statistical_significance'] = {
            'significant_tests': significant_tests,
            'total_tests': len(statistical_results)
        }
        
        return findings
    
    def _evaluate_hypothesis(
        self,
        hypothesis: str,
        findings: Dict[str, Any],
        statistical_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate whether the hypothesis is supported by the findings"""
        
        # Simple heuristic-based hypothesis evaluation
        significant_tests = len(findings.get('statistical_significance', {}).get('significant_tests', []))
        total_tests = findings.get('statistical_significance', {}).get('total_tests', 1)
        
        significance_ratio = significant_tests / max(total_tests, 1)
        
        # Data quality indicators
        success_rate = findings.get('data_summary', {}).get('success_rate', 0)
        
        # Support evaluation
        supported = False
        confidence = 0.0
        
        if significance_ratio >= 0.5 and success_rate >= 0.7:
            supported = True
            confidence = min(95.0, significance_ratio * 80 + success_rate * 15)
        elif significance_ratio >= 0.3:
            confidence = significance_ratio * 60 + success_rate * 10
        else:
            confidence = success_rate * 20
        
        # Extract p-value if available
        p_value = None
        for test_results in statistical_results.values():
            if isinstance(test_results, dict) and 'p_value' in test_results:
                p_value = test_results['p_value']
                break
        
        return {
            'supported': supported,
            'confidence': confidence,
            'significance_ratio': significance_ratio,
            'p_value': p_value,
            'significant': p_value < 0.05 if p_value else False,
            'evidence_strength': 'strong' if confidence > 80 else 'moderate' if confidence > 50 else 'weak'
        }
    
    def _generate_recommendations(
        self,
        findings: Dict[str, Any],
        hypothesis_result: Dict[str, Any]
    ) -> List[str]:
        """Generate research recommendations based on findings"""
        
        recommendations = []
        
        # Data quality recommendations
        success_rate = findings.get('data_summary', {}).get('success_rate', 0)
        if success_rate < 0.8:
            recommendations.append(
                "Improve data collection procedures to increase success rate above 80%"
            )
        
        # Sample size recommendations
        valid_samples = findings.get('data_summary', {}).get('valid_samples', 0)
        if valid_samples < 20:
            recommendations.append(
                "Increase sample size to at least 20 valid data points for more robust analysis"
            )
        
        # Statistical significance recommendations
        if not hypothesis_result.get('supported', False):
            recommendations.append(
                "Consider refining the hypothesis or adjusting experimental design"
            )
        
        # Performance recommendations
        performance = findings.get('performance', {})
        if performance.get('quality_consistency', 0) < 0.7:
            recommendations.append(
                "Investigate factors causing quality inconsistency across responses"
            )
        
        # Model-specific recommendations
        if 'model_comparison' in findings:
            model_perf = findings['model_comparison']
            best_model = max(model_perf.keys(), key=lambda k: model_perf[k]['avg_quality'])
            recommendations.append(
                f"Consider using {best_model} for optimal performance based on quality metrics"
            )
        
        # General research recommendations
        if hypothesis_result.get('confidence', 0) < 70:
            recommendations.append(
                "Conduct additional studies with larger sample sizes to increase confidence"
            )
        
        if not recommendations:
            recommendations.append(
                "Current research design and findings are satisfactory for the stated objectives"
            )
        
        return recommendations
    
    def _generate_study_summary(self, study_results: Dict[str, Any]) -> str:
        """Generate executive summary of the research study"""
        
        hypothesis_result = study_results.get('hypothesis_result', {})
        findings = study_results.get('findings', {})
        
        # Basic study info
        sample_size = findings.get('data_summary', {}).get('valid_samples', 0)
        success_rate = findings.get('data_summary', {}).get('success_rate', 0)
        
        # Hypothesis evaluation
        supported = hypothesis_result.get('supported', False)
        confidence = hypothesis_result.get('confidence', 0)
        
        # Generate summary
        summary_parts = []
        
        summary_parts.append(f"Research study completed with {sample_size} valid data points")
        summary_parts.append(f"(success rate: {success_rate:.1%}).")
        
        if supported:
            summary_parts.append(f"The hypothesis was SUPPORTED with {confidence:.1f}% confidence.")
        else:
            summary_parts.append(f"The hypothesis was NOT SUPPORTED (confidence: {confidence:.1f}%).")
        
        # Performance insights
        performance = findings.get('performance', {})
        if performance:
            avg_quality = performance.get('avg_quality_score', 0)
            summary_parts.append(f"Average response quality score was {avg_quality:.2f}.")
        
        # Statistical significance
        stat_sig = findings.get('statistical_significance', {})
        significant_tests = len(stat_sig.get('significant_tests', []))
        total_tests = stat_sig.get('total_tests', 0)
        
        if total_tests > 0:
            summary_parts.append(
                f"{significant_tests} out of {total_tests} statistical tests showed significance."
            )
        
        return " ".join(summary_parts)
