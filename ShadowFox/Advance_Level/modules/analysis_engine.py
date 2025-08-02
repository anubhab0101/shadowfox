import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import time
import statistics
from datetime import datetime
import re

from modules.gemini_client import GeminiClient
from utils.text_processor import TextProcessor
from utils.rate_limiter import RateLimiter

class AnalysisEngine:
    """Core analysis engine for language model evaluation"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
        self.text_processor = TextProcessor()
        self.rate_limiter = RateLimiter(calls_per_minute=30)  # Conservative rate limiting
        
        # Analysis modules
        self.sentiment_analyzer = None
        self.complexity_analyzer = None
        
        self._initialize_analyzers()
    
    def _initialize_analyzers(self):
        """Initialize text analysis components"""
        try:
            # Initialize basic sentiment analysis
            self.sentiment_analyzer = self._basic_sentiment_analysis
            self.complexity_analyzer = self._complexity_analysis
        except Exception as e:
            print(f"Warning: Could not initialize all analyzers: {e}")
    
    def analyze_prompt(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 500,
        temperature: float = 0.7,
        analyze_sentiment: bool = True,
        analyze_complexity: bool = True,
        analyze_tokens: bool = True,
        analyze_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single prompt-response pair
        """
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Get response from Gemini
            api_response = self.client.simple_query(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract basic response data
            response_content = self.client.extract_response_content(api_response)
            citations = self.client.extract_citations(api_response)
            usage_info = self.client.get_usage_info(api_response)
            response_time = api_response.get('_response_time', 0)
            
            # Check if response content is empty
            if not response_content or len(response_content.strip()) == 0:
                response_content = "No response generated"
            
            # Initialize analysis results
            analysis_results = {
                'response': response_content,
                'citations': citations,
                'usage': usage_info,
                'response_time': response_time,
                'model': model,
                'analysis': {}
            }
            
            # Perform text analysis
            text_analysis = {}
            
            if analyze_tokens:
                text_analysis.update(self._analyze_tokens(response_content))
            
            if analyze_sentiment and self.sentiment_analyzer:
                text_analysis['sentiment'] = self.sentiment_analyzer(response_content)
            
            if analyze_complexity and self.complexity_analyzer:
                text_analysis['complexity'] = self.complexity_analyzer(response_content)
            
            if analyze_citations:
                text_analysis['citations'] = self._analyze_citations(citations)
            
            # Additional metrics
            text_analysis.update(self._calculate_quality_metrics(response_content, citations))
            
            analysis_results['analysis'] = text_analysis
            
            return analysis_results
            
        except Exception as e:
            return {
                'error': str(e),
                'prompt': prompt,
                'model': model
            }
    
    def compare_models(
        self,
        prompt: str,
        models: List[str],
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple models on the same prompt
        """
        comparison_results = []
        
        for model in models:
            try:
                result = self.analyze_prompt(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Extract key metrics for comparison
                if 'analysis' in result:
                    analysis = result['analysis']
                    comparison_data = {
                        'model': model,
                        'response_length': analysis.get('token_count', 0),
                        'word_count': analysis.get('word_count', 0),
                        'sentiment_score': analysis.get('sentiment', {}).get('score', 0),
                        'complexity_score': analysis.get('complexity', {}).get('score', 0),
                        'citation_count': analysis.get('citations', {}).get('count', 0),
                        'response_time': result.get('response_time', 0),
                        'quality_score': analysis.get('quality_score', 0)
                    }
                    comparison_results.append(comparison_data)
                
            except Exception as e:
                comparison_results.append({
                    'model': model,
                    'error': str(e)
                })
        
        return comparison_results
    
    def run_performance_benchmark(
        self,
        models: List[str],
        test_suite: str = "Quick Evaluation",
        iterations: int = 3,
        metrics: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run comprehensive performance benchmark
        """
        if metrics is None:
            metrics = ["Response Time", "Token Efficiency", "Citation Quality"]
        
        # Define test prompts based on suite
        test_prompts = self._get_test_prompts(test_suite)
        
        results = {}
        total_tests = len(models) * len(test_prompts) * iterations
        current_test = 0
        
        for model in models:
            model_results = {
                'response_times': [],
                'token_counts': [],
                'citation_scores': [],
                'quality_scores': [],
                'consistency_scores': [],
                'errors': 0
            }
            
            # Run tests for this model
            model_responses = []
            
            for prompt in test_prompts:
                prompt_responses = []
                
                for iteration in range(iterations):
                    current_test += 1
                    
                    if progress_callback:
                        progress = current_test / total_tests
                        status = f"Testing {model} - Prompt {len(prompt_responses)+1}/{len(test_prompts)} - Iteration {iteration+1}/{iterations}"
                        progress_callback(progress, status)
                    
                    try:
                        result = self.analyze_prompt(
                            prompt=prompt,
                            model=model,
                            max_tokens=500,
                            temperature=0.7
                        )
                        
                        if 'error' not in result:
                            prompt_responses.append(result)
                            
                            # Collect metrics
                            analysis = result.get('analysis', {})
                            model_results['response_times'].append(result.get('response_time', 0))
                            model_results['token_counts'].append(analysis.get('token_count', 0))
                            model_results['citation_scores'].append(analysis.get('citations', {}).get('quality_score', 0))
                            model_results['quality_scores'].append(analysis.get('quality_score', 0))
                        else:
                            model_results['errors'] += 1
                    
                    except Exception as e:
                        model_results['errors'] += 1
                
                model_responses.append(prompt_responses)
            
            # Calculate consistency scores
            for prompt_responses in model_responses:
                if len(prompt_responses) > 1:
                    consistency = self._calculate_consistency(prompt_responses)
                    model_results['consistency_scores'].append(consistency)
            
            # Calculate aggregate metrics
            results[model] = self._calculate_aggregate_metrics(model_results)
        
        return results
    
    def _get_test_prompts(self, test_suite: str) -> List[str]:
        """Get test prompts based on suite type"""
        prompts = {
            "Quick Evaluation": [
                "Explain the concept of machine learning in simple terms.",
                "What are the main advantages of renewable energy?",
                "Describe the process of photosynthesis."
            ],
            "Comprehensive Analysis": [
                "Explain the concept of machine learning in simple terms.",
                "What are the main advantages of renewable energy?",
                "Describe the process of photosynthesis.",
                "Analyze the impact of social media on modern communication.",
                "Compare different programming paradigms.",
                "Explain quantum computing and its potential applications.",
                "Discuss the ethical implications of artificial intelligence.",
                "Describe the economic factors affecting global trade."
            ],
            "Domain-Specific Testing": [
                "Explain neural networks and deep learning architectures.",
                "Describe advanced calculus concepts and applications.",
                "Analyze molecular biology and genetic engineering.",
                "Discuss quantum mechanics and particle physics.",
                "Explain economic theory and market dynamics.",
                "Describe legal principles and constitutional law."
            ],
            "Custom Benchmark": [
                "Write a creative story about time travel.",
                "Solve this math problem: What is the derivative of x^3 + 2x^2 + x + 1?",
                "Analyze this quote: 'The only constant in life is change.'",
                "Explain how to cook a complex dish step by step.",
                "Describe the history and evolution of the internet."
            ]
        }
        
        return prompts.get(test_suite, prompts["Quick Evaluation"])
    
    def _analyze_tokens(self, text: str) -> Dict[str, Any]:
        """Analyze token and word statistics"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'token_count': len(text.split()),  # Simplified token count
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / max(len(sentences), 1)
        }
    
    def _basic_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis using word matching"""
        # Simple sentiment word lists
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'beneficial', 'effective', 'successful'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'negative', 'harmful', 'ineffective', 'failed', 'poor', 'worst'}
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            sentiment_label = "neutral"
        else:
            sentiment_score = (positive_count - negative_count) / len(words)
            if sentiment_score > 0.01:
                sentiment_label = "positive"
            elif sentiment_score < -0.01:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
        
        return {
            'score': sentiment_score,
            'label': sentiment_label,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def _complexity_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity"""
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return {'score': 0, 'level': 'unknown'}
        
        # Flesch Reading Ease approximation
        avg_sentence_length = len(words) / max(len(sentences), 1)
        syllable_counts = [self._count_syllables(word) for word in words]
        avg_syllables = sum(syllable_counts) / len(syllable_counts) if syllable_counts else 1
        
        # Simplified Flesch formula
        complexity_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        
        # Normalize to 0-1 scale
        normalized_score = max(0.0, min(1.0, float((100 - complexity_score) / 100)))
        
        # Determine reading level
        if complexity_score >= 90:
            level = "very_easy"
        elif complexity_score >= 80:
            level = "easy"
        elif complexity_score >= 70:
            level = "fairly_easy"
        elif complexity_score >= 60:
            level = "standard"
        elif complexity_score >= 50:
            level = "fairly_difficult"
        elif complexity_score >= 30:
            level = "difficult"
        else:
            level = "very_difficult"
        
        return {
            'score': normalized_score,
            'level': level,
            'avg_sentence_length': avg_sentence_length,
            'avg_syllables': avg_syllables
        }
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _analyze_citations(self, citations: List[str]) -> Dict[str, Any]:
        """Analyze citation quality and patterns"""
        if not citations:
            return {
                'count': 0,
                'unique_sources': 0,
                'quality_score': 0,
                'domain_diversity': 0
            }
        
        unique_sources = len(set(citations))
        
        # Analyze domains
        domains = []
        for citation in citations:
            try:
                # Extract domain from URL
                if '://' in citation:
                    domain = citation.split('://')[1].split('/')[0]
                    domains.append(domain)
            except:
                pass
        
        unique_domains = len(set(domains))
        domain_diversity = unique_domains / max(len(citations), 1)
        
        # Quality score based on source diversity and reliability
        quality_score = min(1.0, (unique_sources / max(len(citations), 1)) * 0.5 + domain_diversity * 0.5)
        
        return {
            'count': len(citations),
            'unique_sources': unique_sources,
            'quality_score': quality_score,
            'domain_diversity': domain_diversity,
            'domains': list(set(domains))
        }
    
    def _calculate_quality_metrics(self, text: str, citations: List[str]) -> Dict[str, Any]:
        """Calculate overall quality metrics"""
        # Simple quality scoring
        word_count = len(text.split())
        citation_count = len(citations)
        
        # Quality components
        length_score = min(1.0, word_count / 300)  # Optimal around 300 words
        citation_score = min(1.0, citation_count / 5)  # Good citation count
        
        # Overall quality score
        quality_score = (length_score * 0.4 + citation_score * 0.3 + 0.3)  # Base score of 0.3
        
        return {
            'quality_score': quality_score,
            'length_score': length_score,
            'citation_score': citation_score
        }
    
    def _calculate_consistency(self, responses: List[Dict[str, Any]]) -> float:
        """Calculate consistency between multiple responses"""
        if len(responses) < 2:
            return 1.0
        
        # Compare response lengths
        lengths = [resp.get('analysis', {}).get('token_count', 0) for resp in responses]
        length_variance = np.var(lengths) if lengths else 0
        
        # Compare sentiment scores
        sentiments = [resp.get('analysis', {}).get('sentiment', {}).get('score', 0) for resp in responses]
        sentiment_variance = np.var(sentiments) if sentiments else 0
        
        # Consistency score (lower variance = higher consistency)
        consistency = 1.0 / (1.0 + length_variance * 0.001 + sentiment_variance * 10)
        
        return min(1.0, max(0.0, float(consistency)))
    
    def _calculate_aggregate_metrics(self, model_results: Dict[str, List]) -> Dict[str, Any]:
        """Calculate aggregate metrics from model results"""
        metrics = {}
        
        # Response time metrics
        if model_results['response_times']:
            response_times = model_results['response_times']
            metrics['avg_response_time'] = sum(response_times) / len(response_times)
            metrics['min_response_time'] = min(response_times)
            metrics['max_response_time'] = max(response_times)
            # Calculate standard deviation
            mean_rt = metrics['avg_response_time']
            variance = sum((x - mean_rt) ** 2 for x in response_times) / len(response_times)
            metrics['response_time_std'] = variance ** 0.5
        
        # Token efficiency
        if model_results['token_counts']:
            token_counts = model_results['token_counts']
            metrics['avg_tokens'] = sum(token_counts) / len(token_counts)
            mean_tokens = metrics['avg_tokens']
            variance = sum((x - mean_tokens) ** 2 for x in token_counts) / len(token_counts)
            metrics['token_variance'] = variance
            metrics['token_efficiency'] = 1.0 / (1.0 + variance * 0.001)
        
        # Citation quality
        if model_results['citation_scores']:
            citation_scores = model_results['citation_scores']
            metrics['citation_score'] = sum(citation_scores) / len(citation_scores)
        
        # Quality scores
        if model_results['quality_scores']:
            quality_scores = model_results['quality_scores']
            metrics['quality_score'] = sum(quality_scores) / len(quality_scores)
        
        # Consistency
        if model_results['consistency_scores']:
            consistency_scores = model_results['consistency_scores']
            metrics['consistency_score'] = sum(consistency_scores) / len(consistency_scores)
        
        # Error rate
        num_response_times = len(model_results.get('response_times', []))
        num_errors = model_results.get('errors', 0)
        total_tests = num_response_times + num_errors
        metrics['error_rate'] = (num_errors / max(total_tests, 1)) * 100
        
        # Overall score
        component_scores = [
            metrics.get('token_efficiency', 0) * 0.2,
            metrics.get('citation_score', 0) * 0.2,
            metrics.get('quality_score', 0) * 0.2,
            metrics.get('consistency_score', 0) * 0.2,
            max(0, 1 - metrics.get('error_rate', 0) / 100) * 0.2
        ]
        metrics['overall_score'] = sum(component_scores)
        
        return metrics
