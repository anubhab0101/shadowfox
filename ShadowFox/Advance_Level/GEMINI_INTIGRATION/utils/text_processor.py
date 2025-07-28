import re
import string
import math
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import statistics

# Download required NLTK data (with error handling)
def download_nltk_data():
    """Download required NLTK data with error handling"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass  # Fallback to basic tokenization
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            pass
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        try:
            nltk.download('wordnet', quiet=True)
        except:
            pass
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        try:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
    
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        try:
            nltk.download('maxent_ne_chunker', quiet=True)
        except:
            pass

# Initialize NLTK data
download_nltk_data()

class TextProcessor:
    """Comprehensive text processing utilities for NLP analysis"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stopwords if NLTK data not available
            self.stop_words = set([
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 
                'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 
                'the', 'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 
                'they', 'have', 'had', 'what', 'said', 'each', 'which', 'their'
            ])
        
        try:
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.stemmer = None
            self.lemmatizer = None
        
        # Sentiment word lists (basic implementation)
        self.positive_words = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'best',
            'outstanding', 'superb', 'brilliant', 'perfect', 'awesome', 'incredible',
            'magnificent', 'remarkable', 'exceptional', 'impressive', 'positive',
            'beneficial', 'effective', 'successful', 'valuable', 'helpful', 'useful',
            'important', 'significant', 'powerful', 'strong', 'robust', 'solid'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'poor', 'disappointing',
            'inadequate', 'insufficient', 'weak', 'problematic', 'concerning',
            'troubling', 'alarming', 'dangerous', 'harmful', 'damaging', 'negative',
            'ineffective', 'useless', 'worthless', 'failed', 'broken', 'flawed',
            'corrupt', 'dishonest', 'unfair', 'wrong', 'false', 'incorrect'
        }
        
        # Complexity indicators
        self.complex_indicators = {
            'academic_terms': {
                'furthermore', 'moreover', 'consequently', 'nevertheless', 'nonetheless',
                'therefore', 'accordingly', 'subsequently', 'specifically', 'particularly',
                'essentially', 'fundamentally', 'conceptually', 'theoretically',
                'empirically', 'systematically', 'comprehensively', 'methodology',
                'analysis', 'synthesis', 'hypothesis', 'paradigm', 'phenomenon'
            },
            'technical_terms': {
                'algorithm', 'implementation', 'optimization', 'infrastructure',
                'architecture', 'framework', 'specification', 'configuration',
                'parameter', 'variable', 'function', 'procedure', 'protocol',
                'interface', 'component', 'module', 'system', 'network'
            }
        }
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences"""
        try:
            return sent_tokenize(text)
        except:
            # Fallback sentence tokenization
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        try:
            return word_tokenize(text)
        except:
            # Fallback word tokenization
            # Remove punctuation and split on whitespace
            text = re.sub(r'[^\w\s]', ' ', text)
            return text.split()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
        
        return text.strip()
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Remove stopwords from word list"""
        return [word for word in words if word.lower() not in self.stop_words]
    
    def stem_words(self, words: List[str]) -> List[str]:
        """Apply stemming to words"""
        if not self.stemmer:
            return words
        return [self.stemmer.stem(word) for word in words]
    
    def lemmatize_words(self, words: List[str]) -> List[str]:
        """Apply lemmatization to words"""
        if not self.lemmatizer:
            return words
        return [self.lemmatizer.lemmatize(word) for word in words]
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """Extract keywords from text using frequency analysis"""
        words = self.tokenize_words(self.clean_text(text))
        words = self.remove_stopwords(words)
        
        # Filter out very short words and numbers
        words = [word for word in words if len(word) > 2 and not word.isdigit()]
        
        # Count frequency
        word_freq = Counter(words)
        
        return word_freq.most_common(top_n)
    
    def calculate_readability(self, text: str) -> Dict[str, Any]:
        """Calculate various readability metrics"""
        sentences = self.tokenize_sentences(text)
        words = self.tokenize_words(text)
        
        if not sentences or not words:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'automated_readability_index': 0,
                'difficulty_level': 'unknown'
            }
        
        # Basic metrics
        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        # Avoid division by zero
        avg_sentence_length = total_words / max(total_sentences, 1)
        avg_syllables_per_word = total_syllables / max(total_words, 1)
        
        # Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        
        # Automated Readability Index
        avg_characters_per_word = sum(len(word) for word in words) / max(total_words, 1)
        ari = (4.71 * avg_characters_per_word) + (0.5 * avg_sentence_length) - 21.43
        
        # Determine difficulty level
        if flesch_reading_ease >= 90:
            difficulty = 'very_easy'
        elif flesch_reading_ease >= 80:
            difficulty = 'easy'
        elif flesch_reading_ease >= 70:
            difficulty = 'fairly_easy'
        elif flesch_reading_ease >= 60:
            difficulty = 'standard'
        elif flesch_reading_ease >= 50:
            difficulty = 'fairly_difficult'
        elif flesch_reading_ease >= 30:
            difficulty = 'difficult'
        else:
            difficulty = 'very_difficult'
        
        return {
            'flesch_reading_ease': max(0, min(100, flesch_reading_ease)),
            'flesch_kincaid_grade': max(0, flesch_kincaid_grade),
            'automated_readability_index': max(0, ari),
            'difficulty_level': difficulty,
            'avg_sentence_length': avg_sentence_length,
            'avg_syllables_per_word': avg_syllables_per_word
        }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using a simple heuristic"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        
        if word[0] in vowels:
            count += 1
        
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        
        if word.endswith("e"):
            count -= 1
        
        if count == 0:
            count = 1
        
        return count
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Perform basic sentiment analysis"""
        words = self.tokenize_words(self.clean_text(text))
        
        positive_count = 0
        negative_count = 0
        positive_matches = []
        negative_matches = []
        
        for word in words:
            if word in self.positive_words:
                positive_count += 1
                positive_matches.append(word)
            elif word in self.negative_words:
                negative_count += 1
                negative_matches.append(word)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            sentiment_label = "neutral"
            confidence = 0.5
        else:
            # Calculate sentiment score
            sentiment_score = (positive_count - negative_count) / len(words)
            confidence = total_sentiment_words / max(len(words), 1)
            
            # Determine label
            if sentiment_score > 0.01:
                sentiment_label = "positive"
            elif sentiment_score < -0.01:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
        
        return {
            'score': sentiment_score,
            'label': sentiment_label,
            'confidence': min(1.0, confidence * 10),  # Scale confidence
            'positive_words_found': positive_matches,
            'negative_words_found': negative_matches,
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    
    def analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity beyond readability"""
        words = self.tokenize_words(self.clean_text(text))
        sentences = self.tokenize_sentences(text)
        
        if not words:
            return {
                'complexity_score': 0,
                'complexity_level': 'unknown',
                'factors': {}
            }
        
        factors = {}
        
        # Vocabulary complexity
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / len(words)
        factors['vocabulary_diversity'] = vocabulary_diversity
        
        # Word length complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        factors['avg_word_length'] = avg_word_length
        
        # Academic/Technical term usage
        academic_count = sum(1 for word in words if word in self.complex_indicators['academic_terms'])
        technical_count = sum(1 for word in words if word in self.complex_indicators['technical_terms'])
        
        complex_term_ratio = (academic_count + technical_count) / len(words)
        factors['complex_term_ratio'] = complex_term_ratio
        
        # Sentence structure complexity
        if sentences:
            sentence_lengths = [len(self.tokenize_words(sent)) for sent in sentences]
            sentence_length_variance = statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0
            factors['sentence_length_variance'] = sentence_length_variance
            
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
            factors['avg_sentence_length'] = avg_sentence_length
        else:
            factors['sentence_length_variance'] = 0
            factors['avg_sentence_length'] = 0
        
        # Punctuation complexity (indicates complex sentence structures)
        punctuation_count = sum(1 for char in text if char in ',:;()-')
        punctuation_ratio = punctuation_count / len(text) if text else 0
        factors['punctuation_ratio'] = punctuation_ratio
        
        # Calculate overall complexity score
        complexity_components = [
            vocabulary_diversity * 0.25,
            min(1.0, avg_word_length / 7) * 0.2,  # Normalize word length
            complex_term_ratio * 3 * 0.3,  # Weight academic/technical terms heavily
            min(1.0, factors['sentence_length_variance'] / 50) * 0.15,  # Normalize variance
            min(1.0, punctuation_ratio * 20) * 0.1  # Normalize punctuation
        ]
        
        complexity_score = sum(complexity_components)
        
        # Determine complexity level
        if complexity_score >= 0.8:
            complexity_level = 'very_high'
        elif complexity_score >= 0.6:
            complexity_level = 'high'
        elif complexity_score >= 0.4:
            complexity_level = 'moderate'
        elif complexity_score >= 0.2:
            complexity_level = 'low'
        else:
            complexity_level = 'very_low'
        
        return {
            'complexity_score': min(1.0, complexity_score),
            'complexity_level': complexity_level,
            'factors': factors
        }
    
    def extract_named_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text"""
        try:
            words = word_tokenize(text)
            pos_tags = pos_tag(words)
            chunks = ne_chunk(pos_tags)
            
            entities = []
            current_entity = []
            current_label = None
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):  # It's a named entity
                    if current_label != chunk.label():
                        if current_entity:
                            entities.append({
                                'text': ' '.join(current_entity),
                                'label': current_label
                            })
                        current_entity = [word for word, pos in chunk.leaves()]
                        current_label = chunk.label()
                    else:
                        current_entity.extend([word for word, pos in chunk.leaves()])
                else:  # It's a regular word
                    if current_entity:
                        entities.append({
                            'text': ' '.join(current_entity),
                            'label': current_label
                        })
                        current_entity = []
                        current_label = None
            
            # Add final entity if exists
            if current_entity:
                entities.append({
                    'text': ' '.join(current_entity),
                    'label': current_label
                })
            
            return entities
        
        except Exception:
            # Fallback: simple pattern-based entity extraction
            entities = []
            
            # Simple patterns for common entities
            patterns = {
                'ORGANIZATION': r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Organization|University|Institute)\b',
                'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                'LOCATION': r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*(?: City| State| Country)?\b',
                'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b',
            }
            
            for label, pattern in patterns.items():
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'label': label
                    })
            
            return entities
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get comprehensive text statistics"""
        words = self.tokenize_words(text)
        sentences = self.tokenize_sentences(text)
        
        # Basic counts
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        word_count = len(words)
        sentence_count = len(sentences)
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Advanced metrics
        unique_words = len(set(word.lower() for word in words))
        lexical_diversity = unique_words / max(word_count, 1)
        
        # Average lengths
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Word length distribution
        word_lengths = [len(word) for word in words]
        word_length_stats = {
            'mean': statistics.mean(word_lengths) if word_lengths else 0,
            'median': statistics.median(word_lengths) if word_lengths else 0,
            'mode': statistics.mode(word_lengths) if word_lengths else 0,
            'std': statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0
        }
        
        return {
            'character_count': char_count,
            'character_count_no_spaces': char_count_no_spaces,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'unique_word_count': unique_words,
            'lexical_diversity': lexical_diversity,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'word_length_stats': word_length_stats
        }
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare two texts and return similarity metrics"""
        # Get basic statistics for both texts
        stats1 = self.get_text_statistics(text1)
        stats2 = self.get_text_statistics(text2)
        
        # Tokenize and get word sets
        words1 = set(word.lower() for word in self.tokenize_words(self.clean_text(text1)))
        words2 = set(word.lower() for word in self.tokenize_words(self.clean_text(text2)))
        
        # Jaccard similarity (word overlap)
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_similarity = intersection / max(union, 1)
        
        # Cosine similarity (simplified)
        all_words = words1.union(words2)
        vector1 = [1 if word in words1 else 0 for word in all_words]
        vector2 = [1 if word in words2 else 0 for word in all_words]
        
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))
        
        cosine_similarity = dot_product / max(magnitude1 * magnitude2, 1)
        
        # Statistical comparisons
        length_ratio = min(stats1['word_count'], stats2['word_count']) / max(stats1['word_count'], stats2['word_count'])
        
        return {
            'jaccard_similarity': jaccard_similarity,
            'cosine_similarity': cosine_similarity,
            'word_overlap_count': intersection,
            'unique_to_text1': len(words1 - words2),
            'unique_to_text2': len(words2 - words1),
            'length_ratio': length_ratio,
            'statistical_comparison': {
                'word_count_diff': abs(stats1['word_count'] - stats2['word_count']),
                'avg_word_length_diff': abs(stats1['avg_word_length'] - stats2['avg_word_length']),
                'lexical_diversity_diff': abs(stats1['lexical_diversity'] - stats2['lexical_diversity'])
            }
        }
