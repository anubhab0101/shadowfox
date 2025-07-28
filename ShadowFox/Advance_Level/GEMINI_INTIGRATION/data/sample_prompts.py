"""
Sample prompts and research questions for NLP analysis
"""

SAMPLE_PROMPTS = {
    "Text Generation": [
        "Write a creative story about a robot learning to understand human emotions.",
        "Compose a persuasive essay about the benefits of renewable energy.",
        "Create a technical explanation of how neural networks work.",
        "Write a dialogue between two characters discussing the future of work.",
        "Generate a poem about the intersection of technology and nature."
    ],
    
    "Question Answering": [
        "What are the main differences between supervised and unsupervised learning?",
        "How does climate change affect global food security?",
        "What are the ethical implications of artificial intelligence in healthcare?",
        "Explain the concept of quantum computing and its potential applications.",
        "What factors contribute to economic inequality in modern societies?"
    ],
    
    "Analysis & Reasoning": [
        "Analyze the pros and cons of remote work vs. traditional office environments.",
        "Compare different programming paradigms and their use cases.",
        "Evaluate the impact of social media on political discourse.",
        "Assess the effectiveness of different learning methods for complex subjects.",
        "Examine the relationship between technology advancement and job displacement."
    ],
    
    "Domain-Specific": [
        "Explain the principles of CRISPR gene editing technology.",
        "Describe the mathematical foundations of machine learning algorithms.",
        "Analyze the economic factors affecting cryptocurrency markets.",
        "Discuss the historical significance of the Renaissance period.",
        "Explain the physics behind quantum entanglement."
    ],
    
    "Creative & Open-ended": [
        "If you could design the perfect educational system, what would it look like?",
        "Imagine a world where AI has solved all major global problems. Describe it.",
        "What would be the most important skills for humans in a post-AI world?",
        "Design a solution for sustainable urban transportation in megacities.",
        "How might human society evolve over the next 1000 years?"
    ],
    
    "Complex Reasoning": [
        "A train leaves New York at 3 PM traveling at 80 mph. Another train leaves Boston at 4 PM traveling at 70 mph toward New York. If the distance between cities is 200 miles, when and where will they meet?",
        "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons of water?",
        "If all roses are flowers, and some flowers fade quickly, what can we conclude about roses?",
        "A company's profit increased by 20% in year 1, decreased by 15% in year 2, and increased by 25% in year 3. What was the overall change?",
        "In a group of 100 people, 60 like coffee, 50 like tea, and 30 like both. How many like neither?"
    ]
}

RESEARCH_QUESTIONS = [
    "How does response quality vary with different temperature settings?",
    "What is the relationship between prompt length and response accuracy?",
    "Does the model show consistent performance across different domains?",
    "How does citation quantity correlate with response reliability?",
    "What factors influence the creativity of generated content?",
    "How does model size affect reasoning capability?",
    "What is the impact of context length on response coherence?",
    "Does sentiment in prompts affect response sentiment?",
    "How consistent are responses to identical prompts over time?",
    "What role does prompt structure play in response quality?",
    "How does the model handle ambiguous vs. specific queries?",
    "What is the relationship between response time and quality?",
    "Does the model perform better on factual vs. creative tasks?",
    "How does expertise level of prompts affect response depth?",
    "What factors determine the diversity of generated responses?",
    "How does the model balance accuracy with comprehensiveness?",
    "What is the impact of domain-specific terminology on understanding?",
    "How does response length correlate with information quality?",
    "Does the model show bias in responses across different topics?",
    "What is the relationship between complexity and response confidence?"
]

# Test scenarios for benchmarking
BENCHMARK_SCENARIOS = {
    "contextual_understanding": [
        {
            "prompt": "The man went to the bank. He needed to make a withdrawal.",
            "follow_up": "What type of bank was mentioned?",
            "expected_context": "financial institution"
        },
        {
            "prompt": "The pitcher threw the ball to first base.",
            "follow_up": "What sport is being discussed?",
            "expected_context": "baseball"
        }
    ],
    
    "factual_accuracy": [
        {
            "prompt": "What is the capital of France?",
            "expected_answer": "Paris",
            "difficulty": "easy"
        },
        {
            "prompt": "When was the Treaty of Versailles signed?",
            "expected_answer": "1919",
            "difficulty": "medium"
        },
        {
            "prompt": "What is the molecular formula for caffeine?",
            "expected_answer": "C8H10N4O2",
            "difficulty": "hard"
        }
    ],
    
    "reasoning_tasks": [
        {
            "prompt": "If A > B and B > C, what is the relationship between A and C?",
            "expected_reasoning": "transitive property",
            "answer": "A > C"
        },
        {
            "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
            "expected_reasoning": "literal interpretation",
            "answer": "9"
        }
    ],
    
    "creative_tasks": [
        {
            "prompt": "Write the opening line of a mystery novel.",
            "evaluation_criteria": ["originality", "intrigue", "clarity"],
            "task_type": "creative_writing"
        },
        {
            "prompt": "Invent a new sport that combines elements of chess and basketball.",
            "evaluation_criteria": ["creativity", "feasibility", "detail"],
            "task_type": "creative_problem_solving"
        }
    ]
}

# Domain-specific test prompts
DOMAIN_PROMPTS = {
    "science": [
        "Explain the process of photosynthesis at the molecular level.",
        "Describe the relationship between DNA, RNA, and protein synthesis.",
        "What is the significance of the Higgs boson particle?",
        "How do vaccines work to provide immunity?",
        "Explain the theory of evolution through natural selection."
    ],
    
    "technology": [
        "How do blockchain networks achieve consensus?",
        "What are the key differences between SQL and NoSQL databases?",
        "Explain the principles of object-oriented programming.",
        "How does public key cryptography ensure security?",
        "What is the difference between AI, ML, and deep learning?"
    ],
    
    "mathematics": [
        "Prove that the square root of 2 is irrational.",
        "Explain the fundamental theorem of calculus.",
        "What is the significance of Euler's identity?",
        "Describe the applications of linear algebra in data science.",
        "How do you solve a system of linear equations?"
    ],
    
    "literature": [
        "Analyze the themes in Shakespeare's Hamlet.",
        "Compare the writing styles of Hemingway and Faulkner.",
        "What is the significance of symbolism in poetry?",
        "Discuss the evolution of the novel as a literary form.",
        "How does cultural context influence literary interpretation?"
    ],
    
    "history": [
        "What were the causes of World War I?",
        "Analyze the impact of the Industrial Revolution on society.",
        "How did the Renaissance change European culture?",
        "What role did trade routes play in ancient civilizations?",
        "Discuss the significance of the Scientific Revolution."
    ]
}

# Performance testing templates
PERFORMANCE_TEST_TEMPLATES = {
    "speed_tests": [
        "Quick fact: What is {fact_query}?",
        "Brief explanation: {concept}",
        "Short answer: {question}"
    ],
    
    "complexity_tests": [
        "Comprehensive analysis: {complex_topic}",
        "Detailed comparison: {item1} vs {item2}",
        "In-depth explanation: {advanced_concept}"
    ],
    
    "consistency_tests": [
        "Explain {concept} in simple terms.",
        "What is {concept}?",
        "Describe {concept} for a beginner."
    ]
}

# Citation testing prompts
CITATION_TEST_PROMPTS = [
    "What are the latest developments in renewable energy technology?",
    "Summarize recent research on artificial intelligence safety.",
    "What are the current COVID-19 vaccination recommendations?",
    "Describe the most recent discoveries in astronomy.",
    "What are the latest economic indicators and their implications?",
    "Summarize recent advances in cancer treatment.",
    "What are the current trends in sustainable agriculture?",
    "Describe recent developments in quantum computing research.",
    "What are the latest findings on climate change impacts?",
    "Summarize recent progress in gene therapy treatments."
]
