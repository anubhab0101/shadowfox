import os
import time
from typing import Dict, List, Optional, Any
import json
from google import genai
from google.genai import types

class GeminiClient:
    """Client for interacting with Google Gemini API"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=self.api_key)
        
        # Available models
        self.models = [
            {
                "name": "gemini-2.5-flash",
                "description": "Fast and efficient model for general queries"
            },
            {
                "name": "gemini-2.5-pro", 
                "description": "Most capable model for complex reasoning tasks"
            },
            {
                "name": "gemini-2.0-flash-preview-image-generation",
                "description": "Multimodal model with image generation capabilities"
            }
        ]
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models"""
        return self.models
    
    def check_api_status(self) -> Dict[str, Any]:
        """Check API connectivity and status"""
        try:
            # Test with a simple query
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Test connection"
            )
            
            return {
                "connected": True,
                "status": "operational",
                "credits": "Available (Google API)",
                "response_time": "< 1s"
            }
        except Exception as e:
            return {
                "connected": False,
                "status": "error",
                "error": str(e),
                "credits": "unknown"
            }
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-2.5-flash",
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40
    ) -> Dict[str, Any]:
        """
        Send chat completion request to Gemini API
        """
        try:
            start_time = time.time()
            
            # Convert messages to Gemini format
            contents = []
            system_instruction = None
            
            for message in messages:
                if message["role"] == "system":
                    system_instruction = message["content"]
                elif message["role"] == "user":
                    contents.append(types.Content(
                        role="user", 
                        parts=[types.Part(text=message["content"])]
                    ))
                elif message["role"] == "assistant":
                    contents.append(types.Content(
                        role="model", 
                        parts=[types.Part(text=message["content"])]
                    ))
            
            # Configure generation parameters
            config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction
            )
            
            # Make API call
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            
            response_time = time.time() - start_time
            
            # Format response to match expected structure
            result = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response.text or ""
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                    "total_tokens": response.usage_metadata.total_token_count if response.usage_metadata else 0
                },
                "_response_time": response_time,
                "_model_used": model,
                "citations": []  # Gemini doesn't provide citations like Perplexity
            }
            
            return result
                
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def simple_query(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simple query interface for single prompts
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def extract_response_content(self, api_response: Dict[str, Any]) -> str:
        """Extract the text content from API response"""
        try:
            return api_response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            raise Exception("Invalid API response format")
    
    def extract_citations(self, api_response: Dict[str, Any]) -> List[str]:
        """Extract citations from API response (Gemini doesn't provide citations)"""
        return []  # Gemini API doesn't provide web citations like Perplexity
    
    def get_usage_info(self, api_response: Dict[str, Any]) -> Dict[str, int]:
        """Extract usage information from API response"""
        usage = api_response.get('usage', {})
        return {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0)
        }
    
    def batch_queries(
        self,
        prompts: List[str],
        model: str = "gemini-2.5-flash",
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        delay_between_requests: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple queries with rate limiting
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                result = self.simple_query(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                results.append(result)
                
                # Rate limiting
                if i < len(prompts) - 1:  # Don't delay after last request
                    time.sleep(delay_between_requests)
                    
            except Exception as e:
                results.append({
                    'error': str(e),
                    'prompt': prompt
                })
        
        return results
