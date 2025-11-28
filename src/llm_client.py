"""LLM Client abstraction supporting free LLM providers"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Abstraction for free LLM providers (Ollama, Groq)"""
    
    def __init__(self, provider: str = "groq"):
        """Initialize LLM client"""
        self.provider = provider.lower()
        self.available = True
        self._unavailable_reason = None
        
        if self.provider == "ollama":
            import ollama
            try:
                pass
            except Exception as e:
                self.available = False
                self._unavailable_reason = f"Ollama unavailable: {e}"
                return

            self.client = ollama
            self.model = "llama3.2"
            
        elif self.provider == "groq":
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                self.available = False
                self._unavailable_reason = "GROQ_API_KEY missing"
                return
            
            self.client = Groq(api_key=api_key)
            self.model = "llama-3.1-8b-instant"
            
        else:
            raise ValueError(f"Unknown provider: {provider}. Use: ollama or groq")
    

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt"""
        if not self.is_available():
            raise RuntimeError(f"{self.provider} unavailable: {self._unavailable_reason}")

        if self.provider == "ollama":
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response['message']['content']
            except Exception as e:
                raise ConnectionError(f"Ollama error: {e}. Make sure Ollama is running: 'ollama serve'")
        
        elif self.provider == "groq":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content


    def is_available(self) -> bool:
        return bool(self.available)