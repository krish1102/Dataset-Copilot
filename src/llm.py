import os
import httpx
import json

def get_ollama_models(host: str = "http://localhost:11434") -> tuple[bool, dict[str, str]]:
    """Checks if Ollama is running and catalogs available models by role (fast vs reasoning)."""
    roles = {"fast": "llama3", "reasoning": "deepseek-r1:7b"}
    try:
        url = f"{host.rstrip('/')}/api/tags"
        response = httpx.get(url, timeout=2.0)
        if response.status_code == 200:
            models_data = response.json().get("models", [])
            if models_data:
                names = [m["name"] for m in models_data]
                
                # 1. Look for a reasoning model (DeepSeek, R1, Reasoning)
                reasoning_model = None
                for n in names:
                    n_lower = n.lower()
                    if "deepseek" in n_lower or "r1" in n_lower or "reason" in n_lower:
                        reasoning_model = n
                        break
                
                # 2. Look for a fast model (Qwen, Llama, Gemma, Mistral, Phi)
                fast_model = None
                for n in names:
                    n_lower = n.lower()
                    if any(kw in n_lower for kw in ["qwen", "llama", "gemma", "mistral", "phi"]):
                        fast_model = n
                        break
                
                # Assign with fallbacks
                if fast_model:
                    roles["fast"] = fast_model
                else:
                    roles["fast"] = names[0]
                    
                if reasoning_model:
                    roles["reasoning"] = reasoning_model
                else:
                    roles["reasoning"] = roles["fast"] # Fallback if no reasoning model
                    
                return True, roles
            return True, roles
    except Exception:
        pass
    return False, roles

def query_llm(prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.3, response_format: str | None = None, model_preference: str = "fast") -> str | None:
    """Queries OpenAI (if key present) or local Ollama (routed by model_preference)."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    # 1. Try OpenAI if key is present
    if openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            }
            # Route model based on preference if OpenAI is used
            model = "gpt-4o" if model_preference == "reasoning" else "gpt-4-turbo"
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }
            if response_format == "json":
                payload["response_format"] = {"type": "json_object"} # type: ignore
                
            response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # 2. Try Ollama (routed dynamically)
    ollama_available, roles = get_ollama_models(ollama_host)
    if ollama_available:
        try:
            url = f"{ollama_host.rstrip('/')}/api/generate"
            ollama_model = roles.get(model_preference, roles["fast"])
            
            # Format prompt with system instructions for Ollama's non-chat generate endpoint
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            payload = {
                "model": ollama_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            if response_format == "json":
                payload["format"] = "json"
                
            response = httpx.post(url, json=payload, timeout=45.0)
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception:
            pass
            
    return None

def query_llm_chat(messages: list, temperature: float = 0.5, model_preference: str = "fast") -> str | None:
    """Handles multi-turn conversation messages utilizing OpenAI or local Ollama (routed by model_preference)."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    # 1. Try OpenAI
    if openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            }
            model = "gpt-4o" if model_preference == "reasoning" else "gpt-4-turbo"
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            response = httpx.post(url, headers=headers, json=payload, timeout=20.0)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
            
    # 2. Try Ollama
    ollama_available, roles = get_ollama_models(ollama_host)
    if ollama_available:
        try:
            ollama_model = roles.get(model_preference, roles["fast"])
            
            # Build simple chat prompt format for generate endpoint
            chat_str = ""
            for msg in messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                chat_str += f"{role}: {content}\n"
            chat_str += "Assistant:"
            
            url = f"{ollama_host.rstrip('/')}/api/generate"
            payload = {
                "model": ollama_model,
                "prompt": chat_str,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            response = httpx.post(url, json=payload, timeout=30.0)
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception:
            pass
            
    return None
