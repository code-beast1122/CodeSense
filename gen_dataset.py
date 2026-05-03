import requests
import json
import pandas as pd
import time

# 1. OPENROUTER CONFIGURATION
API_KEY = "Api-Key-Here"
MODEL_ID = "openrouter/free" 

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://localhost.com",
    "X-Title": "AI Complexity Project"
}

complexities = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(V + E)"]
dataset = []

print("🧠 Starting Knowledge Distillation from OpenRouter (Llama 3.3 70B)...")

# 2. GENERATION LOOP
for big_o in complexities:
    print(f"\nGenerating messy, real-world {big_o} algorithms...")
    
    for batch in range(5): 
        prompt = f"""
        You are a senior C++ engineer. Generate 10 highly unique, messy, real-world C++ functions that strictly have a time complexity of {big_o}. 
        Do NOT use standard boring templates (like 'foo' or 'matrix'). Use realistic scenarios (e.g., parsing strings, rendering game frames, database indexing).
        
        Return ONLY a valid JSON array of objects, exactly like this, with NO markdown formatting around it:
        [
            {{"code_snippet": "void process_physics() {{ ... }}", "time_complexity": "{big_o}"}}
        ]
        """
        
        data = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            # Ping the OpenRouter API
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            response_json = response.json()
            
            # --- THE FIX: Catch the actual API error ---
            if 'error' in response_json:
                print(f"   -> API Error: {response_json['error'].get('message', response_json['error'])}")
                time.sleep(3)
                continue
                
            raw_text = response_json['choices'][0]['message']['content'].strip()
            
            # Clean up Markdown backticks if the LLM accidentally adds them
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()
                
            # Parse and save
            batch_data = json.loads(raw_text)
            dataset.extend(batch_data)
            
            print(f"   -> Batch {batch+1}/5 successful! ({len(batch_data)} snippets harvested)")
            
            # Pause so we don't hit the free-tier rate limit
            time.sleep(3) 
            
        except json.JSONDecodeError:
            print("   -> Batch failed: The AI did not return perfectly formatted JSON.")
        except Exception as e:
            print(f"   -> Batch failed. Error: {e}")

# 3. SAVE THE GOLDEN DATASET
print("\nSaving the distilled dataset...")
df = pd.DataFrame(dataset)

# Shuffle the rows so CodeBERT learns logic, not patterns
df = df.sample(frac=1).reset_index(drop=True)

filename = "cpp_data_cleaned_final.csv"
df.to_csv(filename, index=False)

print(f"✅ Distillation Complete! Saved {len(df)} enterprise-grade snippets to {filename}.")