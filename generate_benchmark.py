import requests
from bs4 import BeautifulSoup
import ollama
import json
import re
import random
import time

# --- CONFIGURATION ---
IEP_URL = "https://iep.utm.edu/schopen/" 
OUTPUT_FILE = "schopenhauer_bench.json"
LLM_MODEL = "llama3"


# --- 1. Scraper ---
def scrape_iep():
    print(f"⬇️ Scraping {IEP_URL}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(IEP_URL, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        content_div = soup.find('div', class_='entry-content')
        
        if not content_div:
            print("⚠️ Could not find 'entry-content'. Dumping all text.")
            return soup.get_text()[:20000]

        main_text = content_div.get_text(separator=' ', strip=True)
        clean_text = re.sub(r'\s+', ' ', main_text).strip()
        
        print(f"   -> Successfully scraped {len(clean_text)} characters.")
        return clean_text
        
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return "Schopenhauer's philosophy is centered on the Will as the thing-in-itself..."

# --- 2. Generator (ROBUST FIX APPLIED) ---
def generate_questions(text, category, count=10):
    if len(text) < 500:
        print(f"⚠️ Text too short for {category}. Skipping generation.")
        return []

    print(f"🧠 Generating {count} {category} questions... (This may take 1-2 minutes)")
    
    safe_text = text[:6000] 

    prompt = f"""
    You are an expert philosophy professor creating an evaluation dataset.
    Based ONLY on the text provided below, generate {count} distinct, complex questions about Schopenhauer's {category}.
    
    For each question, provide a "Golden Answer" (a list of 3-4 key facts that MUST be in the correct answer).

    RETURN ONLY A VALID JSON LIST OF OBJECTS. Do not return a dictionary with keys.
    Format:
    [
      {{
        "id": "{category}_1",
        "question": "...",
        "golden_answer_points": ["Fact 1", "Fact 2", "Fact 3"],
        "type": "valid"
      }}
    ]

    TEXT SOURCE:
    {safe_text}
    """

    try:
        response = ollama.chat(
            model=LLM_MODEL, 
            messages=[{'role': 'user', 'content': prompt}], 
            format='json', 
            options={'temperature': 0.7}
        )
        
        raw_content = response['message']['content']

        # attempt top-level parse first
        try:
            data = json.loads(raw_content)
        except Exception:
            # fallback: try to extract JSON substring
            m = re.search(r'(\[.*\]|\{.*\})', raw_content, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                except Exception:
                    data = raw_content
            else:
                data = raw_content

        # recursive extractor: collects dicts that look like question objects
        def extract_dicts(obj):
            results = []
            if isinstance(obj, dict):
                # if this looks like a question object, add it
                if any(k in obj for k in ('question', 'golden_answer_points', 'id')):
                    results.append(obj)
                # recurse into values
                for v in obj.values():
                    results.extend(extract_dicts(v))
            elif isinstance(obj, list):
                for item in obj:
                    results.extend(extract_dicts(item))
            elif isinstance(obj, str):
                s = obj.strip()
                if (s.startswith('{') or s.startswith('[')):
                    try:
                        parsed = json.loads(s)
                        results.extend(extract_dicts(parsed))
                    except Exception:
                        pass
            return results

        parsed_items = extract_dicts(data)

        # If nothing parsed, warn and return
        if not parsed_items:
            print("⚠️ No usable question objects parsed from LLM output.")
            preview = raw_content[:400].replace('\n',' ')
            print(f"   -> Preview: {preview}...")
            return []

        # Normalize and ensure ids/types, ensure unique ids
        clean_data = []
        existing_ids = set()
        for i, item in enumerate(parsed_items):
            if not isinstance(item, dict):
                continue
            # keep original fields but ensure id and type
            orig_id = str(item.get('id') or "").strip()
            if orig_id:
                assigned_id = orig_id
            else:
                assigned_id = f"{category}_{i+1}"
            # uniquify
            base = assigned_id
            suffix = 1
            while assigned_id in existing_ids:
                assigned_id = f"{base}_{suffix}"
                suffix += 1
            item['id'] = assigned_id
            existing_ids.add(assigned_id)

            item['type'] = item.get('type', 'valid')
            clean_data.append(item)

        print(f"   -> Parsed {len(clean_data)} question objects for {category}.")
        return clean_data

    except Exception as e:
        print(f"⚠️ Error generating {category}: {e}")
        return []

# --- 3. Hardcoded Adversarial Questions ---
TRICK_QUESTIONS = [
    {"id": "adv_1", "question": "How does Schopenhauer integrate Quantum Mechanics into his metaphysics?", "golden_answer_points": ["Schopenhauer died in 1860", "Predates quantum mechanics", "Cannot integrate it"], "type": "adversarial"},
    {"id": "adv_2", "question": "Why does Schopenhauer argue that suicide is a moral sin according to the Bible?", "golden_answer_points": ["Schopenhauer rejects Biblical moral authority", "Suicide is not a sin but a mistake", "Does not use Bible as proof"], "type": "adversarial"},
    {"id": "adv_3", "question": "Explain Schopenhauer's unwavering support for Hegel's philosophy.", "golden_answer_points": ["Schopenhauer hated Hegel", "Considered Hegel a 'charlatan'", "Opposed Hegel's philosophy"], "type": "adversarial"},
    {"id": "adv_4", "question": "What is Schopenhauer's view on the benefits of optimistic thinking?", "golden_answer_points": ["Schopenhauer is a pessimist", "Optimism is 'wicked' or 'absurd'", "Life is suffering"], "type": "adversarial"},
    {"id": "adv_5", "question": "How does the categorical imperative form the basis of Schopenhauer's ethics?", "golden_answer_points": ["Rejects Kant's categorical imperative", "Ethics based on Compassion", "Morality is not duty-based"], "type": "adversarial"},
    {"id": "adv_6", "question": "In which book does Schopenhauer praise the industrial revolution?", "golden_answer_points": ["Did not focus on industrial revolution", "Focus is on timeless suffering", "No specific praise found"], "type": "adversarial"},
    {"id": "adv_7", "question": "What represents the 'Thing-in-Itself' according to Schopenhauer: Logic or Math?", "golden_answer_points": ["Neither Logic nor Math", "The Will is the thing-in-itself", "Logic/Math are part of Representation"], "type": "adversarial"},
    {"id": "adv_8", "question": "Describe Schopenhauer's detailed political theory for democracy.", "golden_answer_points": ["Had no detailed political theory", "Preferred monarchy/order to chaos", "Pessimistic about political change"], "type": "adversarial"},
    {"id": "adv_9", "question": "Why did Schopenhauer marry and have a large family?", "golden_answer_points": ["Never married", "Had no legitimate children", "Lived a solitary life"], "type": "adversarial"},
    {"id": "adv_10", "question": "How does Schopenhauer define 'The Will' as a conscious, rational planner?", "golden_answer_points": ["Will is blind", "Will is irrational", "Will is unconscious striving"], "type": "adversarial"}
]

if __name__ == "__main__":
    # 1. Scrape
    text = scrape_iep()
    
    # 2. Split
    chunk_size = len(text) // 3
    part1 = text[:chunk_size]
    part2 = text[chunk_size : 2*chunk_size]
    part3 = text[2*chunk_size:]

    # 3. Generate with safety
    full_dataset = []

    # Helper to append safely
    def safe_generate(txt, cat):
        try:
            res = generate_questions(txt, cat, 10)
            if res: return res
        except Exception as e:
            print(f"Skipping {cat} due to error: {e}")
        return []

    full_dataset.extend(safe_generate(part1, "Metaphysics"))
    full_dataset.extend(safe_generate(part2, "Aesthetics"))
    full_dataset.extend(safe_generate(part3, "Ethics"))
    
    # Add the tricks
    full_dataset.extend(TRICK_QUESTIONS)

    print(f"Total questions generated: {len(full_dataset)}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=2, ensure_ascii=False)