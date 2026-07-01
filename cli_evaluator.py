import requests
import json

API_URL = "http://127.0.0.1:8000/chat"

def main():
    print("--- SHL Recommender CLI Evaluator ---")
    print("Ensure you have run 'uvicorn main:app --reload' in another terminal.")
    print("Type 'exit' to stop.\n")
    
    messages = []
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        payload = {"messages": messages}
        
        try:
            # We enforce a 30s timeout here just like the SHL Evaluator harness
            response = requests.post(API_URL, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                print(f"\nAgent: {data.get('reply')}")
                
                recs = data.get('recommendations', [])
                if recs:
                    print(f"\n[Shortlist Provided ({len(recs)} items):]")
                    for i, r in enumerate(recs, 1):
                        print(f"  {i}. {r.get('name')} (Type: {r.get('test_type')})")
                        print(f"     {r.get('url')}")
                
                if data.get('end_of_conversation'):
                    print("\n[Agent flagged: END_OF_CONVERSATION]")
                
                print("-" * 50)
                
                # Append assistant reply to the state history
                messages.append({"role": "assistant", "content": data.get('reply', '')})
                
            else:
                print(f"\n[API Error {response.status_code}]: {response.text}")
                messages.pop() # Remove failed message
                
        except requests.exceptions.Timeout:
            print("\n[FAILED] Request exceeded the 30-second timeout limit!")
            messages.pop()
        except requests.exceptions.ConnectionError:
            print("\n[FAILED] Could not connect to API. Is uvicorn running?")
            messages.pop()
        except Exception as e:
            print(f"\n[FAILED] Request error: {e}")
            messages.pop()

if __name__ == "__main__":
    main()
