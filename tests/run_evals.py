import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.agent.graph import app
import pandas as pd
import re
from difflib import SequenceMatcher

load_dotenv()

eval_samples = [
    {
        "question": "What is the recommended tire pressure?",
        "expected": "The recommended tire pressure is 240 kPa (35 psi) for front and 230 kPa (33 psi) for rear under normal load.",
        "key_facts": ["240", "35", "230", "33", "kPa", "psi"]
    },
    {
        "question": "What type of engine oil is recommended?",
        "expected": "SAE 0W-20 (API Latest, ILSAC Latest) is recommended for better fuel economy.",
        "key_facts": ["SAE", "0W-20", "API", "ILSAC"]
    },
    {
        "question": "What is the wheel lug nut torque specification?",
        "expected": "The wheel lug nut torque is 11~13 kgf·m (79~94 lbf·ft, 107~127 N·m).",
        "key_facts": ["11", "13", "79", "94", "107", "127", "kgf", "lbf", "N·m"]
    },
]

def improved_faithfulness_check(response: str, key_facts: list) -> float:
    """Check if key facts from expected answer appear in response."""
    response_lower = response.lower()
    matches = sum(1 for fact in key_facts if fact.lower() in response_lower)
    return matches / len(key_facts) if key_facts else 1.0

def text_similarity(text1: str, text2: str) -> float:
    """Calculate overall text similarity."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def extract_numbers(text: str) -> set:
    """Extract all numbers from text."""
    return set(re.findall(r'\d+(?:\.\d+)?', text))

def number_accuracy(response: str, expected: str) -> float:
    """Check if key numbers match."""
    response_nums = extract_numbers(response)
    expected_nums = extract_numbers(expected)
    
    if not expected_nums:
        return 1.0
    
    matches = len(response_nums & expected_nums)
    return matches / len(expected_nums)

def run_improved_evaluation():
    """
    Run improved custom evaluation with better metrics.
    """
    results = []
    
    print("="*60)
    print("Running Improved Custom Evaluation")
    print("="*60)
    
    for idx, sample in enumerate(eval_samples, 1):
        print(f"\n[{idx}/{len(eval_samples)}] Question: {sample['question']}")
        print(f"Expected: {sample['expected'][:80]}...")
        
        try:
            response = app.invoke(
                {"messages": [HumanMessage(content=sample['question'])]},
                config={"configurable": {"thread_id": f"eval_{idx}"}}
            )
            
            if response and "messages" in response:
                last_message = response["messages"][-1]
                generated = last_message.content if hasattr(last_message, 'content') else str(last_message)
            else:
                generated = "No response generated"
            
            print(f"Generated: {generated[:150]}...")
            
            # Calculate improved metrics
            key_fact_score = improved_faithfulness_check(generated, sample['key_facts'])
            similarity_score = text_similarity(generated, sample['expected'])
            number_score = number_accuracy(generated, sample['expected'])
            
            # Overall score (weighted average)
            overall_score = (key_fact_score * 0.5 + number_score * 0.3 + similarity_score * 0.2)
            
            results.append({
                "question": sample['question'],
                "expected": sample['expected'],
                "generated": generated,
                "key_fact_accuracy": key_fact_score,
                "number_accuracy": number_score,
                "text_similarity": similarity_score,
                "overall_score": overall_score
            })
            
            print(f"   ✅ Key Facts: {key_fact_score:.1%}")
            print(f"   🔢 Numbers: {number_score:.1%}")
            print(f"   📝 Similarity: {similarity_score:.1%}")
            print(f"   🎯 Overall: {overall_score:.1%}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                "question": sample['question'],
                "expected": sample['expected'],
                "generated": "Error",
                "key_fact_accuracy": 0.0,
                "number_accuracy": 0.0,
                "text_similarity": 0.0,
                "overall_score": 0.0
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv("improved_eval_results.csv", index=False)
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)
    
    print(f"\n📁 Results saved to: improved_eval_results.csv")
    
    # Summary table
    print("\n📊 Detailed Results:")
    summary_df = df[['question', 'key_fact_accuracy', 'number_accuracy', 'overall_score']].copy()
    summary_df.columns = ['Question', 'Key Facts', 'Numbers', 'Overall']
    print(summary_df.to_string(index=False))
    
    # Averages
    print(f"\n📈 Average Scores:")
    print(f"   Key Fact Accuracy: {df['key_fact_accuracy'].mean():.1%}")
    print(f"   Number Accuracy:   {df['number_accuracy'].mean():.1%}")
    print(f"   Text Similarity:   {df['text_similarity'].mean():.1%}")
    print(f"   Overall Score:     {df['overall_score'].mean():.1%}")
    
    avg_overall = df['overall_score'].mean()
    
    if avg_overall > 0.8:
        print("\n🎉 Excellent! Your RAG agent is performing very well!")
    elif avg_overall > 0.6:
        print("\n✅ Good! Your RAG agent is working correctly.")
    else:
        print("\n⚠️ Needs improvement. Consider:")
        print("   - Improving chunk size in Pinecone")
        print("   - Strengthening the RAG prompt")
        print("   - Increasing retrieval top_k")

if __name__ == "__main__":
    run_improved_evaluation()
