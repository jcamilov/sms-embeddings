#!/usr/bin/env python3
"""
Test script for semantic_search_sms function
This script demonstrates how to use the semantic search functionality
"""

import sys
import os

# Add the scripts directory to the path so we can import the function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_search import semantic_search_sms

def test_semantic_search():
    """Test the semantic search function with various SMS examples."""
    
    # Test SMS examples
    test_sms_list = [
        "Your bank account has been suspended. Click here to verify: http://fake-bank.com",
        "Hi mom, can you pick me up from school?",
        "URGENT: Your package delivery failed. Call 1-800-FAKE-NOW to reschedule",
        "Meeting at 3pm today. Don't forget!",
        # "Congratulations! You've won $1000. Claim now: http://fake-prize.com",
        # "Thanks for the birthday wishes everyone!",
        # "Your Netflix subscription will expire. Renew now: http://fake-netflix.com",
        # "Dinner tonight at 7pm? Let me know!",
        # "SECURITY ALERT: Unusual login detected. Verify account: http://fake-security.com",
        # "Happy birthday! Hope you have a great day!"
    ]
    
    print("=" * 80)
    print("SEMANTIC SEARCH FUNCTION TEST")
    print("=" * 80)
    print("This script tests the semantic_search_sms function with various SMS examples.")
    print("Each test will show the top 3 most similar SMS for both smishing and benign classes.\n")
    
    for i, test_sms in enumerate(test_sms_list, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {test_sms[:50]}{'...' if len(test_sms) > 50 else ''}")
        print(f"{'='*60}")
        
        try:
            # Perform semantic search
            results = semantic_search_sms(test_sms, top_k=3)
            
            # Display smishing results
            if results['smishing']:
                print(f"\n--- TOP 3 SMISHING RESULTS ---")
                for j, result in enumerate(results['smishing'], 1):
                    print(f"{j}. Similarity: {result['similarity']:.3f}")
                    print(f"   SMS ID: {result['sms_id']}")
                    print(f"   Text: {result['text']}")
                    print()
            else:
                print("\n--- NO SMISHING RESULTS FOUND ---")
            
            # Display benign results
            if results['benign']:
                print(f"--- TOP 3 BENIGN RESULTS ---")
                for j, result in enumerate(results['benign'], 1):
                    print(f"{j}. Similarity: {result['similarity']:.3f}")
                    print(f"   SMS ID: {result['sms_id']}")
                    print(f"   Text: {result['text']}")
                    print()
            else:
                print("--- NO BENIGN RESULTS FOUND ---")
                
        except Exception as e:
            print(f"Error during search: {e}")
            print("Make sure you have generated the embeddings first using:")
            print("python scripts/generate_embeddings.py")
        
        print("\n" + "-"*60)

def main():
    """Main function to run the tests."""
    print("Starting semantic search function test...")
    test_semantic_search()
    print("\nTest completed!")

if __name__ == "__main__":
    main() 