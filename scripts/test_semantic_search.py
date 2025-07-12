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
        """North Carolina State Department of Vehicles(DMV) Final Notice: Enforcement Penalties Begin on July 8. Our records show that as of today, you still have an outstanding traffic ticket. In accordance with North Carolina State Administrative Code 15C-16.003, If you do not complete payment by July 7, 2025, we will take the following actions: 1. Report to the DMV violation database 2. Suspend your vehicle registration starting July 8 3. Suspend driving privileges for 30 days 4. Transfer to a toll booth and charge a 35% service fee 5. You may be prosecuted and your credit score will be affected Pay Now: https://ncdot.com-dkv.live/pay Please pay immediately before enforcement to avoid license suspension and further legal disputes. (Reply Y and re-open this message to click the link, or copy it to your browser.) """,
        """Apple Account Security Alert We have detected unusual activity on your Apple ID associated with a transaction at "Apple Store - CA" for $143.95, made via Apple Pay (pre- authorization). Additionally, we observed suspicious sign-in attempts and an Apple Pay activation request. To protect your account, these actions have been temporarily placed on hold. Please note: your photos, data, payment information, and linked cards. could be at risk. If you did not initiate this activity, we strongly recommend contacting Apple Support immediately to verify your account and prevent unauthorized charges. Call Apple Support: +18083060531 Visit: https://support.apple.com/billing Failure to act promptly may result in the charge being processed and irreversible. Thank you for your attention to this matter. - Apple Support Team""",
        """IRS2024IDME: An update was made to your 2024 TAX-PROFILE Navigate here https://vo.la/QcKTTOA ONLY if you haven't file 2024 TAX to avoid profile suspension"""

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