"""
Performance Testing Script for TKR Chatbot
Tests response times before and after optimization
"""
import requests
import time
import json
from statistics import mean, median, stdev

# Configuration
BASE_URL = "http://localhost:5000"
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"

# Test questions
TEST_QUESTIONS = [
    "What is machine learning?",
    "Explain data structures",
    "What is database normalization?",
    "Define artificial intelligence",
    "What are the types of algorithms?",
    "Explain object-oriented programming",
    "What is cloud computing?",
    "Define software engineering",
    "What is computer networking?",
    "Explain operating systems"
]

def test_chat_response_time(question, subject_id=None):
    """Test a single chat request and measure response time"""
    payload = {
        "message": question,
        "subject_id": subject_id
    }
    
    start_time = time.time()
    try:
        response = requests.post(CHAT_ENDPOINT, json=payload, timeout=30)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'time': elapsed_time,
                'question': question,
                'answer_length': len(data.get('answer', '')),
                'sources': len(data.get('sources', []))
            }
        else:
            return {
                'success': False,
                'time': elapsed_time,
                'question': question,
                'error': response.text
            }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'success': False,
            'time': elapsed_time,
            'question': question,
            'error': str(e)
        }

def run_performance_test():
    """Run complete performance test suite"""
    print("=" * 70)
    print("TKR CHATBOT PERFORMANCE TEST")
    print("=" * 70)
    print(f"\nTesting {len(TEST_QUESTIONS)} questions...")
    print(f"Endpoint: {CHAT_ENDPOINT}\n")
    
    # Check if server is running
    try:
        health_response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server is not responding. Please start the backend server.")
            return
    except:
        print("❌ Cannot connect to server. Please start the backend server.")
        print("   Run: cd backend && python app.py")
        return
    
    print("✅ Server is running\n")
    
    results = []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] Testing: {question[:50]}...")
        result = test_chat_response_time(question)
        results.append(result)
        
        if result['success']:
            print(f"    ✅ Response time: {result['time']:.2f}s")
        else:
            print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Calculate statistics
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    if not successful_results:
        print("\n❌ All tests failed. Please check the server logs.")
        return
    
    response_times = [r['time'] for r in successful_results]
    
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"\nTotal Tests: {len(TEST_QUESTIONS)}")
    print(f"Successful: {len(successful_results)} ✅")
    print(f"Failed: {len(failed_results)} ❌")
    
    print(f"\n📊 Response Time Statistics:")
    print(f"   Average:  {mean(response_times):.2f}s")
    print(f"   Median:   {median(response_times):.2f}s")
    print(f"   Min:      {min(response_times):.2f}s")
    print(f"   Max:      {max(response_times):.2f}s")
    
    if len(response_times) > 1:
        print(f"   Std Dev:  {stdev(response_times):.2f}s")
    
    # Performance rating
    avg_time = mean(response_times)
    print(f"\n🎯 Performance Rating:")
    if avg_time < 2:
        print(f"   ⭐⭐⭐⭐⭐ EXCELLENT (< 2s)")
    elif avg_time < 3:
        print(f"   ⭐⭐⭐⭐ VERY GOOD (< 3s)")
    elif avg_time < 5:
        print(f"   ⭐⭐⭐ GOOD (< 5s)")
    elif avg_time < 8:
        print(f"   ⭐⭐ FAIR (< 8s)")
    else:
        print(f"   ⭐ NEEDS IMPROVEMENT (> 8s)")
    
    # Cache effectiveness test
    print(f"\n🔄 Testing Cache Effectiveness...")
    print(f"   Asking same question twice...")
    
    test_q = TEST_QUESTIONS[0]
    first_result = test_chat_response_time(test_q)
    time.sleep(0.5)
    second_result = test_chat_response_time(test_q)
    
    if first_result['success'] and second_result['success']:
        speedup = first_result['time'] / second_result['time']
        print(f"   First request:  {first_result['time']:.2f}s")
        print(f"   Second request: {second_result['time']:.2f}s")
        print(f"   Speedup: {speedup:.1f}x faster ✅")
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)

if __name__ == "__main__":
    run_performance_test()
