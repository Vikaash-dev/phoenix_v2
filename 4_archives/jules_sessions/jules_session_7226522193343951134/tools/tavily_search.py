import os
import sys
import json
import urllib.request
import urllib.error

def tavily_search(query, max_results=5):
    """
    Performs a search using the Tavily API using standard library urllib.
    Requires TAVILY_API_KEY environment variable.
    """
    url = "https://api.tavily.com/search"
    
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY environment variable not set."}
    
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": max_results
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP Error {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tavily_search.py <query>")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    result = tavily_search(query)
    print(json.dumps(result, indent=2))
