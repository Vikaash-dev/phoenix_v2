
import os
import pandas as pd
from tavily import TavilyClient

# Set the Tavily API key
os.environ["TAVILY_API_KEY"] = "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"

def search_tavily(query, max_results=10):
    """
    Searches Tavily for research papers matching the query.
    """
    client = TavilyClient()
    response = client.search(query=query, search_depth="advanced", max_results=max_results)
    return pd.DataFrame(response['results'])

if __name__ == "__main__":
    # Define search queries
    queries = {
        "hybrid_architectures": "novel hybrid architectures for brain tumor detection combining CNNs with transformers or attention mechanisms",
        "advanced_conv_layers": "advanced convolutional layers for medical imaging, such as dynamic snake convolution or deformable convolution, applied to brain tumor detection",
        "novel_training": "novel training methodologies, loss functions, or data augmentation techniques for brain tumor detection"
    }

    all_results = {}
    for name, query in queries.items():
        print(f"Searching for: {name}...")
        df = search_tavily(query, max_results=20)
        all_results[name] = df
        print(f"Found {len(df)} results for {name}.")

    # Save results to CSV
    for name, df in all_results.items():
        df.to_csv(f"tavily_results_{name}.csv", index=False)
        print(f"Saved results to tavily_results_{name}.csv")

    print("Tavily search complete.")
