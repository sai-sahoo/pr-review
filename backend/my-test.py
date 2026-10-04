import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Review a GitHub PR with an LLM.")
    parser.add_argument("pr_url", nargs="?")
    parser.add_argument("--show-graph", action="store_true", help="print the graph and exit")
    args = parser.parse_args()
    print(args)
    
if __name__ == "__main__":
    main()