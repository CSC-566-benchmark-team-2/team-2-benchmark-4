import sys

from tests.benchmarks import run_benchmarks
from src.main import create_agent
from src.send_results import UserResults


def benchmarks():
    if create_agent("agent1") is None:
        print("Please implement the `create_agent` function in src/main.py")
        return
    # soln = Solution()
    results = run_benchmarks(create_agent)
    print(results)
    if len(sys.argv) > 1:  # send results if on the server
        try:
            user_results = UserResults(username=sys.argv[1], datasets=results)
            user_results.send_results()
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    benchmarks()
    sys.exit()
