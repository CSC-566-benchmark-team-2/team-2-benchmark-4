import sys

from tests.benchmarks import run_benchmarks
from src.main import Solution
from src.send_results import UserResults


def benchmarks():
    if Solution is None:
        print("No solution class provided")
        return
    soln = Solution()
    results = run_benchmarks(soln)
    print(results)
    if len(sys.argv) > 1:  # send results if on the server
        try:
            user_results = UserResults(username=sys.argv[1], datasets=results)
            user_results.send_results()
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    benchmarks()
