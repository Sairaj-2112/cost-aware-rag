import json
import os


METRICS_FILE = "metrics.json"


def create_metrics():

    if os.path.exists(METRICS_FILE):

        with open(METRICS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    return {
        "total_queries": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "total_tokens": 0,
        "total_cost": 0.0
    }


def save_metrics(metrics):

    with open(METRICS_FILE, "w", encoding="utf-8") as file:

        json.dump(
            metrics,
            file,
            indent=4
        )


def print_metrics(metrics):

    print("\n========== METRICS ==========")

    print("Total queries:", metrics["total_queries"])
    print("Cache hits:", metrics["cache_hits"])
    print("Cache misses:", metrics["cache_misses"])
    print("Total tokens:", metrics["total_tokens"])
    print("Total cost: $", round(metrics["total_cost"], 8))