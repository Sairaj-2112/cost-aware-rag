def calculate_cost(
    prompt_tokens,
    completion_tokens,
    prompt_cost_per_million,
    completion_cost_per_million
):

    prompt_cost = (
        prompt_tokens / 1_000_000
    ) * prompt_cost_per_million

    completion_cost = (
        completion_tokens / 1_000_000
    ) * completion_cost_per_million

    total_cost = prompt_cost + completion_cost

    return total_cost