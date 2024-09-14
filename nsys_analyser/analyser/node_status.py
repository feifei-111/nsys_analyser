def nodes_time_cover(nodes):
    start_times = [node.start for node in nodes]
    end_times = [node.end for node in nodes]

    start_time_idx = 0
    end_time_idx = 0

    last_start = None
    counter = 0

    total_time_cost = 0

    # we get the small time from the head of start_times or end_times
    # the idx show where the head is

    # if still have time not used
    while start_time_idx < len(start_times) or end_time_idx < len(end_times):
        next_start_time = (
            start_times[start_time_idx]
            if start_time_idx < len(start_times)
            else float("inf")
        )
        next_end_time = (
            end_times[end_time_idx]
            if end_time_idx < len(end_times)
            else float("inf")
        )

        if next_start_time < next_end_time:
            if counter == 0:
                last_start = next_start_time
            counter += 1
            start_time_idx += 1
        else:
            counter -= 1
            if counter == 0:
                time_cost = next_end_time - last_start
                total_time_cost += time_cost
                last_start = None
            end_time_idx += 1

    return total_time_cost
