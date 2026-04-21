def select_candidates(results, thresholds, verbose=True):
    candidates = []
    fight_threshold = thresholds["fight_candidate"]
    fall_threshold = thresholds.get("fall_candidate")

    for result in results:
        fight_score = result["scores"]["fight_candidate_score"]
        fall_score = result["scores"].get("fall_candidate_score")

        candidate_types = []
        if fight_score >= fight_threshold:
            candidate_types.append("fight")
        if fall_threshold is not None and fall_score is not None and fall_score >= fall_threshold:
            candidate_types.append("fall")

        if not candidate_types:
            continue

        result["candidate_types"] = candidate_types
        if fall_score is None:
            result["candidate_score"] = fight_score
        else:
            result["candidate_score"] = max(fight_score, fall_score)
        candidates.append(result)

    if verbose:
        print(f" candidate clip 개수: {len(candidates)} / {len(results)}")
    return candidates
