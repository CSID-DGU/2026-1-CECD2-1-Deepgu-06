def filter_clips(results, threshold=0.4):
    """
    results: main_pipeline 결과 리스트
    """

    candidates = []

    for r in results:
        score = r["scores"]["final_score"]

        if score > threshold:
            candidates.append(r)

    print(f" candidate clip 개수: {len(candidates)} / {len(results)}")

    return candidates