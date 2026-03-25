from config import EVENT_THRESHOLD, MERGE_GAP

def detect_segments(scores):

    segments = []
    start = None

    for i,score in enumerate(scores):

        if score > EVENT_THRESHOLD:

            if start is None:
                start = i

        else:

            if start is not None:
                segments.append((start,i))
                start = None

    if start is not None:
        segments.append((start,len(scores)-1))

    return segments


def merge_segments(segments):

    if not segments:
        return []

    merged = []
    cur_start,cur_end = segments[0]

    for start,end in segments[1:]:

        if start-cur_end <= MERGE_GAP:

            cur_end = end

        else:

            merged.append((cur_start,cur_end))
            cur_start,cur_end = start,end

    merged.append((cur_start,cur_end))

    return merged