from trajectory import build_trajectories
from motion import motion_signal
from interaction import interaction_signal
from scoring import normalize, combine_scores, smooth_signal, persistence_filter, temporal_aggregation
from smoothing import temporal_smoothing
from segments import detect_segments, merge_segments
from clip_generator import generate_candidate_clips


def candidate_pipeline(frames, frame_objects):

    print("[pipeline] start")

    trajectories = build_trajectories(frame_objects)
    print("[pipeline] trajectories:", len(trajectories))

    motion_scores = []
    interaction_scores = []
    trajectory_scores = []

    for i in range(1, len(frames)):

        motion = motion_signal(frames[i - 1], frames[i])
        objects = frame_objects.get(i, []) or frame_objects.get(str(i), [])

        interaction = interaction_signal(objects)

        traj_score = 0
        for traj in trajectories.values():
            for f, cx, cy in traj:
                if f == i:
                    traj_score += 1
                    break

        motion_scores.append(motion)
        interaction_scores.append(interaction)
        trajectory_scores.append(traj_score)

    print("[pipeline] raw motion max:", max(motion_scores) if motion_scores else None)
    print("[pipeline] raw interaction max:", max(interaction_scores) if interaction_scores else None)

    motion_scores = normalize(motion_scores)
    interaction_scores = normalize(interaction_scores)
    trajectory_scores = normalize(trajectory_scores)

    interaction_scores = smooth_signal(interaction_scores)
    interaction_scores = persistence_filter(interaction_scores)

    scores = []
    for m, i, t in zip(motion_scores, interaction_scores, trajectory_scores):
        scores.append(combine_scores(m, i, t))

    print("[pipeline] score max:", max(scores) if scores else None)

    scores = temporal_smoothing(scores)
    scores = temporal_aggregation(scores)

    segments = detect_segments(scores)
    print("[pipeline] segments:", segments)

    segments = merge_segments(segments)
    print("[pipeline] merged segments:", segments)

    if not segments:
        print("[pipeline] WARNING: no segments detected")

    clips = generate_candidate_clips(frames, segments)
    print("[pipeline] clips generated:", len(clips))

    return clips