def build_trajectories(frame_objects):

    trajectories = {}

    for frame_id, objects in frame_objects.items():

        frame_id = int(frame_id)

        for obj in objects:

            track_id = obj["track_id"]
            x1,y1,x2,y2 = obj["bbox"]

            cx = (x1+x2)/2
            cy = (y1+y2)/2

            if track_id not in trajectories:
                trajectories[track_id] = []

            trajectories[track_id].append((frame_id,cx,cy))

    return trajectories