def build_binary_fight_prompt(motion_summary, num_frames=None):
    frame_note = ""
    if num_frames is not None:
        frame_note = f"You are given {int(num_frames)} sampled frames from one short CCTV clip.\n"

    return (
        f"{frame_note}"
        "You are reviewing CCTV frames for physical violence.\n"
        "Decide whether the clip shows any sign of physical aggression, even if it is subtle, partial, "
        "brief, or only visible in some frames.\n"
        "Count as fight if there is any visible striking, kicking, lunging, grabbing, forceful pushing, "
        "raising fists to attack, chasing to attack, or other aggressive physical confrontation.\n"
        "Do not require a full fight sequence. Subtle or incomplete attack cues should still be marked as fight.\n"
        "Return strict JSON only with keys: label, confidence, reasoning.\n"
        'Use label="fight" or label="non_fight".\n'
        "Use confidence as a float between 0.0 and 1.0.\n"
        "Keep reasoning short, under 20 words.\n"
        'Question: "Does this clip show any sign of physical aggression, even subtle or partial?"\n'
        f"Motion summary: {motion_summary}"
    )
