def build_event_fight_prompt(num_frames, duration_sec=None):
    duration_note = f" spanning approximately {duration_sec:.1f} seconds" if duration_sec else ""
    return (
        f"You are given {int(num_frames)} frames sampled from a CCTV event{duration_note}.\n"
        "An automated detector flagged this event as a potential physical fight.\n"
        "Review the full sequence and decide: is this an actual fight or a false alarm?\n"
        "Count as fight: visible striking, kicking, grabbing, forceful pushing, chasing to attack.\n"
        "Count as non_fight: bystanders watching, someone helping an injured/fallen person, "
        "argument or confrontation without physical contact, post-fight scene with no active violence "
        "(e.g. person lying on ground being assisted, police restraining after the fact), "
        "or normal crowd movement.\n"
        "Focus on whether ACTIVE physical violence is happening, not its aftermath.\n"
        "Return strict JSON only with keys: label, confidence, reasoning.\n"
        'Use label="fight" or label="non_fight".\n'
        "Use confidence as a float between 0.0 and 1.0.\n"
        "Keep reasoning short, under 20 words.\n"
        '"Does this event show active physical violence (not just its aftermath or crowd)?"'
    )


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


def build_event_fight_prompt_v3(num_frames, duration_sec=None):
    duration_note = (
        f" spanning approximately {duration_sec:.1f} seconds"
        if duration_sec else ""
    )
    return (
        f"You are given {int(num_frames)} frames from a CCTV event{duration_note}.\n"
        "An automated system flagged this segment as a potential fight incident.\n\n"
        "DEFINITION: A fight incident covers the full arc of a violence-related event:\n"
        "verbal confrontation → aggressive approach → pushing or physical contact"
        " → active fighting → immediate aftermath.\n\n"
        "Label as 'fight' if ANY of the following is visible:\n"
        "  - Verbal argument or heated confrontation between individuals\n"
        "  - Aggressive approaching, blocking, or threatening posture\n"
        "  - Pushing, shoving, grabbing, or physical contact with aggressive intent\n"
        "  - Active punching, kicking, or physical assault\n"
        "  - Immediate aftermath: restraining, intervening, or"
        " attending to an injured person at the scene\n\n"
        "Label as 'non_fight' ONLY if ALL of the following are true:\n"
        "  - No argument, confrontation, or aggressive interaction visible\n"
        "  - No signs of physical contact, injury, or bystander reaction to an incident\n"
        "  - People are walking, waiting, or engaging in casual conversation\n"
        "  - Ordinary crowd movement with no conflict indicators\n\n"
        "When uncertain, choose 'fight'.\n\n"
        "Return strict JSON only:\n"
        '{"label": "fight"|"non_fight",'
        ' "confidence": <float 0.0-1.0>,'
        ' "reasoning": "<under 20 words>"}'
    )