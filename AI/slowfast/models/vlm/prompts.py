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


def build_event_fight_prompt_3label(num_frames, duration_sec=None):
    duration_note = f" spanning approximately {duration_sec:.1f} seconds" if duration_sec else ""
    return (
        f"You are given {int(num_frames)} frames sampled from a CCTV event{duration_note}.\n"
        "An automated detector flagged this event as a possible fight.\n"
        "Classify the event into ONE of three levels:\n\n"
        "Level 0 — CLEARLY NORMAL: Only ordinary daily activity. "
        "Examples: people walking, standing, talking calmly, empty or near-empty scene, "
        "routine crowd movement. Nothing unusual.\n\n"
        "Level 1 — AMBIGUOUS OR ELEVATED: Anything between normal and a clear fight. "
        "Examples: argument, shouting, heated confrontation, people gathering around an incident, "
        "someone being restrained or assisted, a person lying on the ground, "
        "police or bystanders managing aftermath, poor visibility, unclear or partial context. "
        "Choose Level 1 whenever you are unsure.\n\n"
        "Level 2 — CLEAR FIGHT: Unambiguous active physical violence. "
        "Examples: visible punching, kicking, striking, forceful grabbing, "
        "aggressive physical assault with clear aggressor and victim.\n\n"
        "Return strict JSON only:\n"
        '{"level": <0|1|2>, "confidence": <float 0.0-1.0>, "reasoning": "<under 20 words>"}'
    )


def build_event_fight_prompt_v3(num_frames, duration_sec=None):
    duration_note = (
        f" spanning approximately {duration_sec:.1f} seconds"
        if duration_sec else ""
    )
    return (
        f"You are given {int(num_frames)} frames from a CCTV event{duration_note}.\n"
        "An automated system flagged this segment as a potential fight incident.\n\n"
        "STEP 1 — OBSERVE FIRST. Before judging, write 'scene_description' in KOREAN\n"
        "(under 35 words): a neutral, factual account of the people, their actions, and\n"
        "the setting. Do NOT say whether it is a fight — only describe what is visible.\n\n"
        "STEP 2 — THEN DECIDE the label.\n"
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
        "'reasoning' must be written in KOREAN, explaining the label decision (under 20 words).\n"
        "Return strict JSON only, with scene_description FIRST:\n"
        '{"scene_description": "<한국어, 35단어 이내, 중립적 장면 묘사>",'
        ' "label": "fight"|"non_fight",'
        ' "confidence": <float 0.0-1.0>,'
        ' "reasoning": "<한국어, 20단어 이내, 판단 근거>"}'
    )


def build_event_fight_prompt_v4(num_frames, duration_sec=None):
    """v4: VERA식 guiding sub-question 분해 후 결론.

    단일 판단 대신 5개 폭력 지표(언쟁/위협자세/군중반응/사후/물리접촉)를 각각
    yes/no로 먼저 답하게 하여, 모델이 'physical contact 없음=non_fight'로 붕괴하지
    않고 사건 arc 전체(접촉 없는 정황 포함)를 보도록 유도한다.
    label/confidence 의미는 v3와 동일(점수 환산 변경 없음).
    """
    duration_note = (
        f" spanning approximately {duration_sec:.1f} seconds"
        if duration_sec else ""
    )
    return (
        f"You are given {int(num_frames)} frames from a CCTV event{duration_note}.\n"
        "An automated system flagged this segment as a potential fight incident.\n\n"
        "A fight incident covers the FULL ARC of a violence-related event:\n"
        "verbal confrontation → aggressive approach → pushing or physical contact"
        " → active fighting → immediate aftermath.\n"
        "A real fight can be present even when no clear physical contact is visible"
        " in these frames (e.g. heated standoff, the moment just before or after"
        " contact, or a crowd reacting to an off-frame assault).\n\n"
        "First answer each indicator question as true or false, based ONLY on what"
        " is visible:\n"
        "  - aggressive_confrontation: Is there an aggressive confrontation or heated"
        " argument between people?\n"
        "  - threatening_posture: Is there a threatening posture, blocking, chasing,"
        " or aggressive approach toward someone?\n"
        "  - crowd_reaction: Is there a crowd gathering, encircling, or visibly"
        " reacting to a possible conflict?\n"
        "  - aftermath: Is someone on the ground, being restrained, assisted,"
        " attended to, or apprehended after an incident?\n"
        "  - physical_contact: Is there visible physical contact or assault"
        " (striking, kicking, grabbing, forceful pushing)?\n\n"
        "Then decide the label:\n"
        "  - label = 'fight' if ANY indicator above is true.\n"
        "  - label = 'non_fight' ONLY if ALL five indicators are false (people merely"
        " walking, waiting, or in casual conversation with no conflict).\n"
        "When uncertain, choose 'fight'.\n\n"
        "Return strict JSON only:\n"
        '{"aggressive_confrontation": <true|false>,'
        ' "threatening_posture": <true|false>,'
        ' "crowd_reaction": <true|false>,'
        ' "aftermath": <true|false>,'
        ' "physical_contact": <true|false>,'
        ' "label": "fight"|"non_fight",'
        ' "confidence": <float 0.0-1.0>,'
        ' "reasoning": "<under 25 words>"}'
    )
