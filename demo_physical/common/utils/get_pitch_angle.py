def get_pitch_angle(stage, time, pitch_angle):
    if stage == 1:
        pitch_angle = 90
    if stage == 2:
        if time <= 44:
            pitch_angle -= 126 / 30
        if 80 > time > 44:
            pitch_angle += 36 / 36
    if stage == 3:
        pitch_angle = 0
    if stage == 4:
        pitch_angle = 0
    return pitch_angle
