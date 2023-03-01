def get_attack_angle(stage, time):
    attack_angle = 0
    if stage == 1:
        attack_angle = 0
    if stage == 2:
        if time < 24:
            attack_angle = -(time - 14)
        if 44 > time >= 24:
            attack_angle = (time - 24) * 1 - 10
        if 80 > time >= 44:
            attack_angle = -(time - 80) * (36 / 6) + 10
    if stage == 3:
        attack_angle = 4
    if stage == 4:
        attack_angle = -10
    return attack_angle
