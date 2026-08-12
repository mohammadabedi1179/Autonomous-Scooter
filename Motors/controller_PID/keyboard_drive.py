import pygame
import serial
import time

# ================= SERIAL CONFIG =================
PORT = '/dev/ttyACM0'
BAUD = 115200
SEND_HZ = 20
SEND_INTERVAL = 1.0 / SEND_HZ

STOP_CMD = 'SP#+000#+000#+000#+000\n'

# ================= SPEED CONFIG =================
# Adjust these to taste
LINEAR_RPM = 80   # RPM for forward/backward
TURN_RPM   = 60   # RPM for rotation (lower = gentler turns)

# ================= INIT =================
ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(2)
ser.reset_input_buffer()
ser.reset_output_buffer()
ser.write(STOP_CMD.encode())

pygame.init()
screen = pygame.display.set_mode((360, 180))
pygame.display.set_caption("Robot Keyboard Control")
font = pygame.font.SysFont(None, 28)

def make_cmd(rpm_l, rpm_r):
    """
    Build SP command from signed left/right RPM values.
    Left wheels: base sign +, Right wheels: base sign - (opposite winding).
    """
    sign_l = '+' if rpm_l >= 0 else '-'
    sign_r = '-' if rpm_r >= 0 else '+'
    return f'SP#{sign_l}{abs(rpm_l):03d}#{sign_r}{abs(rpm_r):03d}#{sign_l}{abs(rpm_l):03d}#{sign_r}{abs(rpm_r):03d}\n'

CMDS = {
    'forward':  make_cmd( LINEAR_RPM,  LINEAR_RPM),
    'backward': make_cmd(-LINEAR_RPM, -LINEAR_RPM),
    'left':     make_cmd(-TURN_RPM,    TURN_RPM),
    'right':    make_cmd( TURN_RPM,   -TURN_RPM),
}

LABELS = {
    'forward':  'W / ↑  →  Forward',
    'backward': 'S / ↓  →  Backward',
    'left':     'A / ←  →  Rotate Left',
    'right':    'D / →  →  Rotate Right',
    'stop':     'Release  →  Stop',
}

# ================= MAIN LOOP =================
last_send  = 0
last_cmd   = STOP_CMD
last_label = 'stopped'

print("Keyboard control active. Use W/A/S/D or arrow keys. Ctrl+C or close window to quit.")

running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            cmd, label = CMDS['forward'],  'forward'
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            cmd, label = CMDS['backward'], 'backward'
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            cmd, label = CMDS['left'],     'left'
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            cmd, label = CMDS['right'],    'right'
        else:
            cmd, label = STOP_CMD,         'stop'

        now = time.time()
        if now - last_send >= SEND_INTERVAL:
            ser.write(cmd.encode())
            last_send = now

        # ---- Draw HUD ----
        if label != last_label:
            screen.fill((30, 30, 30))
            title = font.render("Robot Keyboard Control", True, (200, 200, 200))
            screen.blit(title, (20, 15))
            y = 55
            for key, text in LABELS.items():
                color = (80, 200, 120) if key == label else (140, 140, 140)
                surf = font.render(text, True, color)
                screen.blit(surf, (20, y))
                y += 26
            pygame.display.flip()
            last_label = label

        time.sleep(0.005)

except KeyboardInterrupt:
    pass

finally:
    print("\nStopping robot...")
    ser.write(STOP_CMD.encode())
    ser.close()
    pygame.quit()
    print("Done.")
