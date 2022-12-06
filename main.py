import pygame as pg
import pygame.gfxdraw
import os
import numpy as np

FPS = 60
WIDTH, HEIGHT = 900, 900
DISPLAY = pg.display.set_mode((WIDTH, HEIGHT))
WIN = pg.surface.Surface((WIDTH, HEIGHT))
keys = [False]*6
HEALTH_BAR_WIDTH = 700
HEALTH_BAR_HEIGHT = 25
FONT_SIZES = 120, 80, 25, 40
TRAIL_FREQ = 2
MAX_INT = 10_000
IMAGE_DICT = {
    "BACKGROUND": pg.image.load(os.path.join('Assets', 'background_cropped.jpg')),
    "SPACESHIP": [pg.image.load(os.path.join('Assets', 'ship{}.png'.format(i+1))) for i in range(4)],
    "BULLET": pg.image.load(os.path.join('Assets', 'bullet.png')),
    "CRYSTAL": pg.image.load(os.path.join('Assets', 'crystal.png')).convert_alpha(),
    "ASTEROID": pg.image.load(os.path.join('Assets', 'asteroid.png')),
    "BOSS": [pg.image.load(os.path.join('Assets', 'boss{}.png'.format(i+1))).convert_alpha() for i in range(2)],
    "EXPLOSION": pg.image.load(os.path.join('Assets', 'explosion.png'))
}
SIZE_DICT = {
    "BACKGROUND": (900, 900),
    "SPACESHIP": (50, 50),
    "BULLET": (20, 20),
    "CRYSTAL": (40, 40),
    "ASTEROID": (70, 70),
    "BOSS": (150, 150),
    "SMALL EXPLOSION": (60, 60),
    "LARGE EXPLOSION": (160, 160)
}
COLOR_DICT = {
    "WHITE": (255, 255, 255, 0),
    "GRAY": (200, 200, 200, 0),
    "PURPLE": (130, 30, 230, 0),
    "LASER": (140, 40, 40, 0),
    "LIGHT GREEN": (60, 200, 120, 0),
    "DARK RED": (100, 0, 0, 0),
    "LIGHT RED": (255, 60, 60, 0),
    "LIGHT BLUE": (0, 200, 230, 0),
    "ORANGE": (255, 128, 0, 0),
    "YELLOW": (255, 255, 0, 0),
    "GOLD": (160, 130, 0, 0)
}


class BasicSprite (pg.sprite.Sprite):
    def __init__(self, size, pos, image, vel, facing):
        super().__init__()
        self.image = pg.transform.scale(image, size)
        self.clean_img = self.image
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.vel = vel
        self.facing = facing


class Background (BasicSprite):
    def __init__(self, size, pos, image, vel, facing):
        super().__init__(size, pos, image, vel, facing)


class Hero (BasicSprite):
    def __init__(self, size, pos, image, vel, facing, bullet_cooldown):
        super().__init__(size, pos, image, vel, facing)
        self.bullet_cooldown = bullet_cooldown
        self.curr_bullet_cooldown = bullet_cooldown
        self.img_phase = 4
        self.curr_img_phase = 1


class Missile (BasicSprite):
    def __init__(self, size, pos, image, vel, facing, type, delay=0, spawn_time=0, spawn_prot=0, osc_phase=0, leave_trail=True):
        super().__init__(size, pos, image, vel, facing)
        self.curr_vel = vel
        self.type = type
        self.delay = delay
        self.curr_delay = delay
        self.spawn_pt = pos
        self.spawn_time = spawn_time
        self.spawn_prot = spawn_prot
        self.osc_phase = osc_phase
        self.leave_trail = leave_trail


class Boss (BasicSprite):
    def __init__(self, size, pos, image, vel, facing, phase, phase_hp, iframe_cooldown, move_cooldown_range, transition_cooldown):
        super().__init__(size, pos, image, vel, facing)
        self.curr_vel = vel
        self.phase = phase
        self.phase_hp = phase_hp
        self.total_hp = np.sum(phase_hp)
        self.iframe_cooldown = iframe_cooldown
        self.curr_iframe_cooldown = iframe_cooldown
        self.move_cooldown_range = move_cooldown_range
        self.curr_move_cooldown = np.random.uniform(move_cooldown_range[0], move_cooldown_range[1])*FPS
        self.move_matrix = self.reset_move_matrix()
        self.transitioning = 0
        self.transition_cooldown = transition_cooldown
        self.curr_transition_cooldown = transition_cooldown

    def reset_move_matrix(self):
        """
        Resets the boss's matrix of moves
        """
        self.move_matrix = np.ones((2, 5))  # There are 5 possible moves. First row is each move's weight
        self.move_matrix[1, :] = self.move_matrix[0, :]/self.move_matrix.sum(axis=1)[0]  # Second row is each move's weighted probability
        return self.move_matrix


class Laser (pg.sprite.Sprite):
    def __init__(self, color, start_pos, end_pos, width, lifespan, delay):
        super().__init__()
        self.color = color
        self.curr_color = color
        self.start_pos = start_pos
        self.curr_start_pos = start_pos
        self.end_pos = end_pos
        self.curr_end_pos = start_pos
        self.width = width
        self.lifespan = lifespan
        self.curr_lifespan = lifespan
        self.delay = delay
        self.curr_delay = delay


class Shockwave (pg.sprite.Sprite):
    def __init__(self, color, center, radius, lifespan):
        super().__init__()
        self.color = color
        self.radius = radius
        self.curr_radius = 0
        self.center = center
        self.lifespan = lifespan
        self.curr_lifespan = lifespan


class Blackhole (Shockwave):
    def __init__(self, color, center, radius, lifespan, vel, facing):
        super().__init__(color, center, radius, lifespan)
        self.curr_color = color
        self.vel = vel
        self.curr_vel = vel
        self.facing = facing


class Trail (Shockwave):
    def __init__(self, color, center, radius, lifespan, vel, facing):
        super().__init__(color, center, radius, lifespan)
        self.curr_color = color
        self.vel = vel
        self.facing = facing


def set_image(image, size):
    """
    Returns two images scaled to a given size
    """
    image = pg.transform.scale(image, size)
    return image, image


def get_rot_matrix(angle_rad):
    """
    Returns a rotation matrix for a specified angle in radians
    """
    return np.asarray(((np.cos(angle_rad), -np.sin(angle_rad)), (np.sin(angle_rad), np.cos(angle_rad))))


def normalize(vec):
    """
    Normalizes a vector
    """
    norm = np.linalg.norm(vec)
    if norm != 0:  # Avoid dividing by zero
        return vec/norm
    return vec


def rotate(img, angle):
    """
    Rotates an image by a specified angle in degrees
    """
    if not np.isnan(angle):
        return pg.transform.rotate(img, angle)
    return img


def create_text(surface, text, color, font, x_pos, y_pos, drop_shadow_dist=5):
    """
    Displays text to a surface with a drop shadow
    """
    text_main = font.render(text, True, color)
    text_shadow = font.render(text, True, (0, 0, 0, 0))
    textbox = text_main.get_rect()
    textbox.center = x_pos, y_pos
    surface.blit(text_shadow, np.add(textbox, drop_shadow_dist))
    surface.blit(text_main, textbox)


def boss_move(move_matrix):
    """
    Returns a move from the boss's move matrix, as well as the updated matrix
    """
    choice = np.random.choice(move_matrix.shape[1], 1, p=move_matrix[1])[0]  # Choose move based on weighted probabilities
    move_matrix[0, :choice] += 1  # Add weight to each move except the chosen one
    move_matrix[0, choice+1:] += 1
    move_matrix[1, :] = move_matrix[0, :]/move_matrix.sum(axis=1)[0]  # Recalculate weighted probabilities
    return move_matrix, choice


def screen_shake_logic(screen_shake_timer, screen_shake_pos):
    """
    Shakes the screen for a time screen_shake_timer from a position shake_screen_pos
    """
    if screen_shake_timer > 0:
        screen_shake_timer -= 1
        if screen_shake_timer > FPS/10:  # Shake the screen
            shake_mult = 10
            diff = (np.random.rand(2)*2-1)*shake_mult
            screen_shake_pos = np.add(screen_shake_pos, diff)
        else:  # Below a time, make screen return to its original position
            screen_shake_pos = np.add(screen_shake_pos, 4*normalize(np.subtract((0, 0), screen_shake_pos)))
    else:
        screen_shake_pos = (0, 0)
    return screen_shake_timer, screen_shake_pos


def game_logic(game_counter, keys, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, bg_group):
    """
    Handles the logic of the game. Returns the status of the game, an incremented game counter, and the screen shake duration
    """
    game_status = 0  # Game status values: 0 = continue, -1 = lose, 1 = win, -2 = game has not started
    screen_shake_timer = 0
    # Background logic
    for bg in bg_group:
        if bg.rect.top >= HEIGHT:  # Background reaches height limit
            bg.kill()
        if bg.rect.top == 0:  # Spawn new background
            new_bg = Background(SIZE_DICT["BACKGROUND"], (WIDTH/2, -HEIGHT/2), IMAGE_DICT["BACKGROUND"], 1, (0, 1))
            bg_group.add(new_bg)
        bg.rect.center = np.add(bg.rect.center, bg.vel * np.asarray(bg.facing))
    # Hero logic
    mouse_pos = pg.mouse.get_pos()
    hero_pos = hero.rect.center
    hero.facing = normalize(np.subtract(mouse_pos, hero_pos))
    hero.image = rotate(hero.clean_img, 270-np.arctan2(hero.facing[1], hero.facing[0])*180/np.pi)  # Hero faces mouse
    if game_counter % 50 == 0:  # Hero cycles through images
        hero.curr_img_phase += 1
        if hero.curr_img_phase >= 4:
            hero.curr_img_phase = 0
        hero.image, hero.clean_img = set_image(IMAGE_DICT["SPACESHIP"][hero.curr_img_phase], SIZE_DICT["SPACESHIP"])
    if hero.curr_bullet_cooldown > 0:  # Cooldown before hero can shoot again
        hero.curr_bullet_cooldown -= 1
    if keys[0] and hero.rect.top > 0:  # Movement of hero
        hero.rect.centery -= hero.vel
    if keys[1] and hero.rect.left > 0:
        hero.rect.centerx -= hero.vel
    if keys[2] and hero.rect.bottom < HEIGHT:
        hero.rect.centery += hero.vel
    if keys[3] and hero.rect.right < WIDTH:
        hero.rect.centerx += hero.vel
    if keys[4] and hero.curr_bullet_cooldown <= 0:  # Hero shoots bullet and bullet cooldown is reset
        hero.curr_bullet_cooldown = hero.bullet_cooldown*FPS
        bullet = Missile(SIZE_DICT["BULLET"], np.add(hero.rect.center, 35*np.asarray(hero.facing)), IMAGE_DICT["BULLET"], 10, hero.facing, 'linear')
        bullets.add(bullet)
    # Boss logic
    boss_pos = boss.rect.center
    origin_pos = (WIDTH/2, HEIGHT/2)
    boss.rect.center = np.add(boss_pos, boss.curr_vel*np.asarray(boss.facing))  # Boss moves in its facing direction
    boss.facing = (0, 1)
    if boss.rect.colliderect(hero.rect) and np.linalg.norm(np.subtract(boss_pos, hero_pos)) <= 100:  # Boss collides with hero and game is over
        game_status = -1
    if boss.transitioning == 1:  # Phase transition step 1: Boss moves to center
        boss.facing = normalize(np.subtract(origin_pos, boss_pos))
        if np.linalg.norm(np.subtract(boss_pos, origin_pos)) <= 6:  # Boss becomes stationary
            boss.transitioning = 2
            boss.curr_vel = 0
    elif boss.transitioning == 2:  # Phase transition step 2: Boss is momentarily stationary
        if boss.curr_transition_cooldown > 0:
            boss.curr_transition_cooldown -= 1
        else:  # Boss attacks with asteroids
            screen_shake_timer = FPS/2  # Add a screen shake
            num_asteroids = 16
            angles_rad = np.linspace(0, 2*np.pi, num_asteroids, endpoint=False)  # Generate num_asteroids angles from [0, 2pi)
            rot_matrix_list = get_rot_matrix(angles_rad).swapaxes(0, 2)  # Change rotation matrix list shape to (num_asteroids, 2, 2)
            facing_list = rot_matrix_list@boss.facing  # Apply rotations to boss's facing direction
            for facing in facing_list:
                crystal_pos = np.add(boss_pos, 100*facing)  # Spawn crystals at a distance from boss
                crystal_facing = np.asarray(crystal_pos-boss_pos)
                crystal_facing[[0, 1]] = crystal_facing[[1, 0]]  # Make crystals orthogonal to initially outward direction
                crystal_facing[0] = -crystal_facing[0]
                crystal_facing = normalize(crystal_facing)  # Normalize facing direction
                new_asteroid = Missile(SIZE_DICT["ASTEROID"], crystal_pos, IMAGE_DICT["ASTEROID"], 10, crystal_facing, 'linear')
                crystals.add(new_asteroid)
            boss.transitioning = 3
            boss.curr_transition_cooldown = boss.transition_cooldown
            if boss.phase == 1:  # Change image of boss
                boss.image, boss.clean_img = set_image(IMAGE_DICT["BOSS"][1], SIZE_DICT["BOSS"])
    elif boss.transitioning == 3:  # Phase transition step 3: Boss is momentarily stationary
        if boss.curr_transition_cooldown > 0:
            boss.curr_transition_cooldown -= 1
        else:  # Boss's regular velocity is restored
            boss.transitioning = 0
            boss.curr_transition_cooldown = boss.transition_cooldown
            boss.curr_vel = boss.vel
    elif boss.transitioning == 0:  # Boss is not transitioning phases
        boss.facing = normalize(np.subtract(hero_pos, boss_pos))  # Boss faces hero
        if boss.curr_move_cooldown > 0:  # Cooldown before boss can perform another move
            boss.curr_move_cooldown -= 1
            if boss.curr_move_cooldown < 0.5*FPS:  # Boss is stationary before performing a move
                boss.curr_vel = 0
        else:  # Boss performs a move
            boss.curr_vel = boss.vel  # Boss's regular velocity is restored
            boss.move_matrix, choice = boss_move(boss.move_matrix)  # Update boss's move matrix and make a choice
            boss.curr_move_cooldown = np.random.uniform(boss.move_cooldown_range[0], boss.move_cooldown_range[1])*FPS  # Reset boss's move cooldown
            if choice == 0:  # Chosen move is different based on its current phase
                if boss.phase == 0:  # Move (Phase 1, Choice 1): Fast Crystal
                    new_crystal = Missile(SIZE_DICT["CRYSTAL"], np.add(boss_pos, 60 * np.asarray(boss.facing)), IMAGE_DICT["CRYSTAL"], 20, boss.facing, 'linear')
                    crystals.add(new_crystal)
                elif boss.phase == 1:  # Move (Phase 2, Choice 1): Shockwave
                    new_shockwave = Shockwave(COLOR_DICT["PURPLE"], boss_pos, 300, 2*FPS)
                    shockwaves.add(new_shockwave)
            elif choice == 1:
                if boss.phase == 0:  # Move (Phase 1, Choice 2): Curving Crystal
                    new_crystal = Missile(SIZE_DICT["CRYSTAL"], np.add(boss_pos, 60*np.asarray(boss.facing)), IMAGE_DICT["CRYSTAL"], 10, boss.facing, 'curved')
                    crystals.add(new_crystal)
                elif boss.phase == 1:  # Move (Phase 2, Choice 2): Black Holes
                    num_blackholes = 3
                    max_offset = 90*np.pi/180  # Maximum angle variation
                    variation = (2*np.random.rand(num_blackholes)-1)*max_offset  # Random angle variation
                    angles_rad = np.linspace(0, 2*np.pi, num_blackholes, endpoint=False)  # Generate num_blackholes angles from [0, 2pi)
                    angles_rad = np.add(angles_rad, variation)  # Add variation to angles
                    rot_matrix_list = get_rot_matrix(angles_rad).swapaxes(0, 2)  # Change rotation matrix list shape to (num_blackholes, 2, 2)
                    facing_list = rot_matrix_list@boss.facing  # Apply rotations to boss's facing direction
                    for facing in facing_list:
                        blackhole_pos = np.add(boss_pos, 10*facing)  # Spawn black holes at a distance from boss
                        blackhole_facing = normalize(blackhole_pos-np.asarray(boss_pos))
                        new_blackhole = Blackhole((0, 0, 0), blackhole_pos, 50, 3*FPS, 3, blackhole_facing)
                        blackholes.add(new_blackhole)
            elif choice == 2:
                if boss.phase == 0:  # Move (Phase 1, Choice 3): Oscillating Crystals
                    new_crystal_1 = Missile(SIZE_DICT["CRYSTAL"], np.add(boss_pos, 60*np.asarray(boss.facing)), IMAGE_DICT["CRYSTAL"], 20, boss.facing, 'oscillating', spawn_time=game_counter, osc_phase=0)
                    new_crystal_2 = Missile(SIZE_DICT["CRYSTAL"], np.add(boss_pos, 60*np.asarray(boss.facing)), IMAGE_DICT["CRYSTAL"], 20, boss.facing, 'oscillating', spawn_time=game_counter, osc_phase=-np.pi)
                    crystals.add(new_crystal_1)
                    crystals.add(new_crystal_2)
                elif boss.phase == 1:  # Move (Phase 2, Choice 3): Astral Current
                    new_laser = Laser(COLOR_DICT["PURPLE"], np.add(boss_pos, 40*np.asarray(boss.facing)), np.add(boss_pos, WIDTH*np.sqrt(2)*np.asarray(boss.facing)), 20, 2*FPS, FPS/2)
                    lasers.add(new_laser)
            elif choice == 3:
                if boss.phase == 0:  # Move (Phase 1, Choice 4): Charged Converging Crystals
                    gap = 50
                    for i in range(5):
                        perp_ray = normalize((boss.facing[1], -boss.facing[0]))  # Get vector orthogonal to boss's facing direction
                        new_crystal = Missile(SIZE_DICT["CRYSTAL"], np.asarray(boss_pos)+60*np.asarray(boss.facing)+perp_ray*gap*(i-2), IMAGE_DICT["CRYSTAL"], 30, boss.facing, 'linear', FPS*(1+i*0.1))
                        crystals.add(new_crystal)
                elif boss.phase == 1:  # Move (Phase 2, Choice 4): Asteroid Shower
                    screen_shake_timer = FPS/8  # Add a screen shake
                    num_asteroids = 5
                    # Spawn asteroids near (an offset of) the closest screen border behind the boss's facing direction
                    if boss.facing[0] != 0:  # Avoid dividing by zero
                        slope = (boss.facing[1]/boss.facing[0])  # Get slope of boss's facing direction
                    else:
                        slope = (boss.facing[1]/0.01)
                    y_intercept = boss_pos[1]-slope*boss_pos[0]  # Get y intercept of boss's position vector
                    back_y = 0
                    back_x = 0
                    max_offset = 200  # Maximum angle variation
                    variation = np.random.rand(num_asteroids, 2)*max_offset  # Random angle variation
                    if boss.facing[1] >= boss.facing[0]:  # y is dominant
                        if boss.facing[1] >= 0:  # y is positive
                            back_y = 0-max_offset
                        else:  # y is negative
                            back_y = HEIGHT+max_offset
                        back_x = (back_y-y_intercept)/slope
                    else:  # x is dominant
                        if boss.facing[0] >= 0:  # x is positive
                            back_x = 0-max_offset
                        else:  # x is negative
                            back_x = WIDTH+max_offset
                        back_y = slope*back_x+y_intercept
                    for i in range(num_asteroids):
                        new_asteroid = Missile(SIZE_DICT["ASTEROID"], np.add(variation[i], (back_x, back_y)), IMAGE_DICT["ASTEROID"], np.random.uniform(20, 25), boss.facing, 'linear', spawn_prot=4*FPS)
                        crystals.add(new_asteroid)
            elif choice == 4:
                if boss.phase == 0:  # Move (Phase 1, Choice 5): Quintuple Crystal Blast
                    for i in range(-60, 61, 30):
                        i_rad = np.pi*i/180
                        new_crystal = Missile(SIZE_DICT["CRYSTAL"], np.add(boss_pos, 60*np.asarray(boss.facing)), IMAGE_DICT["CRYSTAL"], 10, get_rot_matrix(i_rad)@np.asarray(boss.facing), 'linear')
                        crystals.add(new_crystal)
                elif boss.phase == 1:  # Move (Phase 2, Choice 5): Triple Asteroid Blast
                    screen_shake_timer = FPS/8  # Add a screen shake
                    for i in range(-30, 31, 30):
                        i_rad = np.pi * i / 180
                        new_asteroid = Missile(SIZE_DICT["ASTEROID"], np.add(boss_pos, 60 * np.asarray(boss.facing)), IMAGE_DICT["ASTEROID"], 10, get_rot_matrix(i_rad) @ np.asarray(boss.facing), 'linear')
                        crystals.add(new_asteroid)
    boss.image = rotate(boss.clean_img, 90-np.arctan2(boss.facing[1], boss.facing[0])*180/np.pi)  # Boss rotates in facing direction
    if boss.curr_iframe_cooldown > 0 and boss.transitioning == 0:  # Overlay a red tint on boss when damaged
        iframe_mult = 1-np.abs(boss.curr_iframe_cooldown-boss.iframe_cooldown*FPS/2)/(boss.iframe_cooldown*FPS/2)  # Maximum of 1 at half the cooldown
        boss.image.fill((255*iframe_mult, 0, 0, 0), special_flags=pg.BLEND_RGBA_ADD)
        boss.curr_iframe_cooldown -= 1
    elif boss.transitioning in (2, 3):  # Overlay a purple tint when boss is transitioning to next phase
        transition_mult = boss.curr_transition_cooldown/boss.transition_cooldown  # Maximum of 1 at max cooldown
        if boss.transitioning == 2:
            transition_mult = 1-transition_mult
        boss.image.fill(np.multiply(transition_mult, COLOR_DICT["PURPLE"]), special_flags=pg.BLEND_RGBA_ADD)
    # Bullet logic
    for bullet in bullets:
        bullet.rect.center = np.add(bullet.rect.center, bullet.curr_vel*np.asarray(bullet.facing))  # Bullet moves in facing direction
        bullet.image = rotate(bullet.clean_img, 270-np.arctan2(bullet.facing[1], bullet.facing[0])*180/np.pi)  # Bullet rotates toward facing direction
        if bullet.leave_trail and game_counter % TRAIL_FREQ == 0:  # Leave a trail particle at a frequency depending on TRAIL_FREQ
            trails.add(Trail(COLOR_DICT["GOLD"], bullet.rect.center, 0.25*bullet.rect.width, FPS, 0, (0, 1)))
        if len(blackholes) > 0:  # Slow hero's bullets down when a black hole is present
            bullet.curr_vel = bullet.vel/2
        else:
            bullet.curr_vel = bullet.vel
        for blackhole in blackholes:
            grav_constant = 1000
            bullet_to_blackhole = np.subtract(blackhole.center, bullet.rect.center)
            dot = np.dot(normalize(bullet_to_blackhole), bullet.facing)+1  # dot is 0 when bullet faces away from black hole, 2 when facing black hole, and 1 when facing orthogonally
            mag_grav_pull = grav_constant*blackhole.curr_radius*dot/np.linalg.norm(bullet_to_blackhole)**2  # Calculate magnitude of gravitational pull
            bullet.rect.center = np.add(bullet.rect.center, np.multiply(normalize(bullet_to_blackhole), mag_grav_pull))  # Move bullet toward black hole
            if np.linalg.norm(np.subtract(blackhole.center, bullet.rect.center)) <= blackhole.curr_radius:  # Bullet is sucked into black hole and destroyed
                bullet.kill()
                break
        if bullet.rect.top <= 0 or bullet.rect.left <= 0 or bullet.rect.bottom >= HEIGHT or bullet.rect.right >= WIDTH:  # Bullet reaches screen border and is destroyed
            bullet.kill()
        elif bullet.rect.colliderect(boss.rect):  # Bullet collides with boss and generates sparks from impact point
            max_spark_size = 20
            num_sparks = int(np.random.uniform(3, 6))  # Generate random number from [3, 5]
            spark_sizes = (np.random.rand(num_sparks)*max_spark_size).astype(int)  # Generate num_sparks integer values of [0, max_spark_size)
            lifespans = (np.random.random(num_sparks)*FPS).astype(int)  # Generate num_sparks integer values of [0, FPS)
            angles_rad = np.linspace(-np.pi/3, np.pi/3, num_sparks)  # Generate num_sparks angles from [-pi/3, pi/3]
            variation = (np.random.random(num_sparks)*2-1)*np.pi/6  # Random angle variation
            angles_rad = np.add(angles_rad, variation)  # Add angle variation to angles_rad
            rot_matrix_list = get_rot_matrix(angles_rad).swapaxes(0, 2)  # Change rotation matrix list shape to (num_sparks, 2, 2)
            facing_list = rot_matrix_list@np.negative(bullet.facing)  # Apply rotations to the opposite of boss's facing direction
            for ind, facing in enumerate(facing_list):
                spark_pos = np.add(boss_pos, 60*facing)  # Spawn sparks at a distance from boss
                new_spark = Trail(COLOR_DICT["DARK RED"], spark_pos, spark_sizes[ind], lifespans[ind], 5, normalize(facing))
                trails.add(new_spark)
            bullet.kill()  # Destroy the bullet upon impact on the boss
            if boss.curr_iframe_cooldown <= 0:  # Reset boss's short invincibility cooldown
                boss.curr_iframe_cooldown = boss.iframe_cooldown*FPS
                boss.phase_hp[boss.phase] -= 1
                if boss.phase_hp[boss.phase] <= 0:  # Boss's current phase health is 0, so enter phase transition
                    screen_shake_timer = FPS/4  # Add a screen shake
                    boss.transitioning = 1
                    boss.curr_vel = boss.vel*4  # Boss's velocity is temporarily increased
                    boss.reset_move_matrix()  # Boss's move matrix is reset
                    boss.phase += 1
                    if boss.phase >= len(boss.phase_hp):  # Hero wins the game when no more boss phases are left
                        game_status = 1
    # Crystal logic
    for crystal in crystals:
        delay_mult = 0
        if crystal.curr_delay > 0:  # Crystal does not shoot yet because there is a delay
            delay_mult = crystal.curr_delay/crystal.delay  # Maximum of 1 at max delay
            crystal.curr_delay -= 1
            crystal.facing = normalize(hero_pos-np.asarray(crystal.rect.center))  # Crystal faces toward hero
            if crystal.curr_delay == 0:  # Crystal is ready to shoot
                screen_shake_timer = FPS/8  # Add a screen shake
        else:
            if crystal.spawn_prot > 0:  # Crystal is temporarily protected from being destroyed by screen border
                crystal.spawn_prot -= 1
            if crystal.type == 'linear':
                crystal.rect.center = np.add(crystal.rect.center, crystal.vel*np.asarray(crystal.facing))
            elif crystal.type == 'curved':
                crystal_to_hero = normalize(np.subtract(hero_pos, crystal.rect.center))
                crystal.rect.center = np.add(crystal.rect.center, crystal.vel*np.asarray(crystal.facing)+9*crystal_to_hero)
            elif crystal.type == 'oscillating':
                amp = 2
                freq = 0.25
                # Rotate a sine wave by an angle to receive a parametric equation
                angle_rad = -np.arctan2(crystal.facing[1], -crystal.facing[0])  # Angle to rotate the sine wave by
                t = (crystal.spawn_time-game_counter)  # Difference in time from crystal's spawn time to game counter time
                transformation = get_rot_matrix(angle_rad)@np.asarray((t, amp*np.sin(freq*t+crystal.osc_phase)))  # Get parametric equation for rotated sine wave
                crystal.rect.center = np.add(crystal.spawn_pt, crystal.vel*transformation)  # Move the crystal along the path
            if crystal.leave_trail and game_counter % TRAIL_FREQ == 0:  # Leave a trail particle at a frequency depending on TRAIL_FREQ
                trail_color = COLOR_DICT["ORANGE"]
                if crystal.rect.height == SIZE_DICT["CRYSTAL"][0]:
                    trail_color = COLOR_DICT["LIGHT BLUE"]
                trails.add(Trail(trail_color, crystal.rect.center, 0.25*crystal.rect.width, FPS, 0, (0, 1)))
        crystal.image = rotate(crystal.clean_img, 270-np.arctan2(crystal.facing[1], crystal.facing[0])*180/np.pi)  # Crystal rotates toward facing direction
        crystal.image.fill((255*delay_mult, 255*delay_mult, 255*delay_mult, 0), special_flags=pg.BLEND_RGBA_ADD)
        if crystal.spawn_prot <= 0 and (crystal.rect.top <= 0 or crystal.rect.left <= 0 or crystal.rect.bottom >= HEIGHT or crystal.rect.right >= WIDTH):  # Crystal reaches screen border and is destroyed
            crystal.kill()
        if crystal.rect.colliderect(hero.rect) and np.linalg.norm(np.subtract(crystal.rect.center, hero_pos)) <= 30:  # Crystal collides with hero and game is over
            game_status = -1
    # Laser logic
    for laser in lasers:
        if laser.curr_delay > 0:
            laser.curr_delay -= 1
            laser.curr_color = COLOR_DICT["LASER"]
            laser.curr_end_pos = np.add(laser.start_pos, np.subtract(laser.end_pos, laser.start_pos)*(1-(laser.curr_delay/laser.delay)**2))  # Current end point moves toward eventual end point
            if laser.curr_delay == 0:
                screen_shake_timer = FPS/8  # Add a screen shake
        elif laser.curr_lifespan > 0:
            laser.curr_lifespan -= 1
            laser.curr_color = np.subtract(255, np.subtract(255, laser.color)*(1-laser.curr_lifespan/laser.lifespan))  # Fades from white to laser's color
            laser.curr_start_pos = np.add(laser.start_pos, np.subtract(laser.end_pos, laser.start_pos)*(1-(laser.curr_lifespan/laser.lifespan)**2))  # Current start point moves toward end point
            coords_list = np.linspace(laser.curr_start_pos, laser.curr_end_pos, 100)  # Generate collision points of laser
            for coords in coords_list:
                if hero.rect.collidepoint(coords[0], coords[1]):  # Laser collides with hero and game is over
                    game_status = -1
        else:  # Laser's lifespan has reached zero and is destroyed
            laser.kill()
    # Shockwave logic
    for shockwave in shockwaves:
        if shockwave.curr_lifespan > 0:
            shockwave.curr_lifespan -= 1
            shockwave.center = boss_pos  # Shockwave is always centered at boss's center
            radius_mult = 1-(np.abs(shockwave.curr_lifespan-shockwave.lifespan/2)/(shockwave.lifespan/2))**2  # Maximum of 1 at half of shockwave's lifespan
            shockwave.curr_radius = radius_mult*shockwave.radius  # Multiply normalized multiplier with shockwave's maximum radius
            if np.linalg.norm(np.subtract(boss_pos, hero_pos)) <= shockwave.curr_radius:  # Shockwave collides with hero and game is over
                game_status = -1
        else:  # Shockwave's lifespan has reached zero and is destroyed
            shockwave.kill()
    # Black Hole logic
    for blackhole in blackholes:
        if blackhole.curr_lifespan > 0:
            color_mult = (np.sin(game_counter/10)+1)/2  # Black hole's color depends on sine function of game counter time
            blackhole.curr_color = np.tile(55+200*color_mult, 3)  # All RGB channels are equal
            blackhole.curr_lifespan -= 1
            lifespan_frac = blackhole.curr_lifespan/blackhole.lifespan  # Maximum of 1 at black hole's max lifespan
            blackhole.curr_vel = blackhole.vel*lifespan_frac  # Multiply normalized multiplier by black hole's maximum velocity
            blackhole.center = np.add(blackhole.center, blackhole.vel*np.asarray(blackhole.facing))  # Move black hole toward its facing direction
            if lifespan_frac > 0.75:  # Black hole expands
                radius_mult = ((blackhole.lifespan-blackhole.curr_lifespan)/(blackhole.lifespan*0.25))**2
            elif lifespan_frac <= 0.25:  # Black hole shrinks
                radius_mult = 1-((blackhole.lifespan*0.25-blackhole.curr_lifespan)/(blackhole.lifespan*0.25))**2
            else:  # Black hole's size stays at its maximum
                radius_mult = 1
            blackhole.curr_radius = blackhole.radius*radius_mult  # Black hole's current radius is its maximum radius multiplied by a normalized multiplier
            if np.linalg.norm(np.subtract(blackhole.center, hero_pos)) <= blackhole.curr_radius:  # Black hole collides with hero and game is over
                game_status = -1
        else:  # Black hole's lifespan has reached zero and is destroyed
            blackhole.kill()
    # Trail logic
    for trail in trails:
        if trail.curr_lifespan > 0:
            trail.curr_lifespan -= 1
            radius_mult = (trail.curr_lifespan/trail.lifespan)**2  # Maximum of 1 at trail's max lifespan
            trail.curr_radius = radius_mult*trail.radius  # Trail's radius is its maximum radius multiplied by a normalized multiplier
            trail.curr_color = np.add(trail.color, np.subtract(255, trail.color)*(1-radius_mult))  # Trail's color fades to white
            if trail.vel > 0:
                trail.center = np.add(trail.center, trail.vel*trail.facing)
        else:  # Trail's lifespan has reached zero and is destroyed
            trail.kill()
    return game_status, game_counter+1, screen_shake_timer


def draw_window(game_status, fonts, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, bg_group, screen_shake_pos):
    """
    Draws objects to the screen
    """
    DISPLAY.blit(WIN, screen_shake_pos)
    for bg in bg_group:
        WIN.blit(bg.image, np.subtract(bg.rect.center, np.asarray(bg.rect.size)/2))
    for shockwave in shockwaves:
        pg.draw.circle(WIN, shockwave.color, shockwave.center, shockwave.curr_radius)
    for blackhole in blackholes:
        pg.draw.circle(WIN, blackhole.curr_color, blackhole.center, blackhole.curr_radius)
    for laser in lasers:
        pg.draw.line(WIN, laser.curr_color, laser.curr_start_pos, laser.curr_end_pos, width=laser.width)
    for trail in trails:
        if np.abs(int(trail.center[0])) < MAX_INT and np.abs(int(trail.center[1])) < MAX_INT and int(trail.curr_radius) < MAX_INT:  # Avoid integer overflows
            pygame.gfxdraw.filled_circle(WIN, int(trail.center[0]), int(trail.center[1]), int(trail.curr_radius), trail.curr_color)
    for bullet in bullets:
        WIN.blit(bullet.image, np.subtract(bullet.rect.center, np.asarray(bullet.image.get_rect().size)/2))
    for crystal in crystals:
        WIN.blit(crystal.image, np.subtract(crystal.rect.center, np.asarray(crystal.image.get_rect().size)/2))
    pg.draw.rect(WIN, COLOR_DICT["DARK RED"], pg.rect.Rect(WIDTH/2-HEALTH_BAR_WIDTH/2, HEIGHT-50, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT), 0, 20, 20, 20, 20)
    pg.draw.rect(WIN, COLOR_DICT["LIGHT RED"], pg.rect.Rect(WIDTH/2-HEALTH_BAR_WIDTH/2, HEIGHT-50, HEALTH_BAR_WIDTH*(np.sum(boss.phase_hp)/boss.total_hp), HEALTH_BAR_HEIGHT), 0, 20, 20, 20, 20)
    if boss.phase == 0:
        boss_text = 'Undead Dragonfly'
    elif boss.phase >= 1:
        boss_text = 'King of the Cosmos'
    create_text(WIN, boss_text, COLOR_DICT["WHITE"], fonts[2], WIDTH/2, HEIGHT-35, 2)
    WIN.blit(boss.image, np.subtract(boss.rect.center, np.asarray(boss.image.get_rect().size)/2))
    WIN.blit(hero.image, np.subtract(hero.rect.center, np.asarray(hero.image.get_rect().size)/2))
    if game_status != 0:
        if game_status == -2:
            create_text(WIN, 'ONE LIFE', COLOR_DICT["YELLOW"], fonts[0], WIDTH/2, HEIGHT/2-150)
            create_text(WIN, 'Press P to play', COLOR_DICT["WHITE"], fonts[1], WIDTH/2, HEIGHT/2-50)
            create_text(WIN, 'Use mouse to aim', COLOR_DICT["GRAY"], fonts[3], WIDTH/2, HEIGHT/2+70)
            create_text(WIN, 'Press WASD to move', COLOR_DICT["GRAY"], fonts[3], WIDTH/2, HEIGHT/2+120)
            create_text(WIN, 'Hold SPACE to shoot', COLOR_DICT["GRAY"], fonts[3], WIDTH/2, HEIGHT/2+170)
        else:
            if game_status == -1:
                explosion = BasicSprite(SIZE_DICT["SMALL EXPLOSION"], hero.rect.center, IMAGE_DICT["EXPLOSION"], 0, (0, 1))
                result_text = 'GAME OVER!'
                result_color = COLOR_DICT["LIGHT RED"]
            elif game_status == 1:
                explosion = BasicSprite(SIZE_DICT["LARGE EXPLOSION"], boss.rect.center, IMAGE_DICT["EXPLOSION"], 0, (0, 1))
                result_text = 'YOU WIN!'
                result_color = COLOR_DICT["LIGHT GREEN"]
            WIN.blit(explosion.image, np.subtract(explosion.rect.center, np.asarray(explosion.image.get_rect().size)/2))
            create_text(WIN, result_text, result_color, fonts[0], WIDTH/2, HEIGHT/2-60)
            create_text(WIN, 'Press P to replay', COLOR_DICT["WHITE"], fonts[1], WIDTH/2, HEIGHT/2+60)
    pg.display.update()  # Display must be updated every game iteration


def init_game():
    """
    Sets up the default variables for the game
    """
    bg_group = pg.sprite.Group()
    bg = Background(SIZE_DICT["BACKGROUND"], (WIDTH/2, HEIGHT/2), IMAGE_DICT["BACKGROUND"], 1, (0, 1))
    bg_group.add(bg)
    hero = Hero(SIZE_DICT["SPACESHIP"], (WIDTH/2, HEIGHT-100), IMAGE_DICT["SPACESHIP"][0], 7, (0, 1), 0.4)
    boss = Boss(SIZE_DICT["BOSS"], (WIDTH/2, 100), IMAGE_DICT["BOSS"][0], 2, (0, -1), 0, [50, 50], 0.2, (1, 2), FPS)
    bullets = pg.sprite.Group()
    crystals = pg.sprite.Group()
    lasers = pg.sprite.Group()
    shockwaves = pg.sprite.Group()
    blackholes = pg.sprite.Group()
    trails = pg.sprite.Group()
    clock = pg.time.Clock()
    fonts = [pg.font.SysFont('bahnschrift', font_size) for font_size in FONT_SIZES]
    game_status = 0
    game_counter = 0
    screen_shake_timer = 0
    screen_shake_pos = 0, 0
    return bg_group, bg, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, clock, fonts, game_status, game_counter, screen_shake_timer, screen_shake_pos


def main():
    """
    Initializes the default variables, then runs the game loop
    """
    pg.font.init()
    pg.display.set_caption("ONE LIFE")
    bg_group, bg, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, clock, fonts, game_status, game_counter, screen_shake_timer, screen_shake_pos = init_game()
    game_status = -2  # A game status of -2 means the title screen will be displayed
    game_running = True
    while game_running:  # Game loop
        clock.tick(FPS)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                game_running = False  # Exits the game loop
            if event.type == pg.KEYDOWN:  # Key is being pressed
                if event.key == pg.K_w: keys[0] = True
                elif event.key == pg.K_a: keys[1] = True
                elif event.key == pg.K_s: keys[2] = True
                elif event.key == pg.K_d: keys[3] = True
                elif event.key == pg.K_SPACE: keys[4] = True
                elif event.key == pg.K_p: keys[5] = True
            if event.type == pg.KEYUP:  # Pressed key is released
                if event.key == pg.K_w: keys[0] = False
                elif event.key == pg.K_a: keys[1] = False
                elif event.key == pg.K_s: keys[2] = False
                elif event.key == pg.K_d: keys[3] = False
                elif event.key == pg.K_SPACE: keys[4] = False
                elif event.key == pg.K_p: keys[5] = False
        if game_status == 0:  # Game is currently running
            game_status, game_counter, screen_shake_timer_temp = game_logic(game_counter, keys, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, bg_group)
            if screen_shake_timer_temp > 0:  # Time is added to screen shake duration if the value is nonzero
                screen_shake_timer = screen_shake_timer_temp
            if game_status != 0:
                screen_shake_timer = FPS/4  # Whenever the hero wins or loses, the screen shakes
        elif keys[5]:
            bg_group, bg, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, clock, fonts, game_status, game_counter, screen_shake_timer, screen_shake_pos = init_game()
            game_status = 0  # Initiate the game
        screen_shake_timer, screen_shake_pos = screen_shake_logic(screen_shake_timer, screen_shake_pos)  # Apply screen shake
        draw_window(game_status, fonts, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, bg_group, screen_shake_pos)
    pg.quit()  # Call pygame's quit function before the game is closed


if __name__ == "__main__":
    main()
