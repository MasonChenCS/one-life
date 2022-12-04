import pygame as pg
import pygame.gfxdraw
import os
import numpy as np

WIDTH, HEIGHT = 900, 900
DISPLAY = pg.display.set_mode((WIDTH, HEIGHT))
WIN = pg.surface.Surface((WIDTH, HEIGHT))
WHITE = (255, 255, 255)
FPS = 60
BACKGROUND_SIZE = (900, 900)
BACKGROUND_IMAGE = pg.image.load(os.path.join('Assets', 'background_cropped.jpg'))
SPACESHIP_SIZE = (50, 50)
SPACESHIP_IMAGES = []
for i in range(4):
    SPACESHIP_IMAGES.append(pg.image.load(os.path.join('Assets', 'ship{}.png'.format(i+1))))
BULLET_SIZE = (20, 20)
BULLET_IMAGE = pg.image.load(os.path.join('Assets', 'bullet.png'))
CRYSTAL_SIZE = (40, 40)
CRYSTAL_IMAGE = pg.image.load(os.path.join('Assets', 'crystal.png')).convert_alpha()
ASTEROID_SIZE = (70, 70)
ASTEROID_IMAGE = pg.image.load(os.path.join('Assets', 'asteroid.png'))
BOSS_SIZE = (150, 150)
BOSS_IMAGE_1 = pg.image.load(os.path.join('Assets', 'boss1.png')).convert_alpha()
BOSS_IMAGE_2 = pg.image.load(os.path.join('Assets', 'boss2.png')).convert_alpha()
SMALL_EXPLOSION_SIZE = (60, 60)
LARGE_EXPLOSION_SIZE = (160, 160)
EXPLOSION_IMAGE = pg.image.load(os.path.join('Assets', 'explosion.png'))
keys = [False, False, False, False, False, False]
WHITE_COLOR = (255, 255, 255, 0)
GRAY_COLOR = (200, 200, 200, 0)
PURPLE_COLOR = (130, 30, 230, 0)
LASER_COLOR = (140, 40, 40, 0)
LIGHT_GREEN_COLOR = (60, 200, 120, 0)
DARK_RED_COLOR = (100, 0, 0, 0)
LIGHT_RED_COLOR = (255, 60, 60, 0)
LIGHT_BLUE_COLOR = (0, 200, 230, 0)
ORANGE_COLOR = (255, 128, 0, 0)
YELLOW_COLOR = (255, 255, 0, 0)
GOLD_COLOR = (160, 130, 0, 0)
HEALTH_BAR_WIDTH = 700
HEALTH_BAR_HEIGHT = 25
FONT_SIZES = (120, 80, 25, 40)
TRAIL_FREQ = 2
MAX_INT = 10_000


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

    def set_image(self, image):
        self.image = pg.transform.scale(image, SPACESHIP_SIZE)
        self.clean_img = self.image


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
        self.move_matrix = np.ones((2, 5))
        self.move_matrix[1, :] = self.move_matrix[0, :] / self.move_matrix.sum(axis=1)[0]
        return self.move_matrix

    def set_image(self, image):
        self.image = pg.transform.scale(image, BOSS_SIZE)
        self.clean_img = self.image


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


def get_rot_matrix(angle_rad):
    return np.asarray(((np.cos(angle_rad), -np.sin(angle_rad)), (np.sin(angle_rad), np.cos(angle_rad))))


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm != 0:
        return vec/norm
    return vec


def rotate(img, angle):
    if not np.isnan(angle):
        return pg.transform.rotate(img, angle)
    return img


def create_text(surface, text, color, font, x_pos, y_pos, drop_shadow_dist=5):
    text_main = font.render(text, True, color)
    text_shadow = font.render(text, True, (0, 0, 0, 0))
    textbox = text_main.get_rect()
    textbox.center = x_pos, y_pos
    surface.blit(text_shadow, np.add(textbox, drop_shadow_dist))
    surface.blit(text_main, textbox)


def boss_move(move_matrix):
    choice = np.random.choice(move_matrix.shape[1], 1, p=move_matrix[1])[0]
    move_matrix[0, :choice] += 0.5
    move_matrix[0, choice+1:] += 0.5
    move_matrix[1, :] = move_matrix[0, :]/move_matrix.sum(axis=1)[0]
    return move_matrix, choice


def screen_shake_logic(screen_shake_timer, screen_shake_pos):
    if screen_shake_timer > 0:
        screen_shake_timer -= 1
        if screen_shake_timer > FPS/10:
            shake_mult = 10
            diff = (np.random.rand(2)*2-1)*shake_mult
            screen_shake_pos = np.add(screen_shake_pos, diff)
        elif FPS/10:
            screen_shake_pos = np.add(screen_shake_pos, 4*normalize(np.subtract((0, 0), screen_shake_pos)))
    else:
        screen_shake_pos = (0, 0)
    return screen_shake_timer, screen_shake_pos


def game_logic(game_counter, keys, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, bg_group):
    game_status = 0  # 0 = continue, -1 = lose, 1 = win
    screen_shake_timer = 0
    # Background logic
    for bg in bg_group:
        if bg.rect.top >= HEIGHT:
            bg.kill()
        if bg.rect.top == 0:
            new_bg = Background(BACKGROUND_SIZE, (WIDTH/2, -HEIGHT/2), BACKGROUND_IMAGE, 1, (0, 1))
            bg_group.add(new_bg)
        bg.rect.center = np.add(bg.rect.center, bg.vel * np.asarray(bg.facing))
    # Hero logic
    mouse_pos = pg.mouse.get_pos()
    hero_pos = hero.rect.center
    hero.facing = normalize(np.subtract(mouse_pos, hero_pos))
    hero.image = rotate(hero.clean_img, 270-np.arctan2(hero.facing[1], hero.facing[0])*180/np.pi)
    if game_counter % 50 == 0:
        hero.curr_img_phase += 1
        if hero.curr_img_phase >= 4:
            hero.curr_img_phase = 0
        hero.set_image(SPACESHIP_IMAGES[hero.curr_img_phase])
    if hero.curr_bullet_cooldown > 0:
        hero.curr_bullet_cooldown -= 1
    if keys[0] and hero.rect.top > 0:
        hero.rect.centery -= hero.vel
    if keys[1] and hero.rect.left > 0:
        hero.rect.centerx -= hero.vel
    if keys[2] and hero.rect.bottom < HEIGHT:
        hero.rect.centery += hero.vel
    if keys[3] and hero.rect.right < WIDTH:
        hero.rect.centerx += hero.vel
    if keys[4] and hero.curr_bullet_cooldown <= 0:
        hero.curr_bullet_cooldown = hero.bullet_cooldown*FPS
        bullet = Missile(BULLET_SIZE, np.add(hero.rect.center, 35*np.asarray(hero.facing)), BULLET_IMAGE, 10, hero.facing, 'linear')
        bullets.add(bullet)
    # Boss logic
    boss_pos = boss.rect.center
    origin_pos = (WIDTH/2, HEIGHT/2)
    boss.rect.center = np.add(boss_pos, boss.curr_vel * np.asarray(boss.facing))
    boss.facing = (0, 1)
    if boss.rect.colliderect(hero.rect) and np.linalg.norm(np.subtract(boss_pos, hero_pos)) <= 100:
        game_status = -1
    if boss.transitioning == 1:
        boss.facing = normalize(np.subtract(origin_pos, boss_pos))
        if np.linalg.norm(np.subtract(boss_pos, origin_pos)) <= 6:
            boss.transitioning = 2
            boss.curr_vel = 0
    elif boss.transitioning == 2:
        if boss.curr_transition_cooldown > 0:
            boss.curr_transition_cooldown -= 1
        else:
            screen_shake_timer = FPS/2
            num_asteroids = 16
            angles_rad = np.linspace(0, 2 * np.pi, num_asteroids, endpoint=False)
            rot_matrix_list = get_rot_matrix(angles_rad).swapaxes(0, 2)
            facing_list = rot_matrix_list@boss.facing
            for facing in facing_list:
                crystal_pos = np.add(boss_pos, 100*facing)
                crystal_facing = np.asarray(crystal_pos-boss_pos)
                crystal_facing[[0, 1]] = crystal_facing[[1, 0]]
                crystal_facing[0] = -crystal_facing[0]
                crystal_facing = normalize(crystal_facing)
                new_asteroid = Missile(ASTEROID_SIZE, crystal_pos, ASTEROID_IMAGE, 10, crystal_facing, 'linear')
                crystals.add(new_asteroid)
            boss.transitioning = 3
            boss.curr_transition_cooldown = boss.transition_cooldown
            if boss.phase == 1:
                boss.set_image(BOSS_IMAGE_2)
    elif boss.transitioning == 3:
        if boss.curr_transition_cooldown > 0:
            boss.curr_transition_cooldown -= 1
        else:
            boss.transitioning = 0
            boss.curr_transition_cooldown = boss.transition_cooldown
            boss.curr_vel = boss.vel
    elif boss.transitioning == 0:
        boss.facing = normalize(np.subtract(hero_pos, boss_pos))
        if boss.curr_move_cooldown > 0:
            boss.curr_move_cooldown -= 1
            if boss.curr_move_cooldown < 0.5*FPS:
                boss.curr_vel = 0
        else:
            boss.curr_vel = boss.vel
            boss.move_matrix, choice = boss_move(boss.move_matrix)
            boss.curr_move_cooldown = np.random.uniform(boss.move_cooldown_range[0], boss.move_cooldown_range[1])*FPS
            if choice == 0:
                if boss.phase == 0:
                    new_crystal = Missile(CRYSTAL_SIZE, np.add(boss_pos, 60 * np.asarray(boss.facing)), CRYSTAL_IMAGE, 20, boss.facing, 'linear')
                    crystals.add(new_crystal)
                elif boss.phase == 1:
                    new_shockwave = Shockwave(PURPLE_COLOR, boss_pos, 300, 2*FPS)
                    shockwaves.add(new_shockwave)
            elif choice == 1:
                if boss.phase == 0:
                    new_crystal = Missile(CRYSTAL_SIZE, np.add(boss_pos, 60*np.asarray(boss.facing)), CRYSTAL_IMAGE, 10, boss.facing, 'curved')
                    crystals.add(new_crystal)
                elif boss.phase == 1:
                    num_blackholes = 3
                    offset = 90*np.pi/180
                    variation = (2*np.random.rand(num_blackholes)-1)*offset
                    angles_rad = np.linspace(0, 2*np.pi, num_blackholes, endpoint=False)
                    angles_rad = np.add(angles_rad, variation)
                    rot_matrix_list = get_rot_matrix(angles_rad).swapaxes(0, 2)
                    facing_list = rot_matrix_list@boss.facing
                    for facing in facing_list:
                        blackhole_pos = np.add(boss_pos, 10*facing)
                        blackhole_facing = normalize(blackhole_pos-np.asarray(boss_pos))
                        new_blackhole = Blackhole((0, 0, 0), blackhole_pos, 50, 3*FPS, 3, blackhole_facing)
                        blackholes.add(new_blackhole)
            elif choice == 2:
                if boss.phase == 0:
                    new_crystal_1 = Missile(CRYSTAL_SIZE, np.add(boss_pos, 60*np.asarray(boss.facing)), CRYSTAL_IMAGE, 20, boss.facing, 'oscillating', spawn_time=game_counter, osc_phase=0)
                    new_crystal_2 = Missile(CRYSTAL_SIZE, np.add(boss_pos, 60*np.asarray(boss.facing)), CRYSTAL_IMAGE, 20, boss.facing, 'oscillating', spawn_time=game_counter, osc_phase=-np.pi)
                    crystals.add(new_crystal_1)
                    crystals.add(new_crystal_2)
                elif boss.phase == 1:
                    new_laser = Laser(PURPLE_COLOR, np.add(boss_pos, 40*np.asarray(boss.facing)), np.add(boss_pos, WIDTH*np.sqrt(2)*np.asarray(boss.facing)), 20, 2*FPS, FPS/2)
                    lasers.add(new_laser)
            elif choice == 3:
                if boss.phase == 0:
                    gap = 50
                    for i in range(5):
                        perp_ray = normalize((boss.facing[1], -boss.facing[0]))
                        new_crystal = Missile(CRYSTAL_SIZE, np.asarray(boss_pos)+60*np.asarray(boss.facing)+perp_ray*gap*(i-2), CRYSTAL_IMAGE, 30, boss.facing, 'linear', FPS*(1+i*0.1))
                        crystals.add(new_crystal)
                elif boss.phase == 1:
                    screen_shake_timer = FPS/8
                    num_asteroids = 5
                    if boss.facing[0] != 0:
                        slope = (boss.facing[1]/boss.facing[0])
                    else:
                        slope = (boss.facing[1]/0.01)
                    y_intercept = boss_pos[1]-slope*boss_pos[0]
                    back_y = 0
                    back_x = 0
                    offset = 200
                    variation = np.random.rand(num_asteroids, 2)*offset
                    if boss.facing[1] >= boss.facing[0]:  # y is dominant
                        if boss.facing[1] >= 0:  # y is positive
                            back_y = 0-offset
                        else:  # y is negative
                            back_y = HEIGHT+offset
                        back_x = (back_y-y_intercept)/slope
                    else:  # x is dominant
                        if boss.facing[0] >= 0:  # x is positive
                            back_x = 0-offset
                        else:  # x is negative
                            back_x = WIDTH+offset
                        back_y = slope*back_x+y_intercept
                    for i in range(num_asteroids):
                        new_asteroid = Missile(ASTEROID_SIZE, np.add(variation[i], (back_x, back_y)), ASTEROID_IMAGE, np.random.uniform(20, 25), boss.facing, 'linear', spawn_prot=4*FPS)
                        crystals.add(new_asteroid)
            elif choice == 4:
                if boss.phase == 0:
                    for i in range(-60, 61, 30):
                        i_rad = np.pi*i/180
                        new_crystal = Missile(CRYSTAL_SIZE, np.add(boss_pos, 60*np.asarray(boss.facing)), CRYSTAL_IMAGE, 10, get_rot_matrix(i_rad)@np.asarray(boss.facing), 'linear')
                        crystals.add(new_crystal)
                elif boss.phase == 1:
                    screen_shake_timer = FPS/8
                    for i in range(-30, 31, 30):
                        i_rad = np.pi * i / 180
                        new_asteroid = Missile(ASTEROID_SIZE, np.add(boss_pos, 60 * np.asarray(boss.facing)), ASTEROID_IMAGE, 10, get_rot_matrix(i_rad) @ np.asarray(boss.facing), 'linear')
                        crystals.add(new_asteroid)
    boss.image = rotate(boss.clean_img, 90 - np.arctan2(boss.facing[1], boss.facing[0]) * 180 / np.pi)
    if boss.curr_iframe_cooldown > 0 and boss.transitioning == 0:
        iframe_mult = 1 - np.abs(boss.curr_iframe_cooldown - boss.iframe_cooldown*FPS/2)/(boss.iframe_cooldown*FPS/2)
        boss.image.fill((255*iframe_mult, 0, 0, 0), special_flags=pg.BLEND_RGBA_ADD)
        boss.curr_iframe_cooldown -= 1
    elif boss.transitioning in (2, 3):
        transition_mult = boss.curr_transition_cooldown/boss.transition_cooldown
        if boss.transitioning == 2:
            transition_mult = 1-transition_mult
        boss.image.fill(np.multiply(transition_mult, PURPLE_COLOR), special_flags=pg.BLEND_RGBA_ADD)
    # Bullet logic
    for bullet in bullets:
        bullet.rect.center = np.add(bullet.rect.center, bullet.curr_vel*np.asarray(bullet.facing))
        bullet.image = rotate(bullet.clean_img, 270-np.arctan2(bullet.facing[1], bullet.facing[0])*180/np.pi)
        if bullet.leave_trail and game_counter % TRAIL_FREQ == 0:
            trails.add(Trail(GOLD_COLOR, bullet.rect.center, 0.25*bullet.rect.width, FPS, 0, (0, 1)))
        if len(blackholes) > 0:
            bullet.curr_vel = bullet.vel/2
        else:
            bullet.curr_vel = bullet.vel
        for blackhole in blackholes:
            grav_constant = 1000
            bullet_to_blackhole = np.subtract(blackhole.center, bullet.rect.center)
            dot = np.dot(normalize(bullet_to_blackhole), bullet.facing)+1
            bullet.rect.center = np.add(bullet.rect.center, np.multiply(normalize(bullet_to_blackhole), grav_constant*blackhole.curr_radius*dot/np.linalg.norm(bullet_to_blackhole)**2))
            if np.linalg.norm(np.subtract(blackhole.center, bullet.rect.center)) <= blackhole.curr_radius:
                bullet.kill()
                break
        if bullet.rect.top <= 0 or bullet.rect.left <= 0 or bullet.rect.bottom >= HEIGHT or bullet.rect.right >= WIDTH:
            bullet.kill()
        elif bullet.rect.colliderect(boss.rect):
            max_size = 20
            spark_sizes = (np.random.rand(int(np.random.uniform(3, 6)))*max_size).astype(int)
            lifespans = (np.random.random(len(spark_sizes))*FPS).astype(int)
            angles_rad = np.linspace(-np.pi/3, np.pi/3, len(spark_sizes))
            variation = (np.random.random(len(spark_sizes))*2-1)*np.pi/6
            angles_rad = np.add(angles_rad, variation)
            rot_matrix_list = get_rot_matrix(angles_rad).swapaxes(0, 2)
            facing_list = rot_matrix_list@np.negative(bullet.facing)
            for ind, facing in enumerate(facing_list):
                spark_pos = np.add(boss_pos, 60*facing)
                new_spark = Trail(DARK_RED_COLOR, spark_pos, spark_sizes[ind], lifespans[ind], 5, normalize(facing))
                trails.add(new_spark)
            bullet.kill()
            if boss.curr_iframe_cooldown <= 0:
                boss.curr_iframe_cooldown = boss.iframe_cooldown*FPS
                boss.phase_hp[boss.phase] -= 1
                if boss.phase_hp[boss.phase] <= 0:  # Phase transition
                    screen_shake_timer = FPS/4
                    boss.transitioning = 1
                    boss.curr_vel = boss.vel*4
                    boss.reset_move_matrix()
                    boss.phase += 1
                    if boss.phase >= len(boss.phase_hp):
                        game_status = 1
    # Crystal logic
    for crystal in crystals:
        delay_mult = 0
        if crystal.curr_delay > 0:
            delay_mult = crystal.curr_delay/crystal.delay
            crystal.curr_delay -= 1
            crystal.facing = normalize(hero_pos-np.asarray(crystal.rect.center))
            if crystal.curr_delay == 0:
                screen_shake_timer = FPS/8
        else:
            if crystal.spawn_prot > 0:
                crystal.spawn_prot -= 1
            if crystal.type == 'linear':
                crystal.rect.center = np.add(crystal.rect.center, crystal.vel*np.asarray(crystal.facing))
            elif crystal.type == 'curved':
                crystal_to_hero = normalize(np.subtract(hero_pos, crystal.rect.center))
                crystal.rect.center = np.add(crystal.rect.center, crystal.vel*np.asarray(crystal.facing)+9*crystal_to_hero)
            elif crystal.type == 'oscillating':
                amp = 2
                freq = 0.25
                angle_rad = -np.arctan2(crystal.facing[1], -crystal.facing[0])
                t = (crystal.spawn_time-game_counter)
                transformation = get_rot_matrix(angle_rad)@np.asarray((t, amp*np.sin(freq*t+crystal.osc_phase)))
                crystal.rect.center = np.add(crystal.spawn_pt, crystal.vel*transformation)
            if crystal.leave_trail and game_counter % TRAIL_FREQ == 0:
                trail_color = ORANGE_COLOR
                if crystal.rect.height == CRYSTAL_SIZE[0]:
                    trail_color = LIGHT_BLUE_COLOR
                trails.add(Trail(trail_color, crystal.rect.center, 0.25*crystal.rect.width, FPS, 0, (0, 1)))
        crystal.image = rotate(crystal.clean_img, 270-np.arctan2(crystal.facing[1], crystal.facing[0])*180/np.pi)
        crystal.image.fill((255*delay_mult, 255*delay_mult, 255*delay_mult, 0), special_flags=pg.BLEND_RGBA_ADD)
        if crystal.spawn_prot <= 0 and (crystal.rect.top <= 0 or crystal.rect.left <= 0 or crystal.rect.bottom >= HEIGHT or crystal.rect.right >= WIDTH):
            crystal.kill()
        if crystal.rect.colliderect(hero.rect) and np.linalg.norm(np.subtract(crystal.rect.center, hero_pos)) <= 30:
            game_status = -1
    # Laser logic
    for laser in lasers:
        if laser.curr_delay > 0:
            laser.curr_delay -= 1
            laser.curr_color = LASER_COLOR
            laser.curr_end_pos = np.add(laser.start_pos, np.subtract(laser.end_pos, laser.start_pos)*(1-(laser.curr_delay/laser.delay)**2))
            if laser.curr_delay == 0:
                screen_shake_timer = FPS/8
        elif laser.curr_lifespan > 0:
            laser.curr_lifespan -= 1
            laser.curr_color = np.subtract(255, np.subtract(255, laser.color)*(1-laser.curr_lifespan/laser.lifespan))
            laser.curr_start_pos = np.add(laser.start_pos, np.subtract(laser.end_pos, laser.start_pos)*(1-(laser.curr_lifespan/laser.lifespan)**2))
            coords_list = np.linspace(laser.curr_start_pos, laser.curr_end_pos, 100)
            for coords in coords_list:
                if hero.rect.collidepoint(coords[0], coords[1]):
                    game_status = -1
        else:
            laser.kill()
    # Shockwave logic
    for shockwave in shockwaves:
        if shockwave.curr_lifespan > 0:
            shockwave.curr_lifespan -= 1
            shockwave.center = boss_pos
            radius_mult = 1-(np.abs(shockwave.curr_lifespan-shockwave.lifespan/2)/(shockwave.lifespan/2))**2
            shockwave.curr_radius = radius_mult*shockwave.radius
            if np.linalg.norm(np.subtract(boss_pos, hero_pos)) <= shockwave.curr_radius:
                game_status = -1
        else:
            shockwave.kill()
    # Black Hole logic
    for blackhole in blackholes:
        if blackhole.curr_lifespan > 0:
            color_mult = (np.sin(game_counter/10)+1)/2
            blackhole.curr_color = np.tile(55+200*color_mult, 3)
            blackhole.curr_lifespan -= 1
            lifespan_frac = blackhole.curr_lifespan/blackhole.lifespan
            blackhole.curr_vel = blackhole.vel*lifespan_frac
            blackhole.center = np.add(blackhole.center, blackhole.vel * np.asarray(blackhole.facing))
            if lifespan_frac > 0.75:
                radius_mult = ((blackhole.lifespan-blackhole.curr_lifespan)/(blackhole.lifespan*0.25))**2
            elif lifespan_frac <= 0.25:
                radius_mult = 1-((blackhole.lifespan*0.25-blackhole.curr_lifespan)/(blackhole.lifespan*0.25))**2
            else:
                radius_mult = 1
            blackhole.curr_radius = blackhole.radius * radius_mult
            if np.linalg.norm(np.subtract(blackhole.center, hero_pos)) <= blackhole.curr_radius:
                game_status = -1
        else:
            blackhole.kill()
    # Trail logic
    for trail in trails:
        if trail.curr_lifespan > 0:
            trail.curr_lifespan -= 1
            radius_mult = (trail.curr_lifespan/trail.lifespan)**2
            trail.curr_radius = radius_mult*trail.radius
            trail.curr_color = np.add(trail.color, np.subtract(255, trail.color)*(1-radius_mult))
            if trail.vel > 0:
                trail.center = np.add(trail.center, trail.vel*trail.facing)
        else:
            trail.kill()
    return game_status, game_counter+1, screen_shake_timer


def draw_window(game_status, fonts, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, bg_group, screen_shake_pos):
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
        if np.abs(int(trail.center[0])) < MAX_INT and np.abs(int(trail.center[1])) < MAX_INT and int(trail.curr_radius) < MAX_INT:
            pygame.gfxdraw.filled_circle(WIN, int(trail.center[0]), int(trail.center[1]), int(trail.curr_radius), trail.curr_color)
    for bullet in bullets:
        WIN.blit(bullet.image, np.subtract(bullet.rect.center, np.asarray(bullet.image.get_rect().size)/2))
    for crystal in crystals:
        WIN.blit(crystal.image, np.subtract(crystal.rect.center, np.asarray(crystal.image.get_rect().size)/2))
    pg.draw.rect(WIN, DARK_RED_COLOR, pg.rect.Rect(WIDTH/2-HEALTH_BAR_WIDTH/2, HEIGHT-50, HEALTH_BAR_WIDTH, HEALTH_BAR_HEIGHT), 0, 20, 20, 20, 20)
    pg.draw.rect(WIN, LIGHT_RED_COLOR, pg.rect.Rect(WIDTH/2-HEALTH_BAR_WIDTH/2, HEIGHT-50, HEALTH_BAR_WIDTH*(np.sum(boss.phase_hp)/boss.total_hp), HEALTH_BAR_HEIGHT), 0, 20, 20, 20, 20)
    if boss.phase == 0:
        boss_text = 'Undead Dragonfly'
    elif boss.phase >= 1:
        boss_text = 'King of the Cosmos'
    create_text(WIN, boss_text, WHITE_COLOR, fonts[2], WIDTH/2, HEIGHT-35, 2)
    WIN.blit(boss.image, np.subtract(boss.rect.center, np.asarray(boss.image.get_rect().size)/2))
    WIN.blit(hero.image, np.subtract(hero.rect.center, np.asarray(hero.image.get_rect().size)/2))
    if game_status != 0:
        if game_status == -2:
            create_text(WIN, 'ONE LIFE', YELLOW_COLOR, fonts[0], WIDTH/2, HEIGHT/2-150)
            create_text(WIN, 'Press P to play', WHITE_COLOR, fonts[1], WIDTH/2, HEIGHT/2-50)
            create_text(WIN, 'Use mouse to aim', GRAY_COLOR, fonts[3], WIDTH/2, HEIGHT/2+70)
            create_text(WIN, 'Press WASD to move', GRAY_COLOR, fonts[3], WIDTH/2, HEIGHT/2+120)
            create_text(WIN, 'Hold SPACE to shoot', GRAY_COLOR, fonts[3], WIDTH/2, HEIGHT/2+170)
        else:
            if game_status == -1:
                explosion = BasicSprite(SMALL_EXPLOSION_SIZE, hero.rect.center, EXPLOSION_IMAGE, 0, (0, 1))
                result_text = 'GAME OVER!'
                result_color = LIGHT_RED_COLOR
            elif game_status == 1:
                explosion = BasicSprite(LARGE_EXPLOSION_SIZE, boss.rect.center, EXPLOSION_IMAGE, 0, (0, 1))
                result_text = 'YOU WIN!'
                result_color = LIGHT_GREEN_COLOR
            WIN.blit(explosion.image, np.subtract(explosion.rect.center, np.asarray(explosion.image.get_rect().size)/2))
            create_text(WIN, result_text, result_color, fonts[0], WIDTH/2, HEIGHT/2-60)
            create_text(WIN, 'Press P to replay', WHITE_COLOR, fonts[1], WIDTH/2, HEIGHT/2+60)
    pg.display.update()


def init_game():
    bg_group = pg.sprite.Group()
    bg = Background(BACKGROUND_SIZE, (WIDTH/2, HEIGHT/2), BACKGROUND_IMAGE, 1, (0, 1))
    bg_group.add(bg)
    hero = Hero(SPACESHIP_SIZE, (WIDTH/2, HEIGHT-100), SPACESHIP_IMAGES[0], 7, (0, 1), 0.4)
    boss = Boss(BOSS_SIZE, (WIDTH/2, 100), BOSS_IMAGE_1, 2, (0, -1), 0, [50, 50], 0.2, (1, 2), FPS)
    bullets = pg.sprite.Group()
    crystals = pg.sprite.Group()
    lasers = pg.sprite.Group()
    shockwaves = pg.sprite.Group()
    blackholes = pg.sprite.Group()
    trails = pg.sprite.Group()
    clock = pg.time.Clock()
    fonts = []
    for font_size in FONT_SIZES:
        fonts.append(pg.font.SysFont('bahnschrift', font_size))
    game_status = 0
    game_counter = 0
    screen_shake_timer = 0
    screen_shake_pos = (0, 0)
    return bg_group, bg, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, clock, fonts, game_status, game_counter, screen_shake_timer, screen_shake_pos


def main():
    pg.font.init()
    pg.display.set_caption("ONE LIFE")
    bg_group, bg, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, clock, fonts, game_status, game_counter, screen_shake_timer, screen_shake_pos = init_game()
    game_status = -2
    game_running = True
    while game_running:
        clock.tick(FPS)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                game_running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_w: keys[0] = True
                elif event.key == pg.K_a: keys[1] = True
                elif event.key == pg.K_s: keys[2] = True
                elif event.key == pg.K_d: keys[3] = True
                elif event.key == pg.K_SPACE: keys[4] = True
                elif event.key == pg.K_p: keys[5] = True
            if event.type == pg.KEYUP:
                if event.key == pg.K_w: keys[0] = False
                elif event.key == pg.K_a: keys[1] = False
                elif event.key == pg.K_s: keys[2] = False
                elif event.key == pg.K_d: keys[3] = False
                elif event.key == pg.K_SPACE: keys[4] = False
                elif event.key == pg.K_p: keys[5] = False
        if game_status == 0:
            game_status, game_counter, screen_shake_timer_temp = game_logic(game_counter, keys, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, bg_group)
            if screen_shake_timer_temp > 0:
                screen_shake_timer = screen_shake_timer_temp
            if game_status != 0:
                screen_shake_timer = FPS/4
        elif keys[5]:
            bg_group, bg, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, clock, fonts, game_status, game_counter, screen_shake_timer, screen_shake_pos = init_game()
            game_status = 0
        screen_shake_timer, screen_shake_pos = screen_shake_logic(screen_shake_timer, screen_shake_pos)
        draw_window(game_status, fonts, hero, boss, bullets, crystals, lasers, shockwaves, blackholes, trails, bg_group, screen_shake_pos)
    pg.quit()


if __name__ == "__main__":
    main()
