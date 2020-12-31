import engine3D
import pygame

shape = engine3D.Obj.loadObj("cube.obj")
cam = engine3D.Camera()
screen = pygame.display.set_mode((1000, 1000))
renderer = engine3D.Render(screen)
clock = pygame.time.Clock()
running = True

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    elapsedTime = 1 / (clock.get_fps() if clock.get_fps() else 1)

    forward = cam.dir * (8 * elapsedTime)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        cam.pos.y += 8 * elapsedTime
    if keys[pygame.K_DOWN]:
        cam.pos.y -= 8 * elapsedTime
    if keys[pygame.K_w]:
        cam.pos = cam.pos + forward
    if keys[pygame.K_s]:
        cam.pos = cam.pos - forward
    if keys[pygame.K_a]:
        cam.yaw = cam.yaw + 2 * elapsedTime
    if keys[pygame.K_d]:
        cam.yaw = cam.yaw - 2 * elapsedTime

    screen.fill((0, 0, 0))
    renderer.render(shape, camera=cam, offset=0, wireframe=False, clippingColors=False)

    pygame.display.update()
    pygame.display.set_caption(str(clock.get_fps()))
    clock.tick(30)
