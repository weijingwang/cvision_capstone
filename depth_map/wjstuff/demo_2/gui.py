import pygame
import time
from my_constants import *


def quit_app():
    pygame.quit()
    exit()


def handle_gui_events(square_rect, last_click_time):
    # Initialize state variables
    button_is_pressed = False
    is_held = False
    is_double_clicked = False

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_app()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if square_rect.collidepoint(event.pos):
                current_time = time.time()
                if current_time - last_click_time <= DOUBLE_CLICK_THRESHOLD:
                    is_double_clicked = True
                last_click_time = current_time
                button_is_pressed = True
                is_held = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if square_rect.collidepoint(event.pos):
                button_is_pressed = False
                is_held = False

    return button_is_pressed, is_held, is_double_clicked


def render_gui(screen, square_rect, text_surface, text_rect, objects, button_is_pressed, is_double_clicked, clock):
    # Handle double click
    if is_double_clicked:
        print("LISTEN FOR VOICE INSTRUCTIONS")
        is_double_clicked = False

    # Handle button press
    if button_is_pressed:
        print(f"{objects}")

    # Render the screen
    color = DARK_GREEN if button_is_pressed else GREEN
    screen.fill(WHITE)  # Clear screen
    pygame.draw.rect(screen, color, square_rect)  # Draw green square
    screen.blit(text_surface, text_rect)  # Draw text on the screen
    pygame.display.flip()

    # Limit FPS to 60
    clock.tick(PYGAME_FPS)  # Returns the time passed since the last frame in milliseconds
    pygame_fps = clock.get_fps()  # Get the current frames per second

    # Update window title with FPS
    pygame.display.set_caption(f"Your phone - FPS: {pygame_fps:.3f}")