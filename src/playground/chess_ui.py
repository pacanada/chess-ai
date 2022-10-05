import pyglet
window = pyglet.window.Window(480,480)
#background image
background = pyglet.resource.image("resources/board.svg.png")
pieces_image = pyglet.resource.image("resources/pieces.svg.png", )
pieces_sprite = pyglet.sprite.Sprite(pieces_image, x= 200, y = 200)

@window.event
def on_draw():
    window.clear()
    background.blit(0,0)
    pieces_sprite.draw()

pyglet.app.run()