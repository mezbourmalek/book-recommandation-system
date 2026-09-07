from PIL import Image, ImageDraw, ImageFont

WIDTH = 800
HEIGHT = 1290

img = Image.new("RGB", (WIDTH, HEIGHT), "#A13502")
draw = ImageDraw.Draw(img)


margin = 60

draw.rectangle(
    [margin, margin, WIDTH - margin, HEIGHT - margin],
    outline="#9C7907",
    width=3
)

# Windows Arial Bold
font = ImageFont.truetype(
    r"C:\Windows\Fonts\arialbd.ttf",
    75
)

lines = ["Cover", "not", "found"]

y = 300

for text in lines:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]

    x = (WIDTH - text_width) // 2

    draw.text(
        (x, y),
        text,
        fill="#F8D563",
        font=font
    )

    y += 120

img.save("cover-not-found.jpg", quality=95)

print("Cover created successfully!")