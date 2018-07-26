from PIL import Image

im = Image.open('test_img.jpg')

im = im.resize((250,250))
im.save('test_img.jpg')
