from PIL import Image, ImageFilter, ImageEnhance

def clean_and_resize_image(image_path, output_path, target_size=(416, 416)):
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')
        cleaned_image = clean_and_enhance_image(image)
        resized_image = cleaned_image.resize(target_size)
        resized_image.save(output_path)
        print("Image cleaned and enhanced successfully!")
    except Exception as e:
        print("Error:", e)

def clean_and_enhance_image(image):
    cleaned_image = image.filter(ImageFilter.MedianFilter(size=3))
    enhancer = ImageEnhance.Sharpness(cleaned_image)
    cleaned_image = enhancer.enhance(2.0)  # enhancement factor
    enhancer = ImageEnhance.Contrast(cleaned_image)
    cleaned_image = enhancer.enhance(1.5)  # enhancement factor
    
    return cleaned_image