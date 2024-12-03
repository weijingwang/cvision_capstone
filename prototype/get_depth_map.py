import ml_depth_pro.src.depth_pro as depth_pro


def get_dm(image_path):
    print("ayoo")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    image, _, f_px, = depth_pro.load_rgb(image_path)
    image = transform(image)

    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    print("it worked")

    return depth, focallength_px


get_dm("misc/cropped_car_street.png")