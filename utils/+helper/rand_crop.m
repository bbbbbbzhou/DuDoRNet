function img = rand_crop(img, crop_size)
    image_size = size(img);
    px = randi([1, image_size(1) - crop_size(1) + 1]);
    py = randi([1, image_size(2) - crop_size(2) + 1]);

    img = img(px:px+crop_size(1) - 1, py:py+crop_size(2) - 1, :);
end