function coefficient = convert2coefficient(hu)
    % assume miu_water = 0.0192, miu_air = 0.0
    coefficient = hu * 0.0192 / 1000 + 0.0192;
end
