% Make DIC Data for Mat
x_range = linspace(0, 6, 20);
y_range = linspace(0, 4, 20);
[X, Y] = meshgrid(x_range, y_range);

T = 275 + 20 * X + 10 * Y;
U_x = X / 20;
U_y = 0 * Y;
E = 200. * ones(size(X));
save("simple_2D_dic2.mat",'E',"T","U_x", "U_y", "X", "Y", '-v7')