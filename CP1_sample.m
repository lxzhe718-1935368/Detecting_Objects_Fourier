clear all; close all; clc

load Kraken.mat

L = 10; % spatial domain
n = 64; % Fourier modes
threshold = 0.6; % Change this threshold to see what happens to the plot

x2 = linspace(-L, L, n+1); x = x2(1:n); y = x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) - n/2: -1]; ks = fftshift(k);

[X, Y, Z] = meshgrid(x, y, z);
[Kx, Ky, Kz] = meshgrid(ks, ks, ks);

for j = 1:49
    Un(:, :, :) = reshape(Kraken(:, j), n, n, n); % We need to reshape our data into a tensor, which represents a cube of Fourier modes in x-y-z space
    M = max(abs(Un), [], 'all');
    close all, isosurface(X, Y, Z, abs(Un)/M, threshold)
    axis([min(x) max(x) min(y) max(y) min(z) max(z)]), grid on, drawnow
    pause % You have to press any key to get the next plot
end