% Membaca file input
data = csvread('data.csv');

% Memisahkan kolom X dan Y
X = data(:, 1);
Y = data(:, 2);

% Menghitung jumlah data
n = length(X);

% Menghitung rata-rata
mean_X = mean(X);
mean_Y = mean(Y);

% Menghitung koefisien regresi
numerator = sum((X - mean_X) .* (Y - mean_Y));
denominator = sum((X - mean_X) .^ 2);
b1 = numerator / denominator;
b0 = mean_Y - b1 * mean_X;

% Menampilkan koefisien regresi
disp(['Koefisien Regresi (b1): ', num2str(b1)]);
disp(['Intersepsi (b0): ', num2str(b0)]);

% Menampilkan persamaan regresi
disp(['Persamaan Regresi: Y = ', num2str(b0), ' + ', num2str(b1), ' * X']);

% Menampilkan hasil prediksi
X_new = 10; % Contoh nilai X yang baru
Y_new = b0 + b1 * X_new;
disp(['Prediksi Y untuk X = ', num2str(X_new), ': ', num2str(Y_new)]);
