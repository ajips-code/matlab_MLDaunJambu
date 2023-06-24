% Membuat GUI untuk tampilan
fig = uifigure('Name', 'Klasifikasi Daun Jambu');
fig.Position(3:4) = [300 200];

% Membuat panel untuk memilih folder dataset
dataPanel = uipanel(fig, 'Title', 'Data Training');
dataPanel.Position = [20 100 260 80];

dataFolderLabel = uilabel(dataPanel, 'Position', [10 40 100 20], 'Text', 'Folder Dataset:');
dataFolderEdit = uieditfield(dataPanel, 'Position', [120 40 120 20], 'Value', 'dataset/', 'Editable', 'off');

dataFolderButton = uibutton(dataPanel, 'Position', [200 10 50 20], 'Text', 'Pilih', 'ButtonPushedFcn', @(btn,event) selectDataFolder());

% Membuat panel untuk memilih folder testing
testPanel = uipanel(fig, 'Title', 'Data Testing');
testPanel.Position = [20 20 260 70];

testFolderLabel = uilabel(testPanel, 'Position', [10 30 100 20], 'Text', 'Folder Testing:');
testFolderEdit = uieditfield(testPanel, 'Position', [120 30 120 20], 'Value', 'testing/', 'Editable', 'off');

testFolderButton = uibutton(testPanel, 'Position', [200 10 50 20], 'Text', 'Pilih', 'ButtonPushedFcn', @(btn,event) selectTestFolder());

% Membuat tombol untuk memulai klasifikasi
classifyButton = uibutton(fig, 'Position', [110 40 80 30], 'Text', 'Klasifikasi', 'ButtonPushedFcn', @(btn,event) classifyLeaves());

% Deklarasi variabel global
global dataFolderEdit testFolderEdit

% Fungsi untuk memilih folder dataset
function selectDataFolder()
    folder = uigetdir();
    dataFolderEdit.Value = folder;
end

% Fungsi untuk memilih folder testing
function selectTestFolder()
    folder = uigetdir();
    testFolderEdit.Value = folder;
end

% Fungsi untuk melakukan klasifikasi
function classifyLeaves()
    % Mendapatkan path folder dataset dan testing
    dataFolderPath = dataFolderEdit.Value;
    testFolderPath = testFolderEdit.Value;
    
    % Membaca data training
    dataFiles = dir(fullfile(dataFolderPath, '*.jpg'));
    numData = numel(dataFiles);
    features = zeros(numData, 8);
    labels = cell(numData, 1);
    
    for i = 1:numData
        imgPath = fullfile(dataFolderPath, dataFiles(i).name);
        img = imread(imgPath);
        
        % Ekstraksi fitur (misal: GLCM)
        % Implementasikan di sini sesuai dengan metode yang Anda gunakan
        
        % Menyimpan fitur dan label
        features(i, :) = extractedFeatures;
        labels{i} = 'daunjambubiji'; % Ganti dengan label yang sesuai
    end
    
    % Membaca data testing
    testFiles = dir(fullfile(testFolderPath, '*.jpg'));
    numTest = numel(testFiles);
    testFeatures = zeros(numTest, 8);
    
    for i = 1:numTest
        imgPath = fullfile(testFolderPath, testFiles(i).name);
        img = imread(imgPath);
        
        % Ekstraksi fitur (misal: GLCM)
        % Implementasikan di sini sesuai dengan metode yang Anda gunakan
        
        % Menyimpan fitur
        testFeatures(i, :) = extractedFeatures;
    end
    
    % Normalisasi fitur
    % Implementasikan di sini jika diperlukan
    
    % Melakukan klasifikasi menggunakan KNN
    k = 3; % Jumlah tetangga terdekat
    model = fitcknn(features, labels, 'NumNeighbors', k);
    predictedLabels = predict(model, testFeatures);
    
    % Menampilkan hasil klasifikasi
    resultText = uilabel(fig, 'Position', [20 150 260 30], 'Text', 'Hasil Klasifikasi:');
    resultEdit = uieditfield(fig, 'Position', [20 120 260 30], 'Value', predictedLabels{1}, 'Editable', 'off');
end
