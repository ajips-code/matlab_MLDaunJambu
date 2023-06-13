classdef LeafClassifierApp < matlab.apps.AppBase
    
    properties (Access = private)
        UIFigure               matlab.ui.Figure
        RunButton              matlab.ui.control.Button
        ResultLabel            matlab.ui.control.Label
        FileEditFieldLabel     matlab.ui.control.Label
        FileEditField          matlab.ui.control.EditField
        BrowseButton           matlab.ui.control.Button
    end
    
    methods (Access = private)
        
        function classifyLeaves(app)
            % Read the dataset
            file = 'features.xlsx';
            dataset = readtable(file);
            glcm_properties = {'dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy'};
            
            % Get the features and labels from the dataset
            fitur = dataset{:, 2:end-1};
            kelas = dataset{:, end};
            
            % Read and preprocess the test image
            file_name = app.FileEditField.Value;
            src = imread(file_name);
            tmp = rgb2gray(src);
            mask = tmp > 127;
            mask = imdilate(mask, strel('disk', 10));
            mask = imerode(mask, strel('disk', 10));
            cropped = src .* uint8(mask);
            gray = rgb2gray(cropped);
            
            % HSV
            hsv_image = rgb2hsv(cropped);
            dom_color = impixel(hsv_image, 1, 1);
            
            % GLCM
            glcm = graycomatrix(gray, 'Offset', [0 1], 'Symmetric', true);
            glcm_props = greycoprops(glcm, glcm_properties);
            
            % Shape
            mask_label = logical(mask);
            props = regionprops(mask_label);
            eccentricity = props.Eccentricity;
            area = props.Area;
            perimeter = props.Perimeter;
            metric = (4 * pi * area) / (perimeter * perimeter);
            
            % Combine the test features
            tes_fitur = [dom_color glcm_props metric eccentricity];
            
            % Scale the features
            scaler = StandardScaler();
            scaler.fit(fitur);
            fitur = scaler.transform(fitur);
            tes_fitur = scaler.transform(tes_fitur);
            
            % Train the classifier
            classifier = fitcknn(fitur, kelas, 'NumNeighbors', 13);
            
            % Predict the class
            kelas_pred = predict(classifier, tes_fitur);
            
            % Display the result
            app.ResultLabel.Text = ['Class: ' kelas_pred];
        end
        
        function browseFile(app, ~)
            [file, path] = uigetfile({'*.jpg;*.jpeg;*.png', 'Image Files (*.jpg, *.jpeg, *.png)'}, 'Select an image file');
            if isequal(file, 0)
                return;
            end
            app.FileEditField.Value = fullfile(path, file);
        end
        
        function createUI(app)
            % Create the UI components
            app.UIFigure = uifigure('Name', 'Leaf Classifier', 'Position', [100 100 300 150]);
            app.RunButton = uibutton(app.UIFigure, 'push', 'Text', 'Run', 'Position', [120 20 60 22], 'ButtonPushedFcn', @(~,~)classifyLeaves(app));
            app.ResultLabel = uilabel(app.UIFigure, 'Position', [30 80 240 22]);
            app.FileEditFieldLabel = uilabel(app.UIFigure, 'Position', [30 50 80 22], 'Text', 'File:');
            app.FileEditField = uieditfield(app.UIFigure, 'Position', [120 50 150 22]);
            app.BrowseButton = uibutton(app.UIFigure, 'push', 'Text', 'Browse', 'Position', [270 50 60 22], 'ButtonPushedFcn', @(~,~)browseFile(app));
        end
        
    end
    
    methods (Access = 'public')
        
        function app = LeafClassifierApp
            createUI(app);
        end
        
        function run(app)
            % Display the UI
            app.UIFigure.Visible = 'on';
        end
        
    end
    
    methods (Access = 'public', Static)
        
        function main
            app = LeafClassifierApp;
            app.run;
        end
        
    end
    
end
