function varargout = Classification_Leaf_KNN(varargin)
% CLASSIFICATION_LEAF_KNN MATLAB code for Classification_Leaf_KNN.fig
%      CLASSIFICATION_LEAF_KNN, by itself, creates a new CLASSIFICATION_LEAF_KNN or raises the existing
%      singleton*.
%
%      H = CLASSIFICATION_LEAF_KNN returns the handle to a new CLASSIFICATION_LEAF_KNN or the handle to
%      the existing singleton*.
%
%      CLASSIFICATION_LEAF_KNN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CLASSIFICATION_LEAF_KNN.M with the given input arguments.
%
%      CLASSIFICATION_LEAF_KNN('Property','Value',...) creates a new CLASSIFICATION_LEAF_KNN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Classification_Leaf_KNN_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Classification_Leaf_KNN_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Classification_Leaf_KNN

% Last Modified by GUIDE v2.5 25-Jun-2023 01:57:47

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Classification_Leaf_KNN_OpeningFcn, ...
                   'gui_OutputFcn',  @Classification_Leaf_KNN_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Classification_Leaf_KNN is made visible.
function Classification_Leaf_KNN_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Classification_Leaf_KNN (see VARARGIN)

% Choose default command line output for Classification_Leaf_KNN
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Classification_Leaf_KNN wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Classification_Leaf_KNN_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
