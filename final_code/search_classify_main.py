import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from search_classify_hlpr import *
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from falsePos_and_MultDet_filter import *
from scipy.ndimage.measurements import label


''' The following parameters are used to configure which code will be run. '''
relearnData = False  # retrain svc on data set
scv_model_name = 'scvModel_spatSz16_hogAll'
UseFullSet = True  # Only relevant when relearnData is True
procTestImages = False  # process the test images
viz_channels = False  # visualize some images for better understanding of different parameters.
viz_search_grid = False
procVideo = True  # process the video.
vidFileName = r'..\test_video_full.mp4'  # r'..\test_video_full.mp4'

pathToImages = r'C:\Users\ROEE\Google Drive\selfDrivingCourse\20_Object_detection\images'

if relearnData:
    # Read in cars and notcars
    if not UseFullSet:
        images = glob.glob(pathToImages + r'\smallset\*\*\*\*.jpeg', recursive=True)
        delimiterStr = r'smallset'
    else:
        images = glob.glob(pathToImages + r'\fullset\*\*\*\*.png', recursive=True)
        delimiterStr = r'fullset'

    cars = []
    notcars = []
    if not UseFullSet:
        for image in images:
            tmp=image.split(delimiterStr)[-1]
            if 'image' in tmp or 'extra' in tmp:
                notcars.append(image)
            else:
                cars.append(image)
    else:
        for image in images:
            if r'non-vehicles' in image:
                notcars.append(image)
            else:
                cars.append(image)

    # Reduce the sample size because
    sample_size = -1  # 2200
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    if viz_channels:  # visualize data
        # Visualize images:
        visualize_images_colormap(cars, 1)

        visualize_color_channels(cars, 'YCrCb', 4)
        visualize_color_channels(cars, 'HSV', 4)
        visualize_color_channels(cars, 'YUV', 4)

        colorTypes = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
        carNum=3
        feature_image = visualize_color_channels(cars, 'RGB', 1, vis=False, imgNum=carNum)
        plt.figure()
        plt.imshow(feature_image)
        plt.draw()
        plt.pause(0.001)
        for color_space in colorTypes:
            for chnl in range(3):
                feature_image = visualize_color_channels(cars, color_space, 1, vis=False, imgNum=carNum)
                features, hog_image = get_hog_features(feature_image[:, :, chnl], orient=9, pix_per_cell=8,
                                                       cell_per_block=2, vis=True, feature_vec=True)
                plt.figure()
                plt.imshow(hog_image, cmap='gray')
                plt.title('clr spc=' + color_space + ', chnl=' + str(chnl))
                plt.draw()
                plt.pause(0.001)

    # WRITEUP3: feature selection:
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    # WRITEUP1: Extract features from training set:
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)

    # WRITEUP2: Extract features from training set:
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Number of training example: ' + str(len(X_train)))
    print('Number of testing example: ' + str(len(X_test)))
    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    #svc = svm.SVC(C=10.0, kernel='rbf')
    svc = LinearSVC(C=5.0)
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    # # TEST DIFFERENT SVM'S - START
    # from sklearn import svm, datasets
    # from sklearn.model_selection import GridSearchCV
    # parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    # svr = svm.SVC()
    # clf = GridSearchCV(svr, parameters)
    # clf.fit(X_train, y_train)
    # clf.best_params_ # resulted in {'C': 10, 'kernel': 'rbf'}
    # # TEST DIFFERENT SVM'S - END
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    dictSVC = {'svc': svc, 'X_scaler': X_scaler, 'color_space': color_space, 'spatial_size': spatial_size,
               'hist_bins': hist_bins, 'orient': orient, 'pix_per_cell': pix_per_cell,
               'cell_per_block': cell_per_block, 'hog_channel': hog_channel, 'spatial_feat': spatial_feat,
               'hist_feat': hist_feat, 'hog_feat': hog_feat}

    with open(scv_model_name + '.p', 'wb') as fid:
        pickle.dump(dictSVC, fid, pickle.HIGHEST_PROTOCOL)

else:
    with open(scv_model_name + '.p', 'rb') as fid:
        dictSVC = pickle.load(fid)


# define the search area and grid in images:
windows = []
for winSz, y_start_stop in zip([128, 96, 64], [[400, 600], [400, 550], [400, 500]]):
    windows = windows + \
              slide_window((720, 1280, 3), x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(winSz, winSz), xy_overlap=(0.7, 0.7))

if viz_search_grid:
    # visuzlize search area and grid:
    from PIL import Image
    image = np.asarray(Image.open(r'..\..\ObjDet\test_images\test4.jpg'))
    window_img = draw_boxes(image, windows[1::4], color=None, thick=2)
    plt.figure()
    plt.imshow(window_img)
    plt.show()


if procTestImages:
    images = glob.glob(r'..\..\ObjDet\test_images\*.jpg', recursive=True)
    for imageName in images:
        image = np.asarray(Image.open(imageName))

        hot_windows = findBoxes(image, dictSVC, windows)
        window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)

        fig = plt.figure()
        plt.imshow(window_img)
        plt.draw()
        plt.pause(0.001)
        saveFileName = imageName.replace('test_images', 'output_images').split('.jpg')[0] + '_' + scv_model_name + '_allBBox.png'
        plt.savefig(saveFileName, dpi=130)
        # plt.show()

        # Multiple detections and false positive filter:

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        # Add heat to each box in hot_windows
        heat = add_heat(heat, hot_windows)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img, _ = draw_labeled_bboxes(np.copy(image), labels)

        plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.draw()
        plt.pause(0.001)
        saveFileName = imageName.replace('test_images', 'output_images').split('.jpg')[0] + '_heatMapFiltered.png'
        plt.savefig(saveFileName, dpi=130)
    plt.show()


if procVideo:
    import imageio
    vid = imageio.get_reader(vidFileName)
    metaData = vid.get_meta_data()
    L = metaData['nframes']
    print('Video file name:{}'.format(vidFileName))
    print('Video attributes:\nnum frames = {}'.format(L))
    print('source_size = {}'.format(metaData['source_size']))
    print('plugin = {}'.format(metaData['plugin']))
    print('fps = {}\n'.format(metaData['fps']))
    outFileName = vidFileName.replace('.mp4', '_' + scv_model_name + '_out.mp4')
    hot_windows_prev = []
    thresh = 1
    with imageio.get_writer(outFileName, fps=metaData['fps']) as writer:
        for i in range(L):
            im_i = vid.get_data(i)
            print('processing image ' + str(i) + ' of ' + str(L))
            hot_windows = findBoxes(im_i, dictSVC, windows)
            heat = np.zeros_like(im_i[:, :, 0]).astype(np.float)
            if len(hot_windows_prev) > 0:
                heat = add_heat(heat, hot_windows + hot_windows_prev)
                thresh = 2
            else:
                heat = add_heat(heat, hot_windows)
                # thresh = 1
            heat = apply_threshold(heat, thresh)
            heatmap = np.clip(heat, 0, 255)
            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            draw_img, bbox = draw_labeled_bboxes(im_i, labels)
            hot_windows_prev = hot_windows
            writer.append_data(draw_img)

