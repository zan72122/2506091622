import shutil, os
from textwrap import dedent

base = open('source.html').read()

variants = [
    ('ki67_v01_preprocess_gaussian.html', 'Preprocessing with Gaussian blur',
     'reduces noise-induced oversegmentation', 'Gaussian kernel 5x5'),
    ('ki67_v02_preprocess_median.html', 'Median blur preprocessing',
     'handles salt-and-pepper noise to merge fragments', 'Median kernel 5'),
    ('ki67_v03_preprocess_bilateral.html', 'Bilateral filter preprocessing',
     'preserves edges while smoothing', 'diameter 9, sigmaColor 75, sigmaSpace 75'),
    ('ki67_v04_preprocess_equalize.html', 'Histogram equalization on L channel',
     'enhances contrast for better thresholding', 'cv.equalizeHist'),
    ('ki67_v05_preprocess_unsharp.html', 'Unsharp masking after Gaussian blur',
     'sharpens boundaries to avoid small fragments', 'blur kernel 5x5, amount 1.5'),
    ('ki67_v06_preprocess_tophat.html', 'Morphological tophat preprocessing',
     'emphasizes nuclei over background', 'kernel 9x9'),
    ('ki67_v07_preprocess_gradient.html', 'Morphological gradient preprocessing',
     'highlights edges to merge weakly stained parts', 'kernel 3x3'),
    ('ki67_v08_preprocess_gauss_median.html', 'Gaussian then median filter',
     'two-step denoise for robust segmentation', 'gauss 5x5, median 3'),
    ('ki67_v09_preprocess_laplacian.html', 'Laplacian high-pass preprocessing',
     'accentuates edges, reducing split nuclei', 'kernel 3x3'),
    ('ki67_v10_preprocess_opening.html', 'Morphological opening preprocessing',
     'removes small noise before threshold', 'kernel 5x5'),

    ('ki67_v11_segmentation_otsu.html', 'Otsu threshold segmentation',
     'global threshold avoids local noise', 'cv.THRESH_OTSU'),
    ('ki67_v12_segmentation_adaptive_mean.html', 'Adaptive mean threshold (block 51)',
     'better local adaptation', 'blockSize 51, C 5'),
    ('ki67_v13_segmentation_adaptive_small.html', 'Adaptive gaussian small window',
     'preserves detail while smoothing', 'blockSize 41, C 7'),
    ('ki67_v14_segmentation_canny.html', 'Canny edge based segmentation',
     'edge detection to outline nuclei', 'threshold1 50, threshold2 150'),
    ('ki67_v15_segmentation_sobel.html', 'Sobel gradient magnitude segmentation',
     'uses gradient to detect nuclei', 'kernel 3x3'),
    ('ki67_v16_segmentation_laplacian.html', 'Laplacian segmentation',
     'zero-crossing style segmentation', 'kernel 3x3'),
    ('ki67_v17_segmentation_hsv.html', 'HSV color threshold segmentation',
     'uses hue to separate nuclei colors', 'H range 100-140'),
    ('ki67_v18_segmentation_kmeans.html', 'K-means color clustering segmentation',
     'clusters pixels into nuclei/background', 'k=2, attempts=5'),
    ('ki67_v19_segmentation_watershed.html', 'Watershed segmentation',
     'splits touching nuclei carefully', 'distance transform markers'),
    ('ki67_v20_segmentation_floodfill.html', 'Flood fill region growing',
     'expands from seeds to cover nuclei', 'seed step 10'),

    ('ki67_v21_morphology_rect.html', 'Rectangular kernel open->close',
     'different kernel shape may preserve nuclei', 'open 3x3 rect, close 5x5 rect'),
    ('ki67_v22_morphology_close_first.html', 'Close then open sequence',
     'merges fragments then removes noise', 'kernel 5x5'),
    ('ki67_v23_morphology_dilate_erode.html', 'Dilate then erode',
     'fills gaps before shrinking', 'kernel 3x3'),
    ('ki67_v24_morphology_gradient.html', 'Morphological gradient usage',
     'keeps edge emphasis for merging', 'kernel 3x3'),
    ('ki67_v25_morphology_tophat_blackhat.html', 'Tophat + blackhat preprocessing',
     'removes uneven illumination', 'kernel 9x9'),
    ('ki67_v26_morphology_close_twice.html', 'Closing twice with large kernel',
     'aggressively merges split nuclei', 'kernel 11x11 repeated'),
    ('ki67_v27_morphology_open_large.html', 'Opening with large kernel',
     'removes small debris', 'kernel 7x7'),
    ('ki67_v28_morphology_close_small_dilate.html', 'Close small then dilate',
     'connects fragments while expanding', 'close kernel 3x3, dilate 5x5'),
    ('ki67_v29_morphology_distance.html', 'Distance transform threshold',
     'uses distance map to refine segmentation', 'threshold 0.4'),
    ('ki67_v30_morphology_skeleton.html', 'Skeletonization-based cleanup',
     'reduces oversegmentation by merging skeletons', 'iterative erosion'),

    ('ki67_v31_color_hsv.html', 'HSV based classification',
     'better separation of brown/blue', 'H & S channels'),
    ('ki67_v32_color_ycrcb.html', 'YCrCb color space usage',
     'Cr channel distinguishes stains', 'Cr threshold'),
    ('ki67_v33_color_xyz.html', 'XYZ color space classification',
     'uses X/Z ratio for color', 'threshold on Z'),
    ('ki67_v34_color_luv.html', 'Luv color space classification',
     'u component for blue/brown', 'threshold on u'),
    ('ki67_v35_color_hls.html', 'HLS color space classification',
     'lightness helps with uneven staining', 'threshold on H & L'),
    ('ki67_v36_color_yuv.html', 'YUV color space classification',
     'U channel separates hues', 'threshold on U'),
    ('ki67_v37_color_combined.html', 'Combined Lab B and HSV S',
     'two-channel rule improves robustness', 'B>140 || S>90'),
    ('ki67_v38_color_lbp.html', 'Texture via Laplacian',
     'texture aids classification', 'Laplacian kernel 3x3'),
    ('ki67_v39_color_rgbdiff.html', 'RGB difference method',
     'simple difference between channels', 'if R-B>50'),
    ('ki67_v40_color_gray.html', 'Grayscale intensity only',
     'simplified intensity-based separation', 'threshold 128'),

    ('ki67_v41_hybrid_gauss_otsu_grad.html', 'Gaussian+Otsu+gradient hybrid',
     'combines smoothing with robust split', 'gauss 5x5'),
    ('ki67_v42_hybrid_bilat_hsv.html', 'Bilateral + HSV threshold',
     'smooth and color-based segmentation', 'diameter 9'),
    ('ki67_v43_hybrid_clahe_watershed.html', 'Equalize + watershed',
     'contrast enhancement then watershed', 'kernel 3x3'),
    ('ki67_v44_hybrid_median_sobel.html', 'Median + Sobel edges',
     'reduce noise then use gradients', 'kernel 5'),
    ('ki67_v45_hybrid_gauss_canny_merge.html', 'Gaussian + Canny + merge',
     'edge-based merging', 'threshold 50/150'),
    ('ki67_v46_hybrid_hist_open_grad.html', 'HistEq + open + gradient',
     'cleans background then emphasize edges', 'open 3x3'),
    ('ki67_v47_hybrid_unsharp_otsu.html', 'Unsharp masking + Otsu',
     'sharp boundaries then global threshold', 'amount 1.5'),
    ('ki67_v48_hybrid_tophat_kmeans.html', 'Tophat + k-means clustering',
     'background removal then color clustering', 'k=2'),
    ('ki67_v49_hybrid_gauss_color.html', 'Gaussian + color rules',
     'smooth then classify in HSV', 'gauss 5x5, S>100'),
    ('ki67_v50_hybrid_grad_texture.html', 'Gradient + texture merge',
     'uses edges and Laplacian texture', 'gradient 3x3')
]

for filename, desc, improvement, params in variants:
    content = base
    title_line = f'<title>{filename}</title>'
    content = content.replace('<title>Ki67 Cell Counter (v1 - Less Aggressive Close)</title>', title_line)
    content = content.replace('<h1>Ki67 Cell Counter (v1 - Less Aggressive Close)</h1>', f'<h1>{filename}</h1>')
    comment = f"<!--\nApproach: {desc}\nExpected improvement: {improvement}\nParameters: {params}\n-->"\

    content = comment + '\n' + content

    # Add simple modifications based on short keywords
    if 'gaussian' in filename and 'unsharp' not in filename:
        content = content.replace('let lab = new cv.Mat();', 'cv.GaussianBlur(src, src, new cv.Size(5,5), 1);\n  let lab = new cv.Mat();')
    if 'median' in filename:
        content = content.replace('let lab = new cv.Mat();', 'cv.medianBlur(src, src, 5);\n  let lab = new cv.Mat();')
    if 'bilat' in filename:
        content = content.replace('let lab = new cv.Mat();', 'cv.bilateralFilter(src, src, 9, 75, 75);\n  let lab = new cv.Mat();')
    if 'equalize' in filename or 'hist' in filename:
        content = content.replace('cv.split(lab, lChannel);', 'cv.split(lab, lChannel);\n  cv.equalizeHist(lChannel.get(0), lChannel.get(0));')
    if 'unsharp' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let blur = new cv.Mat();\n  cv.GaussianBlur(src, blur, new cv.Size(5,5), 1);\n  cv.addWeighted(src, 1.5, blur, -0.5, 0, src);\n  blur.delete();\n  let lab = new cv.Mat();')
    if 'tophat' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let tophatKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(9,9));\n  cv.morphologyEx(src, src, cv.MORPH_TOPHAT, tophatKernel);\n  tophatKernel.delete();\n  let lab = new cv.Mat();')
    if 'gradient' in filename and 'segmentation' not in filename and 'hybrid_grad_texture' not in filename:
        content = content.replace('let lab = new cv.Mat();', 'let gradKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3,3));\n  cv.morphologyEx(src, src, cv.MORPH_GRADIENT, gradKernel);\n  gradKernel.delete();\n  let lab = new cv.Mat();')
    if 'gauss_median' in filename:
        content = content.replace('let lab = new cv.Mat();', 'cv.GaussianBlur(src, src, new cv.Size(5,5), 1);\n  cv.medianBlur(src, src, 3);\n  let lab = new cv.Mat();')
    if 'laplacian' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let lap = new cv.Mat();\n  cv.Laplacian(src, lap, cv.CV_8U, 3, 1, 0);\n  cv.addWeighted(src, 1, lap, -1, 0, src);\n  lap.delete();\n  let lab = new cv.Mat();')
    if 'opening' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let openK = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5,5));\n  cv.morphologyEx(src, src, cv.MORPH_OPEN, openK);\n  openK.delete();\n  let lab = new cv.Mat();')

    # Segmentation modifications
    if 'segmentation_otsu' in filename:
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'cv.threshold(lChannel.get(0), nucleiMask, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU);')
    if 'segmentation_adaptive_mean' in filename:
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 51, 5);')
    if 'segmentation_adaptive_small' in filename:
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 41, 7);')
    if 'segmentation_canny' in filename:
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'cv.Canny(lChannel.get(0), nucleiMask, 50, 150);\n  cv.threshold(nucleiMask, nucleiMask, 0, 255, cv.THRESH_BINARY);')
    if 'segmentation_sobel' in filename:
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'let gradx = new cv.Mat();\n  let grady = new cv.Mat();\n  cv.Sobel(lChannel.get(0), gradx, cv.CV_16S, 1, 0);\n  cv.Sobel(lChannel.get(0), grady, cv.CV_16S, 0, 1);\n  cv.convertScaleAbs(gradx, gradx);\n  cv.convertScaleAbs(grady, grady);\n  cv.addWeighted(gradx, 0.5, grady, 0.5, 0, nucleiMask);\n  gradx.delete(); grady.delete();\n  cv.threshold(nucleiMask, nucleiMask, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU);')
    if 'segmentation_laplacian' in filename:
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'cv.Laplacian(lChannel.get(0), nucleiMask, cv.CV_16S, 3);\n  cv.convertScaleAbs(nucleiMask, nucleiMask);\n  cv.threshold(nucleiMask, nucleiMask, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU);')
    if 'segmentation_hsv' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let hsv = new cv.Mat();\n  cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);\n  let mask = new cv.Mat();\n  let low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [100,50,20,0]);\n  let high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [140,255,255,255]);\n  cv.inRange(hsv, low, high, mask);\n  hsv.delete(); low.delete(); high.delete();\n  let lab = new cv.Mat();\n  cv.cvtColor(src, lab, cv.COLOR_RGB2Lab);\n  nucleiMask = mask.clone();')
    if 'segmentation_kmeans' in filename:
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'let samples = src.reshape(1, src.rows*src.cols);\n  samples.convertTo(samples, cv.CV_32F);\n  let criteria = new cv.TermCriteria(cv.TermCriteria_EPS+cv.TermCriteria_MAX_ITER, 10, 1.0);\n  let labels = new cv.Mat();\n  let centers = new cv.Mat();\n  cv.kmeans(samples, 2, labels, criteria, 5, cv.KMEANS_RANDOM_CENTERS, centers);\n  nucleiMask = labels.reshape(1, src.rows);\n  nucleiMask.convertTo(nucleiMask, cv.CV_8U);\n  cv.threshold(nucleiMask, nucleiMask, 0, 255, cv.THRESH_BINARY_INV);\n  samples.delete(); labels.delete(); centers.delete();')
    if 'segmentation_watershed' in filename:
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'cv.threshold(lChannel.get(0), nucleiMask, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU);\n  let dist = new cv.Mat();\n  cv.distanceTransform(nucleiMask, dist, cv.DIST_L2, 3);\n  cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX);\n  let distThresh = new cv.Mat();\n  cv.threshold(dist, distThresh, 0.5, 1.0, cv.THRESH_BINARY);\n  distThresh.convertTo(distThresh, cv.CV_8U);\n  let markers = new cv.Mat();\n  cv.connectedComponents(distThresh, markers);\n  cv.watershed(src, markers);\n  nucleiMask = markers.convertTo(cv.CV_8U);\n  cv.threshold(nucleiMask, nucleiMask, 1, 255, cv.THRESH_BINARY);\n  dist.delete(); distThresh.delete(); markers.delete();')
    if 'segmentation_floodfill' in filename:
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'cv.threshold(lChannel.get(0), nucleiMask, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU);\n  let maskFlood = nucleiMask.clone();\n  for (let y=0; y<src.rows; y+=10){ for (let x=0; x<src.cols; x+=10){ if(maskFlood.ucharPtr(y,x)[0]==255){ cv.floodFill(maskFlood, new cv.Point(x,y), new cv.Scalar(128)); } } }\n  cv.threshold(maskFlood, nucleiMask, 127, 255, cv.THRESH_BINARY);\n  maskFlood.delete();')

    # Morphology modifications
    if 'morphology_rect' in filename:
        content = content.replace('const openKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));',
                                  'const openKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3,3));')
        content = content.replace('const defragKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(11, 11));',
                                  'const defragKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5,5));')
    if 'morphology_close_first' in filename:
        content = content.replace('cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_OPEN, openKernel);',
                                  'cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_CLOSE, openKernel);')
    if 'morphology_dilate_erode' in filename:
        content = content.replace('cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_OPEN, openKernel);',
                                  'cv.dilate(nucleiMask, nucleiMask, openKernel);\n  cv.erode(nucleiMask, nucleiMask, openKernel);')
    if 'morphology_gradient' in filename and 'preprocess' not in filename:
        content = content.replace('cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_CLOSE, defragKernel);',
                                  'cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_GRADIENT, defragKernel);')
    if 'morphology_tophat_blackhat' in filename:
        content = content.replace('cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_OPEN, openKernel);',
                                  'cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_TOPHAT, openKernel);\n  cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_BLACKHAT, openKernel);')
    if 'morphology_close_twice' in filename:
        content = content.replace('cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_CLOSE, defragKernel);',
                                  'cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_CLOSE, defragKernel);\n  cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_CLOSE, defragKernel);')
    if 'morphology_open_large' in filename:
        content = content.replace('const openKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));',
                                  'const openKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(7,7));')
    if 'morphology_close_small_dilate' in filename:
        content = content.replace('const defragKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(11, 11));',
                                  'const defragKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3,3));')
        content = content.replace('cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_CLOSE, defragKernel);',
                                  'cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_CLOSE, defragKernel);\n  cv.dilate(nucleiMask, nucleiMask, defragKernel);')
    if 'morphology_distance' in filename:
        content = content.replace('cv.findContours(nucleiMask, allContours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);',
                                  'let dist = new cv.Mat();\n  cv.distanceTransform(nucleiMask, dist, cv.DIST_L2, 3);\n  cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX);\n  cv.threshold(dist, nucleiMask, 0.4, 1.0, cv.THRESH_BINARY);\n  nucleiMask.convertTo(nucleiMask, cv.CV_8U);\n  cv.findContours(nucleiMask, allContours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);\n  dist.delete();')
    if 'morphology_skeleton' in filename:
        content = content.replace('cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_CLOSE, defragKernel);',
                                  'let skel = cv.Mat.zeros(nucleiMask.rows, nucleiMask.cols, cv.CV_8U);\n  let temp = new cv.Mat();\n  let element = cv.getStructuringElement(cv.MORPH_CROSS, new cv.Size(3,3));\n  do {\n    cv.morphologyEx(nucleiMask, temp, cv.MORPH_OPEN, element);\n    cv.bitwise_not(temp, temp);\n    cv.bitwise_and(nucleiMask, temp, temp);\n    cv.bitwise_or(skel, temp, skel);\n    cv.erode(nucleiMask, nucleiMask, element);\n  } while (cv.countNonZero(nucleiMask) !== 0);\n  nucleiMask = skel;')

    # Color modifications
    if 'color_hsv' in filename or 'hybrid_gauss_color' in filename or 'hybrid_bilat_hsv' in filename:
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'let hsv = new cv.Mat();\n    cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);\n    const meanHSV = cv.mean(hsv, tempMask);\n    hsv.delete();\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanHSV[1] > 100 || meanB > 140) {')
    if 'color_ycrcb' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let ycc = new cv.Mat();\n  cv.cvtColor(src, ycc, cv.COLOR_RGB2YCrCb);\n  let lab = new cv.Mat();\n  cv.cvtColor(src, lab, cv.COLOR_RGB2Lab);')
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'const meanYCC = cv.mean(ycc, tempMask);\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanYCC[1] > 150 || meanL < 100) {')
        content = content.replace('  [src, lab, nucleiMask, openKernel, defragKernel, allContours, hierarchy,', '  [src, lab, ycc, nucleiMask, openKernel, defragKernel, allContours, hierarchy,')
    if 'color_xyz' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let xyz = new cv.Mat();\n  cv.cvtColor(src, xyz, cv.COLOR_RGB2XYZ);\n  let lab = new cv.Mat();\n  cv.cvtColor(src, lab, cv.COLOR_RGB2Lab);')
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'const meanXYZ = cv.mean(xyz, tempMask);\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanXYZ[2] > 150 || meanL < 100) {')
        content = content.replace('  [src, lab, nucleiMask, openKernel, defragKernel, allContours, hierarchy,', '  [src, lab, xyz, nucleiMask, openKernel, defragKernel, allContours, hierarchy,')
    if 'color_luv' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let luv = new cv.Mat();\n  cv.cvtColor(src, luv, cv.COLOR_RGB2Luv);\n  let lab = new cv.Mat();\n  cv.cvtColor(src, lab, cv.COLOR_RGB2Lab);')
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'const meanLuv = cv.mean(luv, tempMask);\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanLuv[1] > 100 || meanB > 140) {')
        content = content.replace('  [src, lab, nucleiMask, openKernel, defragKernel, allContours, hierarchy,', '  [src, lab, luv, nucleiMask, openKernel, defragKernel, allContours, hierarchy,')
    if 'color_hls' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let hls = new cv.Mat();\n  cv.cvtColor(src, hls, cv.COLOR_RGB2HLS);\n  let lab = new cv.Mat();\n  cv.cvtColor(src, lab, cv.COLOR_RGB2Lab);')
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'const meanHLS = cv.mean(hls, tempMask);\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanHLS[0] > 100 || meanB > 140) {')
        content = content.replace('  [src, lab, nucleiMask, openKernel, defragKernel, allContours, hierarchy,', '  [src, lab, hls, nucleiMask, openKernel, defragKernel, allContours, hierarchy,')
    if 'color_yuv' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let yuv = new cv.Mat();\n  cv.cvtColor(src, yuv, cv.COLOR_RGB2YUV);\n  let lab = new cv.Mat();\n  cv.cvtColor(src, lab, cv.COLOR_RGB2Lab);')
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'const meanYUV = cv.mean(yuv, tempMask);\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanYUV[2] > 120 || meanB > 140) {')
        content = content.replace('  [src, lab, nucleiMask, openKernel, defragKernel, allContours, hierarchy,', '  [src, lab, yuv, nucleiMask, openKernel, defragKernel, allContours, hierarchy,')
    if 'color_combined' in filename:
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'let hsv = new cv.Mat();\n    cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);\n    const meanHSV = cv.mean(hsv, tempMask);\n    hsv.delete();\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanB > 140 || meanHSV[1] > 90) {')
    if 'color_lbp' in filename or 'hybrid_grad_texture' in filename:
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'let lap = new cv.Mat();\n    cv.Laplacian(src, lap, cv.CV_8U, 3);\n    const meanLap = cv.mean(lap, tempMask);\n    lap.delete();\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanLap[0] > 20 || meanB > 140) {')
    if 'color_rgbdiff' in filename:
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'const meanRGB = cv.mean(src, tempMask);\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanRGB[0]-meanRGB[2] > 50 || meanL < 100) {')
    if 'color_gray' in filename:
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanL < 128) {')
    if 'hybrid_gauss_otsu_grad' in filename:
        content = content.replace('let lab = new cv.Mat();', 'cv.GaussianBlur(src, src, new cv.Size(5,5), 1);\n  let lab = new cv.Mat();')
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'cv.threshold(lChannel.get(0), nucleiMask, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU);')
        content = content.replace('cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_CLOSE, defragKernel);',
                                  'cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_GRADIENT, defragKernel);')
    if 'hybrid_bilat_hsv' in filename:
        content = content.replace('let lab = new cv.Mat();', 'cv.bilateralFilter(src, src, 9, 75, 75);\n  let lab = new cv.Mat();')
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'let hsv = new cv.Mat();\n    cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);\n    const meanHSV = cv.mean(hsv, tempMask);\n    hsv.delete();\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanHSV[1] > 100 || meanB > 140) {')
    if 'hybrid_clahe_watershed' in filename:
        content = content.replace('cv.split(lab, lChannel);', 'cv.split(lab, lChannel);\n  cv.equalizeHist(lChannel.get(0), lChannel.get(0));')
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'cv.threshold(lChannel.get(0), nucleiMask, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU);\n  let dist = new cv.Mat();\n  cv.distanceTransform(nucleiMask, dist, cv.DIST_L2, 3);\n  cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX);\n  let distThresh = new cv.Mat();\n  cv.threshold(dist, distThresh, 0.5, 1.0, cv.THRESH_BINARY);\n  distThresh.convertTo(distThresh, cv.CV_8U);\n  let markers = new cv.Mat();\n  cv.connectedComponents(distThresh, markers);\n  cv.watershed(src, markers);\n  nucleiMask = markers.convertTo(cv.CV_8U);\n  cv.threshold(nucleiMask, nucleiMask, 1, 255, cv.THRESH_BINARY);\n  dist.delete(); distThresh.delete(); markers.delete();')
    if 'hybrid_median_sobel' in filename:
        content = content.replace('let lab = new cv.Mat();', 'cv.medianBlur(src, src, 5);\n  let lab = new cv.Mat();')
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'let gradx = new cv.Mat();\n  let grady = new cv.Mat();\n  cv.Sobel(lChannel.get(0), gradx, cv.CV_16S, 1, 0);\n  cv.Sobel(lChannel.get(0), grady, cv.CV_16S, 0, 1);\n  cv.convertScaleAbs(gradx, gradx);\n  cv.convertScaleAbs(grady, grady);\n  cv.addWeighted(gradx, 0.5, grady, 0.5, 0, nucleiMask);\n  gradx.delete(); grady.delete();\n  cv.threshold(nucleiMask, nucleiMask, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU);')
    if 'hybrid_gauss_canny_merge' in filename:
        content = content.replace('let lab = new cv.Mat();', 'cv.GaussianBlur(src, src, new cv.Size(5,5), 1);\n  let lab = new cv.Mat();')
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'cv.Canny(lChannel.get(0), nucleiMask, 50, 150);\n  cv.dilate(nucleiMask, nucleiMask, defragKernel);')
    if 'hybrid_hist_open_grad' in filename:
        content = content.replace('cv.split(lab, lChannel);', 'cv.split(lab, lChannel);\n  cv.equalizeHist(lChannel.get(0), lChannel.get(0));')
        content = content.replace('cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_CLOSE, defragKernel);',
                                  'cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_OPEN, defragKernel);\n  cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_GRADIENT, defragKernel);')
    if 'hybrid_unsharp_otsu' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let blur = new cv.Mat();\n  cv.GaussianBlur(src, blur, new cv.Size(5,5), 1);\n  cv.addWeighted(src, 1.5, blur, -0.5, 0, src);\n  blur.delete();\n  let lab = new cv.Mat();')
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'cv.threshold(lChannel.get(0), nucleiMask, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU);')
    if 'hybrid_tophat_kmeans' in filename:
        content = content.replace('let lab = new cv.Mat();', 'let tophatKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(9,9));\n  cv.morphologyEx(src, src, cv.MORPH_TOPHAT, tophatKernel);\n  tophatKernel.delete();\n  let lab = new cv.Mat();')
        content = content.replace('cv.adaptiveThreshold(lChannel.get(0), nucleiMask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 81, 7);',
                                  'let samples = src.reshape(1, src.rows*src.cols);\n  samples.convertTo(samples, cv.CV_32F);\n  let criteria = new cv.TermCriteria(cv.TermCriteria_EPS+cv.TermCriteria_MAX_ITER, 10, 1.0);\n  let labels = new cv.Mat();\n  let centers = new cv.Mat();\n  cv.kmeans(samples, 2, labels, criteria, 5, cv.KMEANS_RANDOM_CENTERS, centers);\n  nucleiMask = labels.reshape(1, src.rows);\n  nucleiMask.convertTo(nucleiMask, cv.CV_8U);\n  cv.threshold(nucleiMask, nucleiMask, 0, 255, cv.THRESH_BINARY_INV);\n  samples.delete(); labels.delete(); centers.delete();')
    if 'hybrid_gauss_color' in filename:
        content = content.replace('let lab = new cv.Mat();', 'cv.GaussianBlur(src, src, new cv.Size(5,5), 1);\n  let lab = new cv.Mat();')
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'let hsv = new cv.Mat();\n    cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);\n    const meanHSV = cv.mean(hsv, tempMask);\n    hsv.delete();\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanHSV[1] > 100 || meanB > 140) {')
    if 'hybrid_grad_texture' in filename:
        content = content.replace('cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_CLOSE, defragKernel);',
                                  'cv.morphologyEx(nucleiMask, nucleiMask, cv.MORPH_GRADIENT, defragKernel);')
        content = content.replace('const meanLab = cv.mean(lab, tempMask);', 'let lap = new cv.Mat();\n    cv.Laplacian(src, lap, cv.CV_8U, 3);\n    const meanLap = cv.mean(lap, tempMask);\n    lap.delete();\n    const meanLab = cv.mean(lab, tempMask);')
        content = content.replace('if (meanB > 140 || meanL < 100) {', 'if (meanLap[0] > 20 || meanB > 140) {')

    with open(filename, 'w') as f:
        f.write(content)
