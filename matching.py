import numpy as np
import pandas as pd
import re
import cv2
import matplotlib.pyplot as plt
import os
import os.path as op
import logging


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from preprocess import ImageSignature
from postprocess import confirm_found_match


def find_matches(image_photo, model, df_krant, krant_features, n_max):
    """
    Input
    :photo image-photo 3 channel 
    :df_krant dataframe with newspaper filenames and coordinates for extracted illustrations
    :krant_features list of features from VGG model ordered as in df_krant 
    the matching is exact one-by-one (for large database precompute nearest neighbours)
    :n_max the number of potential matches to preselect;
    each of n_max is then verified for homography transform to finde 'true' matches
    Return: result dictionary and image with matches to be saved if needed
    """
    show_size = (256, 256)
    obj = ImageSignature(image_photo)
    image_cnn = obj.cnn_image()
    features_photo = model.predict(image_cnn)[0]
    features_photo = features_photo/np.linalg.norm(features_photo) 

    # find the closest newspapers
    euclid_dist_norm = np.array([1-np.dot(features_photo, features_krant) for features_krant in krant_features])
    ids_min = np.argsort(euclid_dist_norm)[:n_max]

    img_match = cv2.resize(cv2.cvtColor(image_photo, cv2.COLOR_BGR2GRAY), show_size, interpolation=cv2.INTER_AREA)
    result = {}
    for i, id_min in enumerate(ids_min):
        idx = df_krant.index[id_min]
        file_match_krant = df_krant.loc[idx, 'filename']
        image = cv2.imread(file_match_krant, cv2.IMREAD_GRAYSCALE)
        xtl = df_krant.loc[idx, 'xtl']
        ytl = df_krant.loc[idx, 'ytl']
        ybr = df_krant.loc[idx, 'ybr']
        xbr = df_krant.loc[idx, 'xbr']
        image_krant = image[ytl:ybr, xtl:xbr] 
        # postprocessing: check if sift features correspond
        postprocess_match = confirm_found_match(cv2.cvtColor(image_photo, cv2.COLOR_BGR2GRAY), image_krant)

        score = np.exp(-euclid_dist_norm[id_min])
        result[i] = {
            'paper_id': df_krant.loc[idx, 'paper_id'], 'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr,
            'score': score.tolist(), 'match': bool(postprocess_match)
        }
        if postprocess_match:
            img_krant = cv2.resize(image_krant, show_size,  interpolation=cv2.INTER_AREA)
            img_match = np.hstack((img_match, img_krant))
    return result, img_match


def create_links(photo_list, df_krant, krant_features, config, output_dir):
    """
    :photo_list list of photos to match
    :df_krant dataframe with relevant newspaper inputs from xml-files
    :krant_features precomputed features per illustration in newspaper (same order as df_krant)
    :config configuration file
    output: csv-file with Nbest matches per photo and verified true/false
    """
    match_links_file = op.join(output_dir, "match_links.csv")
    if config['method'] == 'VGG16':
        base_model = VGG16(weights='imagenet')
        # to define the layer to pool features
        model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    else:
        raise ValueError("VGG16 is currently the only supported method")
    
    df = pd.DataFrame(columns=['photo_id', 'photo_subid', 'paper_id',  'xtl', 'ytl', 'xbr', 'ybr', 'score', 'match'])
    df.to_csv(match_links_file)
    n_total = len(photo_list)
    
    for (i, filename) in enumerate(photo_list):
        bname, _ = os.path.splitext(os.path.basename(filename))
        photo_basename = re.match(config["pattern_photoname"], bname).group(0)
        photo_subname = os.path.basename(filename).split(photo_basename)[1]
        logging.info(f'Processing photo {bname} counter {i+1}/{n_total}')

        # photos are extracted already
        try:
            image_photo = cv2.imread(filename)
        except:
            # TODO: Narrow this exception
            # Pass on any issues reading the image
            continue

        # compute N-best matches and true matches
        result, img_match = find_matches(image_photo, model, df_krant, krant_features, config['n_max'])

        true_match = []
        for j, item in result.items():
            item['photo_id'] = photo_basename
            item['photo_subid'] = photo_subname
            df = df.append(item, ignore_index=True)
            true_match.append(item['match'])

        # save result to disc for every photo
        df_tail = df.tail(config['n_max'])
        df_tail.to_csv(match_links_file, mode="a", header=False)

        # save matches as image
        if config['save_images'] and np.any(true_match):
            plt.figure()
            plt.imsave(os.path.join(output_dir, bname+'_match.jpg'), img_match, cmap='gray', dpi=100)

    return df
