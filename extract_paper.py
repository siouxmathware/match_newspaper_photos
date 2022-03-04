import numpy as np
import pandas as pd
import re
import os
import logging
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

from illustration_extractor import IllustrationExtractor
from preprocess import ImageSignature


def compute_features(image, model):
    """
    input: 3-channel image
    output: cnn-features for a chosen method
    """
    obj = ImageSignature(image)
    img_cnn = obj.cnn_image()
    features = model.predict(img_cnn)[0]
    f_abs = np.linalg.norm(features)
    flist = [(f/f_abs).tolist() for f in features]
    return flist


def create_df_newspaper(paper_filenames, xml_filenames, config):
    """
    input: ordered list of newspaper and xml-files
         min_size: in px min dimension of illustration to be saved
         border_cut: if newspapers are scanned with tilt -> remove white border around non-rectangle
         method: the choice to construct features
    output: dataframe with fields: 
            - paper_id
            - block_id
            - bbox coordinates
            - features
    """
    
    if config['method'] == 'VGG16':
        base_model = VGG16(weights='imagenet')
        # to define the layer to pool features
        model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    else:
        raise ValueError(f"Unknown method: {config['method']}")

    illustration_extrator = IllustrationExtractor.factory(config["xml_source"])

    df = pd.DataFrame(
        columns=['filename', 'paper_id', 'paper_counter', 'block_id', 'xtl', 'ytl', 'xbr', 'ybr', 'features_json']
    )
    features_krant = []
    
    for (i, (paper_file, xml_file)) in enumerate(zip(paper_filenames, xml_filenames)): 
        # check that paper and  xml-files agree 
        basename_paper = re.match(config["pattern_papername"], os.path.basename(paper_file)).group(0)
        basename_xml = re.match(config["pattern_papername"], os.path.basename(xml_file)).group(0)
        assert basename_paper == basename_xml, 'Filenames paper-xml disagree'
        logging.info(f'Processing newspaper {basename_paper}')
        # 3-channel illustration
        illustrations, bboxes, block_ids = illustration_extrator.extract(paper_file, xml_file)

        counter = 0
        for (img, bbox, id_block) in zip(illustrations, bboxes, block_ids):
            if min(img.shape[:2]) > config['min_size']:
                counter += 1
                json_name = basename_paper+'_'+str(counter)+'.json'
                ser = {
                    'filename': paper_file, 'paper_id': basename_paper, 'paper_counter': str(i), 'block_id': id_block,
                    'xtl': bbox[0], 'ytl': bbox[1], 'xbr': bbox[2], 'ybr': bbox[3], 'features_json': json_name
                }

                # compensate for scan tilt cut border
                border_cut = config['border_cut']
                img_cut = img[border_cut:-border_cut, border_cut:-border_cut, :]
                features = compute_features(img_cut, model)
                
                features_krant.append(features)
                df = df.append(ser, ignore_index=True)
    return df, features_krant
