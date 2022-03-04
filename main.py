import argparse
import json
import os
import time
import logging
import os.path as op

from extract_paper import create_df_newspaper
from matching import create_links
import filelist


def main(input_dir: str, output_dir: str):
    """
    Find match links between photo and newspapers 
    input:  folder with newspapers and xml-files, folder with extracted photos from strips
    output: dataframe with found matches and/or saved matches
    """
    logging.basicConfig(level=logging.INFO)
    
    # INPUT
    with open(op.join(input_dir, "config.json")) as config_file:
        config = json.load(config_file)
    paper_filenames, xml_filenames, photo_list = filelist.filelist_frompath(input_dir, config)
    logging.info(f'configuration: {config}')
    logging.info(f'Photos to match: {len(photo_list)}, with newspapers pages: {len(paper_filenames)}')

    # PREPARE OUTPUT
    os.makedirs(output_dir, exist_ok=True)

    # DATABASE NEWSPAPERS
    time0 = time.time()
    df_krant, krant_features = create_df_newspaper(paper_filenames, xml_filenames, config)
    logging.info(f'Time to compute features: {(time.time() - time0):.3f} seconds for {len(paper_filenames)} newspapers')
    if config['save_features']:
        df_krant.to_csv(op.join(output_dir, "newspaper_record.csv"))

    # FIND MATCHES
    time0 = time.time()
    create_links(photo_list, df_krant, krant_features, config, output_dir)
    logging.info(f'Total time to compute matches: {(time.time() - time0)/len(photo_list):.3f} per photo')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=op.join("tests", "data", "GrArchief"))
    parser.add_argument("--output_dir", type=str, default="output")
    args, unknown = parser.parse_known_args()
    argument_dict = vars(args)
    main(**argument_dict)
