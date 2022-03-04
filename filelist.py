import numpy as np
import os
import glob


def filelist_frompath(input_dir, config):
    """
    assume that the ordered list of newspapers images and xml files, checked later
    """
    path_newspapers = os.path.join(input_dir, f'newspaper/{config["newspaper_pattern"]}')
    path_xml = os.path.join(input_dir, f'newspaper/{config["xml_pattern"]}')
    path_photos = os.path.join(input_dir, f'photo/{config["photos_pattern"]}')

    paper_filenames = np.sort(glob.glob(path_newspapers, recursive=True))
    xml_filenames = np.sort(glob.glob(path_xml, recursive=True))
    assert len(paper_filenames) == len(xml_filenames), 'Number of papers and corresponding xml-files differ'
    photo_list = np.sort(glob.glob(path_photos, recursive=True))
    assert len(paper_filenames) > 0, "No papers found"
    assert len(photo_list) > 0, "No photos found"
    return paper_filenames, xml_filenames, photo_list
