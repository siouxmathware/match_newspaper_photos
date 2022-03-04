from abc import ABCMeta, abstractmethod
from xml.etree import ElementTree as ElTree
import cv2
import numpy as np


class IllustrationExtractor(metaclass=ABCMeta):
    """
    Abstract base class for extracting relevant data from an XML file.
    """
    @classmethod
    def factory(cls, source):
        if source == 'ABBYY':
            return IllustrationExtractorABBYY()
        elif source == 'altoABBYY':
            return IllustrationExtractorAltoABBYY()

    @abstractmethod
    def extract(self, paper_file, xml_file):
        """
        extract illustrations from newspaper-page using xml_file
        """
        return NotImplementedError


class IllustrationExtractorAltoABBYY(IllustrationExtractor):
    def extract(self, paper_file, xml_file):
        # read page
        img = cv2.imread(paper_file)
        xtree = ElTree.parse(xml_file)
        xroot = xtree.getroot()
        attributes = [elem.attrib for elem in xroot.iter()]
        illustrations = []
        bboxes = []
        block_ids = []
        for attribute in attributes:
            if 'TYPE' in attribute.keys() and attribute['TYPE'] == 'Illustration':
                xtl = int(attribute['HPOS'])
                ytl = int(attribute['VPOS'])
                xbr = xtl + int(attribute['WIDTH'])
                ybr = ytl + int(attribute['HEIGHT'])
                img_crop = img[ytl:ybr, xtl:xbr, :]
                illustrations.append(np.array(img_crop))
                bboxes.append([xtl, ytl, xbr, ybr])
                block_ids.append(attribute['ID'])
        return illustrations, bboxes, block_ids


class IllustrationExtractorABBYY(IllustrationExtractor):
    def extract(self, paper_file, xml_file):
        # read page
        img = cv2.imread(paper_file)
        xtree = ElTree.parse(xml_file)
        xroot = xtree.getroot()
        attributes = [elem.attrib for elem in xroot.iter()]
        illustrations = []
        bboxes = []
        block_ids = []
        for attribute in attributes:
            if 'blockType' in attribute.keys() and attribute['blockType'] == 'Picture':
                xtl = int(attribute['l'])
                ytl = int(attribute['t'])
                xbr = int(attribute['r'])
                ybr = int(attribute['b'])
                img_crop = img[ytl:ybr, xtl:xbr, :]
                illustrations.append(np.array(img_crop))
                bboxes.append([xtl, ytl, xbr, ybr])
                block_ids.append('None')

        return illustrations, bboxes, block_ids
