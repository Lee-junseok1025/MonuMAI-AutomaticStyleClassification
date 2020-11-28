import numpy as np

class Monument:
    ELEMENT_DIC = {'Bird': ['Torso', 'Tail', 'Neck', 'Eye', 'Leg', 'Beak', 'Animal_Wing', 'Head'], 'Aeroplane': ['Stern', 'Engine', 'Wheel', 'Artifact_Wing', 'Body'], 'Cat': ['Ear'], 'Dog': ['Muzzle', 'Nose'], 'Sheep': ['Horn'], 'Train': ['Locomotive', 'Coach', 'Headlight'], 'Bicycle': ['Chain_Wheel', 'Saddle', 'Handlebar'], 'Horse': ['Hoof'], 'Bottle': ['Cap'], 'Person': ['Ebrow', 'Foot', 'Arm', 'Hair', 'Hand', 'Mouth'], 'Car': ['License_plate', 'Door', 'Bodywork', 'Mirror', 'Window'], 'diningtable': ['diningtable'], 'Pottedplant': ['Pot', 'Plant'], 'Motorbike': [], 'Sofa': ['Sofa'], 'Boat': ['Boat'], 'Cow': [], 'Chair': ['Chair'], 'Bus': [], 'Tvmonitor': ['Screen','Tvmonitor']}
    TRUE_ELEMENT_DIC = {'Bird': ['Torso', 'Tail', 'Neck', 'Eye', 'Leg', 'Beak', 'Animal_Wing', 'Head'],
        'Aeroplane': ['Stern', 'Engine', 'Wheel', 'Artifact_Wing', 'Body'],
        'Cat': ['Torso', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'],
        'Dog': ['Torso', 'Muzzle', 'Nose', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'],
        'Sheep': ['Torso', 'Tail', 'Muzzle', 'Neck', 'Eye', 'Horn', 'Leg', 'Ear', 'Head'],
        'Train': ['Locomotive', 'Coach', 'Headlight'],
        'Bicycle': ['Chain_Wheel', 'Saddle', 'Wheel', 'Handlebar'],
        'Horse': ['Hoof', 'Torso', 'Muzzle', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'], 
        'Bottle': ['Cap', 'Body'], 
        'Person': ['Ebrow', 'Foot', 'Arm', 'Torso', 'Nose', 'Hair', 'Hand', 'Neck', 'Eye', 'Leg', 'Ear', 'Head', 'Mouth'], 
        'Car': ['License_plate', 'Door', 'Wheel', 'Headlight', 'Bodywork', 'Mirror', 'Window'], 
        'diningtable': ['diningtable'], 
        'Pottedplant': ['Pot', 'Plant'], 
        'Motorbike': ['Wheel', 'Headlight', 'Saddle', 'Handlebar'], 
        'Sofa': ['Sofa'], 
        'Boat': ['Boat'], 
        'Cow': ['Torso', 'Muzzle', 'Tail', 'Horn', 'Eye', 'Neck', 'Leg', 'Ear', 'Head'], 
        'Chair': ['Chair'], 
        'Bus': ['License_plate', 'Door', 'Wheel', 'Headlight', 'Bodywork', 'Mirror', 'Window'], 
        'Tvmonitor': ['Screen','Tvmonitor']}

    XML_ELEMENT_DIC = {'Bird': ['Torso', 'Tail', 'Neck', 'Eye', 'Leg', 'Beak', 'Animal_Wing', 'Head'],
        'Aeroplane': ['Stern', 'Engine', 'Wheel', 'Artifact_Wing', 'Body'],
        'Cat': ['Torso', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'],
        'Dog': ['Torso', 'Muzzle', 'Nose', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'],
        'Sheep': ['Torso', 'Tail', 'Muzzle', 'Neck', 'Eye', 'Horn', 'Leg', 'Ear', 'Head'],
        'Train': ['Locomotive', 'Coach', 'Headlight'],
        'Bicycle': ['Chain_Wheel', 'Saddle', 'Wheel', 'Handlebar'],
        'Horse': ['Hoof', 'Torso', 'Muzzle', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'], 
        'Bottle': ['Cap', 'Body'], 
        'Person': ['Ebrow', 'Foot', 'Arm', 'Torso', 'Nose', 'Hair', 'Hand', 'Neck', 'Eye', 'Leg', 'Ear', 'Head', 'Mouth'], 
        'Car': ['License_plate', 'Door', 'Wheel', 'Headlight', 'Bodywork', 'Mirror', 'Window'], 
        'diningtable': ['diningtable'], 
        'Pottedplant': ['Pot', 'Plant'], 
        'Motorbike': ['Wheel', 'Headlight', 'Saddle', 'Handlebar'], 
        'Sofa': ['Sofa'], 
        'Boat': ['Boat'], 
        'Cow': ['Torso', 'Muzzle', 'Tail', 'Horn', 'Eye', 'Neck', 'Leg', 'Ear', 'Head'], 
        'Chair': ['Chair'], 
        'Bus': ['License_plate', 'Door', 'Wheel', 'Headlight', 'Bodywork', 'Mirror', 'Window'], 
        'Tvmonitor': ['Screen','Tvmonitor']}
    # Hot-one encode order for styles

    STYLES_HOTONE_ENCODE = list(ELEMENT_DIC.keys())    # M(Hispanic-muslim)=0, G(Gothic)=1, R(Renaissance)=2, B(Baroque)=3

    def __init__(self, metadata):
        self.__elements = {}        # dictipnary to group objects by elements
        self.__aggregation = {}
        self.__metadata = metadata
        self.__upload_metadata()

    # load metadata, import and order elements into a dictionary
    def __upload_metadata(self):
        self.__metadata.load_metadata()     # load metadata into the object
        styles = self.ELEMENT_DIC.keys()
        for stl in styles:      # each style
            self.__elements[stl] = {}      # insert style key
            elems = self.ELEMENT_DIC[stl]
            for e in elems:         # each element
                scores_e = self.__get_element_scores(e)     # scores of the element
                self.__elements[stl][e] = scores_e          # save scores in dictionary style-elements

    def __get_element_scores(self, element):
        # get index of element and return their scores
        elem_indx = [i for i,x in enumerate(self.__metadata.object_classes) if x==element]      # indexes of element
        scores = np.asarray(self.__metadata.object_scores, dtype=np.float)
        return scores[elem_indx]

    # return array contains the score aggregation
    def aggregation_score_sum(self):
        self.__aggregate_scores_sum()       # perform the element aggregation
        aggregation = np.array([])      # vector to store aggregated scores
        # insert in an array each element aggregated score
        for stl in self.__aggregation.keys():
            for e in self.__aggregation[stl]:
                aggregation = np.append(aggregation, float(self.__aggregation[stl][e]))
        return aggregation

    # generate the aggregation dictionary of elements with the score sum
    def __aggregate_scores_sum(self):
        styles = self.__elements.keys()
        for stl in styles:      # each style
            self.__aggregation[stl] = {}  # insert style key
            elems = self.__elements[stl]
            for e in elems:         # each element
                tot_score = np.sum(self.__elements[stl][e])     # aggregate scores
                self.__aggregation[stl][e] = tot_score