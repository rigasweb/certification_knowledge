import requests
import pandas as pd
import requests
from collections import Counter
from nltk.stem.porter import PorterStemmer
from iteration_utilities import duplicates

class CertificateMatcher:
    """
    matches certifications through the escoe taxonomy
    """

    def __init__(self, dataset, escoe_knowledge, classes):
        self.dataset = dataset
        self.escoe_knowledge = escoe_knowledge
        self.classes = classes


    def load_data(self):
        """
        read data and keep only desired columns

        :return: <DataFrame> the load dataset
        """
        certifications = pd.read_csv('certifications_final.csv', index_col=0)
        certifications.drop(certifications.columns.difference(['cert_name','isco3']),axis=1,inplace=True)
        certifications = certifications[certifications['isco3'].isin(self.classes)]
        certifications.reset_index(drop=True, inplace=True)

        return certifications


    def wextract(text):
        """        
        extract skills from a given text (certification title/description) using WEX

        :param text: <str> text to be analyzed
        :return: <DataFrame> returns all the extracted skills by WEX
        """
        response = requests.post('https://sortinghat-load-balancer-dev.bluemix.net/skills/parse/text', data={"text": text})

        extracted_skills = pd.json_normalize(response.json(),record_path=['skills'])
        extracted_skills.drop(extracted_skills.columns.difference(['text', 'confidence']), axis=1, inplace = True)
        extracted_skills.rename(columns={'text':'skill'}, inplace=True)
        return extracted_skills

        
    def match_skill(self, skill):
        """
        search for skills in escoe taxonomy and retrieve the corresponding occupation code

        :param skill: <str> skill to be searched
        :return: <list> returns all the extracted occupation codes
        """
        skills2escoe = pd.read_csv(self.escoe_knowledge, index_col=0)

        skill = skill.lower()
        if skills2escoe[skills2escoe['skill']==skill]['codes'].empty:
            return 0
        else:
            x = skills2escoe[skills2escoe['skill']==skill]['codes'].values[0]
            return x


    def string_match(string1, string2):
        """
        compares two given strings and raises a flag if the 
        similarity is above a predefined threshold
        
        :param string1: <str> string 1 to be matched
        :param strign2: <str> string 2 to be matched
        :return: <int> returns the how many simalar words exist between the two strings
        """
        stemmer = PorterStemmer()
        matching_score = 0

        tokenized_text1 = string1.split(" ")
        tokenized_text2 = string2.split(" ")
        stemmed_text1 = [stemmer.stem(token) for token in tokenized_text1]
        stemmed_text2 = [stemmer.stem(token) for token in tokenized_text2]               
        
        counter1 = Counter(stemmed_text1)      #count occurrences of each n-gram
        counter2 = Counter(stemmed_text2)
        
        s = [k for k,v in counter1.items() if counter2[k]>0] # find occurances of the words between the two strings
        
        if len(s)>0: 
            matching_score = len(s) / len(stemmed_text2)  # calculate the score between skills and normalise it depending on the skill length
        return matching_score


if __name__ =='__main__':
    classes = [251, 243, 214, 522, 242,334]
    matcher = CertificateMatcher('data/certifications_final.csv','data/escoe-knowledge_final.csv',classes)
    certifications = matcher.load_data()

    df = pd.read_csv('data/escoe-knowledge_final.csv', index_col=0)
    skills = []
    occupation_codes = []
    confidence = []

    for i in range(len(certifications)):
        codes = []
        cert_title =  certifications['cert_name'][i]
        extracted_skills = matcher.wextract(cert_title)

        if extracted_skills.empty: continue

        for extracted_skill in extracted_skills.skill:
            skill_codes = []
            for taxonomy_skill in range(len(df)):
                if matcher.string_match(df['skill'][taxonomy_skill], extracted_skill)>0 : 
                    codes_as_string = df['codes'][taxonomy_skill]        # codes are stored as lists of strings
                    codes_as_string = codes_as_string.replace("[","")   
                    codes_as_string = codes_as_string.replace("]","")
                    codes_as_string = codes_as_string.split()
                    for code in codes_as_string:
                        if int(code) in classes and int(code) not in skill_codes:               # check whether code belongs at the desired classes 
                            skill_codes.append(int(code))
                            skills.append(df['skill'][taxonomy_skill])
            codes.append(skill_codes)

        codes = [item for sublist in codes for item in sublist]         # flatten the list of codes
        if list(duplicates(codes)):                                     # if there are duplicates, it means more than one skill were matched to the same code
            codes = list(duplicates(codes))
            conf = 1
        elif len(codes)<4 and len(codes)>0: 
            conf = 0.5
        else:
            conf = 0
            codes = []
        
        occupation_codes.append(codes)
        confidence.append(conf)
