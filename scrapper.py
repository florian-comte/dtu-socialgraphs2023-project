import requests
import os, re
from datetime import datetime
from bs4 import BeautifulSoup
import pdfkit
import json
import random

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Scraper:
    def __init__(self):
        self.base_url = "https://www.nature.com"
        self.universities = {}
        self.collabs = {}
        self.mapping_univs = {}
        self.links_saved = []
    
    def load_collabs(self, filename):
        with open(filename, 'r') as file:
            self.collabs = json.load(file)

    def load_universities(self, filename):
        with open(filename, 'r') as file:
            self.universities = json.load(file)

    def save_universities_in_file(self):
        with open('universities.json', 'w') as fp:
            json.dump(self.universities, fp)

    def save_collabs_in_file(self):
        with open('collabs.json', 'w') as fp:
            json.dump(self.collabs, fp)
    
    def save_links_in_file(self):
        with open('links.json', 'w') as fp:
            json.dump(self.links_saved, fp)

    def retrieve_url(self, url: str) -> BeautifulSoup:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup

    def create_map(self):
        for new_name in self.universities.keys():        
            for old_name in self.universities[new_name]['init_names']:
                if old_name not in self.mapping_univs:
                    self.mapping_univs[old_name] = new_name

    def extract_universities(self):
        articles_counter = 1
        page = 1

        while True:
            search_url = f'https://www.nature.com/search?journal=srep&article_type=research&subject=mathematics-and-computing&date_range=last_year&order=relevance&page={page}'
            soup = self.retrieve_url(search_url)

            all_atags = soup.find_all('a', {'data-track-action': 'view article'})

            if len(all_atags) == 0:
                break

            for atag in all_atags:
                random.seed(articles_counter)

                if(random.random() < 0.5):
                    self.parse_page(self.base_url + atag['href'], univs=True, collabs=False)
                    print(f"[EXTRACTING UNIVS] {articles_counter} articles and {len(self.universities)} universities found")
                else:
                    self.links_saved.append(atag['href'])
                    print(f"[EXTRACTING UNIVS] {articles_counter} articles found and {len(self.links_saved)} links saved")

                articles_counter = articles_counter + 1            
            page = page + 1

    def extract_collabs(self):
        articles_counter = 1
        page = 1
        while True:
            search_url = f'https://www.nature.com/search?journal=srep&article_type=research&subject=mathematics-and-computing&date_range=last_year&order=relevance&page={page}'
            soup = self.retrieve_url(search_url)

            all_atags = soup.find_all('a', {'data-track-action': 'view article'})

            if len(all_atags) == 0:
                break

            for atag in all_atags:
                random.seed(articles_counter)

                if(random.random() < 0.5):
                    self.parse_page(self.base_url + atag['href'], univs=False, collabs=True)
                    print(f"[EXTRACTING COLLABS] {articles_counter} articles done")
                articles_counter = articles_counter + 1  
            page = page + 1

    def print_stats(self):
        print(f'Number of nodes: {len(self.universities)}')
        links = 0
        links_with_fields = 0
        for u1 in self.collabs:
            for u2 in self.collabs[u1]:
                for u3 in self.collabs[u1][u2]:
                    links_with_fields += 1
                links += 1
        print(f'Number of links (without fields): {links}')
        print(f'Number of links (with fields): {links_with_fields}')

    def parse_page(self, url: str, univs=True, collabs=True):
        soup = self.retrieve_url(url)

        try:
            # Find universities and update dictionnaries
            authors_all = soup.findAll('p', class_='c-article-author-affiliation__address')
            subjects_all = soup.findAll('meta', {'name': 'dc.subject'})
        
            if univs:
                for s in authors_all:        
                    university = s.text        
                    if university not in self.universities:
                        self.universities[university] = 1
                    else:
                        self.universities[university] += 1

            if collabs:
                for s in authors_all:
                    try:
                        university = self.mapping_univs[s.text]

                        if university not in self.collabs:
                            self.collabs[university] = {}

                        alreadyAddedUniv = []
                        for s2 in authors_all:
                            try:
                                other_university = self.mapping_univs[s2.text]
                                if other_university not in alreadyAddedUniv:
                                    for sub in subjects_all:
                                        subject = sub.get('content')
                                        if university != other_university:
                                            if other_university not in self.collabs[university]:
                                                self.collabs[university][other_university] = {}

                                            if subject not in self.collabs[university][other_university]:
                                                self.collabs[university][other_university][subject] = 0

                                            self.collabs[university][other_university][subject] += 1
                                    alreadyAddedUniv.append(other_university)
                            except:
                                continue
                    except:
                        continue

        except Exception as error:
            print("[ERROR] Parse page: ", error) 

    def preprocess_universities(self):
        universities = {}

        for entry, count in self.universities.items():
            # Split the entry by commas and iterate through the parts
            parts = entry.split(',')

            # Look for patterns that match university names
            university = None
            for part in reversed(parts):
                if any(keyword in part for keyword in ["Polytechnic", "NIT", "MIT", "Politecnico", "Escuela", "École", "Institut", "Universitat", "Università", "Universität", "Universidad", "University", "Institute", "Ecole", "Universiti", "Université", "College", "School", "MIT"]):
                    university = part.strip()
                    break

            # If a university name is found, store the entry in the dictionary
            if university:
                if university in universities:
                    universities[university]['init_names'].append(entry)
                    universities[university]['count'] += count
                else:
                    universities[university] = {"init_names": [entry], "count": count}
            else:
                print(f"Not accepted this uni: {entry}")
            
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        university_names = list(universities.keys())  # Use the universities dictionary instead of self.universities.keys()

        embeddings = model.encode(university_names, convert_to_tensor=True)

        similarity_matrix = cosine_similarity(embeddings, embeddings)

        threshold = 0.92
        similar_pairs = []

        for i in range(len(university_names)):
            for j in range(i + 1, len(university_names)):
                if similarity_matrix[i, j] > threshold:
                    similar_pairs.append((university_names[i], university_names[j]))

        similar_groups = []

        for pair in similar_pairs:
            found = False
            for group in similar_groups:
                if pair[0] in group or pair[1] in group:
                    group.add(pair[0])
                    group.add(pair[1])
                    found = True
                    break
            if not found:
                similar_groups.append({pair[0], pair[1]})

        similar_groups = [list(group) for group in similar_groups]

        for gr in similar_groups:
            first_uni = gr[0]

            for uni in gr[1:]:
                if first_uni in universities and uni in universities:
                    if 'init_names' not in universities[first_uni]:
                        universities[first_uni]['init_names'] = []
                    if 'count' not in universities[first_uni]:
                        universities[first_uni]['count'] = 0
                    universities[first_uni]['init_names'].extend(universities[uni].get('init_names', []))
                    universities[first_uni]['count'] = universities.get(first_uni, {}).get('count', 0) + universities.get(uni, {}).get('count', 0)
                    universities.pop(uni)

        universities = {key.strip(): value for key, value in universities.items()}

        self.universities = universities


# -------------------- EXAMPLE -----------------------
# scraper = Scraper()
# scraper.load_universities("universities.json")
# # scraper.extract_universities()
# # print(f"# universities before preprocess: {len(scraper.universities)}")
# # with open('univ_withouth_preprocess.json', 'w') as output_file:
# #     json.dump(scraper.universities, output_file, indent=2)

# # scraper.preprocess_universities()
# # print(f"# universities after preprocess: {len(scraper.universities)}")
# # scraper.save_universities_in_file()

# scraper.create_map()
# scraper.extract_collabs()
# scraper.save_collabs_in_file()
# scraper.save_links_in_file()

scraper = Scraper()
scraper.load_universities("universities.json")
scraper.load_collabs("collabs.json")
scraper.print_stats()

# ------------------------------- TESTING ---------------------

# # universities = {}
# # prev_universities = {}

# # mapping_univs = {}

# # with open('universities.json', 'r') as file:
# #     prev_universities = json.load(file)

# # for new_name in prev_universities.keys():        
# #     for old_name in prev_universities[new_name]['init_names']:
# #         if old_name not in mapping_univs:
# #             mapping_univs[old_name] = new_name

# # print("# of unis in V0:", len(prev_universities))

# # for entry, count in prev_universities.items():
# # # Split the entry by commas and iterate through the parts
# #     parts = entry.split(',')

# #     # Look for patterns that match university names
# #     university = None
# #     for part in reversed(parts):
# #         if "University" in part or "Institute" in part or "Ecole" in part:
# #             university = part
# #             break

# #     # If a university name is found, store the entry in the dictionary
# #     if university:
# #         if university in universities:
# #             universities[university]['init_names'].append(entry)
# #             universities[university]['count'] += count
# #         else:
# #             universities[university] = {"init_names": [entry], "count": count}

# # with open('univv1.json', 'w') as output_file:
# #     json.dump(universities, output_file, indent=2)

# # print("# of unis in V1:", len(universities))

# # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # university_names = list(universities.keys())

# # embeddings = model.encode(university_names, convert_to_tensor=True)

# # similarity_matrix = cosine_similarity(embeddings, embeddings)

# # threshold = 0.92
# # similar_pairs = []

# # for i in range(len(university_names)):
# #     for j in range(i + 1, len(university_names)):
# #         if similarity_matrix[i, j] > threshold:
# #             similar_pairs.append((university_names[i], university_names[j]))

# # similar_groups = []

# # for pair in similar_pairs:
# #     found = False
# #     for group in similar_groups:
# #         if pair[0] in group or pair[1] in group:
# #             group.add(pair[0])
# #             group.add(pair[1])
# #             found = True
# #             break
# #     if not found:
# #         similar_groups.append({pair[0], pair[1]})

# # similar_groups = [list(group) for group in similar_groups]

# # with open('similarities.json', 'w') as output_file:
# #     json.dump(similar_groups, output_file, indent=2)

# # for gr in similar_groups:
# #     first_uni = 0

# #     for uni in gr:
# #         if first_uni == 0:
# #             first_uni = uni
# #         elif uni in universities:
# #             universities[first_uni]['init_names'].extend(universities[uni]['init_names'])
# #             universities[first_uni]['count'] += universities[uni]['count']
# #             universities.pop(uni)

# # print("# of unis in V2:", len(universities))
# # universities = {key.strip(): value for key, value in universities.items()}

# # with open('univv2.json', 'w') as output_file:
# #     json.dump(universities, output_file, indent=2)