import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


class DataMaker:
	"""
	Collects data by scrapping the site
	"""

	site_url = None
	first_page = None
	last_page = None
	urls = []
	courses = []
	organizations = []
	learning_products = []
	ratings = []
	num_rated = []
	difficulty = []
	enrolled = []

	def __init__(self, site, first_page, last_page):
		self.site_url = site
		self.first_page = first_page
		self.last_page = last_page

	def scrape_features(self, page_url):
		"""
		Scrapes features from each page

		:param page_url: <str> URL of the page
		"""

		# create the soup with a certain page URL
		course_page = requests.get(page_url)
		course_soup = BeautifulSoup(course_page.content,'html.parser')

		# pick course name
		cnames = course_soup.select(".headline-1-text")
		for i in range(10):
			self.courses.append(cnames[i].text)

		# pick partner name
		pnames = course_soup.select(".horizontal-box > .partner-name")
		for i in range(10):
			self.organizations.append(pnames[i].text)

		# pick URLs
		root = "https://www.coursera.org"
		links = course_soup.select(
			".ais-InfiniteHits > .ais-InfiniteHits-list > .ais-InfiniteHits-item" 
		)
		for i in range(10):
			self.urls.append(root+links[i].a["href"])

		# pick learning product
		for i in range(10):
			learn_pdcts = course_soup.find_all('div', '_jen3vs _1d8rgfy3')
			self.learning_products.append(learn_pdcts[i].text)

		

	def crawler(self):
		"""
		Traverses between the first and last pages
		
		:param base_url: <str> Base url
		"""

		for page in range(self.first_page, self.last_page+1):
			print("\nCrawling Page " + str(page))
			page_url = self.site_url + "?page=" + str(page) +\
			           "&index=prod_all_products_term_optimization"
			
			self.scrape_features(page_url)


	def make_dataset(self):
		"""
		Make the dataset
		"""

		# initiate crawler
		self.crawler()

		data_dict = {
			"Course URL":self.urls,
			"Course Name":self.courses,
			"Learning Product Type":self.learning_products,
			"Course Provided By":self.organizations
		}

		data = pd.DataFrame(data_dict)

		return data
		

def main():

	dm = DataMaker("https://coursera.org/courses", 1,100)
	df = dm.make_dataset()
	destination_path = os.path.join("data/coursera-courses-overview.csv")
	df.to_csv(destination_path, index=False)

if __name__=="__main__":
	main()