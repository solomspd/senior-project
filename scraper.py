import requests
from bs4 import BeautifulSoup


URL = "https://en.wikipedia.org/wiki/List_of_Java_bytecode_instructions"

page = requests.get(URL)


soup = BeautifulSoup(page.content, "html.parser")

lists = soup.find("tbody").find_all("tr")[1:]


for element in lists:
    print(element.find("td").text.strip(), end = " ,")



