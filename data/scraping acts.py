import requests
from bs4 import BeautifulSoup as Soup 
import re
base='https://indiankanoon.org'
invalid_chars_pattern = r'[<>:"/\\|?*]'
for year in range(1999,2023):

    for i in range (1,10):
        url = f"https://indiankanoon.org/search/?formInput=doctypes%3A%20union-act%20fromdate%3A%201-1-{year}%20todate%3A%2031-12-{year}&pagenum={i}"
        req=requests.get(url)
        soup1 = Soup(req.content, 'html.parser')
        results= soup1.find_all(class_="result_title")
        if not results: 
            print("end of pages")
            break
        for result in results:

            title= result.find('a').text
            if len(title)>=255:
                title=title[:240]
            cleantitle = re.sub(invalid_chars_pattern, '', title)
            acturl = base + result.find('a')['href']
            form_data = {"type": "pdf"}
            response = requests.post(acturl, data=form_data)
            if response.status_code == 200:
                with open(f"D:/personal/projects/AI_legal_advisor/Union acts/{cleantitle}.pdf", "wb") as f:
                    f.write(response.content)
                    print("Document downloaded successfully as document.pdf")
            else:
                print("Failed to download the document. Status code:", response.status_code)
