import json
import time
import requests
from bs4 import BeautifulSoup

def scrape_content(url: str):
    res = requests.get(url)
    time.sleep(0.33)
    html = res.content

    soup = BeautifulSoup(html, 'html.parser')
    if soup is None:
        return None

    tag = soup.select_one('div[id="summaryContentDiv"]')
    if tag is None:
        return None
    
    return tag.text.strip()


def send_request(index: int):
    API_URL = "https://open.assembly.go.kr/portal/openapi/TVBPMBILL11?"
    q_key = "KEY=69f59dc3512b4941b82e3645c7d4a22a&"
    q_type = "Type=json&"
    q_pIndex = "pIndex=%d&" % index
    q_pSize = "pSize=1000"

    url = API_URL + q_key + q_type + q_pIndex + q_pSize
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'Content-Language': 'ko-KR',
    }

    try:
        response = requests.get(url, headers=headers)

        proposal_list = response.json()["TVBPMBILL11"][1]["row"]

        code = response.json()["TVBPMBILL11"][0]["head"][1]["RESULT"]["CODE"]
        if code == "INFO-200" or code == "INFO-300":
            return None

        for proposal_dict in proposal_list:
            scraped_content_data = scrape_content(proposal_dict['LINK_URL'])
            if scraped_content_data is None:
                continue
            proposal_dict.update({'DETAIL_CONTENT': scraped_content_data})

        return proposal_list

    except Exception as ex:
        print(ex)


documents = {
    "docs": []
}

for i in range(1, 100):
    response = send_request(i)
    if response is None:
        continue
    documents["docs"].extend(response)

with open("law.json", "w", encoding='UTF-8-sig') as json_file:
    json.dump(documents, json_file, ensure_ascii=False)
