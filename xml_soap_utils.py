from bs4 import BeautifulSoup
import re
import os
import requests
from pattern3.web import plaintext

host = vip.1gl.ru

def get_content_from_soap(host, mod_id, doc_id):
    body = """
    <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Body>
        <XDocument xmlns="http://glavbukh.ru.1gl.ru">
          <ModId>{ModId}</ModId>
          <DocId>{DocId}</DocId>
          <Actual>false</Actual>
        </XDocument>
      </soap:Body>
    </soap:Envelope>
    """

    body_text = body.format(ModId=mod_id, DocId=doc_id)
    body = body_text.encode('utf-8')
    session = requests.session()
    session.headers = {"Content-Type": "text/xml; charset=utf-8", "SOAPAction": "http://glavbukh.ru.1gl.ru/XDocument"}
    session.headers.update({"Content-Length": str(len(body))})
    url = 'https://' + host + "/service/glavbukh.ru.asmx" + "?op=PublicationSummary"
    response = session.post(url=url, data=body, verify=False)

    session.close()
    res = response.content.decode("utf-8")

    # Below is a rendering of the page up to the first error.
    if response.status_code != 200:
        print('status_code', response.status_code)
        print('reason', response.reason)
        print(f"system {host} mod {mod_id} doc {doc_id}")
        return response.reason
    return res


def clear_text(text):
    if text != text:
        return ''
    if not text:
        return ''
    alphabet_ru = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    alphabet_en = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = '0123456789'
    other = '-IVX.,'
    filter_char = alphabet_ru + digits + other
    text = re.sub(f"[^{filter_char}\n]", " ", text)
    text = re.sub(" +", " ", text)
    return text


def find_in_soup(soup, select_attrs, find_attrs):
    find_elemen = ''
    for select_attr in select_attrs:
        res = soup.select(select_attr)
        if res:
            for res_element in res:
                find_elemen = find_elemen + plaintext(res_element.get_text(separator=" ")) + ' '
        if find_elemen:
            find_elemen = clear_text(find_elemen)
            return find_elemen

    for find_attr in find_attrs:
        res = soup.find_all(find_attr[0], find_attr[1])
        if res:
            for res_element in res:
                find_elemen = find_elemen + plaintext(res_element.get_text(separator=".")) + ' '
        if find_elemen:
            find_elemen = clear_text(find_elemen)
            return find_elemen
    return find_elemen


def get_title_and_text_from_xml(file_name):
    if not file_name:
        return '', ''

    if not os.path.isfile(file_name):
#         print('Not found file', file_name)
        return '', ''

    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace("</p>", "\n\n</p>")
    soup = BeautifulSoup(content, 'xml')

    # FIND TITLE
    select_attrs = []
    select_attrs.append('docinfo>artname')
    select_attrs.append('articleItem>documentTitle')
    select_attrs.append('docinfo>documentTitle')
    select_attrs.append('docinfo>Title')
    find_attrs = []
    find_attrs.append(['h4', {'class': 'article__mainheader'}])

    title = find_in_soup(soup, select_attrs, find_attrs)
    title = title.strip()

    # FIND TEXT
    select_attrs = []
    select_attrs.append('docinfo>artbody')
    select_attrs.append('xmlcontent')
    select_attrs.append('doc-body')
    select_attrs.append('StepList')
    select_attrs.append('xmlContent')
    select_attrs.append('docinfo')
    find_attrs = []

    text = find_in_soup(soup, select_attrs, find_attrs)
    text = text.strip()

    return title, text
